"""
eval/hotpotqa/build_index.py -- build an isolated MindMesh index for a HotpotQA benchmark.

    python eval/hotpotqa/build_index.py --model mistral:latest
    python eval/hotpotqa/build_index.py --model mistral:latest --limit 3   # smoke test

Output (never touches data/cache):
    <bench_dir>/index/<model_slug>/
        extractions.jsonl        MindMesh extractor output, one record per chunk (resumable)
        entities/ relationships/ chunks/     MindMesh Zvec collections
        relationship_records.json            rel_id -> provenance chunk_ids
        graph.graphml  graph_metrics.json  index_meta.json

The SAME collections and graph serve all four benchmark pipelines:
pipeline_a (chunks), pipeline_b (entities + chunks + graph), lightrag_mix /
lightrag_mix_opt (entities + relationships + chunks + graph).

Extraction is resumable: chunks already in extractions.jsonl are skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from mindmesh.benchmarks.hotpotqa.dataset import load_corpus  # noqa: E402
from mindmesh.benchmarks.hotpotqa.env import CHUNKING_TAG, activate_index  # noqa: E402
from mindmesh.benchmarks.hotpotqa.runtime import (  # noqa: E402
    GpuLock, hardware_info, ollama_ps, resolve_local_model, unload_model,
)

DEFAULT_BENCH = ROOT / "data" / "benchmarks" / "hotpotqa" / "dev_distractor_20q_seed42"


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def run_extraction(paras, model, out_jsonl: Path, gleanings: int, limit: int | None) -> dict:
    from graphrag_lite.config import get_config
    from graphrag_lite.ingestion.extractor import extract_entities_relationships
    from graphrag_lite.ingestion.gleaning import glean
    from graphrag_lite.llm.ollama_client import OllamaClient

    cfg = get_config()
    done = set()
    if out_jsonl.exists():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["chunk_id"])
    todo = [p for p in paras if p.chunk_id not in done]
    if limit is not None:
        todo = todo[: max(0, limit - len(done))]
    print(f"[extract] {len(done)} chunks already extracted, {len(todo)} to go (model={model}, gleanings={gleanings})")

    from mindmesh.model_config import think_for
    think = think_for(model)   # qwen3*: False; others: configured default (unchanged)
    client = OllamaClient(
        model=model, temperature=cfg.EXTRACTION_TEMPERATURE, seed=cfg.EXTRACTION_SEED,
        timeout=cfg.LLM_TIMEOUT_S, num_predict=cfg.LLM_NUM_PREDICT, num_ctx=cfg.LLM_NUM_CTX,
        think=think, frequency_penalty=cfg.LLM_FREQUENCY_PENALTY,
    )
    t_all = time.monotonic()
    failures = 0
    with open(out_jsonl, "a", encoding="utf-8") as fh:
        for i, p in enumerate(todo, 1):
            chunk = {"chunk_id": p.chunk_id, "doc_id": p.doc_id, "chunk_index": 0,
                     "text": p.text, "token_count": len(p.text.split())}
            t0 = time.monotonic()
            rec = {"chunk_id": p.chunk_id, "doc_id": p.doc_id, "chunk_index": 0,
                   "text": p.text, "title": p.title, "extraction_model": model}
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    ext = extract_entities_relationships(chunk, client)
                    if gleanings > 0:
                        ext = glean(chunk, ext, client, max_gleanings=gleanings)
                rec["entities"] = ext.get("entities", [])
                rec["relationships"] = ext.get("relationships", [])
                rec["warnings"] = [str(w.message)[:200] for w in caught]
            except Exception as exc:  # noqa: BLE001 -- recorded, never hidden
                failures += 1
                rec["entities"], rec["relationships"] = [], []
                rec["_parse_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            rec["latency_s"] = round(time.monotonic() - t0, 3)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  [{i}/{len(todo)}] {p.title[:40]!r}: {len(rec['entities'])} ent, "
                  f"{len(rec['relationships'])} rel, {rec['latency_s']:.1f}s"
                  + (" PARSE-ERROR" if "_parse_error" in rec else ""), flush=True)
    return {"extracted_now": len(todo), "failures_now": failures,
            "extraction_wall_s": round(time.monotonic() - t_all, 1)}


def run_extraction_openrouter(paras, settings, index_dir: Path, out_jsonl: Path, limit: int | None,
                              workers: int) -> dict:
    """OpenRouter extraction into a provider-specific cache; NO local fallback.

    The cache (one entry per paragraph content hash) is the source of truth and makes the build
    resumable without duplicate requests. extractions.jsonl is rewritten from the cache in corpus
    order after every invocation. A failed paragraph (truncated at max_tokens, invalid JSON after
    retries, provider error) is recorded with _parse_error and contributes no graph records."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from graphrag_lite.retrieval.provider_extraction import OpenRouterExtractionCache
    cache = OpenRouterExtractionCache(index_dir / "extraction_cache", settings)
    todo = [p for p in paras if cache.get(p.text) is None]
    if limit is not None:
        n_done = sum(cache.get(p.text) is not None for p in paras)
        todo = todo[: max(0, limit - n_done)]
    print(f"[extract] openrouter {settings.model}: {len(paras) - len(todo)} cached, {len(todo)} to go, {workers} workers")
    t_all = time.monotonic()
    as_para = lambda p: {"chunk_id": p.chunk_id, "doc_id": p.doc_id, "title": p.title, "text": p.text}  # noqa: E731
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(cache.extract, as_para(p), "offline_build"): p for p in todo}
        for i, f in enumerate(as_completed(futs), 1):
            e = f.result(); p = futs[f]
            print(f"  [{i}/{len(todo)}] {p.title[:40]!r}: {e['status']} {len(e['entities'])} ent, "
                  f"{len(e['relationships'])} rel, {e['request_wall_s']:.1f}s, ${e['cost_usd'] or 0:.5f}", flush=True)
    wall = round(time.monotonic() - t_all, 1)
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for p in paras:
            e = cache.get(p.text)
            if e is None:
                continue
            rec = {"chunk_id": p.chunk_id, "doc_id": p.doc_id, "chunk_index": 0, "text": p.text, "title": p.title,
                   "extraction_model": settings.model, "extraction_provider": "openrouter",
                   "actual_model": e["actual_model"], "upstream_provider": e["upstream_provider"],
                   "cache_version": e["cache_version"], "entities": e["entities"], "relationships": e["relationships"],
                   "latency_s": e["request_wall_s"], "attempts": e["attempts"], "prompt_tokens": e["prompt_tokens"],
                   "completion_tokens": e["completion_tokens"], "cost_usd": e["cost_usd"]}
            if e["status"] != "ok":
                rec["_parse_error"] = f"{e['status']}: {e['error'][:300]}"
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"extracted_now": len(todo), "extraction_wall_s": wall, "workers": workers,
            "cache_dir": str(cache.dir), "cache_version": cache.version}


def graph_metrics(G) -> dict:
    import networkx as nx
    cross_nodes = sum(1 for _, d in G.nodes(data=True) if len(d.get("source_docs", set())) >= 2)
    # an edge is cross-document if its provenance chunks span >1 document, or
    # its endpoints were extracted from disjoint document sets
    cross_edges = 0
    for u, v, d in G.edges(data=True):
        prov_docs = {ss.get("doc_id") for ss in d.get("source_sentences", []) if ss.get("doc_id")}
        ud, vd = G.nodes[u].get("source_docs", set()), G.nodes[v].get("source_docs", set())
        if len(prov_docs) > 1 or (ud and vd and not (ud & vd)):
            cross_edges += 1
    comps = list(nx.weakly_connected_components(G))
    # documents linked through at least one shared entity node
    linked_docs = set()
    for _, d in G.nodes(data=True):
        sd = d.get("source_docs", set())
        if len(sd) >= 2:
            linked_docs |= sd
    return {
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "weak_components": len(comps), "largest_component": max((len(c) for c in comps), default=0),
        "isolated_nodes": sum(1 for n in G if G.degree(n) == 0),
        "cross_document_nodes": cross_nodes, "cross_document_edges": cross_edges,
        "documents_linked_via_shared_entities": len(linked_docs),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="exact local Ollama tag used for entity extraction "
                         "(default: 'extraction' in config/models.json)")
    ap.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--gleanings", type=int, default=0,
                    help="gleaning passes per chunk (default 0 for the benchmark; canonical MindMesh uses 1)")
    ap.add_argument("--limit", type=int, default=None, help="extract at most N chunks in total (smoke test)")
    ap.add_argument("--skip-extraction", action="store_true", help="rebuild graph/collections from existing extractions")
    ap.add_argument("--provider", choices=["ollama", "openrouter"], default="ollama",
                    help="openrouter: model from config/extraction_provider.json, no local fallback")
    ap.add_argument("--workers", type=int, default=8, help="parallel OpenRouter requests")
    args = ap.parse_args(argv)

    args.bench_dir = args.bench_dir.resolve()
    if args.model is None:
        from mindmesh.model_config import load_model_assignments
        args.model = load_model_assignments().extraction
    or_settings = None
    if args.provider == "openrouter":
        from graphrag_lite.retrieval import provider_extraction as PE
        or_settings = PE.load_provider_settings().openrouter
        if not PE.api_key_available(or_settings):
            print(f"[REFUSED] {or_settings.api_key_env} not set", file=sys.stderr)
            return 1
        model = or_settings.model
        model_info = {"provider": "openrouter", "digest": None, "base_url": or_settings.base_url}
    else:
        model, model_info = resolve_local_model(args.model)
    # Leakage guard: offline graph construction reads ONLY corpus.jsonl (paragraph text + IDs),
    # never questions.json, answers or supporting-fact maps.
    paragraphs = load_corpus(args.bench_dir)
    index_dir = args.bench_dir / "index" / (slug(model) if or_settings is None else "openrouter__" + slug(model))
    index_dir.mkdir(parents=True, exist_ok=True)
    activate_index(index_dir)
    from graphrag_lite.config import get_config
    cfg = get_config()
    print(f"[isolation] ZVEC_ROOT={cfg.ZVEC_ROOT}  CACHE_DIR={cfg.CACHE_DIR}")

    ext_path = index_dir / "extractions.jsonl"
    timings = {}
    with GpuLock(f"hotpotqa build_index {model}"):
        if not args.skip_extraction:
            if or_settings is not None:
                timings.update(run_extraction_openrouter(paragraphs, or_settings, index_dir, ext_path,
                                                         args.limit, args.workers))
            else:
                timings.update(run_extraction(paragraphs, model, ext_path, args.gleanings, args.limit))
        ps_after_extract = ollama_ps()
        if or_settings is None:
            unload_model(model)

        records = [json.loads(l) for l in ext_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        clean = [r for r in records if "_parse_error" not in r]
        n_fail = len(records) - len(clean)
        # all chunks are indexed for vector search, even if extraction failed
        extracted_ids = {r["chunk_id"] for r in records}
        chunk_records = list(records) + [
            {"chunk_id": p.chunk_id, "doc_id": p.doc_id, "chunk_index": 0, "text": p.text}
            for p in paragraphs if p.chunk_id not in extracted_ids and args.limit is None
        ]
        if args.limit is not None:
            # smoke test: index only the extracted subset
            chunk_records = list(records)

        from graphrag_lite.graph.builder import build_graph
        buf = io.StringIO()
        t0 = time.monotonic()
        with contextlib.redirect_stdout(buf):
            G = build_graph(clean)
        timings["graph_build_s"] = round(time.monotonic() - t0, 3)
        gm = graph_metrics(G)
        print(f"[graph] {gm}")

        from run_model_experiment import build_experiment_collections
        t0 = time.monotonic()
        coll = build_experiment_collections(G, chunk_records, index_dir, model=model)
        timings["collections_build_s"] = round(time.monotonic() - t0, 3)
        for kind in ("entities", "chunks", "relationships"):
            (index_dir / kind / "chunking_strategy.txt").write_text(CHUNKING_TAG)

        import networkx as nx
        Gx = G.copy()
        for _, d in Gx.nodes(data=True):
            for k in list(d):
                if not isinstance(d[k], (str, int, float, bool)):
                    d[k] = json.dumps(sorted(d[k]) if isinstance(d[k], set) else d[k], ensure_ascii=False, default=str)
        for _, _, d in Gx.edges(data=True):
            for k in list(d):
                if not isinstance(d[k], (str, int, float, bool)):
                    d[k] = json.dumps(d[k], ensure_ascii=False, default=str)
        nx.write_graphml(Gx, index_dir / "graph.graphml")

    ext_lat = [r.get("latency_s", 0) for r in records]
    meta = {
        "benchmark_dir": str(args.bench_dir.relative_to(ROOT)),
        "extraction_model": model,
        "extraction_model_info": model_info,
        "embedding_model": cfg.EMBEDDING_MODEL,
        "embedding_dim": coll.get("embedding_dim"),
        "gleanings": args.gleanings,
        "chunks_total": len(paragraphs),
        "chunks_extracted": len(records),
        "chunks_parse_failed": n_fail,
        "chunks_indexed": coll["chunk_count"],
        "entities_indexed": coll["entity_count"],
        "relationships_indexed": coll["relationship_count"],
        "documents": len({r["doc_id"] for r in chunk_records}),
        "graph": gm,
        "timings": timings,
        "extraction_latency_sum_s": round(sum(ext_lat), 1),
        "extraction_latency_mean_s": round(sum(ext_lat) / len(ext_lat), 2) if ext_lat else 0,
        "ollama_ps_during_extraction": ps_after_extract,
        "hardware": hardware_info(),
        "smoke_test_limit": args.limit,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # Offline build costs, kept separate from any online (query-time) measurement.
    offline = {
        "label": "OFFLINE graph construction (done once, before evaluation); not part of query latency",
        "extraction_model": model, "extraction_model_digest": model_info.get("digest"),
        "inputs_read": ["corpus.jsonl"], "questions_or_labels_read": False,
        "extraction": {"paragraphs": len(records), "failed": n_fail,
                       "time_s_total": round(sum(ext_lat), 1),
                       "time_s_mean_per_paragraph": round(sum(ext_lat) / len(ext_lat), 2) if ext_lat else 0,
                       "this_invocation": {k: v for k, v in timings.items() if k.startswith("extract")}},
        "graph_construction_s": timings.get("graph_build_s"),
        "embedding_and_collections_s": timings.get("collections_build_s"),
        "persisted": ["extractions.jsonl", "graph.graphml", "entities/", "relationships/", "chunks/",
                      "relationship_records.json"],
    }
    if or_settings is not None:
        offline["extraction"].update({
            "provider": "openrouter", "fallback": "none (failures recorded, never substituted)",
            "note": "time_s_total sums per-request wall times; requests ran in parallel, see this_invocation",
            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in records),
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in records),
            "cost_usd": round(sum(r.get("cost_usd") or 0 for r in records), 6),
            "failed_ids": [r["chunk_id"] for r in records if "_parse_error" in r]})
    (index_dir / "offline_build_report.json").write_text(json.dumps(offline, indent=2), encoding="utf-8")
    (index_dir / "graph_metrics.json").write_text(json.dumps(gm, indent=2), encoding="utf-8")
    (index_dir / "index_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("chunks_indexed", "entities_indexed", "relationships_indexed",
                                           "chunks_parse_failed", "graph", "timings")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
