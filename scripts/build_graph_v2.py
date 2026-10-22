"""
build_graph_v2.py -- build graph v2 from the ladder arm, with and without
the conditional endpoint filter, into data/cache_v2 with its own ZVEC_ROOT.

Work-order step 4 (2026-10-22).  The promotion rule it is measured against
is DOCUMENT.md Item 12, which was committed BEFORE this script was run.

WHAT THIS BUILDS
    arm "off"  the endpoint filter as it is today (the dangling edge is
               dropped)          -> data/cache_v2/entities_v2
    arm "on"   the conditional filter (the endpoint is recovered as a
               stub entity when its name is in the chunk text)
                                   -> data/cache_v2/entities_v2_stub
    both arms share one chunks collection (the chunk texts do not depend on
    the flag)                       -> data/cache_v2/chunks_v2

SOURCE
    The ladder arm's own cached raw responses
    (CACHE_DIR=data/cache/exp_reextract_ladder).  The raw responses are
    re-parsed twice -- once per arm -- so the two arms are the SAME
    responses with the flag as the only difference.  NO NEW LLM CALLS.

    The ladder arm's first-pass extraction only: its gleaning pass was
    driven by a first-pass result that differs between the arms, so
    including it would make the two arms incomparable and would cost
    fresh LLM calls.  This is stated in the report as a confound.

CANONICAL IS NEVER TOUCHED
    The canonical graph is re-measured read-only from
    data/cache/extractions.jsonl with the same code, so the comparison is
    like-for-like.  Nothing under data/cache/ is written, moved or deleted.

Run:  python scripts/build_graph_v2.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ------------------------------------------------------------------ #
# The ladder arm's own cache.  Read-only.  NEVER the canonical cache.
# ------------------------------------------------------------------ #
LADDER_CACHE = "data/cache/exp_reextract_ladder"
LADDER_RECORDS = "data/cache/exp_reextract_ladder.jsonl"
V2_ROOT = "data/cache_v2"

os.environ["CACHE_DIR"] = LADDER_CACHE
os.environ["ZVEC_ROOT"] = V2_ROOT
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
os.environ["EXTRACTION_MODEL"] = "qwen2.5:1.5b"
os.environ["LLM_NUM_PREDICT"] = "4096"
os.environ["LLM_NUM_CTX"] = "8192"
os.environ["LLM_THINK"] = "off"
os.environ["EXTRACTION_STUB_ENDPOINTS"] = "0"       # the flag is passed per arm
os.environ["ZVEC_COLLECTION_CHUNKS"] = "chunks_v2"

import diskcache  # noqa: E402

from graphrag_lite.config import get_config  # noqa: E402

get_config.cache_clear()

import networkx as nx  # noqa: E402

from graphrag_lite.cache.llm_cache import _make_key  # noqa: E402
from graphrag_lite.embeddings.embedder import embed_batch  # noqa: E402
from graphrag_lite.graph.builder import build_graph, load_extractions  # noqa: E402
from graphrag_lite.graph.ranking import rank_pagerank  # noqa: E402
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer  # noqa: E402
from graphrag_lite.ingestion.extractor import (  # noqa: E402
    _EXTRACTION_PROMPT_TEMPLATE,
    _extract_json,
    _parse_extraction,
)
from graphrag_lite.ingestion.loader import load_documents  # noqa: E402
from graphrag_lite.vectorstore.zvec_client import (  # noqa: E402
    insert_chunks,
    insert_entities,
    reset_collection,
)

PENALTIES = (0.25, 0.5, 1.0)


def tag(penalty: float) -> str:
    return f"t=0.0;seed=42;np=4096;ctx=8192;fp={penalty};think=False"


# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #

def graph_metrics(G: nx.DiGraph, total_chunks: int) -> dict:
    """
    Every number the work order asks for, measured the same way for every
    graph so the comparison is like-for-like.

    cross-document node : a node whose source_docs spans more than one doc
    cross-document edge : the two endpoints share NO common document
                          (the edge cannot be attributed to a single doc)
    usable chunk        : a chunk that contributes >= 1 edge to the graph
                          (read off the edges' own source_sentences, so
                          self-loops and dropped edges are excluded)
    """
    weak = nx.weakly_connected_components(G)
    comps = list(weak)
    largest = max((len(c) for c in comps), default=0)

    cross_nodes = [n for n, d in G.nodes(data=True) if len(d.get("source_docs", ())) > 1]
    singletons = [n for n in G.nodes if G.degree(n) == 0]

    cross_edges = 0
    for s, t in G.edges():
        sd = G.nodes[s].get("source_docs", set())
        td = G.nodes[t].get("source_docs", set())
        if not (sd & td):
            cross_edges += 1

    type_counts: Counter = Counter()
    for _, _, ed in G.edges(data=True):
        for t in ed.get("types", []):
            type_counts[t] += 1

    usable_chunks = set()
    for _, _, ed in G.edges(data=True):
        for ss in ed.get("source_sentences", []):
            usable_chunks.add(ss.get("chunk_id"))

    stub_nodes = [n for n, d in G.nodes(data=True) if d.get("stub")]

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": len(comps),
        "largest_component": largest,
        "singletons": len(singletons),
        "cross_document_nodes": len(cross_nodes),
        "cross_document_edges": cross_edges,
        "stub_nodes": len(stub_nodes),
        "usable_chunks": len(usable_chunks),
        "total_chunks": total_chunks,
        "usable_chunk_pct": (100.0 * len(usable_chunks) / total_chunks) if total_chunks else 0.0,
        "type_shares": dict(type_counts.most_common()),
        "n_types": len(type_counts),
    }


def entity_texts_for(G: nx.DiGraph) -> tuple[list[dict], list[str]]:
    """The same text construction the canonical entities_v1 was built with."""
    entity_dicts = []
    for node in G.nodes():
        nd = dict(G.nodes[node])
        nd["name"] = node
        entity_dicts.append(nd)
    texts = []
    for ed in entity_dicts:
        d = ed.get("descriptions", [])
        desc = d[0]["text"] if d else ""
        texts.append(f"{ed['name']}: {desc}" if desc else ed["name"])
    return entity_dicts, texts


# ------------------------------------------------------------------ #
# Rebuild the ladder arm's extractions, once per arm
# ------------------------------------------------------------------ #

def build_arm_records() -> tuple[dict[str, list[dict]], dict[str, dict], list[dict]]:
    """
    Return (arms, records_by_chunk, chunks).

    arms["off"] / arms["on"] are lists of extraction records, one per
    chunk, built from the SAME cached raw responses.
    """
    cfg = get_config()
    model = cfg.effective_extraction_model()
    cache = diskcache.Cache(LADDER_CACHE)

    tokenizer = get_tokenizer(cfg.TOKENIZER)
    chunks: list[dict] = []
    for doc in load_documents(str(ROOT / "data" / "raw")):
        chunks.extend(chunk_document(doc, cfg, tokenizer))
    by_id = {c["chunk_id"]: c for c in chunks}

    recs: dict[str, dict] = {}
    with (ROOT / LADDER_RECORDS).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["chunk_id"]] = r

    arms: dict[str, list[dict]] = {"off": [], "on": []}
    missing = 0
    for cid, rec in recs.items():
        chunk = by_id.get(cid)
        if chunk is None:
            missing += 1
            continue
        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
        raw = None
        for pen in PENALTIES:
            candidate = cache.get(_make_key(prompt, model, tag(pen)), default=None)
            if candidate is None:
                continue
            try:
                _extract_json(candidate)
            except Exception:
                continue
            raw = candidate
            break
        if raw is None:
            missing += 1
            continue
        obj = _extract_json(raw)
        for arm, allow in (("off", False), ("on", True)):
            res = _parse_extraction(obj, cid, rec["doc_id"],
                                    chunk_text=chunk["text"],
                                    allow_stub_endpoints=allow)
            arms[arm].append({
                "chunk_id": cid,
                "doc_id": rec["doc_id"],
                "entities": res["entities"],
                "relationships": res["relationships"],
            })
    if missing:
        print(f"[warn] {missing} ladder records had no usable cached response")
    return arms, recs, chunks


def main() -> None:
    warnings.simplefilter("ignore", UserWarning)
    t_start = time.perf_counter()

    cfg = get_config()
    v2_dir = ROOT / V2_ROOT
    v2_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v2] ZVEC_ROOT      = {V2_ROOT}")
    print(f"[v2] CACHE_DIR      = {LADDER_CACHE}  (read-only, the ladder arm's own)")
    print(f"[v2] ZVEC_ROOT      = {cfg.ZVEC_ROOT}")

    # ---- 1. canonical, re-measured with the same code ---------------- #
    print()
    print("[v2] re-measuring the canonical graph (read-only) ...")
    canon_metrics = {}
    for name, path in (("canonical (extractions.jsonl)", "data/cache/extractions.jsonl"),
                       ("canonical-alt (extractions_v2.jsonl)", "data/cache/extractions_v2.jsonl")):
        p = ROOT / path
        if not p.exists():
            continue
        recs = load_extractions(str(p))
        G = build_graph(recs)
        canon_metrics[name] = graph_metrics(G, total_chunks=len(recs))
        canon_metrics[name]["_graph"] = G
        print(f"       {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ---- 2. the two v2 arms ------------------------------------------- #
    arms, ladder_recs, chunks = build_arm_records()
    n_chunks = len(arms["off"])
    print(f"[v2] rebuilt {n_chunks} chunks of extraction records per arm "
          f"(same cached raw responses, no new LLM calls)")

    results = {}
    for arm, collection in (("off", "entities_v2"), ("on", "entities_v2_stub")):
        t0 = time.perf_counter()
        records = arms[arm]
        out_path = v2_dir / f"extractions_v2_{arm}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

        G = build_graph(records)
        m = graph_metrics(G, total_chunks=n_chunks)
        m["extraction_seconds"] = sum(
            float(ladder_recs[r["chunk_id"]].get("seconds") or 0.0) for r in records
        )
        m["_graph"] = G

        # ---- 3. Zvec entities collection for this arm ----------------- #
        entity_dicts, texts = entity_texts_for(G)
        vecs = embed_batch(texts)
        pr = rank_pagerank(G)

        object.__setattr__(cfg, "ZVEC_COLLECTION_ENTITIES", collection)
        epath = str(v2_dir / collection)
        reset_collection(epath)
        insert_entities(entity_dicts, vecs, pagerank_scores=pr)

        m["zvec_entities"] = collection
        m["build_seconds"] = time.perf_counter() - t0
        results[arm] = m
        print(f"[v2] arm={arm}: {m['nodes']} nodes, {m['edges']} edges, "
              f"largest={m['largest_component']}, usable chunks="
              f"{m['usable_chunks']}/{m['total_chunks']} "
              f"({m['usable_chunk_pct']:.1f}%), collection={collection}, "
              f"{m['build_seconds']:.1f}s")

    # ---- 4. the chunks collection (shared by both arms) --------------- #
    t0 = time.perf_counter()
    object.__setattr__(cfg, "ZVEC_COLLECTION_ENTITIES", "entities_v2")
    chunk_vecs = embed_batch([c["text"] for c in chunks])
    cpath = str(v2_dir / "chunks_v2")
    reset_collection(cpath)
    insert_chunks(chunks, chunk_vecs)
    chunk_seconds = time.perf_counter() - t0
    print(f"[v2] chunks_v2: {len(chunks)} chunks, {chunk_seconds:.1f}s")

    # ---- 5. the report ------------------------------------------------ #
    summary = {
        "generated": "step 4 of the 2026-10-22 work order",
        "zvec_root": V2_ROOT,
        "canonical": {name: {k: v for k, v in m.items() if k != "_graph"}
                      for name, m in canon_metrics.items()},
        "arms": {},
        "chunks_collection": {"name": "chunks_v2", "count": len(chunks),
                              "seconds": chunk_seconds},
        "total_seconds": time.perf_counter() - t_start,
    }
    for arm, m in results.items():
        summary["arms"][arm] = {k: v for k, v in m.items() if k != "_graph"}

    out_json = v2_dir / "graph_v2_report.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 6. the table -------------------------------------------------- #
    print()
    print("=" * 96)
    print("GRAPH v2 vs CANONICAL")
    print("=" * 96)
    canon = canon_metrics.get("canonical (extractions.jsonl)", {})
    rows = [("canonical", canon)] + [(f"v2 {arm}", results[arm]) for arm in ("off", "on")]
    hdr = ("metric",) + tuple(n for n, _ in rows)
    print(f"{hdr[0]:<28}" + "".join(f"{h:>20}" for h in hdr[1:]))
    print("-" * 96)
    for key, label in (
        ("nodes", "nodes"),
        ("edges", "edges"),
        ("components", "components"),
        ("largest_component", "largest component"),
        ("singletons", "singletons"),
        ("cross_document_nodes", "cross-doc nodes"),
        ("cross_document_edges", "cross-doc edges"),
        ("stub_nodes", "stub nodes"),
        ("usable_chunks", "usable chunks"),
        ("usable_chunk_pct", "usable chunk %"),
        ("n_types", "relationship types"),
    ):
        print(f"{label:<28}" + "".join(f"{m.get(key, 0):>20}" for _, m in rows))
    print("-" * 96)
    print(f"{'usable chunks (n/total)':<28}" + "".join(
        f"{m.get('usable_chunks', 0)}/{m.get('total_chunks', 0)}".rjust(20)
        for _, m in rows))
    print(f"{'extraction seconds (LLM)':<28}" + "".join(
        f"{m.get('extraction_seconds', 0):>20.1f}" for _, m in rows))
    print(f"{'build seconds (this script)':<28}" + "".join(
        f"{m.get('build_seconds', 0):>20.1f}" for _, m in rows))
    print()
    for name, m in rows:
        print(f"--- {name}: relationship type shares ---")
        tot = sum(m.get("type_shares", {}).values()) or 1
        for t, c in sorted(m.get("type_shares", {}).items(), key=lambda kv: -kv[1]):
            print(f"    {t:<24}{c:>5}  {100.0 * c / tot:>5.1f}%")
    print()
    print(f"[v2] report written to {out_json}")
    print(f"[v2] total {summary['total_seconds']:.1f}s")


if __name__ == "__main__":
    main()
