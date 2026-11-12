"""
run_phase2_gleaning.py — Phase 2: qwen2.5:1.5b with gleaning ON.

Isolated experiment. Never touches:
  - data/cache (canonical collections)
  - data/experiments/qwen2_5_1_5b_validate (Phase 1 baseline)
  - eval/gold_test_set_frozen.json

Outputs go to:
  data/experiments/qwen2_5_1_5b_gleaning_on/

Steps:
  1. Read 42 canonical chunks (read-only)
  2. Extract with qwen2.5:1.5b, MAX_GLEANINGS=1, private cache
  3. Build isolated graph; measure metrics
  4. Build Zvec collections (entities, relationships, chunks)
  5. Evaluate all 6 pipelines on 20 dev questions
  6. Write comparison report vs Phase 1 baseline

Usage:
    python eval/run_phase2_gleaning.py
    python eval/run_phase2_gleaning.py --skip-extraction   # reuse existing JSONL
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

EXP_DIR   = ROOT / "data" / "experiments" / "qwen2_5_1_5b_gleaning_on"
CACHE_DIR = str(EXP_DIR / "llm_cache")
PHASE1_JSON = ROOT / "data" / "experiments" / "qwen2_5_1_5b_validate" / "validation_20261109_223804.json"
CANONICAL_EXTRACTIONS = ROOT / "data" / "cache_v2" / "extractions_v2_off.jsonl"  # gleaning-OFF baseline
DEV_SET   = ROOT / "eval" / "gold_dev_set.json"
MODEL     = "qwen2.5:1.5b"
TOP_K     = 6

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("phase2")


# ── Step 1: Read canonical chunks ────────────────────────────────────────────

def read_canonical_chunks() -> list[dict]:
    import zvec
    log.info("Reading 42 canonical chunks from data/cache/chunks_v1 (read-only)...")
    col = zvec.open(str(ROOT / "data" / "cache" / "chunks_v1"))
    try:
        res = col.query(
            queries=[zvec.Query("text_vec", vector=[0.0]*768,
                                param=zvec.HnswQueryParam(ef=300))],
            topk=100,
        )
        chunks = []
        for doc in res:
            f = doc.fields or {}
            source_doc = f.get("source_doc", "")
            chunks.append({
                "chunk_id":    doc.id,
                # Extractor's Chunk TypedDict uses "doc_id"; Zvec stores "source_doc".
                # Keep both keys so downstream code works regardless of which it reads.
                "doc_id":      source_doc,
                "source_doc":  source_doc,
                "chunk_index": int(f.get("chunk_index", 0)),
                "text":        f.get("text", ""),
            })
    finally:
        col.close()
    chunks.sort(key=lambda c: (c["doc_id"], c["chunk_index"]))
    unique = {c["chunk_id"] for c in chunks}
    log.info("Read %d chunks, %d unique IDs", len(chunks), len(unique))
    assert len(unique) == 42, f"Expected 42 unique chunk IDs, got {len(unique)}"
    return chunks


# ── Step 2: Extraction with gleaning ON ──────────────────────────────────────

def run_extraction(chunks: list[dict]) -> tuple[list[dict], dict]:
    """Run extraction + gleaning. Returns (records, stats)."""
    import diskcache
    import graphrag_lite.cache.llm_cache as lc_mod
    from graphrag_lite.config import get_config
    from graphrag_lite.llm.ollama_client import OllamaClient
    from graphrag_lite.ingestion.extractor import extract_entities_relationships
    from graphrag_lite.ingestion.gleaning import glean

    cfg = get_config()
    MAX_GLEANINGS = 1  # explicitly enabled

    os.makedirs(CACHE_DIR, exist_ok=True)
    log.info("Private LLM cache: %s", CACHE_DIR)
    log.info("Model: %s | MAX_GLEANINGS=%d | temperature=%.1f | seed=%d",
             MODEL, MAX_GLEANINGS, cfg.EXTRACTION_TEMPERATURE, cfg.EXTRACTION_SEED)
    log.info("num_predict=%d | num_ctx=%d | fp=%.2f | think=False",
             cfg.LLM_NUM_PREDICT, cfg.LLM_NUM_CTX, cfg.LLM_FREQUENCY_PENALTY)

    # Swap cache to private directory
    old_cache  = lc_mod._cache
    old_hits   = lc_mod._hits
    old_misses = lc_mod._misses
    lc_mod._cache  = diskcache.Cache(CACHE_DIR)
    lc_mod._hits   = 0
    lc_mod._misses = 0

    client = OllamaClient(
        model=MODEL,
        temperature=cfg.EXTRACTION_TEMPERATURE,
        seed=cfg.EXTRACTION_SEED,
        timeout=cfg.LLM_TIMEOUT_S,
        num_predict=cfg.LLM_NUM_PREDICT,
        num_ctx=cfg.LLM_NUM_CTX,
        think=False,               # qwen2.5 family: not a reasoning model, but safe to be explicit
        frequency_penalty=cfg.LLM_FREQUENCY_PENALTY,
    )

    records      = []
    parse_errors = 0
    retry_events = 0
    glean_calls  = 0
    total_start  = time.monotonic()

    try:
        for i, chunk in enumerate(chunks):
            cid = chunk["chunk_id"]
            doc = chunk["source_doc"]
            t0  = time.monotonic()
            log.info("[%02d/%02d] %s  %s", i+1, len(chunks), cid[:12], doc[:20])

            attempt_warnings: list[str] = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    extraction = extract_entities_relationships(chunk, client)
                    for w in caught:
                        attempt_warnings.append(str(w.message))
                        if "retrying" in str(w.message).lower():
                            retry_events += 1

                # Gleaning pass
                glean_added_e = 0
                glean_added_r = 0
                with warnings.catch_warnings(record=True) as caught_g:
                    warnings.simplefilter("always")
                    pre_e = len(extraction.get("entities", []))
                    pre_r = len(extraction.get("relationships", []))
                    extraction = glean(chunk, extraction, client, max_gleanings=MAX_GLEANINGS)
                    glean_added_e = len(extraction.get("entities", [])) - pre_e
                    glean_added_r = len(extraction.get("relationships", [])) - pre_r
                    glean_calls  += 1
                    for w in caught_g:
                        attempt_warnings.append(str(w.message))

                n_e  = len(extraction.get("entities", []))
                n_r  = len(extraction.get("relationships", []))
                dt   = time.monotonic() - t0
                log.info("  -> %d ent / %d rel  (+%d/+%d from glean)  %.1fs  cache h=%d m=%d",
                         n_e, n_r, glean_added_e, glean_added_r, dt,
                         lc_mod._hits, lc_mod._misses)

                records.append({
                    "chunk_id":      cid,
                    "doc_id":        doc,
                    "chunk_index":   chunk["chunk_index"],
                    "text":          chunk["text"],
                    "entities":      extraction.get("entities", []),
                    "relationships": extraction.get("relationships", []),
                    "extraction_model": client.model,
                    "gleaning":      True,
                    "glean_added_entities": glean_added_e,
                    "glean_added_rels":     glean_added_r,
                    "latency_s":     round(dt, 3),
                    "warnings":      attempt_warnings,
                })

            except Exception as exc:
                dt = time.monotonic() - t0
                parse_errors += 1
                log.error("  PARSE ERROR %s: %s (%.1fs)", cid[:12], str(exc)[:100], dt)
                records.append({
                    "chunk_id": cid, "doc_id": doc,
                    "chunk_index": chunk["chunk_index"], "text": chunk["text"],
                    "entities": [], "relationships": [],
                    "extraction_model": client.model, "gleaning": True,
                    "_parse_error": str(exc),
                    "latency_s": round(dt, 3), "warnings": attempt_warnings,
                })

        total_dt = time.monotonic() - total_start
        stats = {
            "chunks": len(chunks), "parse_errors": parse_errors,
            "retry_events": retry_events, "glean_calls": glean_calls,
            "cache_hits": lc_mod._hits, "cache_misses": lc_mod._misses,
            "total_entities": sum(len(r.get("entities", [])) for r in records),
            "total_relationships": sum(len(r.get("relationships", [])) for r in records),
            "total_glean_added_entities": sum(r.get("glean_added_entities", 0) for r in records),
            "total_glean_added_rels": sum(r.get("glean_added_rels", 0) for r in records),
            "total_s": round(total_dt, 1),
        }
        log.info("Extraction complete: %d chunks, %d errors, %d glean calls, %.1fs",
                 len(chunks), parse_errors, glean_calls, total_dt)
        log.info("Entities: %d total (+%d from gleaning)  Rels: %d (+%d)",
                 stats["total_entities"], stats["total_glean_added_entities"],
                 stats["total_relationships"], stats["total_glean_added_rels"])
        log.info("Cache: %d hits, %d misses", lc_mod._hits, lc_mod._misses)
        return records, stats

    finally:
        lc_mod._cache  = old_cache
        lc_mod._hits   = old_hits
        lc_mod._misses = old_misses


# ── Step 3: Build graph ───────────────────────────────────────────────────────

def build_graph(records: list[dict]):
    import io, contextlib, collections
    import networkx as nx
    from graphrag_lite.graph.builder import build_graph as _build

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        G = _build(records)
    for line in buf.getvalue().splitlines():
        if line.strip():
            log.info("  [builder] %s", line)

    # Metrics
    comps = list(nx.weakly_connected_components(G))
    largest = max((len(c) for c in comps), default=0)
    isolated = sum(1 for n in G.nodes() if G.degree(n) == 0)

    cross_doc_nodes = sum(1 for n, d in G.nodes(data=True) if len(d.get("source_docs", set())) >= 2)
    cross_doc_edges = 0
    for u, v, d in G.edges(data=True):
        u_docs = G.nodes[u].get("source_docs", set())
        v_docs = G.nodes[v].get("source_docs", set())
        if u_docs and v_docs and not u_docs.intersection(v_docs):
            cross_doc_edges += 1

    stub_nodes = sum(1 for n, d in G.nodes(data=True) if d.get("stub", False))

    type_counts: dict = collections.Counter()
    for u, v, d in G.edges(data=True):
        for t in d.get("types", []):
            type_counts[t] += 1

    usable_chunks: set = set()
    for u, v, d in G.edges(data=True):
        for ss in d.get("source_sentences", []):
            if ss.get("chunk_id"):
                usable_chunks.add(ss["chunk_id"])

    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": len(comps),
        "largest_component": largest,
        "isolated_nodes": isolated,
        "cross_doc_nodes": cross_doc_nodes,
        "cross_doc_edges": cross_doc_edges,
        "stub_nodes": stub_nodes,
        "usable_chunks": len(usable_chunks),
        "usable_chunk_rate": round(len(usable_chunks) / 42, 4),
        "edge_type_top10": dict(type_counts.most_common(10)),
    }

    log.info("Graph: %d nodes, %d edges, %d components, largest=%d",
             metrics["nodes"], metrics["edges"], metrics["components"], metrics["largest_component"])
    log.info("  cross_doc_nodes=%d  cross_doc_edges=%d  isolated=%d  usable_chunks=%d/%d",
             cross_doc_nodes, cross_doc_edges, isolated, len(usable_chunks), 42)
    return G, metrics


# ── Step 4: Cross-document entity analysis ───────────────────────────────────

def analyze_cross_doc(G, records: list[dict]) -> dict:
    """Investigate cross-doc connections and naming gaps."""
    import collections

    # Entity names per source doc (from extraction records)
    names_by_doc: dict[str, set] = collections.defaultdict(set)
    for rec in records:
        doc = rec.get("doc_id", "")
        for e in rec.get("entities", []):
            name = e.get("name", "").strip().lower()
            if name:
                names_by_doc[doc].add(name)

    # Cross-doc entity overlaps (same normalized name in 2+ docs)
    all_names = {}  # name -> set of docs
    for doc, names in names_by_doc.items():
        for n in names:
            all_names.setdefault(n, set()).add(doc)
    cross_doc_pairs = {n: sorted(docs) for n, docs in all_names.items() if len(docs) >= 2}

    # Graph nodes that appear in 2+ source_docs
    graph_cross_doc = []
    for node, attrs in G.nodes(data=True):
        src_docs = attrs.get("source_docs", set())
        if len(src_docs) >= 2:
            graph_cross_doc.append({"node": node, "docs": sorted(src_docs)})

    log.info("Cross-doc entity name overlaps (by extraction): %d", len(cross_doc_pairs))
    log.info("Graph nodes in 2+ source_docs: %d", len(graph_cross_doc))

    if cross_doc_pairs:
        log.info("Top cross-doc entities:")
        for name, docs in list(cross_doc_pairs.items())[:10]:
            log.info("  %r: %s", name, docs)

    return {
        "cross_doc_name_overlaps": len(cross_doc_pairs),
        "cross_doc_name_examples": list(cross_doc_pairs.items())[:20],
        "graph_cross_doc_nodes": len(graph_cross_doc),
        "graph_cross_doc_node_examples": graph_cross_doc[:10],
    }


# ── Step 5: Build Zvec collections ───────────────────────────────────────────

def build_collections(G, records: list[dict], exp_dir: Path) -> dict:
    sys.path.insert(0, str(ROOT / "eval"))
    from run_model_experiment import build_experiment_collections
    log.info("Building isolated Zvec collections in %s", exp_dir)
    meta = build_experiment_collections(G, records, exp_dir, model=MODEL)
    log.info("Collections: entities=%d, rels=%d, chunks=%d",
             meta["entity_count"], meta["relationship_count"], meta["chunk_count"])
    return meta


# ── Step 6: Evaluate all 6 pipelines ─────────────────────────────────────────

def recall_at_k(ids: list[str], gold: list[str]) -> float:
    if not gold: return 1.0
    return sum(1 for g in gold if g in ids) / len(gold)

def mrr(ids: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for i, cid in enumerate(ids, 1):
        if cid in gold_set: return 1.0 / i
    return 0.0

def precision_at_k(ids: list[str], gold: list[str]) -> float:
    if not ids: return 0.0
    return sum(1 for cid in ids if cid in gold) / len(ids)

def run_evaluation(G, pr: dict, exp_dir: Path) -> dict:
    from graphrag_lite.retrieval.lightrag import (
        retrieve_local, retrieve_global, retrieve_hybrid, retrieve_mix
    )
    from graphrag_lite.retrieval.pipeline_a import retrieve_flat
    from graphrag_lite.retrieval.pipeline_b import retrieve_structural

    with open(DEV_SET, encoding="utf-8") as f:
        queries = json.load(f)["queries"]

    PIPELINES = ["pipeline_a", "pipeline_b",
                 "lightrag_local", "lightrag_global",
                 "lightrag_hybrid", "lightrag_mix"]

    recall_acc   = {p: [] for p in PIPELINES}
    mrr_acc      = {p: [] for p in PIPELINES}
    prec_acc     = {p: [] for p in PIPELINES}
    latency_acc  = {p: [] for p in PIPELINES}
    errors       = {p: 0  for p in PIPELINES}
    unique_gold  = {p: set() for p in PIPELINES}
    per_question = []

    log.info("Evaluating %d queries across 6 pipelines (top_k=%d)...", len(queries), TOP_K)

    for i, q in enumerate(queries, 1):
        qid       = q["query_id"]
        text      = q["query"]
        gold_ids  = q.get("gold_chunk_ids", [])
        q_type    = q.get("type", "single-chunk")
        log.info("[%02d/%02d] %s (%s)", i, len(queries), qid, q_type)

        row: dict = {"query_id": qid, "type": q_type, "gold_chunk_ids": gold_ids}

        for pl in PIPELINES:
            t0  = time.monotonic()
            ids = []
            err = None
            try:
                if pl == "pipeline_a":
                    ev = retrieve_flat(text)
                    ids = [e.chunk_id for e in ev[:TOP_K]]
                elif pl == "pipeline_b":
                    ev = retrieve_structural(text, top_k=TOP_K)
                    ids = [e.chunk_id for e in ev]
                elif pl == "lightrag_local":
                    ev = retrieve_local(text, G, pr, exp_dir, top_k=TOP_K)
                    ids = [e["chunk_id"] for e in ev]
                elif pl == "lightrag_global":
                    ev = retrieve_global(text, G, exp_dir, top_k=TOP_K)
                    ids = [e["chunk_id"] for e in ev]
                elif pl == "lightrag_hybrid":
                    ev = retrieve_hybrid(text, G, pr, exp_dir, top_k=TOP_K)
                    ids = [e["chunk_id"] for e in ev]
                elif pl == "lightrag_mix":
                    ev = retrieve_mix(text, G, pr, exp_dir, top_k=TOP_K)
                    ids = [e["chunk_id"] for e in ev]
            except Exception as exc:
                err = str(exc)
                errors[pl] += 1
                log.warning("  %s ERROR: %s", pl, err[:80])

            lat  = round(time.monotonic() - t0, 3)
            rec  = recall_at_k(ids, gold_ids)
            m    = mrr(ids, gold_ids)
            prec = precision_at_k(ids, gold_ids)

            recall_acc[pl].append(rec)
            mrr_acc[pl].append(m)
            prec_acc[pl].append(prec)
            latency_acc[pl].append(lat)
            for g in gold_ids:
                if g in ids:
                    unique_gold[pl].add(g)

            row[pl] = {"ids": ids, "recall": round(rec, 4),
                       "mrr": round(m, 4), "precision": round(prec, 4),
                       "latency_s": lat, "error": err}

        per_question.append(row)

        # Log D03, D17, D18 explicitly
        if qid in ("D03", "D17", "D18"):
            log.info("  ** %s focal query:", qid)
            for pl in PIPELINES:
                r = row[pl]
                log.info("    %-20s recall=%.2f  ids=%s", pl, r["recall"], r["ids"][:3])

    n = len(queries)
    aggregates = {}
    for pl in PIPELINES:
        avg_rec  = sum(recall_acc[pl]) / n
        avg_mrr  = sum(mrr_acc[pl]) / n
        avg_prec = sum(prec_acc[pl]) / n
        avg_lat  = sum(latency_acc[pl]) / n
        ug       = len(unique_gold[pl])
        aggregates[pl] = {
            "avg_recall_at_k":  round(avg_rec, 4),
            "mrr":              round(avg_mrr, 4),
            "avg_precision_at_k": round(avg_prec, 4),
            "unique_gold_recovered": ug,
            "avg_latency_s":    round(avg_lat, 4),
            "errors":           errors[pl],
        }

    return {"per_question": per_question, "aggregates": aggregates}


# ── Step 7: Comparison report ─────────────────────────────────────────────────

def build_comparison(eval_result: dict) -> dict:
    if not PHASE1_JSON.exists():
        log.warning("Phase 1 validation JSON not found; skipping comparison.")
        return {}

    with open(PHASE1_JSON, encoding="utf-8") as f:
        p1 = json.load(f)

    p1_agg = p1.get("aggregates", {})
    p2_agg = eval_result["aggregates"]

    PIPELINES = ["pipeline_a", "pipeline_b",
                 "lightrag_local", "lightrag_global",
                 "lightrag_hybrid", "lightrag_mix"]
    rows = []
    for pl in PIPELINES:
        a1 = p1_agg.get(pl, {})
        a2 = p2_agg.get(pl, {})
        rows.append({
            "pipeline": pl,
            "p1_recall": a1.get("avg_recall_at_k", None),
            "p2_recall": a2.get("avg_recall_at_k", None),
            "p1_mrr":    a1.get("mrr", None),
            "p2_mrr":    a2.get("mrr", None),
            "p1_ug":     a1.get("unique_gold_chunks_recovered", a1.get("unique_gold_recovered", None)),
            "p2_ug":     a2.get("unique_gold_recovered", None),
        })

    print()
    print("=" * 100)
    print("PHASE 1 (gleaning OFF) vs PHASE 2 (gleaning ON) — qwen2.5:1.5b, top_k=6, 20 dev questions")
    print("=" * 100)
    fmt = "{:<22} {:>10} {:>10} {:>6}  {:>8} {:>8} {:>6}  {}"
    print(fmt.format("Pipeline", "Recall@1", "Recall@2", "Delta",
                     "MRR@1", "MRR@2", "Delta", "UniqueGold (1→2)"))
    print("-" * 100)
    for r in rows:
        dr = (r["p2_recall"] - r["p1_recall"]) if r["p1_recall"] is not None and r["p2_recall"] is not None else None
        dm = (r["p2_mrr"] - r["p1_mrr"]) if r["p1_mrr"] is not None and r["p2_mrr"] is not None else None
        dr_s = f"{dr:+.4f}" if dr is not None else "N/A"
        dm_s = f"{dm:+.4f}" if dm is not None else "N/A"
        ug_s = f"{r['p1_ug']}→{r['p2_ug']}" if r["p1_ug"] is not None else f"?→{r['p2_ug']}"
        p1r = f"{r['p1_recall']:.4f}" if r["p1_recall"] is not None else "N/A"
        p2r = f"{r['p2_recall']:.4f}" if r["p2_recall"] is not None else "N/A"
        p1m = f"{r['p1_mrr']:.4f}" if r["p1_mrr"] is not None else "N/A"
        p2m = f"{r['p2_mrr']:.4f}" if r["p2_mrr"] is not None else "N/A"
        print(fmt.format(r["pipeline"], p1r, p2r, dr_s, p1m, p2m, dm_s, ug_s))
    print("=" * 100)

    # D18 specific
    pq1 = {r["query_id"]: r for r in p1.get("results", {}).get("a", []) + p1.get("results", {}).get("b", [])}
    pq2 = {r["query_id"]: r for r in eval_result["per_question"]}
    d18_p2 = pq2.get("D18", {})
    print()
    print("D18 (cross-doc paraphrase) — gold: 895ee8b439ed7d7b")
    for pl in PIPELINES:
        r = d18_p2.get(pl, {})
        hit = any(g in r.get("ids", []) for g in ["895ee8b439ed7d7b"])
        print(f"  {pl:<22}  gold_retrieved={hit}  recall={r.get('recall', 0):.2f}  ids={r.get('ids', [])[:3]}")

    return {"comparison_rows": rows}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2: gleaning-ON extraction + evaluation")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Load existing extractions.jsonl instead of re-running")
    parser.add_argument("--skip-collections", action="store_true",
                        help="Reuse existing Zvec collections")
    args = parser.parse_args()

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging to file
    fh = logging.FileHandler(str(EXP_DIR / "experiment.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("=== PHASE 2: qwen2.5:1.5b with gleaning ON ===")
    log.info("Output dir: %s", EXP_DIR)
    t_total = time.monotonic()

    # Step 1
    chunks = read_canonical_chunks()

    # Step 2 — extraction
    ext_path = EXP_DIR / "extractions.jsonl"
    runtime: dict = {}

    if args.skip_extraction and ext_path.exists():
        log.info("Loading existing extractions from %s", ext_path)
        records = []
        with open(ext_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    for c in chunks:
                        if c["chunk_id"] == rec.get("chunk_id"):
                            rec["text"] = c["text"]
                            break
                    records.append(rec)
        log.info("Loaded %d records", len(records))
        extraction_stats: dict = {}
    else:
        t_ext = time.monotonic()
        records, extraction_stats = run_extraction(chunks)
        runtime["extraction_s"] = round(time.monotonic() - t_ext, 1)

        with open(ext_path, "w", encoding="utf-8") as f:
            for rec in records:
                out = {k: v for k, v in rec.items() if k != "text"}
                f.write(json.dumps(out, ensure_ascii=False, default=list) + "\n")
        log.info("Saved extractions to %s", ext_path)

    # Step 3 — graph
    t_graph = time.monotonic()
    G, graph_metrics = build_graph(records)
    runtime["graph_s"] = round(time.monotonic() - t_graph, 1)

    from graphrag_lite.graph.ranking import rank_pagerank
    pr     = rank_pagerank(G)
    pr_max = max(pr.values()) if pr else 1.0

    # Step 4 — cross-doc analysis
    cross_doc_analysis = analyze_cross_doc(G, records)

    # Save graph metrics
    with open(EXP_DIR / "graph_metrics.json", "w", encoding="utf-8") as f:
        json.dump({**graph_metrics, **cross_doc_analysis}, f, indent=2, ensure_ascii=False, default=list)

    # Save GraphML
    try:
        import networkx as nx
        G_copy = G.copy()
        for node, attrs in G_copy.nodes(data=True):
            if "source_docs" in attrs and isinstance(attrs["source_docs"], set):
                G_copy.nodes[node]["source_docs"] = sorted(attrs["source_docs"])
            if "descriptions" in attrs:
                G_copy.nodes[node]["descriptions"] = json.dumps(attrs["descriptions"])
            if "source_sentences" in attrs:
                G_copy.nodes[node]["source_sentences"] = json.dumps(attrs["source_sentences"])
        nx.write_graphml(G_copy, str(EXP_DIR / "graph.graphml"))
        log.info("Saved graph.graphml")
    except Exception as exc:
        log.warning("Could not save GraphML: %s", exc)

    # Step 5 — collections
    if args.skip_collections and (EXP_DIR / "entities").exists():
        log.info("Reusing existing collections")
        coll_meta = {"skipped": True}
    else:
        t_coll = time.monotonic()
        coll_meta = build_collections(G, records, EXP_DIR)
        runtime["collections_s"] = round(time.monotonic() - t_coll, 1)

    # Step 6 — evaluate
    t_eval = time.monotonic()
    eval_result = run_evaluation(G, pr, EXP_DIR)
    runtime["eval_s"] = round(time.monotonic() - t_eval, 1)

    with open(EXP_DIR / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)

    # Step 7 — comparison
    comparison = build_comparison(eval_result)

    # Print aggregate table
    agg = eval_result["aggregates"]
    print()
    print("=" * 80)
    print("PHASE 2 AGGREGATE RESULTS — gleaning ON, qwen2.5:1.5b, top_k=6")
    print("=" * 80)
    fmt = "{:<22} {:>10} {:>8} {:>10} {:>12} {:>6}"
    print(fmt.format("Pipeline", "Recall@K", "MRR", "UniqueGold", "AvgLat(s)", "Errors"))
    print("-" * 70)
    for pl, a in agg.items():
        print(fmt.format(pl, f"{a['avg_recall_at_k']:.4f}", f"{a['mrr']:.4f}",
                         str(a["unique_gold_recovered"]),
                         f"{a['avg_latency_s']:.3f}", str(a["errors"])))
    print("=" * 80)

    # Per-question table
    print()
    print("Per-question recall (A / B / Local / Global / Hybrid / Mix):")
    for row in eval_result["per_question"]:
        vals = " / ".join(f"{row[pl]['recall']:.2f}" for pl in
                          ["pipeline_a","pipeline_b","lightrag_local",
                           "lightrag_global","lightrag_hybrid","lightrag_mix"])
        print(f"  {row['query_id']}  gold={len(row['gold_chunk_ids'])}  {vals}")

    # Save runtime
    runtime["total_s"] = round(time.monotonic() - t_total, 1)
    runtime["extraction_stats"] = extraction_stats
    runtime["graph_metrics"] = graph_metrics
    with open(EXP_DIR / "runtime_metrics.json", "w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2, ensure_ascii=False)

    # Save config snapshot
    from graphrag_lite.config import get_config
    cfg = get_config()
    cfg_snap = {
        "model": MODEL, "gleaning": True, "max_gleanings": 1,
        "extraction_temperature": cfg.EXTRACTION_TEMPERATURE,
        "extraction_seed": cfg.EXTRACTION_SEED,
        "llm_num_predict": cfg.LLM_NUM_PREDICT,
        "llm_num_ctx": cfg.LLM_NUM_CTX,
        "llm_frequency_penalty": cfg.LLM_FREQUENCY_PENALTY,
        "llm_retry_penalties": cfg.LLM_RETRY_FREQUENCY_PENALTIES,
        "embedding_model": cfg.EMBEDDING_MODEL,
        "top_k": TOP_K,
        "exp_dir": str(EXP_DIR),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(EXP_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_snap, f, indent=2)

    log.info("=== Phase 2 complete. Total: %.1fs. Results at %s ===",
             runtime["total_s"], EXP_DIR)
    print(f"\nResults saved to {EXP_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
