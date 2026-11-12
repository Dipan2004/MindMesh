"""
validate_pipelines.py — Phase 1 validation: run all 6 pipelines against dev set
using existing canonical extractions (no new LLM calls needed).

This uses data/cache_v2/extractions_v2_off.jsonl (qwen2.5:1.5b, 42 clean records)
to build an isolated experiment graph and Zvec collections, then tests:
  - Pipeline A  (canonical flat vector, unmodified)
  - Pipeline B  (canonical graph retrieval, unmodified)
  - LightRAG Local
  - LightRAG Global
  - LightRAG Hybrid
  - LightRAG Mix

Usage:
    python eval/validate_pipelines.py
    python eval/validate_pipelines.py --model qwen2.5:1.5b
    python eval/validate_pipelines.py --fresh   # rebuild collections even if they exist
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import shutil
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("validate")

EXTRACTIONS_PATH = ROOT / "data" / "cache_v2" / "extractions_v2_off.jsonl"
EXP_DIR = ROOT / "data" / "experiments" / "qwen2_5_1_5b_validate"
DEV_SET = ROOT / "eval" / "gold_dev_set.json"


# ── Load extractions ──────────────────────────────────────────────────────────

def load_extractions(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "_parse_error" not in r:
                records.append(r)
    log.info("Loaded %d clean extraction records from %s", len(records), path)
    return records


# ── Attach chunk texts from canonical store ───────────────────────────────────

def attach_chunk_texts(extractions: list[dict]) -> list[dict]:
    """Add 'text' field from canonical chunks_v1 to each extraction record."""
    import zvec
    col = zvec.open(str(ROOT / "data" / "cache" / "chunks_v1"))
    try:
        res = col.query(
            queries=[zvec.Query("text_vec", vector=[0.0] * 768,
                                param=zvec.HnswQueryParam(ef=300))],
            topk=100,
        )
        text_map = {doc.id: (doc.fields or {}).get("text", "") for doc in res}
    finally:
        col.close()

    attached = 0
    for r in extractions:
        cid = r.get("chunk_id", "")
        if cid in text_map:
            r["text"] = text_map[cid]
            attached += 1
        elif "text" not in r:
            r["text"] = ""
    log.info("Attached text to %d/%d records from canonical chunks_v1", attached, len(extractions))
    return extractions


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph(extractions: list[dict]):
    import io, contextlib
    from graphrag_lite.graph.builder import build_graph as _build
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        G = _build(extractions)
    for line in buf.getvalue().splitlines():
        if line.strip():
            log.info("  [builder] %s", line)
    log.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


# ── Build Zvec collections ────────────────────────────────────────────────────

def build_collections(G, extractions: list[dict], exp_dir: Path):
    from graphrag_lite.retrieval.lightrag.context import query_collection_direct
    # Delegate to the experiment runner's Stage E
    sys.path.insert(0, str(ROOT / "eval"))
    from run_model_experiment import build_experiment_collections
    meta = build_experiment_collections(G, extractions, exp_dir, model="qwen2.5:1.5b")
    log.info("Collections: entities=%d, rels=%d, chunks=%d",
             meta["entity_count"], meta["relationship_count"], meta["chunk_count"])
    return meta


# ── Run all pipelines on one query ───────────────────────────────────────────

def run_all_pipelines(query: str, G, pr: dict, exp_dir: Path, top_k: int = 6) -> dict:
    from graphrag_lite.retrieval.lightrag import (
        retrieve_local, retrieve_global, retrieve_hybrid, retrieve_mix
    )
    results = {}

    # Pipeline A
    try:
        from graphrag_lite.retrieval.pipeline_a import retrieve_flat
        t0 = time.monotonic()
        ev = retrieve_flat(query)
        results["pipeline_a"] = {
            "ids": [e.chunk_id for e in ev[:top_k]],
            "latency": round(time.monotonic() - t0, 3),
        }
    except Exception as e:
        results["pipeline_a"] = {"ids": [], "latency": 0.0, "error": str(e)}

    # Pipeline B
    try:
        from graphrag_lite.retrieval.pipeline_b import retrieve_structural
        t0 = time.monotonic()
        ev = retrieve_structural(query, top_k=top_k)
        results["pipeline_b"] = {
            "ids": [e.chunk_id for e in ev],
            "latency": round(time.monotonic() - t0, 3),
        }
    except Exception as e:
        results["pipeline_b"] = {"ids": [], "latency": 0.0, "error": str(e)}

    # LightRAG Local
    try:
        t0 = time.monotonic()
        ev = retrieve_local(query, G, pr, exp_dir, top_k=top_k)
        results["lightrag_local"] = {
            "ids": [e["chunk_id"] for e in ev],
            "latency": round(time.monotonic() - t0, 3),
        }
    except Exception as e:
        results["lightrag_local"] = {"ids": [], "latency": 0.0, "error": str(e)}

    # LightRAG Global
    try:
        t0 = time.monotonic()
        ev = retrieve_global(query, G, exp_dir, top_k=top_k)
        results["lightrag_global"] = {
            "ids": [e["chunk_id"] for e in ev],
            "latency": round(time.monotonic() - t0, 3),
        }
    except Exception as e:
        results["lightrag_global"] = {"ids": [], "latency": 0.0, "error": str(e)}

    # LightRAG Hybrid
    try:
        t0 = time.monotonic()
        ev = retrieve_hybrid(query, G, pr, exp_dir, top_k=top_k)
        results["lightrag_hybrid"] = {
            "ids": [e["chunk_id"] for e in ev],
            "latency": round(time.monotonic() - t0, 3),
        }
    except Exception as e:
        results["lightrag_hybrid"] = {"ids": [], "latency": 0.0, "error": str(e)}

    # LightRAG Mix
    try:
        t0 = time.monotonic()
        ev = retrieve_mix(query, G, pr, exp_dir, top_k=top_k)
        results["lightrag_mix"] = {
            "ids": [e["chunk_id"] for e in ev],
            "latency": round(time.monotonic() - t0, 3),
        }
    except Exception as e:
        results["lightrag_mix"] = {"ids": [], "latency": 0.0, "error": str(e)}

    return results


# ── Recall helper ─────────────────────────────────────────────────────────────

def recall_at_k(retrieved: list[str], gold: list[str]) -> float:
    if not gold:
        return 1.0
    return sum(1 for g in gold if g in retrieved) / len(gold)


def mrr(retrieved: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for i, cid in enumerate(retrieved, 1):
        if cid in gold_set:
            return 1.0 / i
    return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate all 6 pipelines on dev set")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--fresh", action="store_true",
                        help="Rebuild Zvec collections even if they exist")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    top_k = args.top_k

    log.info("=== Validation: model=%s  exp_dir=%s ===", args.model, EXP_DIR)

    # Load dev set
    with open(DEV_SET, encoding="utf-8") as f:
        dev_data = json.load(f)
    queries = dev_data["queries"]
    log.info("Dev set: %d questions", len(queries))

    # Load extractions
    extractions = load_extractions(EXTRACTIONS_PATH)
    extractions = attach_chunk_texts(extractions)

    # Build graph
    log.info("Building experimental graph from v2_off extractions...")
    G = build_graph(extractions)

    from graphrag_lite.graph.ranking import rank_pagerank
    pr = rank_pagerank(G)
    pr_max = max(pr.values()) if pr else 1.0
    log.info("PageRank computed: %d nodes, max=%.5f", len(pr), pr_max)

    # Build / reuse collections
    need_build = args.fresh or not (EXP_DIR / "entities").exists()
    if need_build:
        log.info("Building Zvec collections...")
        build_collections(G, extractions, EXP_DIR)
    else:
        log.info("Reusing existing collections at %s", EXP_DIR)

    PIPELINES = [
        "pipeline_a", "pipeline_b",
        "lightrag_local", "lightrag_global",
        "lightrag_hybrid", "lightrag_mix",
    ]

    # Per-pipeline accumulators
    recall_acc  = {p: [] for p in PIPELINES}
    mrr_acc     = {p: [] for p in PIPELINES}
    latency_acc = {p: [] for p in PIPELINES}
    errors      = {p: 0  for p in PIPELINES}
    unique_gold_hits = {p: set() for p in PIPELINES}  # all gold IDs recovered

    print()
    print("=" * 110)
    print(f"VALIDATION — qwen2.5:1.5b extractions  top_k={top_k}")
    print("=" * 110)
    hdr = f"{'QID':<6} {'Gold':>4}  " + "  ".join(f"{p[:12]:<12}" for p in PIPELINES)
    print(hdr)
    print("-" * 110)

    all_results = []

    for q in queries:
        qid       = q["query_id"]
        query_txt = q["query"]
        gold_ids  = q.get("gold_chunk_ids", [])

        pl_results = run_all_pipelines(query_txt, G, pr, EXP_DIR, top_k=top_k)

        row = {
            "query_id": qid,
            "type": q.get("type"),
            "gold_chunk_ids": gold_ids,
        }

        per_pl_str = []
        for pl in PIPELINES:
            r = pl_results.get(pl, {})
            ids = r.get("ids", [])
            lat = r.get("latency", 0.0)
            err = r.get("error")

            rec = recall_at_k(ids, gold_ids)
            m   = mrr(ids, gold_ids)

            recall_acc[pl].append(rec)
            mrr_acc[pl].append(m)
            latency_acc[pl].append(lat)
            if err:
                errors[pl] += 1

            for gid in gold_ids:
                if gid in ids:
                    unique_gold_hits[pl].add(gid)

            row[pl] = {
                "ids": ids, "recall": round(rec, 3),
                "mrr": round(m, 3), "latency_s": lat,
            }
            if err:
                row[pl]["error"] = err

            per_pl_str.append(f"R={rec:.2f}/M={m:.2f}")

        all_results.append(row)
        line = f"{qid:<6} {len(gold_ids):>4}  " + "  ".join(f"{s:<12}" for s in per_pl_str)
        print(line)

    # ── Aggregates ────────────────────────────────────────────────────────────
    print()
    print("=" * 110)
    print("AGGREGATE  (Recall@K / MRR / UniqueGold / AvgLatency / Errors)")
    print("=" * 110)
    n = len(queries)

    agg = {}
    for pl in PIPELINES:
        avg_rec = sum(recall_acc[pl]) / n
        avg_mrr = sum(mrr_acc[pl]) / n
        avg_lat = sum(latency_acc[pl]) / n
        ug      = len(unique_gold_hits[pl])
        errs    = errors[pl]
        agg[pl] = {
            "avg_recall_at_k": round(avg_rec, 4),
            "mrr": round(avg_mrr, 4),
            "unique_gold_chunks_recovered": ug,
            "avg_latency_s": round(avg_lat, 4),
            "errors": errs,
        }
        print(
            f"  {pl:<20}  recall={avg_rec:.4f}  mrr={avg_mrr:.4f}  "
            f"gold={ug:>3}  lat={avg_lat:.3f}s  errs={errs}"
        )

    # ── D18 diagnostic ────────────────────────────────────────────────────────
    d18 = next((q for q in queries if q["query_id"] == "D18"), None)
    if d18:
        print()
        print("D18 diagnostic:")
        d18_res = run_all_pipelines(d18["query"], G, pr, EXP_DIR, top_k=top_k)
        gold_d18 = d18.get("gold_chunk_ids", [])
        for pl in PIPELINES:
            ids = d18_res.get(pl, {}).get("ids", [])
            hit = any(g in ids for g in gold_d18)
            print(f"  {pl:<20}  gold_retrieved={hit}  ids={ids[:3]}...")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = EXP_DIR / f"validation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "extraction_source": str(EXTRACTIONS_PATH),
        "top_k": top_k,
        "per_question": all_results,
        "aggregates": agg,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info("Results saved to %s", out_path)
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
