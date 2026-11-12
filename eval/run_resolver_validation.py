"""
run_resolver_validation.py — Run the entity resolver on the gleaning graph
and evaluate retrieval on the dev set across all 3 configurations.

SAFETY:
  - Never modifies the original graph
  - Each config runs on an isolated deep copy
  - Canonical Zvec collections never touched

Usage:
    python eval/run_resolver_validation.py
    python eval/run_resolver_validation.py --extractions data/cache_v2/extractions_v2_off.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "LightRAG"))

import warnings; warnings.filterwarnings("ignore")

from graphrag_lite.entity_resolution.resolver import EntityResolver
from graphrag_lite.entity_resolution.aliases import AliasRegistry


def _recall_mrr(retrieved: list[str], gold: list[str]):
    if not gold: return 1.0, 0.0
    hit = sum(1 for g in gold if g in retrieved) / len(gold)
    mrr = next((1.0/i for i, c in enumerate(retrieved, 1) if c in set(gold)), 0.0)
    return hit, mrr


def evaluate_config_on_devset(G, pr_map, exp_dir: Path, dev_queries: list[dict], top_k=6) -> dict:
    """Evaluate LightRAG Local retrieval on the dev set using the given config graph."""
    from graphrag_lite.retrieval.lightrag import retrieve_local, retrieve_global, retrieve_hybrid

    recalls, mrrs, latencies = [], [], []
    per_q = []

    for q in dev_queries:
        qid = q["query_id"]
        text = q["query"]
        gold = q.get("gold_chunk_ids", [])
        t0 = time.monotonic()
        try:
            ev = retrieve_local(text, G, pr_map, exp_dir, top_k=top_k)
            ids = [e["chunk_id"] for e in ev]
        except Exception as exc:
            ids = []
        lat = time.monotonic() - t0
        rec, mrr = _recall_mrr(ids, gold)
        recalls.append(rec)
        mrrs.append(mrr)
        latencies.append(lat)
        per_q.append({
            "query_id": qid, "recall": round(rec,4), "mrr": round(mrr,4),
            "ids": ids, "gold": gold, "latency_s": round(lat,3)
        })

    n = len(dev_queries)
    return {
        "avg_recall":  round(sum(recalls)/n, 4) if n else 0,
        "mrr":         round(sum(mrrs)/n, 4) if n else 0,
        "avg_latency": round(sum(latencies)/n, 4) if n else 0,
        "unique_gold": len({cid for r in per_q for cid in r["ids"] if cid in r["gold"]}),
        "per_question": per_q,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractions", default=None)
    args = parser.parse_args()

    # Find best extractions
    candidates = [
        ROOT / "data/experiments/qwen2_5_1_5b_genuine_gleaning_on/extractions.jsonl",
        ROOT / "data/cache_v2/extractions_v2_off.jsonl",
        ROOT / "data/cache/extractions.jsonl",
    ]
    ext_path = Path(args.extractions) if args.extractions else next(
        (p for p in candidates if p.exists()), None)
    if not ext_path or not ext_path.exists():
        print("ERROR: No extractions found."); sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  ENTITY RESOLVER VALIDATION")
    print(f"  Source: {ext_path.name}")
    print(f"{'='*70}\n")

    # Build resolver
    registry = AliasRegistry.with_builtins()
    resolver = EntityResolver.from_extractions(ext_path, registry=registry)
    print("Building resolver configs (no LLM calls)...")
    t0 = time.monotonic()
    configs = resolver.run_all_configs()
    build_time = time.monotonic() - t0
    print(f"Built in {build_time:.1f}s\n")

    # Print graph metrics table
    report = resolver.report(configs)

    # Load dev set
    dev_path = ROOT / "eval" / "gold_dev_set.json"
    dev_queries = json.loads(dev_path.read_text(encoding="utf-8"))["queries"]
    print(f"\nEvaluating LightRAG Local on {len(dev_queries)} dev questions per config...")

    # We need Zvec collections for the retrieval — use the Phase 1 validate collections
    exp_dir = ROOT / "data/experiments/qwen2_5_1_5b_validate"
    if not (exp_dir / "entities").exists():
        print("WARNING: Experiment collections not found at expected path.")
        print(f"  Expected: {exp_dir}")
        print("  Skipping retrieval evaluation (graph metrics still valid)")
        retrieval_rows = {}
    else:
        from graphrag_lite.graph.ranking import rank_pagerank
        retrieval_rows = {}
        for name, cfg in configs.items():
            print(f"  Evaluating {name}...")
            pr = rank_pagerank(cfg.G)
            result = evaluate_config_on_devset(cfg.G, pr, exp_dir, dev_queries)
            retrieval_rows[name] = result
            print(f"    Recall={result['avg_recall']:.4f}  MRR={result['mrr']:.4f}  "
                  f"UniqueGold={result['unique_gold']}  Lat={result['avg_latency']:.3f}s")

        # D18 diagnostic
        d18 = next((q for q in dev_queries if q["query_id"] == "D18"), None)
        if d18:
            print(f"\n  D18 ({d18['query'][:60]}...):")
            for name, result in retrieval_rows.items():
                d18_r = next((r for r in result["per_question"] if r["query_id"]=="D18"), {})
                hit = any(g in d18_r.get("ids",[]) for g in d18["gold_chunk_ids"])
                print(f"    {name:<22}  gold_retrieved={hit}  recall={d18_r.get('recall',0):.2f}")

    # Save results
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(ext_path),
        "graph_metrics": {name: cfg.metrics() for name, cfg in configs.items()},
        "retrieval": retrieval_rows,
        "build_time_s": round(build_time, 2),
    }
    out_path = ROOT / "data/experiments/resolver_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # Print retrieval comparison
    if retrieval_rows:
        print(f"\n{'='*70}")
        print("  RETRIEVAL COMPARISON (LightRAG Local, top_k=6)")
        print(f"{'='*70}")
        fmt = "{:<22} {:>10} {:>8} {:>10} {:>12}"
        print(fmt.format("Config", "Recall@6", "MRR", "UniqueGold", "AvgLat(s)"))
        print("-"*65)
        for name, r in retrieval_rows.items():
            print(fmt.format(name, f"{r['avg_recall']:.4f}", f"{r['mrr']:.4f}",
                             str(r["unique_gold"]), f"{r['avg_latency']:.3f}"))


if __name__ == "__main__":
    main()
