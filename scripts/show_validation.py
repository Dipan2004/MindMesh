"""Print the latest validation results."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

p = ROOT / "data/experiments/qwen2_5_1_5b_validate/validation_20261109_223804.json"
with open(p, encoding="utf-8") as f:
    d = json.load(f)

agg = d["aggregates"]
print("=== VALIDATION RESULTS — qwen2.5:1.5b extractions, top_k=6 ===")
print()
fmt = "{:<22} {:>10} {:>8} {:>10} {:>12} {:>6}"
print(fmt.format("Pipeline","Recall@K","MRR","UniqueGold","AvgLat(s)","Errors"))
print("-" * 72)
for pl in ["pipeline_a","pipeline_b","lightrag_local","lightrag_global","lightrag_hybrid","lightrag_mix"]:
    a = agg.get(pl, {})
    recall = a.get("avg_recall_at_k", 0)
    mrr    = a.get("mrr", 0)
    ug     = a.get("unique_gold_chunks_recovered", 0)
    lat    = a.get("avg_latency_s", 0)
    errs   = a.get("errors", 0)
    print(fmt.format(pl, f"{recall:.4f}", f"{mrr:.4f}", str(ug), f"{lat:.3f}", str(errs)))

print()
# D16 FTS fix check
d16errs = sum(
    1 for r in d["per_question"]
    for pl in ["pipeline_a","pipeline_b"]
    if r.get(pl, {}).get("error")
)
print(f"D16 FTS errors after fix: {d16errs}  (0 = fixed)")

print()
# D18 diagnostic
for r in d["per_question"]:
    if r["query_id"] == "D18":
        print("D18 diagnostic (cross-doc hard query):")
        gold = r.get("gold_chunk_ids", [])
        print(f"  gold_chunk_ids: {gold}")
        for pl in ["pipeline_a","pipeline_b","lightrag_local","lightrag_global","lightrag_hybrid","lightrag_mix"]:
            ids = r.get(pl, {}).get("ids", [])
            hit = any(g in ids for g in gold)
            print(f"  {pl:<22} gold_retrieved={hit}  top_ids={ids[:3]}")

print()
# Per-question table
print("Per-question recall (A / B / Local / Global / Hybrid / Mix):")
for r in d["per_question"]:
    qid = r["query_id"]
    gold = r.get("gold_chunk_ids", [])
    vals = []
    for pl in ["pipeline_a","pipeline_b","lightrag_local","lightrag_global","lightrag_hybrid","lightrag_mix"]:
        rec = r.get(pl, {}).get("recall", 0.0)
        vals.append(f"{rec:.2f}")
    print(f"  {qid}  gold={len(gold)}  " + " / ".join(vals))
