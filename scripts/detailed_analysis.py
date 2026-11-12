import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
f = ROOT / "data/experiments/qwen2_5_1_5b_validate/validation_20261109_223804.json"
with open(f, encoding="utf-8") as fh:
    d = json.load(fh)

pq = d["per_question"]
PIPELINES = ["pipeline_a","pipeline_b","lightrag_local","lightrag_global","lightrag_hybrid","lightrag_mix"]

# Unique gold hits per pipeline
hits = {}
for pl in PIPELINES:
    hits[pl] = set()
    for r in pq:
        gold = set(r.get("gold_chunk_ids", []))
        ids  = set(r.get(pl, {}).get("ids", []))
        hits[pl].update(gold & ids)

a_hits = hits["pipeline_a"]
print("=== Unique gold chunks recovered per pipeline ===")
for pl in PIPELINES:
    extra = hits[pl] - a_hits
    print(f"  {pl:<22}  total={len(hits[pl])}  extra_vs_A={len(extra)}  {sorted(extra)}")

print()
print("=== Questions where LightRAG beats Pipeline A (Recall) ===")
for r in pq:
    qid = r["query_id"]
    gold = r.get("gold_chunk_ids", [])
    a_rec = r.get("pipeline_a", {}).get("recall", 0.0)
    for pl in ["lightrag_local","lightrag_global","lightrag_hybrid","lightrag_mix"]:
        pl_rec = r.get(pl, {}).get("recall", 0.0)
        if pl_rec > a_rec:
            print(f"  {qid}  {pl}: {pl_rec:.2f} > A:{a_rec:.2f}")

print()
print("=== Questions where LightRAG misses but Pipeline A hits ===")
for r in pq:
    qid = r["query_id"]
    a_rec = r.get("pipeline_a", {}).get("recall", 0.0)
    for pl in ["lightrag_local","lightrag_global","lightrag_hybrid","lightrag_mix"]:
        pl_rec = r.get(pl, {}).get("recall", 0.0)
        if a_rec > 0 and pl_rec == 0.0:
            print(f"  {qid}  {pl} misses (A={a_rec:.2f})")

print()
print("=== D16 detail ===")
for r in pq:
    if r["query_id"] == "D16":
        print(f"  gold: {r['gold_chunk_ids']}")
        for pl in PIPELINES:
            info = r.get(pl, {})
            err = info.get("error", "")
            rec = info.get("recall", 0)
            ids = info.get("ids", [])
            print(f"  {pl:<22}  recall={rec}  err={err[:70] if err else 'none'}  top3={ids[:3]}")

print()
print("=== Latency comparison ===")
agg = d["aggregates"]
for pl in PIPELINES:
    a = agg.get(pl, {})
    print(f"  {pl:<22}  recall={a.get('avg_recall_at_k',0):.4f}  mrr={a.get('mrr',0):.4f}  lat={a.get('avg_latency_s',0):.3f}s")
