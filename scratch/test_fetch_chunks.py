from pathlib import Path
import json

EXP_DIR = Path("data/experiments/lightrag_qwen2.5_1.5b_lightrag_mix_opt_20261118_203834")
with open(EXP_DIR / "retrieval_results.json", "r", encoding="utf-8") as f:
    res = json.load(f)["results"]

d16 = next(r for r in res if r["query_id"] == "D16")
d18 = next(r for r in res if r["query_id"] == "D18")

print("D16 query:", d16["query"])
print("D16 chunk_ids:", d16["chunk_ids"])
print("D18 query:", d18["query"])
print("D18 chunk_ids:", d18["chunk_ids"])
