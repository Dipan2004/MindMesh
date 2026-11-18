import json
from pathlib import Path

DEV_PATH = Path("eval/gold_dev_set.json")
with open(DEV_PATH, "r", encoding="utf-8") as f:
    gold_dev = json.load(f)

gold_queries = gold_dev["queries"]
gold_by_id = {q["query_id"]: q for q in gold_queries}

EXP_ROOT = Path("data/experiments")
RUNS = {
    "pipeline_a": "lightrag_qwen2.5_1.5b_pipeline_a_20261118_024952",
    "lightrag_mix": "lightrag_qwen2.5_1.5b_lightrag_mix_20261118_031953",
    "lightrag_mix_opt": "lightrag_qwen2.5_1.5b_lightrag_mix_opt_20261118_203834",
}

print(f"{'QID':4s} | {'Type':12s} | {'Gold Chunks':20s} | {'Pipeline A Hits@6':18s} | {'Mix Opt Hits@6':18s} | {'A Ranks':10s} | {'MixOpt Ranks':12s}")
print("-" * 105)

with open(EXP_ROOT / RUNS["pipeline_a"] / "retrieval_results.json", "r", encoding="utf-8") as f:
    res_a = {r["query_id"]: r for r in json.load(f)["results"]}

with open(EXP_ROOT / RUNS["lightrag_mix_opt"] / "retrieval_results.json", "r", encoding="utf-8") as f:
    res_opt = {r["query_id"]: r for r in json.load(f)["results"]}

for q in gold_queries:
    qid = q["query_id"]
    qtype = q["type"]
    gold = q.get("gold_chunk_ids", [])
    
    cids_a = res_a[qid].get("chunk_ids", [])
    cids_opt = res_opt[qid].get("chunk_ids", [])
    
    hits_a = [c for c in cids_a[:6] if c in gold]
    hits_opt = [c for c in cids_opt[:6] if c in gold]
    
    ranks_a = [cids_a.index(c)+1 for c in hits_a]
    ranks_opt = [cids_opt.index(c)+1 for c in hits_opt]
    
    g_str = ",".join([g[:6] for g in gold]) if gold else "NONE (refusal)"
    ha_str = f"{len(hits_a)}/{len(gold)}" if gold else "N/A"
    hopt_str = f"{len(hits_opt)}/{len(gold)}" if gold else "N/A"
    
    ra_str = str(ranks_a) if ranks_a else "-"
    ropt_str = str(ranks_opt) if ranks_opt else "-"
    
    print(f"{qid:4s} | {qtype:12s} | {g_str:20s} | {ha_str:18s} | {hopt_str:18s} | {ra_str:10s} | {ropt_str:12s}")
