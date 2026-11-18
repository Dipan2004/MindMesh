import json, re
from pathlib import Path

DEV_PATH = Path("eval/gold_dev_set.json")
with open(DEV_PATH, "r", encoding="utf-8") as f:
    gold_dev = json.load(f)
gold_queries = {q["query_id"]: q for q in gold_dev["queries"]}

EXP_ROOT = Path("data/experiments")
runs = [
    ("pipeline_a", "lightrag_qwen2.5_1.5b_pipeline_a_20261118_024952"),
    ("pipeline_b", "lightrag_qwen2.5_1.5b_pipeline_b_20261118_030012"),
    ("lightrag_hybrid", "lightrag_qwen2.5_1.5b_lightrag_hybrid_20261118_031118"),
    ("lightrag_hybrid_opt", "lightrag_qwen2.5_1.5b_lightrag_hybrid_opt_20261118_202556"),
    ("lightrag_mix", "lightrag_qwen2.5_1.5b_lightrag_mix_20261118_031953"),
    ("lightrag_mix_opt", "lightrag_qwen2.5_1.5b_lightrag_mix_opt_20261118_203834")
]

IDX_RE = re.compile(r"\[(\d+)\]")

for name, rdir in runs:
    with open(EXP_ROOT / rdir / "retrieval_results.json", encoding="utf-8") as f:
        res = json.load(f)["results"]
    
    valid_index_list = []
    gold_citation_precision_list = []
    
    for r in res:
        qid = r["query_id"]
        ans = r.get("answer", "") or ""
        cids = r.get("chunk_ids", [])
        gold = gold_queries[qid].get("gold_chunk_ids", [])
        
        idx_matches = [int(x) for x in IDX_RE.findall(ans)]
        
        if not idx_matches:
            # No citations made
            valid_index_list.append(1.0)
            gold_citation_precision_list.append(1.0 if not gold else 0.0)
            continue
            
        # Validity: Are cited indices in range of retrieved chunks [0 .. len(cids)-1]?
        valid = all(0 <= idx < len(cids) for idx in idx_matches)
        valid_index_list.append(1.0 if valid else 0.0)
        
        # Ground-Truth Alignment: Do the cited chunks map to actual gold chunks?
        cited_cids = [cids[idx] for idx in idx_matches if 0 <= idx < len(cids)]
        if cited_cids and gold:
            gold_hits = sum(1 for c in cited_cids if c in gold)
            gold_citation_precision_list.append(gold_hits / len(cited_cids))
        elif not gold:
            gold_citation_precision_list.append(0.0) # cited evidence on an unanswerable question
        else:
            gold_citation_precision_list.append(0.0)
            
    print(f"{name:20s} | In-Range Citation Index Rate: {sum(valid_index_list)/len(valid_index_list):.4f} | Gold-Evidence Citation Alignment: {sum(gold_citation_precision_list)/len(gold_citation_precision_list):.4f}")
