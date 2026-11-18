import json
from pathlib import Path
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")

DEV_PATH = Path("eval/gold_dev_set.json")
with open(DEV_PATH, "r", encoding="utf-8") as f:
    gold_dev = json.load(f)

gold_queries = gold_dev["queries"]
gold_by_id = {q["query_id"]: q for q in gold_queries}

EXP_ROOT = Path("data/experiments")
RUNS = {
    "pipeline_a": "lightrag_qwen2.5_1.5b_pipeline_a_20261118_024952",
    "pipeline_b": "lightrag_qwen2.5_1.5b_pipeline_b_20261118_030012",
    "lightrag_hybrid": "lightrag_qwen2.5_1.5b_lightrag_hybrid_20261118_031118",
    "lightrag_hybrid_opt": "lightrag_qwen2.5_1.5b_lightrag_hybrid_opt_20261118_202556",
    "lightrag_mix": "lightrag_qwen2.5_1.5b_lightrag_mix_20261118_031953",
    "lightrag_mix_opt": "lightrag_qwen2.5_1.5b_lightrag_mix_opt_20261118_203834",
}

print("=== 1. VALIDATE GENERATION QUALITY: LIGHTRAG_MIX_OPT ANSWERS ===")
with open(EXP_ROOT / RUNS["lightrag_mix_opt"] / "retrieval_results.json", "r", encoding="utf-8") as f:
    mix_opt_res = json.load(f)["results"]

with open(EXP_ROOT / RUNS["lightrag_mix"] / "retrieval_results.json", "r", encoding="utf-8") as f:
    mix_base_res = json.load(f)["results"]

with open(EXP_ROOT / RUNS["pipeline_a"] / "retrieval_results.json", "r", encoding="utf-8") as f:
    pipe_a_res = json.load(f)["results"]

for idx, item in enumerate(mix_opt_res, 1):
    qid = item["query_id"]
    ans = item.get("answer", "")
    tokens = len(ENC.encode(ans))
    gold_item = gold_by_id[qid]
    
    # Extract required terms
    req_terms = []
    for g in gold_item.get("gold", []):
        req_terms.extend(g.get("required_terms", []))
    req_terms = list(dict.fromkeys(req_terms)) # deduplicate
    
    terms_found = [t for t in req_terms if t.lower() in ans.lower()]
    term_cov = len(terms_found) / len(req_terms) if req_terms else 1.0
    
    # check ending punctuation
    clean_end = ans.strip().endswith((".", "!", "?", "\"", "'", "}", "]", ")"))
    
    print(f"[{qid}] Toks: {tokens:3d} | Clean End: {str(clean_end):5s} | Req Terms: {len(terms_found)}/{len(req_terms)} ({term_cov*100:5.1f}%) | Gen Latency: {item.get('generation_latency_s', 0):.2f}s")
    print(f"  Excerpt: {ans[:90].strip()}... [Tail]: ...{ans[-45:].strip()}")
    if tokens >= 512:
        print(f"  *** WARNING: TOKENS REACHED OR EXCEEDED 512: {tokens} ***")

tok_counts = [len(ENC.encode(x.get("answer", ""))) for x in mix_opt_res]
print(f"\nSummary of Token Lengths in lightrag_mix_opt:")
print(f"Min: {min(tok_counts)}, Max: {max(tok_counts)}, Mean: {sum(tok_counts)/len(tok_counts):.1f}")
print(f"Any >= 512 tokens? {any(t >= 512 for t in tok_counts)}")

print("\n=== 2. COMPARE TERM COVERAGE & COMPLETENESS ===")
def calc_coverage(results):
    covs = []
    for item in results:
        qid = item["query_id"]
        ans = item.get("answer", "") or ""
        gold_item = gold_by_id[qid]
        req_terms = []
        for g in gold_item.get("gold", []):
            req_terms.extend(g.get("required_terms", []))
        req_terms = list(dict.fromkeys(req_terms))
        if not req_terms:
            continue
        found = [t for t in req_terms if t.lower() in ans.lower()]
        covs.append(len(found) / len(req_terms))
    return sum(covs) / len(covs) if covs else 0.0

cov_pipe_a = calc_coverage(pipe_a_res)
cov_mix_base = calc_coverage(mix_base_res)
cov_mix_opt = calc_coverage(mix_opt_res)
print(f"Pipeline A Required Term Coverage:       {cov_pipe_a*100:.2f}%")
print(f"Baseline Mix Required Term Coverage:     {cov_mix_base*100:.2f}%")
print(f"Optimized Mix Required Term Coverage:    {cov_mix_opt*100:.2f}%")

print("\n=== 3. RECOMPUTE RETRIEVAL METRICS (RECALL@1, 3, 6, MRR) ===")
for name, rdir in RUNS.items():
    with open(EXP_ROOT / rdir / "retrieval_results.json", "r", encoding="utf-8") as f:
        res = json.load(f)["results"]
    
    r1_list, r3_list, r6_list, mrr_list = [], [], [], []
    per_q = {}
    for item in res:
        qid = item["query_id"]
        gold_chunks = gold_by_id[qid].get("gold_chunk_ids", [])
        cids = item.get("chunk_ids", [])
        
        if not gold_chunks:
            r1_list.append(1.0)
            r3_list.append(1.0)
            r6_list.append(1.0)
            mrr_list.append(1.0)
            per_q[qid] = {"r1": 1.0, "r3": 1.0, "r6": 1.0, "mrr": 1.0, "hits6": 0, "denom": 0}
            continue
        
        h1 = len(set(cids[:1]).intersection(gold_chunks))
        h3 = len(set(cids[:3]).intersection(gold_chunks))
        h6 = len(set(cids[:6]).intersection(gold_chunks))
        denom = len(gold_chunks)
        
        r1_list.append(h1 / denom)
        r3_list.append(h3 / denom)
        r6_list.append(h6 / denom)
        
        rr = 0.0
        for rank, cid in enumerate(cids, 1):
            if cid in gold_chunks:
                rr = 1.0 / rank
                break
        mrr_list.append(rr)
        per_q[qid] = {"r1": h1/denom, "r3": h3/denom, "r6": h6/denom, "mrr": rr, "hits6": h6, "denom": denom}
        
    print(f"{name:20s} | R@1: {sum(r1_list)/len(r1_list):.4f} | R@3: {sum(r3_list)/len(r3_list):.4f} | R@6: {sum(r6_list)/len(r6_list):.4f} | MRR: {sum(mrr_list)/len(mrr_list):.4f}")

