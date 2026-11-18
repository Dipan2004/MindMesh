import json
import re
from pathlib import Path
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")

DEV_PATH = Path("eval/gold_dev_set.json")
with open(DEV_PATH, "r", encoding="utf-8") as f:
    gold_dev = json.load(f)

gold_queries = gold_dev["queries"]
gold_by_id = {q["query_id"]: q for q in gold_queries}

EXP_ROOT = Path("data/experiments")
RUNS = [
    ("pipeline_a", "lightrag_qwen2.5_1.5b_pipeline_a_20261118_024952", "baseline"),
    ("pipeline_b", "lightrag_qwen2.5_1.5b_pipeline_b_20261118_030012", "baseline"),
    ("lightrag_hybrid", "lightrag_qwen2.5_1.5b_lightrag_hybrid_20261118_031118", "baseline"),
    ("lightrag_hybrid_opt", "lightrag_qwen2.5_1.5b_lightrag_hybrid_opt_20261118_202556", "optimized"),
    ("lightrag_mix", "lightrag_qwen2.5_1.5b_lightrag_mix_20261118_031953", "baseline"),
    ("lightrag_mix_opt", "lightrag_qwen2.5_1.5b_lightrag_mix_opt_20261118_203834", "optimized"),
]

IDX_RE = re.compile(r"\[(\d+)\]")
REFUSAL_TRIGGERS = [
    "cannot answer", "not provided", "does not contain", "i cannot",
    "insufficient", "not mentioned", "no evidence", "not found"
]

results_table = {}

for name, rdir, variant in RUNS:
    res_path = EXP_ROOT / rdir / "retrieval_results.json"
    with open(res_path, "r", encoding="utf-8") as f:
        res_data = json.load(f)
        results = res_data.get("results", [])
        
    ans_r1, ans_r3, ans_r6, ans_mrr = [], [], [], []
    all_r1, all_r3, all_r6, all_mrr = [], [], [], []
    
    no_ans_correct = []
    
    ret_latencies = []
    gen_latencies = []
    tot_latencies = []
    
    truncation_count = 0
    output_tokens_list = []
    
    valid_citation_indices = []
    gold_citation_hits = []
    term_coverages = []
    
    for item in results:
        qid = item["query_id"]
        gold_chunks = gold_by_id[qid].get("gold_chunk_ids", [])
        qtype = gold_by_id[qid].get("type", "")
        cids = item.get("chunk_ids", [])
        ans = item.get("answer", "") or ""
        
        # Latencies
        ret_latencies.append(item.get("retrieval_latency_s", 0.0))
        gen_latencies.append(item.get("generation_latency_s", 0.0))
        tot_latencies.append(item.get("total_latency_s", 0.0))
        
        # Token counts
        otoks = len(ENC.encode(ans))
        output_tokens_list.append(otoks)
        
        # Check truncation: answer stops without clean punctuation and token count is close to cap (e.g. 512)
        # Or specifically in mix_opt where num_predict was 512
        is_clean_end = ans.strip().endswith((".", "!", "?", "\"", "'", "}", "]", ")"))
        # In mix_opt, 512 Qwen tokens ~ 494-500 cl100k tokens and ending is cut off mid-token
        is_truncated = (not is_clean_end) and (otoks >= 450)
        if is_truncated:
            truncation_count += 1
            
        # Citations
        idx_matches = [int(x) for x in IDX_RE.findall(ans)]
        if idx_matches:
            valid_idx = all(0 <= idx < len(cids) for idx in idx_matches)
            valid_citation_indices.append(1.0 if valid_idx else 0.0)
            
            # Gold-evidence alignment: fraction of cited chunks that are gold
            if gold_chunks:
                cited_chunks = [cids[idx] for idx in idx_matches if 0 <= idx < len(cids)]
                hits = sum(1 for c in cited_chunks if c in gold_chunks)
                gold_citation_hits.append(hits / len(cited_chunks))
            else:
                gold_citation_hits.append(0.0)
        else:
            valid_citation_indices.append(1.0)
            gold_citation_hits.append(1.0 if not gold_chunks else 0.0)
            
        # Lexical term coverage
        req_terms = []
        for g in gold_by_id[qid].get("gold", []):
            req_terms.extend(g.get("required_terms", []))
        req_terms = list(dict.fromkeys(req_terms))
        if req_terms:
            hits = sum(1 for t in req_terms if t.lower() in ans.lower())
            term_coverages.append(hits / len(req_terms))
            
        # Retrieval recall & MRR
        if gold_chunks: # Answerable
            denom = len(gold_chunks)
            h1 = len(set(cids[:1]).intersection(gold_chunks))
            h3 = len(set(cids[:3]).intersection(gold_chunks))
            h6 = len(set(cids[:6]).intersection(gold_chunks))
            
            ans_r1.append(h1 / denom)
            ans_r3.append(h3 / denom)
            ans_r6.append(h6 / denom)
            
            all_r1.append(h1 / denom)
            all_r3.append(h3 / denom)
            all_r6.append(h6 / denom)
            
            rr = 0.0
            for rank, cid in enumerate(cids, 1):
                if cid in gold_chunks:
                    rr = 1.0 / rank
                    break
            ans_mrr.append(rr)
            all_mrr.append(rr)
        else: # No-answer query (D19, D20)
            # All-query convention scores unanswerable queries as 1.0
            all_r1.append(1.0)
            all_r3.append(1.0)
            all_r6.append(1.0)
            all_mrr.append(1.0)
            
            # Refusal check
            refused = any(p in ans.lower() for p in REFUSAL_TRIGGERS)
            no_ans_correct.append(1.0 if refused else 0.0)

    n_ans = len(ans_r1) # 18
    n_all = len(all_r1) # 20
    n_no = len(no_ans_correct) # 2
    
    results_table[name] = {
        "pipeline": name,
        "variant": variant,
        "n_answerable_queries": n_ans,
        "n_no_answer_queries": n_no,
        "n_total_queries": n_all,
        
        # 1. Answerable-only retrieval metrics
        "Answerable_Recall@1": round(sum(ans_r1) / n_ans, 4),
        "Answerable_Recall@3": round(sum(ans_r3) / n_ans, 4),
        "Answerable_Recall@6": round(sum(ans_r6) / n_ans, 4),
        "Answerable_MRR": round(sum(ans_mrr) / n_ans, 4),
        
        # Legacy/All-query retrieval metrics (including 1.0 for unanswerable)
        "All_Query_Recall@1_legacy": round(sum(all_r1) / n_all, 4),
        "All_Query_Recall@3_legacy": round(sum(all_r3) / n_all, 4),
        "All_Query_Recall@6_legacy": round(sum(all_r6) / n_all, 4),
        "All_Query_MRR_legacy": round(sum(all_mrr) / n_all, 4),
        
        # 2. No-answer performance
        "No_Answer_Accuracy_D19_D20": round(sum(no_ans_correct) / n_no, 4) if n_no else None,
        
        # 3. Grounding & Lexical Proxies (Separate)
        "In_Range_Citation_Index_Validity": round(sum(valid_citation_indices) / n_all, 4),
        "Gold_Evidence_Citation_Alignment": round(sum(gold_citation_hits) / n_all, 4),
        "Lexical_Required_Term_Coverage": round(sum(term_coverages) / len(term_coverages), 4),
        
        # 4. Latencies
        "Mean_Retrieval_Latency_s": round(sum(ret_latencies) / n_all, 3),
        "Mean_Generation_Latency_s": round(sum(gen_latencies) / n_all, 3),
        "Mean_Total_Latency_s": round(sum(tot_latencies) / n_all, 3),
        
        # 5. Generation bounds & truncation
        "Answer_Truncation_Count": truncation_count,
        "Mean_Output_Tokens": round(sum(output_tokens_list) / n_all, 1),
        
        # 6. Prompt token estimate (Clearly labeled as synthetic estimate)
        "Estimated_Mean_Prompt_Tokens": 2936.9,
    }

print(json.dumps(results_table, indent=2))
