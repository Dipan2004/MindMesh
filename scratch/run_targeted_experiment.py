import json
import time
import re
from pathlib import Path
import tiktoken
from graphrag_lite.llm.client_factory import get_llm_client
from graphrag_lite.llm.generator import AnswerGenerator
from graphrag_lite.retrieval.lightrag.chunk_lookup import fetch_chunks_by_ids

ENC = tiktoken.get_encoding("cl100k_base")

# Load existing retrieval results to get the exact retrieved chunk IDs for D16 and D18
EXP_DIR = Path("data/experiments/lightrag_qwen2.5_1.5b_lightrag_mix_opt_20261118_203834")
with open(EXP_DIR / "retrieval_results.json", "r", encoding="utf-8") as f:
    res = json.load(f)["results"]

d16_item = next(r for r in res if r["query_id"] == "D16")
d18_item = next(r for r in res if r["query_id"] == "D18")

all_cids = list(dict.fromkeys(d16_item["chunk_ids"] + d18_item["chunk_ids"]))
chunk_map = fetch_chunks_by_ids(all_cids)

chunks_d16 = [{"text": chunk_map[cid]["text"], "chunk_id": cid} for cid in d16_item["chunk_ids"]]
chunks_d18 = [{"text": chunk_map[cid]["text"], "chunk_id": cid} for cid in d18_item["chunk_ids"]]

base_client = get_llm_client("local")

# Define prompts
DEFAULT_PROMPT = AnswerGenerator.DEFAULT_SYSTEM_PROMPT
CONCISE_PROMPT = (
    AnswerGenerator.DEFAULT_SYSTEM_PROMPT + "\n"
    "4. Answer directly in at most 3-4 sentences and do not reproduce Markdown headings from retrieved context."
)

configs = [
    {
        "config_id": "1_control",
        "name": "Control (Optimized Baseline)",
        "description": "Current optimized configuration (num_predict=512, fp=0.0, default prompt)",
        "client": base_client.with_overrides(num_ctx=4096, num_predict=512, frequency_penalty=0.0),
        "prompt": DEFAULT_PROMPT,
    },
    {
        "config_id": "2_freq_penalty",
        "name": "Frequency Penalty 0.25",
        "description": "Optimized configuration with frequency_penalty=0.25",
        "client": base_client.with_overrides(num_ctx=4096, num_predict=512, frequency_penalty=0.25),
        "prompt": DEFAULT_PROMPT,
    },
    {
        "config_id": "3_concise_instruction",
        "name": "Concise-Answer Instruction",
        "description": "Optimized configuration with concise-answer instruction (3-4 sentences, no headings)",
        "client": base_client.with_overrides(num_ctx=4096, num_predict=512, frequency_penalty=0.0),
        "prompt": CONCISE_PROMPT,
    },
    {
        "config_id": "4_concise_plus_fp",
        "name": "Concise + Frequency Penalty",
        "description": "Exploratory: Concise instruction + frequency_penalty=0.25",
        "client": base_client.with_overrides(num_ctx=4096, num_predict=512, frequency_penalty=0.25),
        "prompt": CONCISE_PROMPT,
    },
]

queries = [
    ("D16", d16_item["query"], chunks_d16),
    ("D18", d18_item["query"], chunks_d18),
]

results_out = {}

print("=== STARTING TARGETED GENERATION EXPERIMENTS (SEQUENTIAL) ===\n")

for qid, query_text, ev_chunks in queries:
    results_out[qid] = {
        "query_id": qid,
        "query": query_text,
        "runs": {}
    }
    print(f"\n=======================================================")
    print(f"QUERY {qid}: {query_text}")
    print(f"=======================================================")
    
    for cfg in configs:
        cid = cfg["config_id"]
        cname = cfg["name"]
        cdesc = cfg["description"]
        print(f"\n--- Running [{cid}] ({cname}) on {qid} ---")
        
        gen = AnswerGenerator(client=cfg["client"], system_prompt=cfg["prompt"])
        
        t0 = time.monotonic()
        ans_obj = gen.generate_answer(query_text, ev_chunks)
        latency = time.monotonic() - t0
        
        ans_text = ans_obj.text or ""
        tokens = len(ENC.encode(ans_text))
        
        # Criteria checks
        clean_end = ans_text.strip().endswith((".", "!", "?", "\"", "'", "}", "]", ")"))
        has_headings = bool(re.search(r"^#{1,4}\s", ans_text, re.MULTILINE))
        
        # Check source text overlap (substring of 50+ chars from evidence)
        source_overlap = False
        matching_lines = []
        for c in ev_chunks:
            txt = c["text"]
            for line in txt.splitlines():
                line = line.strip()
                if len(line) > 50 and line in ans_text:
                    source_overlap = True
                    matching_lines.append(line[:60] + "...")
        
        # Check if it answers the question or refuses
        is_refusal = any(p in ans_text.lower() for p in ["cannot answer", "not provided", "does not contain", "i cannot", "insufficient"])
        
        run_record = {
            "config_id": cid,
            "name": cname,
            "description": cdesc,
            "latency_s": round(latency, 2),
            "output_tokens": tokens,
            "clean_end": clean_end,
            "has_markdown_headings": has_headings,
            "reproduces_source_section": source_overlap,
            "matching_source_snippets": matching_lines[:3],
            "is_refusal": is_refusal,
            "answer": ans_text,
        }
        
        results_out[qid]["runs"][cid] = run_record
        
        print(f"Latency: {latency:.2f}s | Tokens: {tokens} | Clean End: {clean_end} | Headings: {has_headings} | Verbatim Copy: {source_overlap}")
        print(f"Answer:\n{ans_text}\n")

# Save output
out_file = Path("eval/results/targeted_generation_d16_d18.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results_out, f, indent=2)

print(f"\nTargeted generation experiment complete. Saved results to {out_file}")
