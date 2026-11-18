import json
from pathlib import Path
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
EXP_ROOT = Path("data/experiments")

runs = {
    "pipeline_a": "lightrag_qwen2.5_1.5b_pipeline_a_20261118_024952",
    "lightrag_mix": "lightrag_qwen2.5_1.5b_lightrag_mix_20261118_031953",
    "lightrag_mix_opt": "lightrag_qwen2.5_1.5b_lightrag_mix_opt_20261118_203834",
}

for qid in ["D16", "D18"]:
    print(f"\n==================== {qid} ====================")
    for name, rdir in runs.items():
        with open(EXP_ROOT / rdir / "retrieval_results.json", "r", encoding="utf-8") as f:
            res = json.load(f)["results"]
        item = next(r for r in res if r["query_id"] == qid)
        ans = item.get("answer", "") or ""
        tokens = len(ENC.encode(ans))
        print(f"--- {name} (toks: {tokens}, gen_latency: {item.get('generation_latency_s', 0):.2f}s) ---")
        print(f"Retrieved chunk_ids: {item.get('chunk_ids', [])}")
        print(f"Answer:\n{ans}")
