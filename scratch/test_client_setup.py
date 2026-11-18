import json
import time
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

# Fetch chunks
chunks_d16 = [fetch_chunks_by_ids([cid])[cid] for cid in d16_item["chunk_ids"]]
chunks_d18 = [fetch_chunks_by_ids([cid])[cid] for cid in d18_item["chunk_ids"]]

print(f"Loaded {len(chunks_d16)} chunks for D16, {len(chunks_d18)} chunks for D18.")

CONCISE_SYSTEM_PROMPT = (
    AnswerGenerator.DEFAULT_SYSTEM_PROMPT + "\n"
    "4. Answer directly in at most 3-4 sentences and do not reproduce Markdown headings from retrieved context."
)

base_client = get_llm_client("local")
print(f"Base client model: {base_client.model}")
