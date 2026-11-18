import json, time
from pathlib import Path
import tiktoken
from graphrag_lite.llm.client_factory import get_llm_client
from graphrag_lite.llm.generator import AnswerGenerator
from graphrag_lite.retrieval.lightrag.chunk_lookup import fetch_chunks_by_ids

ENC = tiktoken.get_encoding("cl100k_base")

EXP_DIR = Path("data/experiments/lightrag_qwen2.5_1.5b_lightrag_mix_opt_20261118_203834")
with open(EXP_DIR / "retrieval_results.json", "r", encoding="utf-8") as f:
    res = json.load(f)["results"]

d16_item = next(r for r in res if r["query_id"] == "D16")
chunk_map = fetch_chunks_by_ids(d16_item["chunk_ids"])
chunks_d16 = [{"text": chunk_map[cid]["text"], "chunk_id": cid} for cid in d16_item["chunk_ids"]]

base_client = get_llm_client("local")
client = base_client.with_overrides(num_ctx=4096, num_predict=512, frequency_penalty=0.0)

gen = AnswerGenerator(client=client, system_prompt=AnswerGenerator.DEFAULT_SYSTEM_PROMPT)

t0 = time.monotonic()
ans = gen.generate_answer(d16_item["query"], chunks_d16)
elapsed = time.monotonic() - t0

toks = len(ENC.encode(ans.text))
print(f"D16 Control test: elapsed={elapsed:.2f}s, tokens={toks}")
print(f"Answer excerpt:\n{ans.text[:150]}... [Tail]: ...{ans.text[-80:]}")
