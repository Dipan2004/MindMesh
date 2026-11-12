"""Quick smoke test: one chunk through extract + gleaning with the fixed chunk dict."""
import sys, pathlib
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import zvec
col = zvec.open(str(ROOT / "data/cache/chunks_v1"))
res = col.query(queries=[zvec.Query("text_vec", vector=[0.0]*768,
                param=zvec.HnswQueryParam(ef=10))], topk=1)
chunk = None
for r in res:
    f = r.fields or {}
    src = f.get("source_doc", "")
    chunk = {"chunk_id": r.id, "doc_id": src, "source_doc": src,
             "chunk_index": int(f.get("chunk_index", 0)), "text": f.get("text", "")}
col.close()

print(f"chunk_id: {chunk['chunk_id']}")
print(f"doc_id:   {chunk['doc_id']}")
print(f"keys:     {list(chunk.keys())}")

from graphrag_lite.config import get_config
from graphrag_lite.llm.ollama_client import OllamaClient
from graphrag_lite.ingestion.extractor import extract_entities_relationships
from graphrag_lite.ingestion.gleaning import glean

cfg = get_config()
client = OllamaClient(
    model="qwen2.5:1.5b",
    temperature=cfg.EXTRACTION_TEMPERATURE,
    seed=cfg.EXTRACTION_SEED,
    num_predict=cfg.LLM_NUM_PREDICT,
    num_ctx=cfg.LLM_NUM_CTX,
    think=False,
    frequency_penalty=cfg.LLM_FREQUENCY_PENALTY,
)

print("Running extraction...")
result = extract_entities_relationships(chunk, client)
n_e1 = len(result.get("entities", []))
n_r1 = len(result.get("relationships", []))
print(f"Extraction: {n_e1} entities, {n_r1} relationships")

print("Running gleaning...")
result2 = glean(chunk, result, client, max_gleanings=1)
n_e2 = len(result2.get("entities", []))
n_r2 = len(result2.get("relationships", []))
print(f"After glean: {n_e2} entities, {n_r2} relationships  (+{n_e2-n_e1}/+{n_r2-n_r1})")
print("SMOKE TEST: PASS")
