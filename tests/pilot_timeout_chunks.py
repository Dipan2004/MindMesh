"""Item 2a: Re-run the two timed-out chunks with num_predict=2048."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.makedirs("data/cache/pilot_1.5b", exist_ok=True)
os.environ["CACHE_DIR"] = "data/cache/pilot_1.5b"
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
os.environ["EXTRACTION_MODEL"] = "qwen2.5:1.5b"
os.environ["SLM_MODEL"] = "qwen2.5:1.5b"

from graphrag_lite.config import get_config; get_config.cache_clear()
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import _EXTRACTION_PROMPT_TEMPLATE
from graphrag_lite.llm.ollama_client import OllamaClient
from graphrag_lite.cache.llm_cache import _get_cache

cfg = get_config()
print(f"CACHE_DIR: {cfg.CACHE_DIR}  entries: {len(_get_cache())}")
print(f"Model: {cfg.effective_extraction_model()}  num_predict: {cfg.LLM_NUM_PREDICT}")
print()

tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = load_documents("data/raw")
ta = next(d for d in docs if d["doc_id"] == "transformer_architecture")
chunks = chunk_document(ta, cfg, tokenizer)

# idx=3 (a885be6b, 154 tokens) and idx=4 (4af3dc8c, 275 tokens)
TARGET_IDS = {"a885be6b749e", "4af3dc8c5da8"}
target_chunks = [c for c in chunks if c["chunk_id"][:12] in TARGET_IDS]

client = OllamaClient(cfg.effective_extraction_model(),
                      temperature=0.0, seed=42,
                      timeout=cfg.LLM_TIMEOUT_S,
                      num_predict=cfg.LLM_NUM_PREDICT)

print(f"Testing {len(target_chunks)} previously-timed-out chunks:")
for c in target_chunks:
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=c["text"])
    print(f"\nchunk_id={c['chunk_id'][:12]}  idx={c['chunk_index']}  tokens={c['token_count']}")
    print(f"  first 80 chars: {c['text'][:80].replace(chr(10),' ')!r}")
    t0 = time.perf_counter()
    try:
        raw = client.generate(prompt)
        elapsed = time.perf_counter() - t0
        resp_tokens = len(raw.split())  # rough token count
        # Try parsing
        try:
            import re as _re
            cleaned = _re.sub(r"```(?:json)?", "", raw, flags=_re.IGNORECASE).strip()
            s = cleaned.find("{"); e = cleaned.rfind("}")
            parsed = __import__("json").loads(cleaned[s:e+1]) if s != -1 else {}
            n_ent = len(parsed.get("entities",[]))
            n_rel = len(parsed.get("relationships",[]))
            status = f"PARSED  entities={n_ent}  relationships={n_rel}"
        except Exception as pe:
            status = f"PARSE_FAIL: {pe}"
        print(f"  elapsed={elapsed:.1f}s  resp_tokens~{resp_tokens}  {status}")
        print(f"  raw first 200: {raw[:200]!r}")
    except Exception as ex:
        elapsed = time.perf_counter() - t0
        print(f"  FAILED after {elapsed:.1f}s: {type(ex).__name__}: {ex}")
