"""
Retry extraction for the single failed chunk from the canonical build_index run:
chunk_id=4af3dc8c5da84fad  doc=transformer_architecture  chunk_idx=4

If successful, prints the result and provides patched extractions.jsonl totals.
Cache is NOT cleared — the 41 other chunks remain cached.
"""
import json, os, sys, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
from graphrag_lite.config import get_config; get_config.cache_clear()
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import extract_entities_relationships
from graphrag_lite.ingestion.gleaning import glean
from graphrag_lite.llm.ollama_client import OllamaClient
from graphrag_lite.cache.llm_cache import cache_stats

cfg = get_config()
tokenizer = get_tokenizer("qwen")
docs = load_documents("data/raw")
client = OllamaClient(cfg.effective_extraction_model(),
                      temperature=cfg.EXTRACTION_TEMPERATURE,
                      seed=cfg.EXTRACTION_SEED,
                      timeout=cfg.LLM_TIMEOUT_S)

# Build chunks and find the failed one
all_chunks = []
for d in docs:
    all_chunks.extend(chunk_document(d, cfg, tokenizer))

target_id = "4af3dc8c5da84fad"
chunk = next((c for c in all_chunks if c["chunk_id"] == target_id), None)
if chunk is None:
    print("ERROR: chunk_id not found in current chunking!")
    sys.exit(1)

print(f"Retrying: chunk_id={chunk['chunk_id']}  doc={chunk['doc_id']}"
      f"  idx={chunk['chunk_index']}  tokens={chunk['token_count']}", flush=True)
print(f"timeout={cfg.LLM_TIMEOUT_S}s", flush=True)
print()

t0 = time.perf_counter()
try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = extract_entities_relationships(chunk, client)
        print(f"Extraction OK in {time.perf_counter()-t0:.1f}s: "
              f"entities={len(result['entities'])}", flush=True)
        if cfg.MAX_GLEANINGS > 0:
            result = glean(chunk, result, client, max_gleanings=cfg.MAX_GLEANINGS)
            print(f"Gleaning OK: entities={len(result['entities'])}", flush=True)
    if caught:
        print("Validation warnings:", len(caught))
    print()
    print("Result JSON:")
    print(json.dumps(result, indent=2))
    stats = cache_stats()
    print(f"\ncache hits={stats['hits']}  misses={stats['misses']}")
    print("STATUS: SUCCESS")
except Exception as exc:
    print(f"FAILED after {time.perf_counter()-t0:.1f}s: {type(exc).__name__}: {exc}")
    print("STATUS: FAILED")
