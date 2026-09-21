"""
dedup_single.py — Dedup analysis for one strategy specified via env var.
Usage: DEDUP_STRATEGY=markdown_aware python -u eval/ablations/dedup_single.py

Hardened exception handling:
  - extract and glean have separate try/except blocks so a gleaning failure
    uses the extraction-only result rather than losing the whole chunk
  - catches ValueError, RuntimeError, ConnectionError, TimeoutError, OSError
    (not bare Exception — TASKS.md rule 4: failures must be visible, not swallowed)
  - all print() calls use flush=True so a crash leaves accurate last-known-state
"""
from __future__ import annotations
import os, sys, warnings
from collections import Counter

try:
    import httpx as _httpx
    import httpcore as _httpcore
    _TIMEOUT_EXCS = (ValueError, RuntimeError, ConnectionError, TimeoutError, OSError,
                     _httpx.ReadTimeout, _httpcore.ReadTimeout)
except ImportError:
    _TIMEOUT_EXCS = (ValueError, RuntimeError, ConnectionError, TimeoutError, OSError)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import extract_entities_relationships
from graphrag_lite.ingestion.gleaning import glean
from graphrag_lite.llm.ollama_client import OllamaClient
from graphrag_lite.cache.llm_cache import clear_cache, reset_stats, cache_stats
from graphrag_lite.config import get_config

strategy = os.environ.get("DEDUP_STRATEGY", "markdown_aware")
size_str = "250" if strategy == "small_window" else "450"

os.environ["CHUNKING_STRATEGY"] = strategy
os.environ["CHUNK_SIZE_EXTRACTION"] = size_str
get_config.cache_clear()
cfg = get_config()

tokenizer = get_tokenizer("qwen")
docs = load_documents("data/raw")
client = OllamaClient(
    cfg.effective_extraction_model(),
    temperature=cfg.EXTRACTION_TEMPERATURE,
    seed=cfg.EXTRACTION_SEED,
    timeout=cfg.LLM_TIMEOUT_S,
)

clear_cache()
reset_stats()

all_chunks = []
for d in docs:
    all_chunks.extend(chunk_document(d, cfg, tokenizer))

print(f"Strategy: {strategy}  chunks={len(all_chunks)}", flush=True)

raw_names = []
failed = 0

for i, chunk in enumerate(all_chunks):
    # ── Extraction ────────────────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = extract_entities_relationships(chunk, client)
    except _TIMEOUT_EXCS as exc:
        failed += 1
        print(f"  [{i+1:3d}/{len(all_chunks)}] EXTRACT FAILED ({type(exc).__name__}): "
              f"{str(exc)[:100]}", flush=True)
        continue   # skip to next chunk — nothing to glean on failure

    # ── Gleaning — separate block so failure falls back to extraction result ──
    if cfg.MAX_GLEANINGS > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = glean(chunk, result, client, max_gleanings=cfg.MAX_GLEANINGS)
        except _TIMEOUT_EXCS as exc:
            print(f"  [{i+1:3d}/{len(all_chunks)}] GLEAN FAILED ({type(exc).__name__}): "
                  f"{str(exc)[:100]} — using extraction-only result", flush=True)

    for e in result["entities"]:
        raw_names.append(e["name"].strip().lower())
    print(f"  [{i+1:3d}/{len(all_chunks)}] chunk_id={chunk['chunk_id'][:8]}  "
          f"entities={len(result['entities'])}  raw_total={len(raw_names)}", flush=True)

stats = cache_stats()
raw = len(raw_names)
unique = len(set(raw_names))
ratio = round(unique / raw, 3) if raw else 0
total_src = sum(len(tokenizer.encode(d["text"])) for d in docs)
unique_per_1k = round(unique / total_src * 1000, 2)

print(flush=True)
print(f"RESULT: raw={raw}  unique={unique}  ratio={ratio}  "
      f"unique/1kTok={unique_per_1k}  hits={stats['hits']}  "
      f"misses={stats['misses']}  failed={failed}", flush=True)
print(flush=True)
print("Top-15 most repeated names:", flush=True)
for name, count in Counter(raw_names).most_common(15):
    if count > 1:
        print(f"  count={count}  {name!r}", flush=True)
