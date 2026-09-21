"""
dedup_check.py — Compute deduplicated entity counts from cached extractions.
All LLM calls are served from disk cache — no live model calls.
"""
from __future__ import annotations
import os, sys, statistics, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import extract_entities_relationships
from graphrag_lite.ingestion.gleaning import glean
from graphrag_lite.llm.ollama_client import OllamaClient
from graphrag_lite.cache.llm_cache import reset_stats, cache_stats
from graphrag_lite.config import get_config

STRATEGIES = [
    ("sliding_window",  "450"),
    ("recursive_char",  "450"),
    ("markdown_aware",  "450"),
    ("small_window",    "250"),
]

def run(strategy, size_str, docs, tokenizer):
    os.environ["CHUNKING_STRATEGY"] = strategy
    os.environ["CHUNK_SIZE_EXTRACTION"] = size_str
    get_config.cache_clear()
    cfg = get_config()
    client = OllamaClient(cfg.effective_extraction_model(),
                          temperature=cfg.EXTRACTION_TEMPERATURE,
                          seed=cfg.EXTRACTION_SEED)
    reset_stats()

    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_document(d, cfg, tokenizer))

    raw_entities = []   # all entity names across all chunks
    failed = 0

    for chunk in all_chunks:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = extract_entities_relationships(chunk, client)
                if cfg.MAX_GLEANINGS > 0:
                    result = glean(chunk, result, client, max_gleanings=cfg.MAX_GLEANINGS)
            for e in result["entities"]:
                raw_entities.append(e["name"].strip().lower())
        except ValueError:
            failed += 1

    stats = cache_stats()
    raw_count = len(raw_entities)
    unique_count = len(set(raw_entities))
    ratio = round(unique_count / raw_count, 3) if raw_count else 0

    # Total source tokens (constant across strategies)
    total_src = sum(len(tokenizer.encode(d["text"])) for d in docs)
    unique_per_1k = round(unique_count / total_src * 1000, 2)

    print(f"Strategy: {strategy}")
    print(f"  chunks={len(all_chunks)}  raw_entities={raw_count}  "
          f"unique_entities={unique_count}  ratio={ratio}  "
          f"unique/1kTok={unique_per_1k}  "
          f"cache_hits={stats['hits']}  misses={stats['misses']}  failed={failed}")
    return {
        "strategy": strategy,
        "chunks": len(all_chunks),
        "raw": raw_count,
        "unique": unique_count,
        "ratio": ratio,
        "unique_per_1k": unique_per_1k,
        "failed": failed,
    }

def main():
    tokenizer = get_tokenizer("qwen")
    docs = load_documents("data/raw")
    total_src = sum(len(tokenizer.encode(d["text"])) for d in docs)
    print(f"Corpus: {len(docs)} docs, {total_src} source tokens")
    print()
    results = []
    for strategy, size_str in STRATEGIES:
        r = run(strategy, size_str, docs, tokenizer)
        results.append(r)
        print()

    print("=" * 70)
    print(f"{'Strategy':<20} {'Raw':>6} {'Unique':>8} {'Ratio':>7} {'Uniq/1kTok':>11}")
    print("-" * 70)
    for r in results:
        print(f"{r['strategy']:<20} {r['raw']:>6} {r['unique']:>8} {r['ratio']:>7.3f} {r['unique_per_1k']:>11}")

if __name__ == "__main__":
    main()
