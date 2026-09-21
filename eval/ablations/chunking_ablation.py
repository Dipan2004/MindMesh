"""
chunking_ablation.py — Phase 1 extension.

Runs all 4 chunking strategies through the full pipeline
(chunk -> extract -> glean) on the 5-doc corpus with
temperature=0.0 / seed=42, and prints a comparison table.

Usage:
    python eval/ablations/chunking_ablation.py

Output: console table + writes eval/ablations/chunking_results.json
"""
from __future__ import annotations

import json
import os
import sys
import statistics
import time
import warnings

# Make sure src/ is importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import (
    extract_entities_relationships,
    _EXTRACTION_PROMPT_TEMPLATE,
    _extract_json,
    _parse_extraction,
)
from graphrag_lite.ingestion.gleaning import glean
from graphrag_lite.llm.ollama_client import OllamaClient
from graphrag_lite.cache.llm_cache import clear_cache, reset_stats, cache_stats
from graphrag_lite.config import get_config

# ── Spot-check sentences ────────────────────────────────────────────────────
# Three sentences each containing a real named entity.
# We verify each chunk set has at least one chunk that contains the full
# sentence without splitting it across a boundary.
SPOT_SENTENCES = [
    # 1) FAISS entity — vector_databases.md
    "FAISS (Facebook AI Similarity Search) is an open-source library developed by Meta AI Research "
    "for efficient similarity search and clustering of dense vectors.",
    # 2) Transformer component — transformer_architecture.md
    "The self-attention mechanism allows the model to weigh the importance of different tokens in a "
    "sequence when encoding a particular token.",
    # 3) Wikidata — knowledge_graphs.md
    "Wikidata is a free, collaboratively edited knowledge base maintained by the Wikimedia Foundation.",
]

SPOT_LABELS = ["FAISS (vector_databases)", "Self-Attention (transformer)", "Wikidata (knowledge_graphs)"]


def _sentence_intact(chunks, sentence: str) -> str:
    """
    Return 'intact' if the sentence appears complete in at least one chunk,
    'split' if parts appear across two chunks but never whole,
    'missing' if it doesn't appear at all.
    """
    sentence_clean = sentence.strip()
    for c in chunks:
        if sentence_clean in c["text"]:
            return "intact"
    # Check if parts appear across neighbouring chunks
    parts = sentence_clean.split()
    first_5 = " ".join(parts[:5])
    last_5 = " ".join(parts[-5:])
    has_start = any(first_5 in c["text"] for c in chunks)
    has_end = any(last_5 in c["text"] for c in chunks)
    if has_start and has_end:
        return "split"
    return "missing"


def run_strategy(strategy: str, docs, tokenizer) -> dict:
    """Run chunking + extraction + gleaning for one strategy. Returns metrics dict."""
    # Override the env var so get_config() picks up the new strategy
    os.environ["CHUNKING_STRATEGY"] = strategy
    if strategy == "small_window":
        os.environ["CHUNK_SIZE_EXTRACTION"] = "250"
    else:
        os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
    get_config.cache_clear()
    cfg = get_config()

    client = OllamaClient(
        cfg.effective_extraction_model(),
        temperature=cfg.EXTRACTION_TEMPERATURE,
        seed=cfg.EXTRACTION_SEED,
        timeout=cfg.LLM_TIMEOUT_S,
    )

    # Total source tokens across all docs (for entities/1000-token normalisation)
    total_source_tokens = sum(len(tokenizer.encode(d["text"])) for d in docs)

    # ── Chunking ────────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, cfg, tokenizer))

    token_counts = [c["token_count"] for c in all_chunks]
    chunk_count = len(all_chunks)

    # Measure actual overlaps between consecutive chunks of the same doc
    overlaps = []
    for i in range(1, len(all_chunks)):
        if all_chunks[i]["doc_id"] == all_chunks[i-1]["doc_id"]:
            prev_end = all_chunks[i-1]["end_offset"]
            curr_start = all_chunks[i]["start_offset"]
            overlap = max(0, prev_end - curr_start)
            overlaps.append(overlap)

    # ── Spot-check ──────────────────────────────────────────────────────────
    spot_results = [_sentence_intact(all_chunks, s) for s in SPOT_SENTENCES]
    intact_count = sum(1 for r in spot_results if r == "intact")

    # ── Extraction + Gleaning ───────────────────────────────────────────────
    reset_stats()
    total_entities = 0
    total_rels = 0
    failed = 0
    llm_calls_before = 0  # will use cache_stats misses as proxy

    for chunk in all_chunks:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = extract_entities_relationships(chunk, client)
        except (ValueError, RuntimeError, ConnectionError, TimeoutError, OSError):
            failed += 1
            continue
        if cfg.MAX_GLEANINGS > 0:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = glean(chunk, result, client, max_gleanings=cfg.MAX_GLEANINGS)
            except (ValueError, RuntimeError, ConnectionError, TimeoutError, OSError):
                pass  # use extraction-only result
        total_entities += len(result["entities"])
        total_rels += len(result["relationships"])

    t_elapsed = time.perf_counter() - t_start
    stats = cache_stats()

    # entities per 1000 source tokens (normalised)
    ent_per_1k = (total_entities / total_source_tokens * 1000) if total_source_tokens > 0 else 0

    return {
        "strategy": strategy,
        "chunk_count": chunk_count,
        "token_min": min(token_counts),
        "token_median": statistics.median(token_counts),
        "token_max": max(token_counts),
        "overlap_median": round(statistics.median(overlaps), 1) if overlaps else 0,
        "total_entities": total_entities,
        "total_rels": total_rels,
        "failed_chunks": failed,
        "fail_rate_pct": round(failed / chunk_count * 100, 1) if chunk_count > 0 else 0,
        "ent_per_chunk": round(total_entities / chunk_count, 2) if chunk_count > 0 else 0,
        "ent_per_1k_tokens": round(ent_per_1k, 2),
        "total_llm_calls": stats["hits"] + stats["misses"],
        "cache_hits": stats["hits"],
        "cache_misses": stats["misses"],
        "wall_time_s": round(t_elapsed, 1),
        "spot_intact": intact_count,
        "spot_details": spot_results,
        "total_source_tokens": total_source_tokens,
    }


def print_table(results: list[dict]) -> None:
    cols = [
        ("Strategy", "strategy", 20),
        ("Chunks", "chunk_count", 7),
        ("Tok med", "token_median", 8),
        ("Entities", "total_entities", 9),
        ("Rels", "total_rels", 5),
        ("Fail%", "fail_rate_pct", 6),
        ("Ent/1kTok", "ent_per_1k_tokens", 10),
        ("LLM calls", "total_llm_calls", 10),
        ("Time(s)", "wall_time_s", 8),
        ("Spot(3)", "spot_intact", 8),
    ]
    header = "  ".join(f"{h:{w}}" for h, _, w in cols)
    print(header)
    print("-" * len(header))
    for r in results:
        row = "  ".join(
            f"{str(r[k]):{w}}" for _, k, w in cols
        )
        print(row)


def main():
    tokenizer = get_tokenizer("qwen")
    docs = load_documents("data/raw")
    print(f"Corpus: {len(docs)} documents")
    print(f"temperature=0.0  seed=42  (pinned)")
    print()

    strategies = ["sliding_window", "recursive_char", "markdown_aware", "small_window"]
    results = []

    for strategy in strategies:
        print(f"{'='*60}")
        print(f"Running strategy: {strategy}")
        print(f"{'='*60}")
        clear_cache()
        r = run_strategy(strategy, docs, tokenizer)
        results.append(r)
        print(f"  chunks={r['chunk_count']}  tok_med={r['token_median']}  "
              f"entities={r['total_entities']}  rels={r['total_rels']}  "
              f"failed={r['failed_chunks']}  "
              f"ent/1kTok={r['ent_per_1k_tokens']}  "
              f"llm_calls={r['total_llm_calls']}  time={r['wall_time_s']}s")
        print(f"  spot-check: {r['spot_details']}")
        print()

    print()
    print("=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print_table(results)

    print()
    print("SPOT-CHECK DETAIL")
    print("-" * 60)
    for r in results:
        print(f"  {r['strategy']:20s}", end="")
        for label, detail in zip(SPOT_LABELS, r["spot_details"]):
            print(f"  [{detail:7s}] {label}", end="")
        print()

    # Determine winners
    print()
    print("ANALYSIS")
    print("-" * 60)
    best_quality = max(results, key=lambda r: r["ent_per_1k_tokens"] * (1 - r["fail_rate_pct"]/100))
    best_cost = min(results, key=lambda r: r["total_llm_calls"])
    print(f"  Best extraction quality (ent/1kTok × (1-fail_rate)): {best_quality['strategy']}")
    print(f"  Cheapest (fewest LLM calls):                          {best_cost['strategy']}")
    if best_quality["strategy"] == best_cost["strategy"]:
        print(f"  → Clear winner on both: {best_quality['strategy']}")
    else:
        print(f"  → No single winner — see recommendation below.")

    # Save results
    os.makedirs("eval/ablations", exist_ok=True)
    out_path = "eval/ablations/chunking_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
