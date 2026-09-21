"""
build_index.py — CLI for running the full indexing pipeline.

Stage 1 (--stage extract): loader → chunker → extractor (+gleaning) across
all documents in data/raw/, writing results to data/cache/extractions.jsonl.

Usage:
    python scripts/build_index.py --stage extract
    python scripts/build_index.py --stage extract --data-dir data/raw --out data/cache/extractions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Broad exception set for all network/timeout/parse failures.
# Must be defined before src imports so _CATCH_EXCS is available at module level.
try:
    import httpx as _httpx
    import httpcore as _httpcore
    _CATCH_EXCS = (ValueError, RuntimeError, ConnectionError,
                   TimeoutError, OSError,
                   _httpx.ReadTimeout, _httpcore.ReadTimeout)
except ImportError:
    _CATCH_EXCS = (ValueError, RuntimeError, ConnectionError, TimeoutError, OSError)

# Ensure the src package is importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.cache.llm_cache import cache_stats, reset_stats
from graphrag_lite.config import get_config
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import extract_entities_relationships
from graphrag_lite.ingestion.gleaning import glean
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.llm.ollama_client import OllamaClient


def run_extract(data_dir: str, out_path: str) -> None:
    cfg = get_config()
    tokenizer = get_tokenizer(cfg.TOKENIZER)
    client = OllamaClient(
        cfg.effective_extraction_model(),
        temperature=cfg.EXTRACTION_TEMPERATURE,
        seed=cfg.EXTRACTION_SEED,
        timeout=cfg.LLM_TIMEOUT_S,
    )

    reset_stats()
    t0 = time.perf_counter()

    # ── Load ──────────────────────────────────────────────────────────
    docs = load_documents(data_dir)
    print(f"[load]    {len(docs)} documents loaded from {data_dir!r}")

    # ── Chunk ─────────────────────────────────────────────────────────
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, cfg, tokenizer))
    print(
        f"[chunk]   {len(all_chunks)} chunks "
        f"(size={cfg.CHUNK_SIZE_EXTRACTION} tokens, overlap={cfg.CHUNK_OVERLAP_PCT:.0%})"
    )

    # ── Extract + Glean ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    total_entities = 0
    total_rels = 0
    # chunk_ids where extraction failed — separate from legitimate 0-entity chunks
    failed_chunks: list[str] = []

    with open(out_path, "w", encoding="utf-8") as fh:
        for i, chunk in enumerate(all_chunks):
            print(
                f"[extract] chunk {i+1:3d}/{len(all_chunks)}  "
                f"doc={chunk['doc_id']}  chunk_idx={chunk['chunk_index']}",
                end="  ",
                flush=True,
            )

            # ── Extraction ────────────────────────────────────────────
            try:
                result = extract_entities_relationships(chunk, client)
            except _CATCH_EXCS as exc:
                # Visible failure — TASKS.md rule 4: must not be silent.
                failed_chunks.append(chunk["chunk_id"])
                print(
                    f"FAILED (extract: {type(exc).__name__})  "
                    f"chunk_id={chunk['chunk_id']}\n"
                    f"  [WARN] {str(exc)[:200]}",
                    flush=True,
                )
                result = {
                    "entities": [],
                    "relationships": [],
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "_parse_error": f"{type(exc).__name__}: {str(exc)[:200]}",
                }
            else:
                # ── Gleaning — separate so failure uses extraction-only result ──
                if cfg.MAX_GLEANINGS > 0:
                    try:
                        result = glean(
                            chunk, result, client, max_gleanings=cfg.MAX_GLEANINGS
                        )
                    except _CATCH_EXCS as exc:
                        print(
                            f"\n  [WARN] glean failed chunk_id={chunk['chunk_id']}: "
                            f"{type(exc).__name__}: {str(exc)[:120]}"
                            f" — using extraction-only result",
                            flush=True,
                        )

                print(
                    f"entities={len(result['entities'])}  "
                    f"rels={len(result['relationships'])}",
                    flush=True,
                )

            total_entities += len(result["entities"])
            total_rels += len(result["relationships"])
            fh.write(json.dumps(result) + "\n")

    elapsed = time.perf_counter() - t0
    stats = cache_stats()
    extracted_ok = len(all_chunks) - len(failed_chunks)

    print()
    print("=" * 60)
    print(f"  Documents  : {len(docs)}")
    print(f"  Chunks     : {len(all_chunks)}")
    print(
        f"  Extracted  : {extracted_ok}/{len(all_chunks)} chunks"
        + (f"  |  Failed (error): {len(failed_chunks)}/{len(all_chunks)} chunks"
           if failed_chunks else "  |  Failed: 0")
    )
    if failed_chunks:
        for fid in failed_chunks:
            print(f"    - FAILED chunk_id={fid}")
    print(f"  Entities   : {total_entities}")
    print(f"  Relationships: {total_rels}")
    print(f"  Cache hits : {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
    print(f"  Elapsed    : {elapsed:.1f}s")
    print(f"  Output     : {out_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="MindMesh indexing pipeline")
    parser.add_argument(
        "--stage",
        choices=["extract"],
        required=True,
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing source documents (default: data/raw)",
    )
    parser.add_argument(
        "--out",
        default="data/cache/extractions.jsonl",
        help="Output JSONL path (default: data/cache/extractions.jsonl)",
    )
    args = parser.parse_args()

    if args.stage == "extract":
        run_extract(args.data_dir, args.out)


if __name__ == "__main__":
    main()
