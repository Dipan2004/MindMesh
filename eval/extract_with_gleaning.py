"""
extract_with_gleaning.py — Resumable, per-chunk gleaning-ON extractor.

Processes the 42 canonical chunks one at a time.
Writes each chunk result immediately after completion so any interruption
can be resumed without re-running completed chunks.

DESIGN
  - State file: data/experiments/qwen2_5_1_5b_genuine_gleaning_on/extractions.jsonl
  - Each completed chunk is ONE line in that JSONL file.
  - On restart, already-completed chunk IDs are loaded and skipped.
  - Every LLM call (extraction pass + gleaning passes) is counted and logged.
  - Cache lives in: data/experiments/qwen2_5_1_5b_genuine_gleaning_on/llm_cache/
    (completely isolated from canonical and v2 caches)
  - Extract first pass, then exactly one gleaning pass (MAX_GLEANINGS=1).
    Both counts are recorded per chunk.

CORRECTNESS
  - The chunk dict passed to the extractor contains BOTH 'doc_id' AND 'source_doc'
    to satisfy the Chunk TypedDict contract ('doc_id') while preserving what the
    Zvec collection stored ('source_doc').
  - A chunk is only written to the JSONL if extraction completed without error.
  - A gleaning parse failure raises, logs the error, and marks the chunk as
    extraction_ok=True, gleaning_ok=False. The chunk is still written (with
    the first-pass entities and relationships) so the run can continue.
  - The cache-key model is taken from client.model (the fix applied 2026-11-09).

GENUINE GLEANING
  - glean() is called with max_gleanings=1 for every chunk.
  - The function is in src/graphrag_lite/ingestion/gleaning.py.
  - It issues one additional LLM call asking "did you miss anything?".
  - The response is parsed and merged with the first pass.
  - gleaning_extract_calls == 1 per chunk means gleaning was genuinely called.

Usage:
    python eval/extract_with_gleaning.py           # start or resume
    python eval/extract_with_gleaning.py --status  # show progress only, no extraction
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

EXP_DIR   = ROOT / "data" / "experiments" / "qwen2_5_1_5b_genuine_gleaning_on"
CACHE_DIR = str(EXP_DIR / "llm_cache")
JSONL_OUT = EXP_DIR / "extractions.jsonl"
LOG_FILE  = EXP_DIR / "extraction.log"
MODEL     = "qwen2.5:1.5b"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gleaning_extract")


# ── Read canonical chunks ─────────────────────────────────────────────────────

def read_canonical_chunks() -> list[dict]:
    import zvec
    col = zvec.open(str(ROOT / "data" / "cache" / "chunks_v1"))
    try:
        res = col.query(
            queries=[zvec.Query("text_vec", vector=[0.0]*768,
                                param=zvec.HnswQueryParam(ef=300))],
            topk=100,
        )
        chunks = []
        for doc in res:
            f = doc.fields or {}
            src = f.get("source_doc", "")
            # BOTH keys: 'doc_id' satisfies Chunk TypedDict; 'source_doc' preserves Zvec field
            chunks.append({
                "chunk_id":    doc.id,
                "doc_id":      src,
                "source_doc":  src,
                "chunk_index": int(f.get("chunk_index", 0)),
                "text":        f.get("text", ""),
            })
    finally:
        col.close()
    chunks.sort(key=lambda c: (c["doc_id"], c["chunk_index"]))
    return chunks


# ── Load already-completed chunks ─────────────────────────────────────────────

def load_completed(jsonl_path: Path) -> dict[str, dict]:
    """Return {chunk_id: record} for all lines in the JSONL that are complete."""
    if not jsonl_path.exists():
        return {}
    completed = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cid = rec.get("chunk_id", "")
                if cid:
                    completed[cid] = rec
            except json.JSONDecodeError:
                pass  # corrupted line — will be skipped
    return completed


# ── Single-chunk extraction + gleaning ───────────────────────────────────────

def extract_chunk(chunk: dict, client, cfg) -> dict:
    """
    Run extraction + one gleaning pass on a single chunk.

    Returns a record dict ready for JSONL serialisation.
    Never returns without having genuinely called glean() once.

    LLM call counting:
      extract_calls  = number of attempts by extract_entities_relationships
                       (1 if attempt-0 succeeds, more if the retry ladder fires)
      gleaning_calls = number of gleaning-prompt LLM calls (always 1 when
                       MAX_GLEANINGS=1 and the gleaning pass does not raise)
    """
    import graphrag_lite.cache.llm_cache as lc_mod
    from graphrag_lite.ingestion.extractor import extract_entities_relationships
    from graphrag_lite.ingestion.gleaning import glean

    cid = chunk["chunk_id"]
    doc = chunk["doc_id"]

    hits_before  = lc_mod._hits
    misses_before = lc_mod._misses

    t0 = time.monotonic()
    attempt_warnings: list[str] = []

    # Pass 1 — extraction (with retry ladder if needed)
    extract_calls = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        extraction = extract_entities_relationships(chunk, client)
        for w in caught:
            msg = str(w.message)
            attempt_warnings.append(msg)
            if "retrying" in msg.lower():
                extract_calls += 1
    extract_calls += 1  # always at least 1 successful call

    n_e1 = len(extraction.get("entities", []))
    n_r1 = len(extraction.get("relationships", []))
    t_extract = time.monotonic() - t0

    # Pass 2 — genuine gleaning (always 1 call; raises on parse failure)
    gleaning_ok     = False
    gleaning_calls  = 0
    gleaning_error  = None
    glean_added_e   = 0
    glean_added_r   = 0
    t_glean_start   = time.monotonic()

    with warnings.catch_warnings(record=True) as caught_g:
        warnings.simplefilter("always")
        try:
            result = glean(chunk, extraction, client, max_gleanings=1)
            gleaning_ok    = True
            gleaning_calls = 1
            glean_added_e  = len(result.get("entities", [])) - n_e1
            glean_added_r  = len(result.get("relationships", [])) - n_r1
            for w in caught_g:
                attempt_warnings.append(str(w.message))
        except ValueError as exc:
            gleaning_error = str(exc)
            log.warning("  Gleaning parse error on %s: %s", cid[:12], gleaning_error[:80])
            # Use first-pass result; mark gleaning as failed
            result = extraction

    t_glean = time.monotonic() - t_glean_start
    total_dt = time.monotonic() - t0

    hits_after   = lc_mod._hits
    misses_after = lc_mod._misses

    return {
        "chunk_id":      cid,
        "doc_id":        doc,
        "chunk_index":   chunk["chunk_index"],
        # text is stored but excluded from the JSONL serialisation (retrieved from canonical)
        "text":          chunk["text"],
        "entities":      result.get("entities", []),
        "relationships": result.get("relationships", []),
        "extraction_model": client.model,
        "gleaning": True,
        "gleaning_ok":   gleaning_ok,
        "gleaning_error": gleaning_error,
        # counts
        "extract_calls":    extract_calls,
        "gleaning_calls":   gleaning_calls,
        "entities_after_extract": n_e1,
        "rels_after_extract":     n_r1,
        "entities_final":  len(result.get("entities", [])),
        "rels_final":      len(result.get("relationships", [])),
        "glean_added_entities": glean_added_e,
        "glean_added_rels":     glean_added_r,
        # timings
        "latency_extract_s": round(t_extract, 3),
        "latency_glean_s":   round(t_glean, 3),
        "latency_total_s":   round(total_dt, 3),
        # cache
        "cache_hits":   hits_after - hits_before,
        "cache_misses": misses_after - misses_before,
        # diagnostics
        "warnings": attempt_warnings,
    }


# ── Status report ─────────────────────────────────────────────────────────────

def print_status(completed: dict[str, dict], chunks: list[dict]) -> None:
    total      = len(chunks)
    done       = len(completed)
    remaining  = total - done
    errors     = sum(1 for r in completed.values() if r.get("_parse_error"))
    glean_ok   = sum(1 for r in completed.values() if r.get("gleaning_ok"))
    glean_fail = sum(1 for r in completed.values() if r.get("gleaning_ok") is False and not r.get("_parse_error"))
    t_ents     = sum(r.get("entities_final", len(r.get("entities", []))) for r in completed.values())
    t_rels     = sum(r.get("rels_final", len(r.get("relationships", []))) for r in completed.values())
    g_ents     = sum(r.get("glean_added_entities", 0) for r in completed.values())
    g_rels     = sum(r.get("glean_added_rels", 0) for r in completed.values())
    e_calls    = sum(r.get("extract_calls", 0) for r in completed.values())
    glean_calls= sum(r.get("gleaning_calls", 0) for r in completed.values())
    cache_hits = sum(r.get("cache_hits", 0) for r in completed.values())
    cache_miss = sum(r.get("cache_misses", 0) for r in completed.values())

    print(f"\n=== GENUINE GLEANING EXTRACTION STATUS ===")
    print(f"  Completed: {done}/{total}  Remaining: {remaining}")
    print(f"  Extraction OK: {done - errors}  Parse errors: {errors}")
    print(f"  Gleaning OK: {glean_ok}  Gleaning fail: {glean_fail}")
    print(f"  Extract calls: {e_calls}  Gleaning calls: {glean_calls}  (expected {done - errors} each)")
    print(f"  Total entities (final): {t_ents}  (+{g_ents} from gleaning)")
    print(f"  Total rels (final):     {t_rels}  (+{g_rels} from gleaning)")
    print(f"  Cache hits: {cache_hits}  misses: {cache_miss}")
    if done:
        total_s = sum(r.get("latency_total_s", 0) for r in completed.values())
        avg_s   = total_s / done
        est_rem = avg_s * remaining
        print(f"  Avg time/chunk: {avg_s:.1f}s  Est remaining: {est_rem/60:.1f} min")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Resumable genuine gleaning-ON extraction for qwen2.5:1.5b"
    )
    parser.add_argument("--status", action="store_true",
                        help="Print progress and exit without running extraction")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after processing N new chunks (0=all remaining)")
    args = parser.parse_args()

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    # Add file handler to log
    fh = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("=== extract_with_gleaning.py  model=%s  MAX_GLEANINGS=1 ===", MODEL)
    log.info("Output: %s", JSONL_OUT)
    log.info("Cache:  %s", CACHE_DIR)

    # Step 1: read chunks
    chunks = read_canonical_chunks()
    log.info("Canonical chunks: %d", len(chunks))

    # Step 2: load completed
    completed = load_completed(JSONL_OUT)
    log.info("Already completed: %d/%d", len(completed), len(chunks))

    print_status(completed, chunks)

    if args.status:
        return 0

    pending = [c for c in chunks if c["chunk_id"] not in completed]
    if not pending:
        print("All 42 chunks complete. Nothing to do.")
        return 0

    print(f"Starting extraction of {len(pending)} remaining chunks...")
    if args.limit:
        pending = pending[:args.limit]
        print(f"  (--limit {args.limit}: processing {len(pending)} chunks this run)")

    # Step 3: set up private cache
    import diskcache
    import graphrag_lite.cache.llm_cache as lc_mod
    from graphrag_lite.config import get_config

    os.makedirs(CACHE_DIR, exist_ok=True)
    old_cache  = lc_mod._cache
    old_hits   = lc_mod._hits
    old_misses = lc_mod._misses
    lc_mod._cache  = diskcache.Cache(CACHE_DIR)
    lc_mod._hits   = 0
    lc_mod._misses = 0

    cfg = get_config()

    from graphrag_lite.llm.ollama_client import OllamaClient
    client = OllamaClient(
        model=MODEL,
        temperature=cfg.EXTRACTION_TEMPERATURE,  # 0.0
        seed=cfg.EXTRACTION_SEED,                # 42
        timeout=cfg.LLM_TIMEOUT_S,              # 300s
        num_predict=cfg.LLM_NUM_PREDICT,        # 4096
        num_ctx=cfg.LLM_NUM_CTX,               # 8192
        think=False,
        frequency_penalty=cfg.LLM_FREQUENCY_PENALTY,  # 0.25
    )

    log.info("Client: model=%s  T=%.1f  seed=%d  np=%d  ctx=%d  fp=%.2f  think=False",
             client.model, client.temperature, client.seed,
             client.num_predict, client.num_ctx or 0, client.frequency_penalty)
    log.info("Cache identity: client.model=%r  cache_tag=%r",
             client.model, client.cache_tag)

    # Step 4: extract chunk by chunk
    t_run = time.monotonic()
    newly_done = 0
    parse_errors = 0

    try:
        for chunk in pending:
            cid = chunk["chunk_id"]
            doc = chunk["doc_id"]
            log.info("[%02d/%02d remaining] %s  %s",
                     newly_done + 1, len(pending), cid[:12], doc[:20])

            try:
                rec = extract_chunk(chunk, client, cfg)
                # Write immediately — never lose a completed chunk
                with open(JSONL_OUT, "a", encoding="utf-8") as f:
                    out = {k: v for k, v in rec.items() if k != "text"}
                    f.write(json.dumps(out, ensure_ascii=False, default=list) + "\n")

                completed[cid] = rec
                newly_done += 1

                log.info(
                    "  -> %d ent / %d rel  (+%d/+%d from glean)  "
                    "extract_calls=%d glean_calls=%d  "
                    "%.1fs extract + %.1fs glean = %.1fs total  "
                    "cache: h=%d m=%d",
                    rec["entities_final"], rec["rels_final"],
                    rec["glean_added_entities"], rec["glean_added_rels"],
                    rec["extract_calls"], rec["gleaning_calls"],
                    rec["latency_extract_s"], rec["latency_glean_s"],
                    rec["latency_total_s"],
                    lc_mod._hits, lc_mod._misses,
                )

                if not rec["gleaning_ok"]:
                    log.warning("  Gleaning failed on %s: %s", cid[:12],
                                rec.get("gleaning_error", "?")[:80])

            except Exception as exc:
                parse_errors += 1
                log.error("  EXTRACTION FAILED %s: %s", cid[:12], str(exc)[:120])
                # Write a parse-error record so we know it was attempted
                err_rec = {
                    "chunk_id": cid, "doc_id": doc,
                    "chunk_index": chunk["chunk_index"],
                    "entities": [], "relationships": [],
                    "extraction_model": MODEL,
                    "gleaning": True, "gleaning_ok": False,
                    "_parse_error": str(exc),
                    "extract_calls": 0, "gleaning_calls": 0,
                    "latency_total_s": 0.0,
                }
                with open(JSONL_OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(err_rec, ensure_ascii=False, default=list) + "\n")
                completed[cid] = err_rec
                newly_done += 1

    finally:
        # Restore cache
        lc_mod._cache  = old_cache
        lc_mod._hits   = old_hits
        lc_mod._misses = old_misses

    wall = time.monotonic() - t_run
    log.info("Run complete: %d new chunks in %.1fs  parse_errors=%d",
             newly_done, wall, parse_errors)

    print_status(completed, chunks)

    # Save a summary JSON after each run
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL,
        "completed": len(completed),
        "total": len(chunks),
        "parse_errors": sum(1 for r in completed.values() if r.get("_parse_error")),
        "gleaning_ok": sum(1 for r in completed.values() if r.get("gleaning_ok")),
        "gleaning_fail": sum(1 for r in completed.values()
                             if r.get("gleaning_ok") is False and not r.get("_parse_error")),
        "total_entities": sum(r.get("entities_final", len(r.get("entities", [])))
                              for r in completed.values()),
        "total_rels": sum(r.get("rels_final", len(r.get("relationships", [])))
                         for r in completed.values()),
        "glean_added_entities": sum(r.get("glean_added_entities", 0)
                                    for r in completed.values()),
        "glean_added_rels": sum(r.get("glean_added_rels", 0)
                                for r in completed.values()),
        "total_extract_calls": sum(r.get("extract_calls", 0) for r in completed.values()),
        "total_gleaning_calls": sum(r.get("gleaning_calls", 0) for r in completed.values()),
    }
    with open(EXP_DIR / "extraction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved: %s", EXP_DIR / "extraction_summary.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
