"""
exp_reextract_corpus_ladder.py — the corpus-level failure rate AFTER the
retry-with-escalation ladder.

The corpus re-extraction (`tests/exp_reextract_corpus.py`) parsed 39/42
chunks with `frequency_penalty=0.25` and no retry. The open question the
ladder was built to answer is: how many of those 3 failures does the
ladder actually rescue?

This runs the WHOLE corpus through the PRODUCTION code path
(`extract_entities_relationships`, which now walks
`LLM_RETRY_FREQUENCY_PENALTIES`), so the number it reports is the number
the pipeline would actually produce.

  * own CACHE_DIR (data/cache/exp_reextract_ladder) — never the canonical
  * prints CACHE_DIR and the cache entry count at start
  * own output files, no writes to the canonical cache
  * Zvec collections untouched
  * own log file

Run detached:

    python tests/exp_reextract_corpus_ladder.py

Environment:
    EXP_LADDER   the escalation ladder to use, JSON list. Default [0.5, 1.0].
                 Set to "[]" to reproduce the no-retry baseline.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ------------------------------------------------------------------ #
# Own CACHE_DIR — set BEFORE importing graphrag_lite (config reads .env)
# ------------------------------------------------------------------ #
CACHE_DIR = os.environ.get("EXP_LADDER_CACHE_DIR", "data/cache/exp_reextract_ladder")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["CACHE_DIR"] = CACHE_DIR
os.environ["ZVEC_ROOT"] = "data/cache"          # canonical collections, untouched
os.environ["LLM_RETRY_FREQUENCY_PENALTIES"] = os.environ.get("EXP_LADDER", "[0.5,1.0]")

# ------------------------------------------------------------------ #
# Pin EVERY setting the corpus run used, so the two are comparable.
# .env says EXTRACTION_MODEL=qwen2.5:0.5b and LLM_THINK=auto; the
# corpus re-extraction ran qwen2.5:1.5b with think=False. Without these
# overrides this script would measure a DIFFERENT configuration and the
# comparison would be meaningless.
# ------------------------------------------------------------------ #
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
os.environ["EXTRACTION_MODEL"] = "qwen2.5:1.5b"
os.environ["LLM_NUM_PREDICT"] = "4096"
os.environ["LLM_NUM_CTX"] = "8192"
os.environ["LLM_THINK"] = "off"
os.environ["LLM_FREQUENCY_PENALTY"] = "0.25"

import diskcache  # noqa: E402

from graphrag_lite.config import get_config  # noqa: E402
get_config.cache_clear()      # the env pins above must win
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer  # noqa: E402
from graphrag_lite.ingestion.extractor import extract_entities_relationships  # noqa: E402
from graphrag_lite.ingestion.loader import load_documents  # noqa: E402
from graphrag_lite.llm.ollama_client import OllamaClient  # noqa: E402

OUT_JSONL = "data/cache/exp_reextract_ladder.jsonl"
OUT_JSON = "data/cache/exp_reextract_ladder.json"
LOG = "data/cache/exp_reextract_ladder_log.txt"


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


# Truncate the log on a fresh start (an appended log would mix two runs).
if not os.environ.get("EXP_RESUME", "1") == "1" or not os.path.exists(OUT_JSONL):
    open(LOG, "w", encoding="utf-8").close()


cfg = get_config()
cache = diskcache.Cache(CACHE_DIR)

log("=" * 78)
log("EXP: corpus-level failure rate WITH the retry-with-escalation ladder")
log("=" * 78)
log(f"CACHE_DIR              : {cfg.CACHE_DIR}")
log(f"Cache entries at start : {len(cache)}")
log(f"Output file            : {OUT_JSON}")
log(f"model                  : {cfg.effective_extraction_model()}")
log(f"num_predict            : {cfg.LLM_NUM_PREDICT}")
log(f"num_ctx                : {cfg.LLM_NUM_CTX}")
log(f"frequency_penalty      : {cfg.LLM_FREQUENCY_PENALTY}")
log(f"retry ladder           : {cfg.LLM_RETRY_FREQUENCY_PENALTIES}")
log("=" * 78)

# ------------------------------------------------------------------ #
# Build the same chunk list the corpus experiment used
# ------------------------------------------------------------------ #
tokenizer = get_tokenizer(cfg.TOKENIZER)
all_chunks = []
for doc in load_documents("data/raw"):
    all_chunks.extend(chunk_document(doc, cfg, tokenizer))
log(f"[chunk]   {len(all_chunks)} chunks from {len(load_documents('data/raw'))} documents")

client = OllamaClient(
    cfg.effective_extraction_model(),
    temperature=cfg.EXTRACTION_TEMPERATURE,
    seed=cfg.EXTRACTION_SEED,
    timeout=cfg.LLM_TIMEOUT_S,
    num_predict=cfg.LLM_NUM_PREDICT,
    think=cfg.effective_think(),
    num_ctx=cfg.LLM_NUM_CTX,
    frequency_penalty=cfg.LLM_FREQUENCY_PENALTY,
)

records = []
t_start = time.perf_counter()

RAW_DIR = "data/cache/exp_reextract_ladder_raw"


def _dump_raw_attempts(chunk) -> None:
    """
    Save the RAW response of every attempt for a chunk that failed OR
    that had to be retried.

    Without this a failure is not diagnosable after the fact: the record
    holds only the parse error, and the response that produced it is
    gone. (The corpus-wide experiment hit exactly this — a failure with
    ``done_reason='stop'`` and no repeated line, i.e. malformed JSON
    rather than a repetition loop, with no raw text to inspect.)

    It is written for a chunk that merely RETRIED as well, not only for
    one that failed outright: a chunk that fails at attempt 0 and
    succeeds at attempt 1 has still produced an interesting failure, and
    that failure is exactly what the raw dump exists to preserve.

    The responses are already in this experiment's own cache; this pulls
    them out under their own filenames.
    """
    from graphrag_lite.cache.llm_cache import _make_key
    from graphrag_lite.ingestion.extractor import _EXTRACTION_PROMPT_TEMPLATE

    os.makedirs(RAW_DIR, exist_ok=True)
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
    penalties = [None] + list(cfg.LLM_RETRY_FREQUENCY_PENALTIES)
    for penalty in penalties:
        c = client if penalty is None else client.with_overrides(frequency_penalty=penalty)
        key = _make_key(prompt, cfg.effective_extraction_model(),
                        getattr(c, "cache_tag", ""))
        raw = cache.get(key, default=None)
        if raw is None:
            continue
        # Name the file by the ACTUAL penalty used. `None` means "the
        # client as configured", which already carries
        # frequency_penalty=0.25 -- it does not mean "no penalty".
        suffix = f"fp{c.frequency_penalty}"
        path = os.path.join(RAW_DIR, f"{chunk['chunk_id']}__{suffix}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(raw)


# ------------------------------------------------------------------ #
# Resume support — the JSONL is written incrementally, so an interrupted
# run continues from where it stopped instead of re-issuing the requests
# that already completed. The model is BISTABLE (DOCUMENT.md Item 3), so
# resumed chunks are issued in a later session; that is a documented
# property of the measurement, not a hidden change.
# ------------------------------------------------------------------ #
if os.path.exists(OUT_JSONL):
    with open(OUT_JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if records:
        log(f"[resume]  {len(records)} records already in {OUT_JSONL}")

done_ids = {r["chunk_id"] for r in records}

for i, chunk in enumerate(all_chunks):
    if chunk["chunk_id"] in done_ids:
        continue
    t0 = time.perf_counter()
    status = "ok"
    err = ""
    n_ent = n_rel = -1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = extract_entities_relationships(chunk, client)
            n_ent = len(result["entities"])
            n_rel = len(result["relationships"])
        except Exception as e:                      # noqa: BLE001
            status = "FAIL"
            err = f"{type(e).__name__}: {str(e)[:300]}"
    seconds = time.perf_counter() - t0

    retries = sum(1 for w in caught if "retrying with a higher penalty" in str(w.message))

    # Dump the raw response of every attempt whenever a chunk failed OR
    # had to be retried — a rescued chunk still produced a failure, and
    # that failure is what the dump exists to preserve.
    if status != "ok" or retries > 0:
        _dump_raw_attempts(chunk)

    rec = {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "index": chunk.get("chunk_index", chunk.get("index")),
        "status": status,
        "entities": n_ent,
        "relationships": n_rel,
        "seconds": round(seconds, 1),
        "retries": retries,
        "error": err,
    }
    records.append(rec)
    log(
        f"  [{i + 1:2d}/{len(all_chunks)}] {chunk['doc_id']}/"
        f"{chunk.get('chunk_index', chunk.get('index'))} "
        f"{chunk['chunk_id'][:12]}  {status:4s}  ent={n_ent:3d} rel={n_rel:3d}  "
        f"retries={retries}  {seconds:7.1f}s"
    )
    with open(OUT_JSONL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

wall = time.perf_counter() - t_start
ok = [r for r in records if r["status"] == "ok"]
failed = [r for r in records if r["status"] != "ok"]
retried = [r for r in records if r.get("retries", 0) > 0]

summary = {
    "chunks": len(records),
    "parsed_ok": len(ok),
    "failures": len(failed),
    "failed_ids": [r["chunk_id"] for r in failed],
    "chunks_that_retried": len(retried),
    "retried_ids": [r["chunk_id"] for r in retried],
    "entities": sum(max(r["entities"], 0) for r in records),
    "relationships": sum(max(r["relationships"], 0) for r in records),
    "wall_clock_s": round(wall, 1),
    "ladder": cfg.LLM_RETRY_FREQUENCY_PENALTIES,
    "frequency_penalty": cfg.LLM_FREQUENCY_PENALTY,
}

log("=" * 78)
log(f"  chunks                  : {summary['chunks']}")
log(f"  parsed OK               : {summary['parsed_ok']}")
log(f"  failures                : {summary['failures']}  {summary['failed_ids']}")
log(f"  chunks that RETRIED     : {summary['chunks_that_retried']}  {summary['retried_ids']}")
log(f"  entities extracted      : {summary['entities']}")
log(f"  relationships extracted : {summary['relationships']}")
log(f"  wall clock              : {summary['wall_clock_s']} s")
log("=" * 78)

with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump({"summary": summary, "records": records}, fh, indent=2)

log(f"CACHE_DIR in use         : {cfg.CACHE_DIR}")
log(f"Cache entries at end     : {len(cache)}")
