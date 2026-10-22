"""
exp_ladder_hard.py — does the retry ladder actually RESCUE, or does it just
cost 3x the wall time?

The corpus ladder run (`tests/exp_reextract_corpus_ladder.py`) answers
"what is the corpus-level failure rate of the production path?". It cannot
answer "how often does a retry rescue a chunk that failed", because on a
given run the chunks that happen to fail are whichever ones the bistable
model happened to lose — and on most runs the set of failures is small or
empty.

This script pins the question to the chunks that are KNOWN to be hard, and
repeats, so the rescue rate is a measured number rather than an anecdote.

The four chunks, and why each is in the set:
  4af3dc8c  repetition loop  (failed in the fp=0.25 corpus run; passed 3/3
                                in the isolated probe — bistable)
  24b17cf3  repetition loop  (failed in the fp=0.25 corpus run; 0/2 in the
                                18-run probe at ctx=8192, 2/2 with ctx unset)
  f08417a7  repetition loop  (failed in the fp=0.25 corpus run; 1/2 in the
                                probe)
  0e82efa9  malformed JSON   (failed in the fp=0 control arm with
                                done=stop, x=1, a MISSING comma — the quiet
                                second failure mode, not a loop)

Each chunk is run through the PRODUCTION `extract_entities_relationships`,
which walks the configured ladder. The outcome of each attempt is inferred
from the retry warnings the extractor emits, and the raw response of every
attempt is read back out of this run's own cache so the failure MODE
(repetition loop vs malformed JSON) can be named rather than guessed.

Isolation:
  * own CACHE_DIR, taken from EXP_LADDER_CACHE_DIR — never the canonical one
  * own output file, own log file, own raw-response directory
  * prints CACHE_DIR and the cache entry count at start
  * Zvec collections untouched

Run one repetition:

    EXP_LADDER_CACHE_DIR=data/cache/exp_ladder_hard_r0 \
    EXP_REP=0 python tests/exp_ladder_hard.py

Each repetition MUST use a different EXP_LADDER_CACHE_DIR: the extractor
goes through `cached_call`, so a repeated chunk+settings pair would be
served from the cache and the repetition would measure nothing. A fresh
cache dir means a fresh request to the server, which is a fresh decode.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ------------------------------------------------------------------ #
# Own CACHE_DIR — BEFORE importing graphrag_lite (config reads .env)
# ------------------------------------------------------------------ #
CACHE_DIR = os.environ.get("EXP_LADDER_CACHE_DIR", "data/cache/exp_ladder_hard_r0")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["CACHE_DIR"] = CACHE_DIR
os.environ["ZVEC_ROOT"] = "data/cache"          # canonical collections, read-only
os.environ["LLM_RETRY_FREQUENCY_PENALTIES"] = os.environ.get("EXP_LADDER", "[0.5,1.0]")

# Pin every setting the corpus run used, so the numbers are comparable.
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
from graphrag_lite.cache.llm_cache import _make_key  # noqa: E402
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer  # noqa: E402
from graphrag_lite.ingestion.extractor import _EXTRACTION_PROMPT_TEMPLATE  # noqa: E402
from graphrag_lite.ingestion.extractor import extract_entities_relationships  # noqa: E402
from graphrag_lite.ingestion.loader import load_documents  # noqa: E402
from graphrag_lite.llm.ollama_client import OllamaClient  # noqa: E402

OUT = "data/cache/exp_ladder_hard.jsonl"
LOG = "data/cache/exp_ladder_hard_log.txt"
RAW_DIR = os.path.join(CACHE_DIR, "raw")

REP = os.environ.get("EXP_REP", "0")

HARD_CHUNK_IDS = [
    "4af3dc8c5da84fad",   # repetition loop
    "24b17cf355abdce3",   # repetition loop
    "f08417a79e9ed8e6",   # repetition loop
    "0e82efa9409ff33c",   # malformed JSON (missing comma) - the quiet mode
]


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


cfg = get_config()
cache = diskcache.Cache(CACHE_DIR)
os.makedirs(RAW_DIR, exist_ok=True)

log("=" * 78)
log(f"EXP: does the retry ladder RESCUE?  (repetition {REP})")
log("=" * 78)
log(f"CACHE_DIR              : {cfg.CACHE_DIR}")
log(f"Cache entries at start : {len(cache)}")
log(f"Output file            : {OUT}")
log(f"model                  : {cfg.effective_extraction_model()}")
log(f"num_predict            : {cfg.LLM_NUM_PREDICT}")
log(f"num_ctx                : {cfg.LLM_NUM_CTX}")
log(f"frequency_penalty      : {cfg.LLM_FREQUENCY_PENALTY}")
log(f"retry ladder           : {cfg.LLM_RETRY_FREQUENCY_PENALTIES}")
log("=" * 78)

tokenizer = get_tokenizer(cfg.TOKENIZER)
all_chunks = []
for doc in load_documents("data/raw"):
    all_chunks.extend(chunk_document(doc, cfg, tokenizer))
by_id = {c["chunk_id"]: c for c in all_chunks}
targets = [by_id[cid] for cid in HARD_CHUNK_IDS if cid in by_id]
missing = [cid for cid in HARD_CHUNK_IDS if cid not in by_id]
if missing:
    log(f"[warn] chunk ids not found in the corpus: {missing}")

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

# attempt 0 uses the client exactly as configured (it already carries
# frequency_penalty = 0.25); attempts 1..n escalate. `None` here means
# "as configured", NOT "no penalty" -- the recorded value is the client's
# ACTUAL penalty, read back off the client used for that attempt.
LADDER_SPECS = [None] + [float(v) for v in cfg.LLM_RETRY_FREQUENCY_PENALTIES]


def classify(raw: str) -> str:
    """
    Name the failure MODE of a raw response, so a failure is not reported
    as an anonymous parse error.

      loop   -- the most repeated line appears more than 5 times
      json   -- a malformed-JSON failure: not a loop, the model stopped
                normally and emitted invalid JSON
      ok     -- parses
    """
    if raw is None:
        return "absent"
    try:
        from graphrag_lite.ingestion.extractor import _extract_json
        _extract_json(raw)
        return "ok"
    except Exception:
        pass
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    top = Counter(lines).most_common(1)[0][1] if lines else 0
    if top > 5:
        return "loop"
    return "json"


for chunk in targets:
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
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
        except Exception as e:                       # noqa: BLE001
            status = "FAIL"
            err = f"{type(e).__name__}: {str(e)[:200]}"
    seconds = time.perf_counter() - t0

    # One warning per retry -> the index of the attempt that succeeded.
    retry_warnings = [w for w in caught if "retrying with a higher penalty" in str(w.message)]
    attempts_used = len(retry_warnings) + 1

    # Read back every attempt's raw response and name its mode.
    per_attempt = []
    for i, penalty in enumerate(LADDER_SPECS):
        c = client if penalty is None else client.with_overrides(frequency_penalty=penalty)
        key = _make_key(prompt, cfg.effective_extraction_model(),
                        getattr(c, "cache_tag", ""))
        raw = cache.get(key, default=None)
        mode = classify(raw)
        per_attempt.append({
            "attempt": i,
            "penalty": c.frequency_penalty,
            "mode": mode,
            "chars": len(raw) if raw else 0,
        })
        if raw is not None:
            suffix = f"fp{c.frequency_penalty}"
            with open(os.path.join(RAW_DIR, f"{chunk['chunk_id']}__{suffix}.txt"),
                      "w", encoding="utf-8") as fh:
                fh.write(raw)

    rec = {
        "rep": int(REP),
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "chunk_index": chunk.get("chunk_index"),
        "status": status,
        "attempts_used": attempts_used,
        "rescued": status == "ok" and attempts_used > 1,
        "entities": n_ent,
        "relationships": n_rel,
        "seconds": round(seconds, 1),
        "per_attempt": per_attempt,
        "error": err,
    }
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    modes = " | ".join(f"a{a['attempt']}(fp={a['penalty']}):{a['mode']}" for a in per_attempt)
    if len(modes) > 150:
        modes = modes[:147] + "..."
    log(f"  {chunk['chunk_id'][:12]}  {status:4s}  attempts={attempts_used}  "
        f"ent={n_ent:3d} rel={n_rel:3d}  {seconds:7.1f}s")
    log(f"      attempts: {modes}")
    if err:
        log(f"      error: {err}")

log("=" * 78)
log(f"CACHE_DIR in use         : {cfg.CACHE_DIR}")
log(f"Cache entries at end     : {len(cache)}")
log("=" * 78)
