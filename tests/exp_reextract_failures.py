"""
exp_reextract_failures.py -- why did a chunk that passes in isolation fail in
the corpus run?

Why this script exists (2026-10-22). The full-corpus re-extraction
(tests/exp_reextract_corpus.py, qwen2.5:1.5b, num_predict=4096,
num_ctx=8192, think=False, frequency_penalty=0.25) gave 39/42 chunks
parsed, with 3 failures:

  large_language_models/8            eval=4096  done=length  x74  parse=FAIL
  retrieval_augmented_generation/6   eval=4096  done=length  x35  parse=FAIL
  transformer_architecture/4         eval=4096  done=length  x62  parse=FAIL

The third one is the surprise. `transformer_architecture/4` is 4af3dc8c --
the chunk the frequency penalty fixed, 3/3 parses, 21 entities, in
tests/exp_fp_penalty_choice.py. Same chunk, same model, same seed, same
temperature, same num_predict, same frequency_penalty=0.25. It parsed
there and it looped here.

Exactly ONE setting differed between the two runs:

    isolated probes   num_ctx NOT sent  (the runner's own default)
    corpus run        num_ctx = 8192

and a second difference: the corpus run had already issued 32 other
requests to the same runner before reaching this chunk.

This script separates the two candidate explanations:

  P1  fp=0.25, num_ctx=8192    -- reproduces the corpus run
  P2  fp=0.25, num_ctx unset   -- reproduces the isolated probe
  P3  fp=None,  num_ctx=8192   -- the control: would these chunks fail
                                  without the penalty at all?

Two reps each, on all three failing chunks, in that order, on one runner.

Isolation:
  * own CACHE_DIR (data/cache/exp_failures) -- never the canonical cache
  * own output file: data/cache/exp_failures.json
  * prints CACHE_DIR and cache entry count at start
  * calls the Ollama SDK directly -> no cache writes at all

Run:  python tests/exp_reextract_failures.py
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE_DIR = "data/cache/exp_failures"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["CACHE_DIR"] = CACHE_DIR
os.environ["ZVEC_ROOT"] = "data/cache"
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"

import ollama                                                  # noqa: E402

from graphrag_lite.config import get_config                     # noqa: E402
get_config.cache_clear()
from graphrag_lite.cache.llm_cache import _get_cache             # noqa: E402
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer   # noqa: E402
from graphrag_lite.ingestion.extractor import _EXTRACTION_PROMPT_TEMPLATE   # noqa: E402
from graphrag_lite.ingestion.extractor import _extract_json      # noqa: E402
from graphrag_lite.ingestion.loader import load_documents        # noqa: E402

cfg = get_config()
cache = _get_cache()
tokenizer = get_tokenizer(cfg.TOKENIZER)

print("=" * 78)
print("EXP: why did a chunk that passes in isolation fail in the corpus run?")
print("=" * 78)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print(f"Output file            : data/cache/exp_failures.json")
print("=" * 78)
print(flush=True)

docs = load_documents("data/raw")
index = {}
for d in docs:
    for c in chunk_document(d, cfg, tokenizer):
        index[(d["doc_id"], c["chunk_index"])] = c

TARGETS = [
    ("transformer_architecture", 4),            # 4af3dc8c -- passes in isolation
    ("large_language_models", 8),               # 24b17cf3 -- new failure
    ("retrieval_augmented_generation", 6),      # f08417a7 -- new failure
]

PHASES = [
    ("P1 fp=0.25 num_ctx=8192", 0.25, 8192),
    ("P2 fp=0.25 num_ctx unset", 0.25, None),
    ("P3 fp=None  num_ctx=8192", None, 8192),
]
REPS = 2


def run(doc_id: str, chunk_index: int, fp: float | None,
        num_ctx: int | None) -> dict:
    chunk = index[(doc_id, chunk_index)]
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
    options = {
        "temperature": 0.0,
        "seed": 42,
        "num_predict": 4096,
    }
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if fp is not None:
        options["frequency_penalty"] = fp

    client = ollama.Client(timeout=3600)
    t0 = time.perf_counter()
    try:
        resp = client.generate(model="qwen2.5:1.5b", prompt=prompt,
                               think=False, options=options)
        terr = ""
    except Exception as e:                        # noqa: BLE001
        resp = {"response": "", "eval_count": None,
                "prompt_eval_count": None, "done_reason": "error"}
        terr = f"{type(e).__name__}: {str(e)[:160]}"
    seconds = time.perf_counter() - t0

    text = resp.get("response") or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    counter = Counter(lines)
    top_count = counter.most_common(1)[0][1] if counter else 0

    n_ent = n_rel = -1
    status = "ok"
    perr = ""
    try:
        parsed = _extract_json(text)
        n_ent = len(parsed.get("entities") or [])
        n_rel = len(parsed.get("relationships") or [])
    except Exception as e:                        # noqa: BLE001
        status = "FAIL"
        perr = f"{type(e).__name__}: {str(e)[:120]}"

    return {
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "chunk_id": chunk["chunk_id"],
        "model": "qwen2.5:1.5b",
        "frequency_penalty": fp,
        "num_ctx": num_ctx,
        "seconds": round(seconds, 2),
        "prompt_eval_count": resp.get("prompt_eval_count"),
        "eval_count": resp.get("eval_count"),
        "done_reason": resp.get("done_reason"),
        "response_chars": len(text),
        "most_repeated_line_count": top_count,
        "parse_status": status,
        "parse_error": perr,
        "transport_error": terr,
        "entities": n_ent,
        "relationships": n_rel,
    }


def show(rec: dict) -> None:
    print(f"  {rec['doc_id']}/{rec['chunk_index']} ({rec['chunk_id'][:8]})  "
          f"fp={rec['frequency_penalty']}  num_ctx={rec['num_ctx']}")
    print(f"    secs={rec['seconds']:.1f}  eval={rec['eval_count']}  "
          f"done={rec['done_reason']}  chars={rec['response_chars']}  "
          f"x{rec['most_repeated_line_count']}")
    print(f"    parse={rec['parse_status']}  ent={rec['entities']} "
          f"rel={rec['relationships']}"
          + (f"  err={rec['parse_error']}" if rec['parse_error'] else ""))
    print(flush=True)


records = []

for label, fp, num_ctx in PHASES:
    print(f"--- {label} ---")
    for target in TARGETS:
        for rep in range(REPS):
            rec = run(target[0], target[1], fp, num_ctx)
            rec["rep"] = rep
            rec["phase"] = label
            records.append(rec)
            show(rec)
        print()
    print()

print("=" * 78)
print("SUMMARY")
print("=" * 78)
for target in TARGETS:
    print(f"  {target[0]}/{target[1]}:")
    for label, _, _ in PHASES:
        rs = [r for r in records
              if r["doc_id"] == target[0] and r["chunk_index"] == target[1]
              and r["phase"] == label]
        if not rs:
            continue
        ok = sum(1 for r in rs if r["parse_status"] == "ok")
        print(f"    {label:<26} parsed {ok}/{len(rs)}  "
              f"x={[r['most_repeated_line_count'] for r in rs]}  "
              f"eval={[r['eval_count'] for r in rs]}  "
              f"ent={[r['entities'] for r in rs]}  "
              f"done={[r['done_reason'] for r in rs]}")

with open("data/cache/exp_failures.json", "w", encoding="utf-8") as fh:
    json.dump(records, fh, indent=2)

print()
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print("Wrote data/cache/exp_failures.json")
print("EXP failures analysis complete.")
