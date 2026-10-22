"""
exp_freq_penalty_robustness.py -- is frequency_penalty=0.5 a fix, or a lucky sample?

Why this script exists (2026-10-22). tests/exp_runaway_freq_penalty.py found
that a frequency penalty breaks the 4af3dc8c repetition loop, but also found
the shape of the trade-off:

    BASELINE (no penalty)  done=length  eval=4096  chars=19661  top_line_x35  parse=FAIL
    FP 0.5                  done=stop    eval=489   chars=2096   top_line_x7   parse=ok  (8 ent / 8 rel)
    FP 1.0                  done=stop    eval=398   chars=1631   top_line_x4   parse=FAIL  (cut off mid-field)
    FP 1.5                  done=stop    eval=98    chars=361    top_line_x2   parse=FAIL  (cut off mid-field)

So 0.5 is the only setting that both breaks the loop AND lets the model
finish. But that is ONE sample, and the same probe's no-penalty baseline did
NOT reproduce the earlier probe's baseline (83x / 18515 chars in the
repeat_penalty probe vs 35x / 19661 chars here) -- same prompt, same seed,
same temperature=0.0. qwen2.5:1.5b is therefore not deterministic even with a
fixed seed, and a single sample proves nothing.

This script asks the two questions that decide whether the knob can ship:

  Q1  REPEATABILITY on the pathological chunk. fp in {0.25, 0.5, 0.75}, three
      runs each, on chunk 4af3dc8c. If the parse rate is not ~100% the
      default is not safe.

  Q2  NO REGRESSION on healthy chunks. fp 0.5 vs no penalty on two chunks
      that already parsed cleanly. A penalty that suppresses legitimate
      repetition (an entity mentioned twice, a repeated key) would quietly
      truncate good extractions. Entity/relationship counts are compared.

Isolation:
  * own CACHE_DIR (data/cache/exp_fp_robust) -- never the canonical cache
  * own output file: data/cache/exp_fp_robust.json
  * prints CACHE_DIR and cache entry count at start
  * calls the Ollama SDK directly -> no cache writes at all

Run:  python tests/exp_freq_penalty_robustness.py
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE_DIR = "data/cache/exp_fp_robust"
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
print("EXP: is frequency_penalty=0.5 a fix, or a lucky sample?")
print("=" * 78)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print(f"Output file            : data/cache/exp_fp_robust.json")
print("=" * 78)
print(flush=True)

docs = load_documents("data/raw")
ta = next(d for d in docs if d["doc_id"] == "transformer_architecture")
chunks = {c["chunk_index"]: c for c in chunk_document(ta, cfg, tokenizer)}
BAD = 4          # 4af3dc8c -- the pathological chunk
GOOD = [0, 2]    # chunks that already parse cleanly


def run(chunk_index: int, fp: float | None, num_predict: int = 4096) -> dict:
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunks[chunk_index]["text"])
    options = {
        "temperature": 0.0,
        "seed": 42,
        "num_predict": num_predict,
    }
    if fp is not None:
        options["frequency_penalty"] = fp

    client = ollama.Client(timeout=2400)
    t0 = time.perf_counter()
    resp = client.generate(model="qwen2.5:1.5b", prompt=prompt, options=options)
    seconds = time.perf_counter() - t0

    text = resp.get("response") or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    counter = Counter(lines)
    top_line, top_count = (counter.most_common(1)[0] if counter else ("", 0))

    n_ent = n_rel = -1
    status = "ok"
    err = ""
    try:
        parsed = _extract_json(text)
        n_ent = len(parsed.get("entities") or [])
        n_rel = len(parsed.get("relationships") or [])
    except Exception as e:                        # noqa: BLE001
        status = "FAIL"
        err = str(e)[:140]

    return {
        "chunk_index": chunk_index,
        "chunk_id": chunks[chunk_index].get("chunk_id"),
        "frequency_penalty": fp,
        "seconds": round(seconds, 2),
        "prompt_eval_count": resp.get("prompt_eval_count"),
        "eval_count": resp.get("eval_count"),
        "done_reason": resp.get("done_reason"),
        "response_chars": len(text),
        "nonempty_lines": len(lines),
        "distinct_lines": len(counter),
        "most_repeated_line_count": top_count,
        "parse_status": status,
        "parse_error": err,
        "entities": n_ent,
        "relationships": n_rel,
    }


def show(rec: dict) -> None:
    print(f"  chunk {rec['chunk_index']} ({rec['chunk_id'][:8]})  "
          f"fp={rec['frequency_penalty']}")
    print(f"    secs={rec['seconds']:.1f}  prompt_eval={rec['prompt_eval_count']} "
          f"eval={rec['eval_count']}  done={rec['done_reason']}  "
          f"chars={rec['response_chars']}  top_line_x{rec['most_repeated_line_count']}")
    print(f"    parse={rec['parse_status']}  ent={rec['entities']} "
          f"rel={rec['relationships']}"
          + (f"  err={rec['parse_error']}" if rec['parse_error'] else ""))
    print(flush=True)


records = []

print("--- Q1: repeatability on the pathological chunk 4af3dc8c ---")
for fp in [0.25, 0.5, 0.75]:
    for rep in range(3):
        rec = run(BAD, fp)
        rec["rep"] = rep
        records.append(rec)
        show(rec)
    print()

print("--- Q2: no regression on healthy chunks (fp 0.5 vs none) ---")
for gi in GOOD:
    for fp in [None, 0.5]:
        rec = run(gi, fp)
        rec["rep"] = 0
        records.append(rec)
        show(rec)
    print()

print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("Q1 -- pathological chunk 4af3dc8c, 3 runs per setting:")
for fp in [0.25, 0.5, 0.75]:
    rs = [r for r in records
          if r["chunk_index"] == BAD and r["frequency_penalty"] == fp]
    ok = sum(1 for r in rs if r["parse_status"] == "ok")
    ents = [r["entities"] for r in rs if r["entities"] >= 0]
    rels = [r["relationships"] for r in rs if r["relationships"] >= 0]
    print(f"  fp={fp:<5} parsed {ok}/{len(rs)}  "
          f"top_line_x={[r['most_repeated_line_count'] for r in rs]}  "
          f"eval={[r['eval_count'] for r in rs]}  "
          f"done={[r['done_reason'] for r in rs]}  "
          f"ent={ents} rel={rels}")

print()
print("Q2 -- healthy chunks, fp=none vs fp=0.5:")
for gi in GOOD:
    for fp in [None, 0.5]:
        rs = [r for r in records
              if r["chunk_index"] == gi and r["frequency_penalty"] == fp]
        if rs:
            r = rs[0]
            print(f"  chunk {gi} ({r['chunk_id'][:8]})  fp={str(fp):<5} "
                  f"parse={r['parse_status']}  ent={r['entities']} "
                  f"rel={r['relationships']}  chars={r['response_chars']}  "
                  f"top_line_x{r['most_repeated_line_count']}")

with open("data/cache/exp_fp_robust.json", "w", encoding="utf-8") as fh:
    json.dump(records, fh, indent=2)

print()
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print("Wrote data/cache/exp_fp_robust.json")
print("EXP frequency-penalty robustness complete.")
