"""
exp_runaway_freq_penalty.py -- is a frequency penalty a better lever than a
repeat penalty for chunk 4af3dc8c?

Why this script exists (2026-10-22). The repeat_penalty probe
(tests/exp_runaway_repeat_penalty.py) measured on qwen2.5:1.5b, chunk
4af3dc8c, num_predict=4096:

    BASELINE (no penalty)   top_line_x83   parse=FAIL
    repeat_penalty 1.05     top_line_x37   parse=FAIL

A flat repeat_penalty applies the SAME multiplicative penalty to every
token that has already appeared, no matter how many times it appeared. It
therefore cannot distinguish "this token appeared once, legitimately" from
"this token has appeared 80 times, I am stuck". A frequency_penalty scales
with the COUNT of the token in the context, so the 84th copy of the
"Google" line is penalised far harder than the first -- which is exactly
the shape of the pathology.

Setting:
    BASELINE   no penalty              (reference)
    FP 0.5     frequency_penalty=0.5
    FP 1.0     frequency_penalty=1.0
    FP 1.5     frequency_penalty=1.5

Success criterion (same as the repeat_penalty probe): done_reason == 'stop'
(or eval_count < num_predict) AND the response parses AND the
most-repeated line count is far below the baseline's.

Isolation:
  * own CACHE_DIR (data/cache/exp_runaway_fp) -- never the canonical cache
  * own output file: data/cache/exp_runaway_fp.json
  * prints CACHE_DIR and cache entry count at start
  * calls the Ollama SDK directly -> no cache writes at all

Run:  python tests/exp_runaway_freq_penalty.py
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE_DIR = "data/cache/exp_runaway_fp"
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
print("EXP: does a frequency penalty break the 4af3dc8c repetition loop?")
print("=" * 78)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print(f"Output file            : data/cache/exp_runaway_fp.json")
print("=" * 78)
print(flush=True)

docs = load_documents("data/raw")
ta = next(d for d in docs if d["doc_id"] == "transformer_architecture")
chunks = {c["chunk_index"]: c for c in chunk_document(ta, cfg, tokenizer)}
prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunks[4]["text"])


def run(label: str, frequency_penalty: float | None, num_predict: int = 4096) -> dict:
    options = {
        "temperature": 0.0,
        "seed": 42,
        "num_predict": num_predict,
    }
    if frequency_penalty is not None:
        options["frequency_penalty"] = frequency_penalty

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

    rec = {
        "label": label,
        "frequency_penalty": frequency_penalty,
        "num_predict": num_predict,
        "seconds": round(seconds, 2),
        "prompt_eval_count": resp.get("prompt_eval_count"),
        "eval_count": resp.get("eval_count"),
        "prompt_plus_eval": (resp.get("prompt_eval_count") or 0)
                            + (resp.get("eval_count") or 0),
        "done_reason": resp.get("done_reason"),
        "response_chars": len(text),
        "nonempty_lines": len(lines),
        "distinct_lines": len(counter),
        "most_repeated_line_count": top_count,
        "most_repeated_line": top_line[:200],
        "parse_status": status,
        "parse_error": err,
        "entities": n_ent,
        "relationships": n_rel,
    }
    print(f"  {label}  (frequency_penalty={frequency_penalty})")
    print(f"    secs={seconds:.1f}  prompt_eval={rec['prompt_eval_count']} "
          f"eval={rec['eval_count']}  prompt+eval={rec['prompt_plus_eval']}  "
          f"done={rec['done_reason']}  chars={len(text)}")
    print(f"    lines={rec['nonempty_lines']} distinct={rec['distinct_lines']} "
          f"top_line_x{top_count}")
    print(f"    parse={status}  ent={n_ent}  rel={n_rel}"
          + (f"  err={err}" if err else ""))
    print(flush=True)
    return rec


records = []
for label, fp in [("BASELINE", None), ("FP 0.5", 0.5), ("FP 1.0", 1.0), ("FP 1.5", 1.5)]:
    records.append(run(label, fp))
    print()

print("=" * 78)
print("SUMMARY")
print("=" * 78)
for r in records:
    looped = r["most_repeated_line_count"] > 5
    print(f"  {r['label']:<10} fp={str(r['frequency_penalty']):<5} "
          f"done={str(r['done_reason']):<7} eval={str(r['eval_count']):<5} "
          f"chars={r['response_chars']:<6} top_line_x{r['most_repeated_line_count']:<3} "
          f"parse={r['parse_status']:<4} ent={r['entities']} rel={r['relationships']} "
          f"{'LOOPING' if looped else 'no-loop'}")
print()
print("A frequency penalty fixes the runaway if top_line_x drops to ~1 AND "
      "the response parses.")

with open("data/cache/exp_runaway_fp.json", "w", encoding="utf-8") as fh:
    json.dump(records, fh, indent=2)

print()
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print("Wrote data/cache/exp_runaway_fp.json")
print("EXP runaway-frequency-penalty complete.")
