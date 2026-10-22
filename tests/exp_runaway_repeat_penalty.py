"""
exp_runaway_repeat_penalty.py -- can a repetition penalty break chunk 4af3dc8c?

Context (2026-10-22). The three "timed-out" chunks turned out to be three
different failure modes (DOCUMENT.md, Phase 4 Closure, Item 3):

  a885be6b  ordinary truncation                 -> fixed by the num_predict bump
  a4a679c4  complete JSON, one trailing comma    -> fixed by the JSON repair
  4af3dc8c  DEGENERATE REPETITION LOOP          -> this script

4af3dc8c (transformer_architecture/4) on qwen2.5:1.5b emits the same
entity line over and over:

  {"name": "Google", "type": "ORGANIZATION", "description": "Location where the
   research was conducted", "source_sentence": "The research was conducted at
   Google Brain and Google Research."}

repeated 34x at num_predict=2048 and 83x at 4096, and it is still looping
at 8192 (37427 chars). A bigger cap demonstrably does not help. A repetition
penalty is the standard lever for exactly this failure.

Three settings are tried, all on the same prompt, same seed, same cap:

  BASELINE  no repeat_penalty              (expected: still loops, eval=4096)
  RP 1.05   repeat_penalty=1.05
  RP 1.15   repeat_penalty=1.15

Success criterion: done_reason == 'stop' (or eval_count < num_predict) AND
the response parses AND the most-repeated line count is far below the
baseline's.

Isolation:
  * own CACHE_DIR (data/cache/exp_runaway_rp) -- never the canonical cache
  * own output file: data/cache/exp_runaway_rp.json
  * prints CACHE_DIR and cache entry count at start
  * calls the Ollama SDK directly -> no cache writes at all

Run:  python tests/exp_runaway_repeat_penalty.py
"""
import json
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE_DIR = "data/cache/exp_runaway_rp"
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
print("EXP: does a repetition penalty break the 4af3dc8c repetition loop?")
print("=" * 78)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print(f"Output file            : data/cache/exp_runaway_rp.json")
print("=" * 78)
print(flush=True)

docs = load_documents("data/raw")
ta = next(d for d in docs if d["doc_id"] == "transformer_architecture")
chunks = {c["chunk_index"]: c for c in chunk_document(ta, cfg, tokenizer)}
prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunks[4]["text"])


def run(label: str, repeat_penalty: float | None, num_predict: int = 4096) -> dict:
    options = {
        "temperature": 0.0,
        "seed": 42,
        "num_predict": num_predict,
    }
    if repeat_penalty is not None:
        options["repeat_penalty"] = repeat_penalty

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
        "repeat_penalty": repeat_penalty,
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
    print(f"  {label}  (repeat_penalty={repeat_penalty})")
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
for label, rp in [("BASELINE", None), ("RP 1.05", 1.05), ("RP 1.15", 1.15)]:
    records.append(run(label, rp))
    print()

print("=" * 78)
print("SUMMARY")
print("=" * 78)
base = records[0]
for r in records:
    looped = r["most_repeated_line_count"] > 5
    print(f"  {r['label']:<10} rp={str(r['repeat_penalty']):<5} "
          f"done={str(r['done_reason']):<7} eval={str(r['eval_count']):<5} "
          f"chars={r['response_chars']:<6} top_line_x{r['most_repeated_line_count']:<3} "
          f"parse={r['parse_status']:<4} ent={r['entities']} rel={r['relationships']} "
          f"{'LOOPING' if looped else 'no-loop'}")
print()
print(f"baseline top_line_x{base['most_repeated_line_count']} -> "
      + " -> ".join(f"{r['label']} x{r['most_repeated_line_count']}" for r in records[1:]))
print("A repetition penalty fixes the runaway if top_line_x drops to ~1 AND "
      "the response parses.")

with open("data/cache/exp_runaway_rp.json", "w", encoding="utf-8") as fh:
    json.dump(records, fh, indent=2)

print()
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print("Wrote data/cache/exp_runaway_rp.json")
print("EXP runaway-repeat-penalty complete.")
