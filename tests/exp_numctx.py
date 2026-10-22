"""
exp_numctx.py -- is the binding limit num_predict or the runtime num_ctx?

Findings that motivated this script (2026-10-22):

  * `ollama ps` shows every model loaded with CONTEXT = 4096, which is the
    Ollama 0.40.2 default -- NOT the model's declared context (qwen3.5:2b
    declares 262144).
  * The extraction prompt costs 403 prompt tokens (measured).
  * qwen3.5:2b with think=False and num_predict=4096 stopped with
    eval_count=3693 and done_reason='length'.
        403 (prompt) + 3693 (generated) = 4096 EXACTLY.

So the generation was cut by the CONTEXT WINDOW, not by num_predict. That
is a different failure from the one we had been calling 'truncation', and it
needs a different fix: raise num_ctx so prompt + generation has room.

Two probes, both against the real extraction prompt:

  PROBE A -- qwen2.5:1.5b, chunk 4af3dc8c, num_predict=8192, num_ctx left
             at the default. If eval_count exceeds 4096 the model had room
             and 4096 was a genuine num_predict cap; if it stops near
             4096-prompt_tokens the context wall is real.
  PROBE B -- qwen3.5:2b, chunk 0, num_predict=4096 WITH num_ctx=8192 in
             options. If the context wall was the cause, this now stops
             with done_reason='stop' and parses.

Isolation:
  * own CACHE_DIR (data/cache/exp_numctx) -- never the canonical cache
  * own output file: data/cache/exp_numctx.json
  * prints CACHE_DIR and cache entry count at start
  * calls the Ollama SDK directly -> no cache writes at all

Run:  python tests/exp_numctx.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE_DIR = "data/cache/exp_numctx"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["CACHE_DIR"] = CACHE_DIR
os.environ["ZVEC_ROOT"] = "data/cache"
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"

from graphrag_lite.config import get_config            # noqa: E402
get_config.cache_clear()
from graphrag_lite.cache.llm_cache import _get_cache    # noqa: E402
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer  # noqa: E402
from graphrag_lite.ingestion.extractor import (         # noqa: E402
    _EXTRACTION_PROMPT_TEMPLATE, _extract_json,
)
from graphrag_lite.ingestion.loader import load_documents   # noqa: E402

import ollama                                              # noqa: E402

cfg = get_config()
cache = _get_cache()
tokenizer = get_tokenizer(cfg.TOKENIZER)

print("=" * 78)
print("EXP: is the binding limit num_predict, or the runtime num_ctx?")
print("=" * 78)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print(f"Output file            : data/cache/exp_numctx.json")
print("=" * 78)
print()

docs = load_documents("data/raw")
ta = next(d for d in docs if d["doc_id"] == "transformer_architecture")
chunks = {c["chunk_index"]: c for c in chunk_document(ta, cfg, tokenizer)}


def run(model: str, prompt: str, num_predict: int, num_ctx: int | None,
        think: bool | None, chunk_index: int) -> dict:
    options = {
        "temperature": 0.0,
        "seed": 42,
        "num_predict": num_predict,
    }
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    kwargs = {"model": model, "prompt": prompt, "options": options}
    if think is not None:
        kwargs["think"] = think

    client = ollama.Client(timeout=2400)
    t0 = time.perf_counter()
    resp = client.generate(**kwargs)
    seconds = time.perf_counter() - t0

    text = resp.get("response") or ""
    n_ent = n_rel = -1
    status = "ok"
    err = ""
    try:
        parsed = _extract_json(text)
        n_ent = len(parsed.get("entities") or [])
        n_rel = len(parsed.get("relationships") or [])
    except Exception as e:                       # noqa: BLE001
        status = "FAIL"
        err = str(e)[:140]

    rec = {
        "model": model,
        "chunk_index": chunk_index,
        "num_predict": num_predict,
        "num_ctx": num_ctx,
        "think": think,
        "seconds": round(seconds, 2),
        "prompt_eval_count": resp.get("prompt_eval_count"),
        "eval_count": resp.get("eval_count"),
        "prompt_plus_eval": (resp.get("prompt_eval_count") or 0)
                            + (resp.get("eval_count") or 0),
        "done_reason": resp.get("done_reason"),
        "response_chars": len(text),
        "parse_status": status,
        "parse_error": err,
        "entities": n_ent,
        "relationships": n_rel,
    }
    print(f"  model={model} np={num_predict} num_ctx={num_ctx} think={think}")
    print(f"    secs={seconds:.1f}  prompt_eval={rec['prompt_eval_count']} "
          f"eval={rec['eval_count']}  prompt+eval={rec['prompt_plus_eval']}  "
          f"done={rec['done_reason']}  chars={len(text)}")
    print(f"    parse={status}  ent={n_ent}  rel={n_rel}"
          + (f"  err={err}" if err else ""))
    print(flush=True)
    return rec


records = []

# ---- PROBE A: qwen2.5:1.5b, the runaway chunk, cap raised to 8192 -----------
print("PROBE A -- qwen2.5:1.5b chunk 4af3dc8c, num_predict=8192, default ctx")
prompt_a = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunks[4]["text"])
records.append(run("qwen2.5:1.5b", prompt_a, 8192, None, None, 4))
print()

# ---- PROBE B: qwen3.5:2b with num_ctx raised --------------------------------
print("PROBE B -- qwen3.5:2b chunk 0, num_predict=4096, num_ctx=8192, think=False")
prompt_b = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunks[0]["text"])
records.append(run("qwen3.5:2b", prompt_b, 4096, 8192, False, 0))
print()

print("=" * 78)
print("SUMMARY")
print("=" * 78)
for r in records:
    binding = ("CONTEXT" if (r["prompt_plus_eval"] == 4096
                             and r["eval_count"] < r["num_predict"])
               else "num_predict" if r["done_reason"] == "length"
               else "finished")
    print(f"  {r['model']:<14} np={r['num_predict']:<5} num_ctx={r['num_ctx']} "
          f"eval={r['eval_count']:<5} prompt+eval={r['prompt_plus_eval']:<5} "
          f"done={str(r['done_reason']):<8} parse={r['parse_status']:<4} "
          f"binding_limit={binding}")
print()

out = "data/cache/exp_numctx.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(records, fh, indent=2, ensure_ascii=False)

print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print(f"Wrote {out}")
print("EXP numctx complete.")
