"""
exp_qwen35_2b_thinkoff.py -- follow-up to exp_qwen35_2b.py.

exp_qwen35_2b.py established two facts for qwen3.5:2b:
  * thinking is ON by default (the Ollama response carries a separate
    `thinking` field of ~8000 chars), and with num_predict=2048 the visible
    `response` comes back EMPTY -- the thinking consumed the whole budget,
    so 0/3 chunks parsed.
  * `think=False` removes the `thinking` field entirely, and then 1/3
    chunks parsed. The 2 failures were JSON truncated at ~8300-8800 chars,
    i.e. num_predict truncation, not a parse-logic problem.

This script closes the loop: the same 3 chunks with think=False and
num_predict=4096, to confirm the combination parses 3/3.

It also records the Ollama `done_reason` so "truncated" vs "finished" is
distinguishable, and the raw response length.

Isolation:
  * own CACHE_DIR  (data/cache/exp_qwen35_2b_thinkoff) -- never the canonical cache
  * own output file: data/cache/exp_qwen35_2b_thinkoff.json
  * prints CACHE_DIR and cache entry count at start

Run:  python tests/exp_qwen35_2b_thinkoff.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE_DIR = "data/cache/exp_qwen35_2b_thinkoff"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["CACHE_DIR"] = CACHE_DIR
os.environ["ZVEC_ROOT"] = "data/cache"
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
os.environ["EXTRACTION_MODEL"] = "qwen3.5:2b"
os.environ["SLM_MODEL"] = "qwen3.5:2b"
os.environ["LLM_TIMEOUT_S"] = "1200"

from graphrag_lite.config import get_config        # noqa: E402
get_config.cache_clear()
from graphrag_lite.cache.llm_cache import _get_cache   # noqa: E402
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer   # noqa: E402
from graphrag_lite.ingestion.extractor import (      # noqa: E402
    _EXTRACTION_PROMPT_TEMPLATE, _extract_json,
)
from graphrag_lite.ingestion.loader import load_documents   # noqa: E402

import ollama                                                # noqa: E402

cfg = get_config()
cache = _get_cache()
tokenizer = get_tokenizer(cfg.TOKENIZER)
MODEL = "qwen3.5:2b"
NUM_PREDICT = 4096

print("=" * 76)
print("EXP: qwen3.5:2b -- think=False + num_predict=4096")
print("=" * 76)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print(f"Model                  : {MODEL}")
print(f"think                  : False")
print(f"num_predict            : {NUM_PREDICT}")
print(f"timeout_s              : {cfg.LLM_TIMEOUT_S}")
print("=" * 76)
print()

docs = load_documents("data/raw")
ta = next(d for d in docs if d["doc_id"] == "transformer_architecture")
chunks = chunk_document(ta, cfg, tokenizer)[:3]

records = []
print(f"{'chunk':<8}{'tok':>5}{'secs':>8}{'eval_tok':>10}{'done':>8}"
      f"{'chars':>8}{'think':>7}{'parse':>7}{'ent':>5}{'rel':>5}")
print("-" * 78)

for c in chunks:
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=c["text"])
    client = ollama.Client(timeout=cfg.LLM_TIMEOUT_S)
    t0 = time.perf_counter()
    resp = client.generate(
        model=MODEL,
        prompt=prompt,
        think=False,
        options={
            "temperature": cfg.EXTRACTION_TEMPERATURE,
            "seed": cfg.EXTRACTION_SEED,
            "num_predict": NUM_PREDICT,
        },
    )
    seconds = time.perf_counter() - t0
    text = resp.get("response") or ""
    thinking = resp.get("thinking") or ""

    n_ent = n_rel = -1
    status = "ok"
    err = ""
    try:
        parsed = _extract_json(text)
        n_ent = len(parsed.get("entities") or [])
        n_rel = len(parsed.get("relationships") or [])
    except Exception as e:                      # noqa: BLE001
        status = "FAIL"
        err = str(e)[:140]

    rec = {
        "chunk_index": c["chunk_index"],
        "chunk_id": c["chunk_id"],
        "token_count": c["token_count"],
        "think": False,
        "num_predict": NUM_PREDICT,
        "seconds": round(seconds, 2),
        "eval_count": resp.get("eval_count"),
        "done_reason": resp.get("done_reason"),
        "response_chars": len(text),
        "thinking_chars": len(thinking),
        "parse_status": status,
        "parse_error": err,
        "entities": n_ent,
        "relationships": n_rel,
    }
    records.append(rec)
    print(f"{c['chunk_index']:<8}{c['token_count']:>5}{seconds:>8.1f}"
          f"{str(resp.get('eval_count')):>10}{str(resp.get('done_reason')):>8}"
          f"{len(text):>8}{len(thinking):>7}{status:>7}{n_ent:>5}{n_rel:>5}",
          flush=True)

print()
print("=" * 76)
print("SUMMARY")
print("=" * 76)
ok = sum(1 for r in records if r["parse_status"] == "ok")
stop = sum(1 for r in records if r["done_reason"] == "stop")
print(f"parse_ok               : {ok}/{len(records)}")
print(f"done_reason == 'stop'  : {stop}/{len(records)}")
print(f"avg_secs               : {sum(r['seconds'] for r in records) / len(records):.1f}")
print(f"any thinking text      : {sum(1 for r in records if r['thinking_chars'] > 0)}"
      f"/{len(records)}")

out = "data/cache/exp_qwen35_2b_thinkoff.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(records, fh, indent=2, ensure_ascii=False)

print()
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print(f"Wrote {out}")
print("EXP qwen3.5:2b think=False+4096 complete.")
