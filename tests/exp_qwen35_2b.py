"""
exp_qwen35_2b.py -- Does qwen3.5:2b emit thinking text, and does it parse?

For 3 chunks we call qwen3.5:2b twice with the identical extraction prompt:

  mode "default" : think=None  (model default -- whatever Ollama picks)
  mode "notthink" : think=False (explicitly disable reasoning)

and record for each call:
  * seconds
  * whether the Ollama response carried a separate `thinking` field
  * how many characters of thinking text came back
  * whether the visible `response` text contains a thinking preamble
  * whether the response parses as the expected JSON schema
  * entity / relationship counts

Isolation:
  * own CACHE_DIR  (data/cache/exp_qwen35_2b) -- never the canonical cache
  * own output file: data/cache/exp_qwen35_2b.json
  * prints CACHE_DIR and cache entry count at start

Run:  python tests/exp_qwen35_2b.py
"""
import json
import os
import re
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE_DIR = "data/cache/exp_qwen35_2b"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["CACHE_DIR"] = CACHE_DIR
os.environ["ZVEC_ROOT"] = "data/cache"
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
os.environ["EXTRACTION_MODEL"] = "qwen3.5:2b"
os.environ["SLM_MODEL"] = "qwen3.5:2b"
os.environ["LLM_TIMEOUT_S"] = "900"
os.environ["LLM_NUM_PREDICT"] = "2048"

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

print("=" * 72)
print("EXP: qwen3.5:2b -- thinking text / parsing / seconds per call")
print("=" * 72)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print(f"Model                  : {MODEL}")
print(f"num_predict            : {cfg.LLM_NUM_PREDICT}")
print(f"temperature / seed     : {cfg.EXTRACTION_TEMPERATURE} / {cfg.EXTRACTION_SEED}")
print("=" * 72)
print()

docs = load_documents("data/raw")
ta = next(d for d in docs if d["doc_id"] == "transformer_architecture")
chunks = chunk_document(ta, cfg, tokenizer)[:3]

OPTIONS = {
    "temperature": cfg.EXTRACTION_TEMPERATURE,
    "seed": cfg.EXTRACTION_SEED,
    "num_predict": cfg.LLM_NUM_PREDICT,
}

# Patterns that indicate reasoning text leaked into the visible answer.
_PREAMBLE = re.compile(
    r"(<think>|</think>|^\s*(?:Okay|Let me|I need to|First,|Let me think)\b)",
    re.IGNORECASE | re.MULTILINE,
)


def run(prompt, think_mode):
    client = ollama.Client(timeout=cfg.LLM_TIMEOUT_S)
    kwargs = {"model": MODEL, "prompt": prompt, "options": dict(OPTIONS)}
    if think_mode == "false":
        kwargs["think"] = False
    t0 = time.perf_counter()
    resp = client.generate(**kwargs)
    seconds = time.perf_counter() - t0

    thinking = resp.get("thinking") or ""
    text = resp.get("response") or ""
    has_field = "thinking" in resp
    preamble = bool(_PREAMBLE.search(text))

    n_ent = n_rel = -1
    parse_status = "ok"
    parse_err = ""
    try:
        parsed = _extract_json(text)
        n_ent = len(parsed.get("entities") or [])
        n_rel = len(parsed.get("relationships") or [])
    except Exception as e:                      # noqa: BLE001
        parse_status = "FAIL"
        parse_err = str(e)[:140]

    return {
        "think_mode": think_mode,
        "seconds": round(seconds, 2),
        "response_has_thinking_field": has_field,
        "thinking_chars": len(thinking),
        "thinking_preview": thinking[:300],
        "response_chars": len(text),
        "response_tokens": len(tokenizer.encode(text)) if text else 0,
        "visible_text_has_thinking_preamble": preamble,
        "parse_status": parse_status,
        "parse_error": parse_err,
        "entities": n_ent,
        "relationships": n_rel,
        "raw_response": text,
    }


records = []
for c in chunks:
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=c["text"])
    print(f"--- chunk {c['chunk_index']}  id={c['chunk_id'][:12]}  tokens={c['token_count']} ---")
    for mode in ("default", "false"):
        r = run(prompt, mode)
        r["chunk_index"] = c["chunk_index"]
        r["chunk_id"] = c["chunk_id"]
        r["token_count"] = c["token_count"]
        records.append(r)
        print(f"  think={mode:<8} secs={r['seconds']:>7.1f}  "
              f"thinking_field={str(r['response_has_thinking_field']):<5} "
              f"thinking_chars={r['thinking_chars']:>6}  "
              f"preamble={str(r['visible_text_has_thinking_preamble']):<5} "
              f"parse={r['parse_status']:<4} e={r['entities']} r={r['relationships']}",
              flush=True)
        if r["parse_status"] == "FAIL":
            print(f"      parse error: {r['parse_error']}")
    print()

print("=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"{'chunk':<8}{'mode':<10}{'secs':>8}{'think_field':>13}{'think_chars':>13}"
      f"{'preamble':>10}{'parse':>8}{'ent':>5}{'rel':>5}")
print("-" * 84)
for r in records:
    print(f"{r['chunk_index']:<8}{r['think_mode']:<10}{r['seconds']:>8.1f}"
          f"{str(r['response_has_thinking_field']):>13}{r['thinking_chars']:>13}"
          f"{str(r['visible_text_has_thinking_preamble']):>10}{r['parse_status']:>8}"
          f"{r['entities']:>5}{r['relationships']:>5}")

for mode in ("default", "false"):
    sub = [r for r in records if r["think_mode"] == mode]
    if not sub:
        continue
    parses = sum(1 for r in sub if r["parse_status"] == "ok")
    with_think = sum(1 for r in sub if r["thinking_chars"] > 0)
    preambles = sum(1 for r in sub if r["visible_text_has_thinking_preamble"])
    avg = sum(r["seconds"] for r in sub) / len(sub)
    print()
    print(f"think={mode:<8} parse_ok={parses}/{len(sub)}  "
          f"calls_with_thinking_field={with_think}/{len(sub)}  "
          f"visible_preamble={preambles}/{len(sub)}  avg_secs={avg:.1f}")

out = "data/cache/exp_qwen35_2b.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(records, fh, indent=2, ensure_ascii=False)

print()
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print(f"Wrote {out}")
print("EXP qwen3.5:2b complete.")
