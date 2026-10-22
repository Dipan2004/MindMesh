"""
exp_runaway_model_swap.py -- is the 4af3dc8c repetition loop model-specific?

Why this script exists (2026-10-22). Chunk 4af3dc8c of transformer_architecture
loops degenerately on qwen2.5:1.5b: the same entity line is emitted 83 times
and the response never parses. Measured so far (Item 3 of DOCUMENT.md):

    num_predict 2048          top_line_x34   parse=FAIL
    num_predict 4096          top_line_x83   parse=FAIL
    num_predict 8192          still looping  parse=FAIL
    repeat_penalty 1.05       top_line_x37   parse=FAIL
    repeat_penalty 1.15       top_line_x48   parse=FAIL

A flat repeat penalty does not fix it. The chunk text itself is ordinary
markdown (1461 chars, no repetition in the input), so this looks like a
decoding pathology of the SMALL model rather than a property of the prompt
or the chunk. The other lever is the model: `qwen3:4b` finished pulling
(2.5 GB, present in `ollama list`) and `qwen3.5:2b` is also installed.

Setting: the same chunk, same prompt template, same seed, same
num_predict=4096, on two bigger models, with thinking explicitly disabled
(`think=False` is a TOP-LEVEL Ollama param, sibling of `options`):

    qwen3:4b      think=False
    qwen3.5:2b    think=False

Success criterion: the response parses AND the most-repeated line count is
far below 83.

Isolation:
  * own CACHE_DIR (data/cache/exp_runaway_swap) -- never the canonical cache
  * own output file: data/cache/exp_runaway_swap.json
  * prints CACHE_DIR and cache entry count at start
  * calls the Ollama SDK directly -> no cache writes at all

Run:  python tests/exp_runaway_model_swap.py
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE_DIR = "data/cache/exp_runaway_swap"
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
print("EXP: is the 4af3dc8c repetition loop model-specific?")
print("=" * 78)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print(f"Output file            : data/cache/exp_runaway_swap.json")
print("=" * 78)
print(flush=True)

docs = load_documents("data/raw")
ta = next(d for d in docs if d["doc_id"] == "transformer_architecture")
chunks = {c["chunk_index"]: c for c in chunk_document(ta, cfg, tokenizer)}
prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunks[4]["text"])


def run(label: str, model: str, num_predict: int = 4096, num_ctx: int = 8192) -> dict:
    options = {
        "temperature": 0.0,
        "seed": 42,
        "num_predict": num_predict,
        "num_ctx": num_ctx,
    }

    client = ollama.Client(timeout=3600)
    t0 = time.perf_counter()
    resp = client.generate(model=model, prompt=prompt, think=False, options=options)
    seconds = time.perf_counter() - t0

    text = resp.get("response") or ""
    thinking = resp.get("thinking") or ""
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
        "model": model,
        "num_predict": num_predict,
        "num_ctx": num_ctx,
        "think": False,
        "seconds": round(seconds, 2),
        "prompt_eval_count": resp.get("prompt_eval_count"),
        "eval_count": resp.get("eval_count"),
        "prompt_plus_eval": (resp.get("prompt_eval_count") or 0)
                            + (resp.get("eval_count") or 0),
        "done_reason": resp.get("done_reason"),
        "thinking_chars": len(thinking),
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
    print(f"  {label}  (model={model}, think=False)")
    print(f"    secs={seconds:.1f}  prompt_eval={rec['prompt_eval_count']} "
          f"eval={rec['eval_count']}  prompt+eval={rec['prompt_plus_eval']}  "
          f"done={rec['done_reason']}  chars={len(text)}  thinking_chars={len(thinking)}")
    print(f"    lines={rec['nonempty_lines']} distinct={rec['distinct_lines']} "
          f"top_line_x{top_count}")
    print(f"    parse={status}  ent={n_ent}  rel={n_rel}"
          + (f"  err={err}" if err else ""))
    print(flush=True)
    return rec


records = []
for label, model in [("qwen3:4b", "qwen3:4b"), ("qwen3.5:2b", "qwen3.5:2b")]:
    records.append(run(label, model))
    print()

print("=" * 78)
print("SUMMARY")
print("=" * 78)
for r in records:
    looped = r["most_repeated_line_count"] > 5
    print(f"  {r['label']:<12} done={str(r['done_reason']):<7} "
          f"eval={str(r['eval_count']):<5} chars={r['response_chars']:<6} "
          f"top_line_x{r['most_repeated_line_count']:<3} "
          f"parse={r['parse_status']:<4} ent={r['entities']} rel={r['relationships']} "
          f"{'LOOPING' if looped else 'no-loop'}")
print()
print("Reference: qwen2.5:1.5b on the same chunk gave top_line_x83 and parse=FAIL.")
print("A model that does not loop gives top_line_x<=5 and parse=ok.")

with open("data/cache/exp_runaway_swap.json", "w", encoding="utf-8") as fh:
    json.dump(records, fh, indent=2)

print()
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print("Wrote data/cache/exp_runaway_swap.json")
print("EXP runaway-model-swap complete.")
