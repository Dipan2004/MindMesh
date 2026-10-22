"""
exp_timeout_chunks.py -- Re-run the chunks that failed in the 1.5b pilot.

The two chunks named by the mentor:
  a885be6b749e39a1  transformer_architecture idx=3  (slow: 191.9s, 12 rels)
  4af3dc8c5da84fad  transformer_architecture idx=4  (FAILED: JSON parse error
                        in the 1.5b pilot, ReadTimeout in the canonical 0.5b run)
plus the third failure found in the pilot output:
  a4a679c427829368  retrieval_augmented_generation idx=7  (FAILED: JSON parse error)

Each is run TWICE on qwen2.5:1.5b -- once with num_predict=2048 and once
with num_predict=4096 -- and we log:
  seconds, eval_count (tokens actually generated), done_reason, parse status.

The ollama SDK is called directly so that `done_reason` is available:
`done_reason == "length"` proves the response was cut off by num_predict
rather than by a timeout. The raw response is captured (not just the parsed
result) so the truncation point of a failed response can be inspected.

Why two budgets: the 2026-10-22 pilot re-run showed these chunks do NOT
time out any more -- they return ~9000-9700 characters of malformed JSON.
The open question is whether that is a num_predict truncation, so the 4096
run answers it directly.

Isolation:
  * own CACHE_DIR  (data/cache/exp_timeout_chunks) -- never the canonical cache
  * own output file: data/cache/exp_timeout_chunks.jsonl
  * prints CACHE_DIR and cache entry count at start

Run:  python tests/exp_timeout_chunks.py
"""
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE_DIR = "data/cache/exp_timeout_chunks"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["CACHE_DIR"] = CACHE_DIR
os.environ["ZVEC_ROOT"] = "data/cache"
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
os.environ["EXTRACTION_MODEL"] = "qwen2.5:1.5b"
os.environ["SLM_MODEL"] = "qwen2.5:1.5b"
os.environ["LLM_TIMEOUT_S"] = "900"
os.environ["LLM_NUM_PREDICT"] = "2048"

from graphrag_lite.config import get_config        # noqa: E402
get_config.cache_clear()
from graphrag_lite.cache.llm_cache import _get_cache, cache_stats, reset_stats  # noqa: E402
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer   # noqa: E402
from graphrag_lite.ingestion.extractor import (      # noqa: E402
    _EXTRACTION_PROMPT_TEMPLATE, _extract_json,
)
from graphrag_lite.ingestion.loader import load_documents   # noqa: E402
from graphrag_lite.llm.ollama_client import OllamaClient    # noqa: E402

cfg = get_config()
cache = _get_cache()
tokenizer = get_tokenizer(cfg.TOKENIZER)

print("=" * 72)
print("EXP: re-run of the failed chunks (qwen2.5:1.5b, num_predict=2048)")
print("=" * 72)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print(f"Model                  : {cfg.effective_extraction_model()}")
print(f"num_predict            : {cfg.LLM_NUM_PREDICT}")
print(f"timeout_s              : {cfg.LLM_TIMEOUT_S}")
print("=" * 72)
print()

TARGET_IDS = {
    "a885be6b749e39a1": "transformer_architecture/3 (slow but OK in pilot)",
    "4af3dc8c5da84fad": "transformer_architecture/4 (FAILED: parse error / ReadTimeout)",
    "a4a679c427829368": "retrieval_augmented_generation/7 (FAILED: parse error)",
}

docs = load_documents("data/raw")
chunks = []
for d in docs:
    chunks.extend(chunk_document(d, cfg, tokenizer))
by_id = {c["chunk_id"]: c for c in chunks}
print(f"Corpus chunks: {len(chunks)}  target ids present: "
      f"{[i[:8] for i in TARGET_IDS if i in by_id]}")
print()

client = OllamaClient(
    cfg.effective_extraction_model(),
    temperature=cfg.EXTRACTION_TEMPERATURE,
    seed=cfg.EXTRACTION_SEED,
    timeout=cfg.LLM_TIMEOUT_S,
    num_predict=cfg.LLM_NUM_PREDICT,
)

# The ollama SDK is used DIRECTLY (not through OllamaClient) for these
# diagnostic calls so that `done_reason` and `eval_count` are available.
# done_reason == "length" is the definitive proof that the response was
# cut off by num_predict, which is the hypothesis under test.
import ollama  # noqa: E402

RESULTS = []


def run_direct(prompt: str, num_predict: int) -> dict:
    """One raw ollama call, returning text + the truncation diagnostics."""
    oclient = ollama.Client(timeout=cfg.LLM_TIMEOUT_S)
    options = {
        "temperature": cfg.EXTRACTION_TEMPERATURE,
        "seed": cfg.EXTRACTION_SEED,
        "num_predict": num_predict,
    }
    t0 = time.perf_counter()
    resp = oclient.generate(
        model=cfg.effective_extraction_model(), prompt=prompt, options=options
    )
    seconds = time.perf_counter() - t0
    text = resp.get("response") or ""
    return {
        "raw": text,
        "seconds": round(seconds, 2),
        "done_reason": resp.get("done_reason"),
        "eval_count": resp.get("eval_count"),
        "prompt_eval_count": resp.get("prompt_eval_count"),
    }


results = []
print(f"{'chunk_id':<18}{'doc/idx':<38}{'tok':>5}{'np':>6}{'secs':>8}{'eval_tok':>9}"
      f"{'done_reason':<12}{'status':<26}")
print("-" * 122)

for cid, note in TARGET_IDS.items():
    c = by_id.get(cid)
    if c is None:
        print(f"{cid[:16]:<18}{'NOT FOUND IN CORPUS':<38}")
        results.append({"chunk_id": cid, "status": "chunk_not_found"})
        continue

    label = f"{c['doc_id']}/{c['chunk_index']}"
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=c["text"])

    for num_predict in (cfg.LLM_NUM_PREDICT, 4096):
        reset_stats()
        raw = ""
        status = "ok"
        err = ""
        diag = {"done_reason": None, "eval_count": None}
        t0 = time.perf_counter()
        try:
            d = run_direct(prompt, num_predict)
            raw = d["raw"]
            seconds = d["seconds"]
            diag = d
        except Exception as e:                  # noqa: BLE001 - recorded, not swallowed
            status = f"EXC:{type(e).__name__}"
            err = str(e)[:160]
            seconds = time.perf_counter() - t0

        resp_tokens = len(tokenizer.encode(raw)) if raw else 0
        n_ent = n_rel = 0
        if raw:
            try:
                parsed = _extract_json(raw)
                n_ent = len(parsed.get("entities") or [])
                n_rel = len(parsed.get("relationships") or [])
            except Exception as e:              # noqa: BLE001
                status = f"PARSE_FAIL:{type(e).__name__}"
                err = str(e)[:160]
        elif not err:
            status = "EMPTY_RESPONSE"

        print(f"{cid[:16]:<18}{label:<38}{c['token_count']:>5}{num_predict:>6}"
              f"{seconds:>8.1f}{str(diag['eval_count']):>9}"
              f"{str(diag['done_reason']):<12}{status:<26}", flush=True)

        results.append({
            "chunk_id": cid,
            "note": note,
            "doc_id": c["doc_id"],
            "chunk_index": c["chunk_index"],
            "token_count": c["token_count"],
            "num_predict": num_predict,
            "seconds": round(seconds, 2),
            "response_tokens": resp_tokens,
            "response_chars": len(raw),
            "done_reason": diag["done_reason"],
            "eval_count": diag["eval_count"],
            "prompt_eval_count": diag["prompt_eval_count"],
            "status": status,
            "error": err,
            "entities": n_ent,
            "relationships": n_rel,
            "raw_response": raw,
        })

out = "data/cache/exp_timeout_chunks.jsonl"
with open(out, "w", encoding="utf-8") as fh:
    for r in results:
        fh.write(json.dumps(r) + "\n")

print()
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print(f"Wrote {out} ({len(results)} records)")
print("EXP timeout-chunks complete.")
