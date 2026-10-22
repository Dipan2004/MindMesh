"""
exp_reextract_corpus.py -- does the frequency penalty hold up across the WHOLE
corpus, not just the one pathological chunk?

Why this script exists (2026-10-22). A single-chunk fix is an anecdote. The
frequency-penalty measurements (Item 3 of DOCUMENT.md) established that
frequency_penalty=0.25 breaks the 4af3dc8c repetition loop, recovers ~21 of
the chunk's ~22 real entities, and does not regress the two healthy chunks
that were checked. This script re-extracts EVERY chunk of the corpus with
the settled settings and reports the parse rate.

This is the precondition for open item 3 (a full-corpus re-extraction is
required before an A-vs-B comparison means anything -- the current
72-node / 30-edge graph makes Pipeline B a no-op filter).

Settings, all settled by measurement on 2026-10-22:
    model              qwen2.5:1.5b
    temperature        0.0
    seed               42
    num_predict        4096
    num_ctx            8192
    think              False
    frequency_penalty  0.25

Isolation:
  * own CACHE_DIR (data/cache/exp_reextract) -- never the canonical cache
  * own output file: data/cache/exp_reextract.jsonl (+ .json summary)
  * prints CACHE_DIR and cache entry count at start
  * calls the Ollama SDK directly -> no cache writes at all
  * does NOT touch the Zvec collections (read-only, and none are written)

Run:  python tests/exp_reextract_corpus.py

The frequency penalty can be overridden with the EXP_FP environment
variable, which is how the corpus-level CONTROL arm is run:

    EXP_FP=0    python tests/exp_reextract_corpus.py

The output path is fixed, so run one arm at a time and move the previous
arm's output aside if you need to keep it.
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_FP = os.environ.get("EXP_FP", "0.25")
# Each arm gets its own cache directory, so the two arms can never share
# a cache entry and the printed entry count means what it says.
CACHE_DIR = ("data/cache/exp_reextract" if _FP == "0.25"
             else f"data/cache/exp_reextract_fp{_FP}")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["CACHE_DIR"] = CACHE_DIR
os.environ["ZVEC_ROOT"] = "data/cache"
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
os.environ["EXTRACTION_MODEL"] = "qwen2.5:1.5b"
os.environ["LLM_NUM_PREDICT"] = "4096"
os.environ["LLM_NUM_CTX"] = "8192"
os.environ["LLM_THINK"] = "off"
os.environ["LLM_FREQUENCY_PENALTY"] = _FP

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
print("EXP: full-corpus re-extraction with frequency_penalty=0.25")
print("=" * 78)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print(f"Output file            : data/cache/exp_reextract.jsonl")
print(f"Model                  : {cfg.effective_extraction_model()}")
print(f"num_predict            : {cfg.LLM_NUM_PREDICT}")
print(f"num_ctx                : {cfg.LLM_NUM_CTX}")
print(f"think                  : {cfg.effective_think()}")
print(f"frequency_penalty      : {cfg.LLM_FREQUENCY_PENALTY}")
print("=" * 78)
print(flush=True)

docs = load_documents("data/raw")
all_chunks = []
for d in docs:
    all_chunks.extend(chunk_document(d, cfg, tokenizer))
print(f"[chunk]   {len(all_chunks)} chunks from {len(docs)} documents")
print(flush=True)

client = ollama.Client(timeout=3600)
out_path = "data/cache/exp_reextract.jsonl"
json_path = "data/cache/exp_reextract.json"
fp_tag = cfg.LLM_FREQUENCY_PENALTY
if fp_tag == 0:
    out_path = "data/cache/exp_reextract_fp0.jsonl"
    json_path = "data/cache/exp_reextract_fp0.json"
records = []
t_start = time.perf_counter()

# ------------------------------------------------------------------ #
# Resume support
# ------------------------------------------------------------------ #
# The script writes its JSONL incrementally, so a run that was interrupted
# can be continued without re-issuing the requests that already
# completed. The model is BISTABLE (see DOCUMENT.md Item 3), so the
# resumed chunks are issued in a later session than the earlier ones --
# that is a documented property of the measurement, not a hidden change.
#
# Set EXP_RESUME=0 to force a fresh run and overwrite the output.
resume = os.environ.get("EXP_RESUME", "1") != "0"
if resume and os.path.exists(out_path):
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[resume]  {len(records)} records already in {out_path}")

done_ids = {r["chunk_id"] for r in records}
mode = "a" if (resume and os.path.exists(out_path)) else "w"

with open(out_path, mode, encoding="utf-8") as fh:
    for i, chunk in enumerate(all_chunks):
        if chunk["chunk_id"] in done_ids:
            continue
        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
        options = {
            "temperature": 0.0,
            "seed": 42,
            "num_predict": cfg.LLM_NUM_PREDICT,
            "num_ctx": cfg.LLM_NUM_CTX,
            "frequency_penalty": cfg.LLM_FREQUENCY_PENALTY,
        }

        t0 = time.perf_counter()
        try:
            resp = client.generate(
                model=cfg.effective_extraction_model(),
                prompt=prompt,
                think=False,
                options=options,
            )
            err = ""
        except Exception as e:                        # noqa: BLE001
            resp = {"response": "", "eval_count": None,
                    "prompt_eval_count": None, "done_reason": "error"}
            err = f"{type(e).__name__}: {str(e)[:200]}"
        seconds = time.perf_counter() - t0

        text = resp.get("response") or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        counter = Counter(lines)
        top_count = counter.most_common(1)[0][1] if counter else 0

        n_ent = n_rel = -1
        names = []
        status = "ok"
        perr = ""
        try:
            parsed = _extract_json(text)
            n_ent = len(parsed.get("entities") or [])
            n_rel = len(parsed.get("relationships") or [])
            names = [e.get("name", "") for e in (parsed.get("entities") or [])
                     if isinstance(e, dict)]
        except Exception as e:                        # noqa: BLE001
            status = "FAIL"
            perr = f"{type(e).__name__}: {str(e)[:160]}"

        rec = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "chunk_index": chunk["chunk_index"],
            "seconds": round(seconds, 2),
            "prompt_eval_count": resp.get("prompt_eval_count"),
            "eval_count": resp.get("eval_count"),
            "done_reason": resp.get("done_reason"),
            "response_chars": len(text),
            "nonempty_lines": len(lines),
            "distinct_lines": len(counter),
            "most_repeated_line_count": top_count,
            "parse_status": status,
            "parse_error": perr,
            "transport_error": err,
            "entities": n_ent,
            "relationships": n_rel,
            "entity_names": names,
        }
        records.append(rec)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

        print(f"[{i+1:3d}/{len(all_chunks)}] {chunk['doc_id']:<26} "
              f"idx={chunk['chunk_index']:<3} {seconds:7.1f}s "
              f"eval={str(rec['eval_count']):<5} done={str(rec['done_reason']):<7} "
              f"chars={rec['response_chars']:<6} x{top_count:<3} "
              f"ent={n_ent:<3} rel={n_rel:<3} {status}"
              + (f"  {perr}" if perr else ""), flush=True)

total_secs = time.perf_counter() - t_start
parsed_ok = [r for r in records if r["parse_status"] == "ok"]
failed = [r for r in records if r["parse_status"] != "ok"]
looped = [r for r in records if r["most_repeated_line_count"] > 5]
truncated = [r for r in records if r["done_reason"] == "length"]

summary = {
    "chunks": len(records),
    "parsed_ok": len(parsed_ok),
    "parse_failures": len(failed),
    "looping_chunks": len(looped),
    "truncated_by_cap": len(truncated),
    "total_seconds": round(total_secs, 1),
    # On a resumed run, total_seconds covers only the requests issued in
    # THIS session. The per-chunk "seconds" fields in the records are the
    # per-request truth; their sum is the run's real cost.
    "sum_of_chunk_seconds": round(sum(r["seconds"] for r in records), 1),
    "resumed": len(records) - len(all_chunks) + len(
        [c for c in all_chunks if c["chunk_id"] in done_ids]),
    "entities_total": sum(r["entities"] for r in records if r["entities"] > 0),
    "relationships_total": sum(r["relationships"] for r in records if r["relationships"] > 0),
    "failed_chunk_ids": [r["chunk_id"] for r in failed],
    "looping_chunk_ids": [r["chunk_id"] for r in looped],
}

print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"  chunks                  : {summary['chunks']}")
print(f"  parsed OK               : {summary['parsed_ok']}")
print(f"  parse failures          : {summary['parse_failures']}  "
      f"{summary['failed_chunk_ids']}")
print(f"  looping (top_line_x>5)  : {summary['looping_chunks']}  "
      f"{summary['looping_chunk_ids']}")
print(f"  truncated by cap        : {summary['truncated_by_cap']}")
print(f"  entities extracted      : {summary['entities_total']}")
print(f"  relationships extracted : {summary['relationships_total']}")
print(f"  wall clock              : {summary['total_seconds']} s")
print()
print("For comparison, the previous run had 2 of 42 chunks failing to parse "
      "(a4a679c4 and 4af3dc8c).")

with open(json_path, "w", encoding="utf-8") as fh:
    json.dump({"summary": summary, "records": records}, fh, indent=2)

print()
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print(f"Wrote {out_path} and {json_path}")
print("EXP corpus re-extraction complete.")
