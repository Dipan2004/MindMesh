"""
exp_pilot_15b.py -- Pilot on qwen2.5:1.5b, done properly.

ONE traversal over the two pilot documents. For every chunk we make exactly
two LLM calls:

  1. first-pass extraction   (timed, cache hit/miss counted)
  2. one gleaning pass       (timed, cache hit/miss counted)

The SAME first pass feeds BOTH settings, so "gleaning off" and "gleaning on"
are compared on identical input -- the difference is exactly the gleaning
call and nothing else. The original pilot_1.5b.py ran the two settings as
two separate traversals, so the second one was served entirely from cache
and the comparison was not controlled.

Isolation:
  * own CACHE_DIR  (data/cache/exp_pilot_15b) -- never the canonical cache
  * own output files:
      data/cache/exp_pilot_15b_glean0.jsonl   (first pass only)
      data/cache/exp_pilot_15b_glean1.jsonl   (first pass + gleaning)
      data/cache/exp_pilot_15b_perchunk.jsonl (per-chunk timing ledger)
  * prints CACHE_DIR and cache entry count at start

Run:  python tests/exp_pilot_15b.py
"""
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Must be set BEFORE importing graphrag_lite ──────────────────────────────
CACHE_DIR = "data/cache/exp_pilot_15b"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["CACHE_DIR"] = CACHE_DIR
os.environ["ZVEC_ROOT"] = "data/cache"          # canonical collections stay reachable
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
os.environ["EXTRACTION_MODEL"] = "qwen2.5:1.5b"
os.environ["SLM_MODEL"] = "qwen2.5:1.5b"
os.environ["LLM_TIMEOUT_S"] = "600"
os.environ["LLM_NUM_PREDICT"] = "2048"

from graphrag_lite.config import get_config            # noqa: E402
get_config.cache_clear()
from graphrag_lite.cache.llm_cache import (            # noqa: E402
    cache_stats, _get_cache, reset_stats,
)
from graphrag_lite.ingestion.chunker import (           # noqa: E402
    chunk_document, get_tokenizer,
)
from graphrag_lite.ingestion.extractor import (         # noqa: E402
    extract_entities_relationships,
)
from graphrag_lite.ingestion.gleaning import glean      # noqa: E402
from graphrag_lite.ingestion.loader import load_documents   # noqa: E402
from graphrag_lite.llm.ollama_client import OllamaClient    # noqa: E402

cfg = get_config()
cache = _get_cache()

print("=" * 72)
print("EXP: pilot qwen2.5:1.5b -- one traversal, two settings from one first pass")
print("=" * 72)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print(f"Model                  : {cfg.effective_extraction_model()}")
print(f"num_predict            : {cfg.LLM_NUM_PREDICT}")
print(f"temperature / seed     : {cfg.EXTRACTION_TEMPERATURE} / {cfg.EXTRACTION_SEED}")
print(f"timeout_s              : {cfg.LLM_TIMEOUT_S}")
print(f"ZVEC_ROOT              : {cfg.ZVEC_ROOT} (canonical collections, read-only)")
print("=" * 72)
print()

TARGET_DOCS = {"transformer_architecture", "retrieval_augmented_generation"}


class TimedClient:
    """Wraps OllamaClient and records wall-time + raw length for every call."""

    def __init__(self, inner):
        self.inner = inner
        self.log = []          # list of dicts: {seconds, raw_chars}

    def generate(self, prompt):
        t0 = time.perf_counter()
        raw = self.inner.generate(prompt)
        dt = time.perf_counter() - t0
        self.log.append({"seconds": round(dt, 2), "raw_chars": len(raw)})
        return raw


def _last_call(timed):
    return timed.log[-1] if timed.log else {"seconds": 0.0, "raw_chars": 0}


tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = [d for d in load_documents("data/raw") if d["doc_id"] in TARGET_DOCS]
docs.sort(key=lambda d: d["doc_id"])
print(f"Pilot docs: {[d['doc_id'] for d in docs]}")
print()

all_chunks = []
for d in docs:
    all_chunks.extend(chunk_document(d, cfg, tokenizer))
all_chunks.sort(key=lambda c: (c["doc_id"], c["chunk_index"]))
print(f"Chunks: {len(all_chunks)}")
print()
print(f"{'doc/chunk':<38}{'tok':>5}{'e1':>5}{'e2':>5}{'r1':>5}{'r2':>5}"
      f"{'t_ext':>8}{'t_gl':>8}{'status':>10}")
print("-" * 102)

glean0_records = []
glean1_records = []
ledger = []
failed = 0
t_start = time.perf_counter()

for c in all_chunks:
    label = f"{c['doc_id']}/{c['chunk_index']}"
    base_client = OllamaClient(
        cfg.effective_extraction_model(),
        temperature=cfg.EXTRACTION_TEMPERATURE,
        seed=cfg.EXTRACTION_SEED,
        timeout=cfg.LLM_TIMEOUT_S,
        num_predict=cfg.LLM_NUM_PREDICT,
    )
    timed = TimedClient(base_client)

    # ---- pass 1: first-pass extraction ------------------------------------
    reset_stats()
    t0 = time.perf_counter()
    glean_status = "n/a"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = extract_entities_relationships(c, timed)
        ext_stats = cache_stats()
        t_ext = time.perf_counter() - t0
        ext_call = _last_call(timed)
    except Exception as e:                      # noqa: BLE001 - recorded, not swallowed
        ext_stats = cache_stats()
        t_ext = time.perf_counter() - t0
        failed += 1
        first = {
            "entities": [], "relationships": [],
            "chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
            "_parse_error": f"{type(e).__name__}: {str(e)[:120]}",
        }
        glean_status = "skipped"
        ext_call = _last_call(timed)

    e1 = len(first.get("entities", []))
    r1 = len(first.get("relationships", []))

    # ---- pass 2: gleaning --------------------------------------------------
    t_gl = 0.0
    gl_call = {"seconds": 0.0, "raw_chars": 0}
    if "_parse_error" not in first:
        reset_stats()
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final = glean(c, first, timed, max_gleanings=1)
            t_gl = time.perf_counter() - t0
            gl_call = _last_call(timed)
            glean_status = "ok"
        except Exception as e:                  # noqa: BLE001 - recorded, not swallowed
            t_gl = time.perf_counter() - t0
            gl_call = _last_call(timed)
            final = first                        # gleaning failed -> first pass stands
            glean_status = f"FAIL:{type(e).__name__}"
    else:
        final = first

    e2 = len(final.get("entities", []))
    r2 = len(final.get("relationships", []))

    print(f"{label:<38}{c['token_count']:>5}{e1:>5}{e2:>5}{r1:>5}{r2:>5}"
          f"{t_ext:>8.1f}{t_gl:>8.1f}{glean_status:>10}", flush=True)

    g0 = dict(first)
    g1 = dict(final)
    glean0_records.append(g0)
    glean1_records.append(g1)

    ledger.append({
        "doc_id": c["doc_id"],
        "chunk_index": c["chunk_index"],
        "chunk_id": c["chunk_id"],
        "token_count": c["token_count"],
        "first_pass_entities": e1,
        "first_pass_relationships": r1,
        "post_glean_entities": e2,
        "post_glean_relationships": r2,
        "extract_seconds": round(t_ext, 2),
        "glean_seconds": round(t_gl, 2),
        "extract_cache_hit": ext_stats["hits"] > 0,
        "glean_cache_hit": gl_call["seconds"] == 0.0 and "FAIL" not in glean_status,
        "glean_status": glean_status,
        "first_pass_error": first.get("_parse_error", ""),
    })

elapsed = time.perf_counter() - t_start

# ── write outputs ──────────────────────────────────────────────────────────
out0 = "data/cache/exp_pilot_15b_glean0.jsonl"
out1 = "data/cache/exp_pilot_15b_glean1.jsonl"
outl = "data/cache/exp_pilot_15b_perchunk.jsonl"
for path, rows in ((out0, glean0_records), (out1, glean1_records), (outl, ledger)):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

print()
print("=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print(f"chunks={len(all_chunks)}  failed={failed}  elapsed={elapsed:.0f}s")
tot_e0 = sum(r["first_pass_entities"] for r in ledger)
tot_r0 = sum(r["first_pass_relationships"] for r in ledger)
tot_e1 = sum(r["post_glean_entities"] for r in ledger)
tot_r1 = sum(r["post_glean_relationships"] for r in ledger)
print(f"gleaning OFF: entities={tot_e0}  relationships={tot_r0}")
print(f"gleaning ON : entities={tot_e1}  relationships={tot_r1}")
print(f"delta       : entities=+{tot_e1 - tot_e0}  relationships=+{tot_r1 - tot_r0}")
print()

# ---- per-chunk delta table: this is where "off > on" would show up -------
print(f"{'chunk':<18}{'e_off':>6}{'e_on':>6}{'r_off':>6}{'r_on':>6}"
      f"{'d_e':>5}{'d_r':>5}  {'glean':<22}")
print("-" * 92)
off_better = []
for r in ledger:
    de = r["post_glean_entities"] - r["first_pass_entities"]
    dr = r["post_glean_relationships"] - r["first_pass_relationships"]
    if de < 0 or dr < 0:
        off_better.append(r["chunk_id"])
    print(f"{r['chunk_id'][:16]:<18}{r['first_pass_entities']:>6}"
          f"{r['post_glean_entities']:>6}{r['first_pass_relationships']:>6}"
          f"{r['post_glean_relationships']:>6}{de:>5}{dr:>5}  {r['glean_status']:<22}")
print()
print(f"Chunks where gleaning-OFF > gleaning-ON : {off_better if off_better else 'NONE'}")
print()
print(f"Wrote {out0} ({len(glean0_records)} records)")
print(f"Wrote {out1} ({len(glean1_records)} records)")
print(f"Wrote {outl} ({len(ledger)} records)")
print("EXP pilot complete.")
