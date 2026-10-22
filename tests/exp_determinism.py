"""
exp_determinism.py -- Determinism of the first pass, and the
gleaning-off vs gleaning-on comparison.

PART 1 -- DETERMINISM
  For 3 chunks we call the model TWICE with the identical prompt, identical
  temperature (0.0) and identical seed (42), and compare the two raw
  responses byte-for-byte. The calls go straight to OllamaClient.generate()
  (the same call cached_call makes on a miss) so the disk cache cannot
  mask a difference.

PART 2 -- GLEANING OFF vs ON
  For transformer_architecture chunks 0 and 2 we print the full first-pass
  entity and relationship lists, then the post-gleaning lists, and state
  which direction each count moved. The hypothesis under test is that
  "gleaning off" beats "gleaning on" for some chunks; this script reports
  the actual direction per chunk rather than assuming one.

Isolation:
  * own CACHE_DIR  (data/cache/exp_determinism) -- never the canonical cache
  * own output file: data/cache/exp_determinism.json
  * prints CACHE_DIR and cache entry count at start

Run:  python tests/exp_determinism.py
"""
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE_DIR = "data/cache/exp_determinism"
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
from graphrag_lite.cache.llm_cache import _get_cache   # noqa: E402
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer   # noqa: E402
from graphrag_lite.ingestion.extractor import (      # noqa: E402
    _EXTRACTION_PROMPT_TEMPLATE, _extract_json, extract_entities_relationships,
)
from graphrag_lite.ingestion.gleaning import glean   # noqa: E402
from graphrag_lite.ingestion.loader import load_documents   # noqa: E402
from graphrag_lite.llm.ollama_client import OllamaClient    # noqa: E402

cfg = get_config()
cache = _get_cache()
tokenizer = get_tokenizer(cfg.TOKENIZER)

print("=" * 72)
print("EXP: determinism of the first pass + gleaning off/on comparison")
print("=" * 72)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print(f"Model                  : {cfg.effective_extraction_model()}")
print(f"temperature / seed     : {cfg.EXTRACTION_TEMPERATURE} / {cfg.EXTRACTION_SEED}")
print(f"num_predict            : {cfg.LLM_NUM_PREDICT}")
print("=" * 72)
print()

docs = load_documents("data/raw")
ta_doc = next(d for d in docs if d["doc_id"] == "transformer_architecture")
chunks = chunk_document(ta_doc, cfg, tokenizer)
by_idx = {c["chunk_index"]: c for c in chunks}

DET_IDX = [0, 1, 2]        # determinism: first pass run twice
GLEAN_IDX = [0, 2]         # gleaning off vs on: full lists printed

client = OllamaClient(
    cfg.effective_extraction_model(),
    temperature=cfg.EXTRACTION_TEMPERATURE,
    seed=cfg.EXTRACTION_SEED,
    timeout=cfg.LLM_TIMEOUT_S,
    num_predict=cfg.LLM_NUM_PREDICT,
)

results = {"determinism": [], "gleaning": []}

# ══════════════════════════════════════════════════════════════════════════
# PART 1 -- determinism
# ══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PART 1 -- DETERMINISM: same prompt, same seed, two independent calls")
print("=" * 72)
print(f"{'chunk':<12}{'tok':>5}{'s1':>8}{'s2':>8}{'len1':>7}{'len2':>7}"
      f"{'identical':>11}{'e1':>4}{'e2':>4}")
print("-" * 84)

for idx in DET_IDX:
    c = by_idx[idx]
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=c["text"])

    raws, secs, ents = [], [], []
    for _ in range(2):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = client.generate(prompt)
        secs.append(time.perf_counter() - t0)
        raws.append(raw)
        try:
            ents.append(len(_extract_json(raw).get("entities") or []))
        except Exception:                       # noqa: BLE001
            ents.append(-1)

    identical = raws[0] == raws[1]
    print(f"{c['chunk_id'][:10]:<12}{c['token_count']:>5}{secs[0]:>8.1f}{secs[1]:>8.1f}"
          f"{len(raws[0]):>7}{len(raws[1]):>7}{str(identical):>11}"
          f"{ents[0]:>4}{ents[1]:>4}", flush=True)

    entry = {
        "chunk_index": idx,
        "chunk_id": c["chunk_id"],
        "token_count": c["token_count"],
        "call1_seconds": round(secs[0], 2),
        "call2_seconds": round(secs[1], 2),
        "call1_chars": len(raws[0]),
        "call2_chars": len(raws[1]),
        "identical": identical,
        "call1_entities": ents[0],
        "call2_entities": ents[1],
    }
    if not identical:
        a, b = raws
        n = min(len(a), len(b))
        first_diff = next((i for i in range(n) if a[i] != b[i]), n)
        entry["first_diff_offset"] = first_diff
        entry["common_prefix_chars"] = first_diff
        entry["call1_tail"] = a[max(0, first_diff - 60):first_diff + 120]
        entry["call2_tail"] = b[max(0, first_diff - 60):first_diff + 120]
    results["determinism"].append(entry)

n_ident = sum(1 for r in results["determinism"] if r["identical"])
print()
print(f"Identical raw responses: {n_ident}/{len(results['determinism'])}")
print()

# ══════════════════════════════════════════════════════════════════════════
# PART 2 -- gleaning off vs on
# ══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PART 2 -- GLEANING OFF vs ON (transformer_architecture chunks 0 and 2)")
print("=" * 72)

off_better = []
for idx in GLEAN_IDX:
    c = by_idx[idx]
    print()
    print(f"--- chunk {idx}  id={c['chunk_id']}  tokens={c['token_count']} ---")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = extract_entities_relationships(c, client)

    t0 = time.perf_counter()
    glean_status = "ok"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final = glean(c, first, client, max_gleanings=1)
    except Exception as e:                      # noqa: BLE001
        final = first
        glean_status = f"FAIL:{type(e).__name__}: {str(e)[:80]}"
    t_gl = time.perf_counter() - t0

    off_e, off_r = first["entities"], first["relationships"]
    on_e, on_r = final["entities"], final["relationships"]

    print(f"  GLEANING OFF (first pass only):  {len(off_e)} entities, {len(off_r)} relationships")
    for e in off_e:
        print(f"      E  {e['name']!r}  [{e['type']}]  {e['description'][:70]}")
    for r in off_r:
        print(f"      R  {r['source']!r} --{r['type']}--> {r['target']!r}")

    print(f"  GLEANING ON  (after 1 glean):    {len(on_e)} entities, {len(on_r)} relationships")
    for e in on_e:
        print(f"      E  {e['name']!r}  [{e['type']}]  {e['description'][:70]}")
    for r in on_r:
        print(f"      R  {r['source']!r} --{r['type']}--> {r['target']!r}")

    off_names = {e["name"].strip().lower() for e in off_e}
    on_names = {e["name"].strip().lower() for e in on_e}
    off_trip = {(r["source"].strip().lower(), r["target"].strip().lower(), r["type"].strip().upper()) for r in off_r}
    on_trip = {(r["source"].strip().lower(), r["target"].strip().lower(), r["type"].strip().upper()) for r in on_r}

    new_e = sorted(on_names - off_names)
    new_r = sorted(on_trip - off_trip)
    lost_e = sorted(off_names - on_names)
    lost_r = sorted(off_trip - on_trip)

    d_e = len(on_e) - len(off_e)
    d_r = len(on_r) - len(off_r)
    verdict = "OFF > ON" if (d_e < 0 or d_r < 0) else ("ON > OFF" if (d_e > 0 or d_r > 0) else "TIE")
    if verdict == "OFF > ON":
        off_better.append(c["chunk_id"])

    print(f"  DELTA  entities {d_e:+d}   relationships {d_r:+d}   -> {verdict}")
    print(f"  entities added by gleaning     : {new_e if new_e else 'none'}")
    print(f"  entities lost by gleaning      : {lost_e if lost_e else 'none'}")
    print(f"  relationships added by gleaning: {new_r if new_r else 'none'}")
    print(f"  relationships lost by gleaning : {lost_r if lost_r else 'none'}")
    print(f"  glean status                   : {glean_status}  ({t_gl:.1f}s)")

    results["gleaning"].append({
        "chunk_index": idx,
        "chunk_id": c["chunk_id"],
        "token_count": c["token_count"],
        "off_entities": len(off_e),
        "off_relationships": len(off_r),
        "on_entities": len(on_e),
        "on_relationships": len(on_r),
        "delta_entities": d_e,
        "delta_relationships": d_r,
        "verdict": verdict,
        "glean_status": glean_status,
        "glean_seconds": round(t_gl, 2),
        "off_entity_list": [dict(e) for e in off_e],
        "off_relationship_list": [dict(r) for r in off_r],
        "on_entity_list": [dict(e) for e in on_e],
        "on_relationship_list": [dict(r) for r in on_r],
        "entities_added": new_e,
        "entities_lost": lost_e,
        "relationships_added": new_r,
        "relationships_lost": lost_r,
    })

print()
print("=" * 72)
print(f"Chunks where gleaning-OFF > gleaning-ON : {off_better if off_better else 'NONE'}")
print("=" * 72)

out = "data/cache/exp_determinism.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2, ensure_ascii=False)

print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print(f"Wrote {out}")
print("EXP determinism complete.")
