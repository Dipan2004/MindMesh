"""
exp_fp_penalty_choice.py -- which frequency penalty should be the default?

Why this script exists (2026-10-22). tests/exp_freq_penalty_robustness.py
established that a frequency penalty breaks the 4af3dc8c repetition loop
(9/9 parses at 0.25 / 0.5 / 0.75), but the three settings recover very
different amounts of content:

    fp=0.25   eval=2583  top_line_x1  21 entities  15 relationships  161 s
    fp=0.5    eval=489   top_line_x7  10 entities   8 relationships   25 s
    fp=0.75   eval=353   top_line_x1   8 entities   3 relationships   19 s

Chunk 4af3dc8c names roughly 22 real entities (Ashish Vaswani, Transformer,
Google Brain, Google Research, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan Gomez, Lukasz Kaiser, Illia Polosukhin, BERT, OpenAI,
GPT, T5, ViT, self-attention, Longformer, BigBird, Linformer, Adam), so
fp=0.25 recovers ~the whole set while 0.5 and 0.75 UNDER-extract. The
headline "0.5 is 8.5x faster" was misleading: 0.5 is fast because the
model gives up early, not because the penalty is cheap.

This script settles the choice with two questions:

  Q1  Does fp=0.25 hold up on the HEALTHY chunks, 3 runs each? The penalty
      must not suppress legitimate repetition (an entity mentioned twice, a
      key repeated in a list) in chunks that already parsed cleanly.

  Q2  Is the penalty CONFERring determinism? Chunk 2 (a5123f46) is the
      chunk that diverged between two earlier runs with the same seed and
      temperature (896 chars / 2 entities vs 5152 chars / 14 entities).
      If 3 runs at fp=0.25 agree and 3 runs at no penalty do not, that is
      direct evidence that the penalty stabilises decoding.

  Q3  What does fp=0.25 actually EXTRACT from the pathological chunk? The
      entity names are printed for inspection -- a penalty that produced 21
      entities would be useless if they were all near-duplicates.

Isolation:
  * own CACHE_DIR (data/cache/exp_fp_choice) -- never the canonical cache
  * own output file: data/cache/exp_fp_choice.json
  * prints CACHE_DIR and cache entry count at start
  * calls the Ollama SDK directly -> no cache writes at all

Run:  python tests/exp_fp_penalty_choice.py
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE_DIR = "data/cache/exp_fp_choice"
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
print("EXP: which frequency penalty should be the default?")
print("=" * 78)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print(f"Output file            : data/cache/exp_fp_choice.json")
print("=" * 78)
print(flush=True)

docs = load_documents("data/raw")
ta = next(d for d in docs if d["doc_id"] == "transformer_architecture")
chunks = {c["chunk_index"]: c for c in chunk_document(ta, cfg, tokenizer)}
BAD = 4
HEALTHY = [0, 2]
FP = 0.25


def run(chunk_index: int, fp: float | None, num_predict: int = 4096) -> dict:
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunks[chunk_index]["text"])
    options = {
        "temperature": 0.0,
        "seed": 42,
        "num_predict": num_predict,
    }
    if fp is not None:
        options["frequency_penalty"] = fp

    client = ollama.Client(timeout=2400)
    t0 = time.perf_counter()
    resp = client.generate(model="qwen2.5:1.5b", prompt=prompt, options=options)
    seconds = time.perf_counter() - t0

    text = resp.get("response") or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    counter = Counter(lines)
    top_line, top_count = (counter.most_common(1)[0] if counter else ("", 0))

    n_ent = n_rel = -1
    names = []
    status = "ok"
    err = ""
    try:
        parsed = _extract_json(text)
        ents = parsed.get("entities") or []
        rels = parsed.get("relationships") or []
        n_ent = len(ents)
        n_rel = len(rels)
        names = [e.get("name", "") for e in ents if isinstance(e, dict)]
    except Exception as e:                        # noqa: BLE001
        status = "FAIL"
        err = str(e)[:140]

    return {
        "chunk_index": chunk_index,
        "chunk_id": chunks[chunk_index].get("chunk_id"),
        "frequency_penalty": fp,
        "seconds": round(seconds, 2),
        "prompt_eval_count": resp.get("prompt_eval_count"),
        "eval_count": resp.get("eval_count"),
        "done_reason": resp.get("done_reason"),
        "response_chars": len(text),
        "response_sha": __import__("hashlib").sha256(text.encode()).hexdigest()[:12],
        "nonempty_lines": len(lines),
        "distinct_lines": len(counter),
        "most_repeated_line_count": top_count,
        "parse_status": status,
        "parse_error": err,
        "entities": n_ent,
        "relationships": n_rel,
        "entity_names": names,
    }


def show(rec: dict) -> None:
    print(f"  chunk {rec['chunk_index']} ({rec['chunk_id'][:8]})  "
          f"fp={rec['frequency_penalty']}")
    print(f"    secs={rec['seconds']:.1f}  eval={rec['eval_count']}  "
          f"done={rec['done_reason']}  chars={rec['response_chars']}  "
          f"sha={rec['response_sha']}  top_line_x{rec['most_repeated_line_count']}")
    print(f"    parse={rec['parse_status']}  ent={rec['entities']} "
          f"rel={rec['relationships']}"
          + (f"  err={rec['parse_error']}" if rec['parse_error'] else ""))
    print(flush=True)


records = []

print("--- Q1: healthy chunks at fp=0.25, 3 runs each ---")
for gi in HEALTHY:
    for rep in range(3):
        rec = run(gi, FP)
        rec["rep"] = rep
        records.append(rec)
        show(rec)
    print()

print("--- Q2: determinism -- chunk 2 (a5123f46) with and without the penalty ---")
for fp in [None, FP]:
    for rep in range(3):
        rec = run(2, fp)
        rec["rep"] = rep
        records.append(rec)
        show(rec)
    print()

print("--- Q3: what does fp=0.25 extract from the pathological chunk? ---")
rec = run(BAD, FP)
rec["rep"] = 0
records.append(rec)
show(rec)
print("    entity names:")
for nm in rec["entity_names"]:
    print(f"      - {nm}")
print()

print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("Q1 -- healthy chunks at fp=0.25, 3 runs each:")
for gi in HEALTHY:
    rs = [r for r in records
          if r["chunk_index"] == gi and r["frequency_penalty"] == FP]
    shas = {r["response_sha"] for r in rs}
    print(f"  chunk {gi} ({rs[0]['chunk_id'][:8]}): parsed "
          f"{sum(1 for r in rs if r['parse_status'] == 'ok')}/{len(rs)}  "
          f"ent={[r['entities'] for r in rs]}  rel={[r['relationships'] for r in rs]}  "
          f"identical_responses={len(shas) == 1}")

print()
print("Q2 -- determinism on chunk 2 (the chunk that diverged before):")
for fp in [None, FP]:
    rs = [r for r in records
          if r["chunk_index"] == 2 and r["frequency_penalty"] == fp]
    shas = {r["response_sha"] for r in rs}
    print(f"  fp={str(fp):<5} distinct_responses={len(shas)}  "
          f"shas={sorted(shas)}  ent={[r['entities'] for r in rs]}  "
          f"chars={[r['response_chars'] for r in rs]}")

print()
print(f"Q3 -- pathological chunk at fp={FP}: "
      f"{rec['entities']} entities, {rec['relationships']} relationships")
print(f"  distinct entity names: {len(set(rec['entity_names']))} of {rec['entities']}")

with open("data/cache/exp_fp_choice.json", "w", encoding="utf-8") as fh:
    json.dump(records, fh, indent=2)

print()
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print("Wrote data/cache/exp_fp_choice.json")
print("EXP frequency-penalty choice complete.")
