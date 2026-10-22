"""
exp_backfill_raw_counts.py -- make the three arms comparable.

The three corpus arms counted DIFFERENT things:

  A / B  (tests/exp_reextract_corpus.py)
        counted the RAW JSON: len(parsed["relationships"])
  C      (tests/exp_reextract_corpus_ladder.py)
        counted the PARSED, FILTERED result:
        len(result["relationships"]) from extract_entities_relationships

The extractor's endpoint-consistency filter drops any relationship whose
source or target is not in the chunk's own entity list, so arm C's counts
are post-filter and arms A/B's are pre-filter.  Comparing them directly
manufactures a relationship "drop" that never happened.

This script re-derives arm C's RAW counts from the run's own cache (the
raw responses are still there) and writes them into the records as
``raw_entities`` / ``raw_relationships``, so every arm can be compared on
the same basis.

It does not invent anything: every number it writes is read out of the
raw LLM response that the run itself cached.

Run:  python tests/exp_backfill_raw_counts.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CACHE_DIR = "data/cache/exp_reextract_ladder"
JSONL = ROOT / "data/cache/exp_reextract_ladder.jsonl"

os.environ["CACHE_DIR"] = CACHE_DIR
os.environ["ZVEC_ROOT"] = "data/cache"
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
os.environ["EXTRACTION_MODEL"] = "qwen2.5:1.5b"
os.environ["LLM_NUM_PREDICT"] = "4096"
os.environ["LLM_NUM_CTX"] = "8192"
os.environ["LLM_THINK"] = "off"

import diskcache  # noqa: E402

from graphrag_lite.config import get_config  # noqa: E402

get_config.cache_clear()

from graphrag_lite.cache.llm_cache import _make_key  # noqa: E402
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer  # noqa: E402
from graphrag_lite.ingestion.extractor import (  # noqa: E402
    _EXTRACTION_PROMPT_TEMPLATE,
    _extract_json,
)
from graphrag_lite.ingestion.loader import load_documents  # noqa: E402

PENALTIES = (0.25, 0.5, 1.0)


def tag(penalty: float) -> str:
    return f"t=0.0;seed=42;np=4096;ctx=8192;fp={penalty};think=False"


def main() -> None:
    cfg = get_config()
    model = cfg.effective_extraction_model()
    cache = diskcache.Cache(CACHE_DIR)

    tokenizer = get_tokenizer(cfg.TOKENIZER)
    chunks = []
    for doc in load_documents("data/raw"):
        chunks.extend(chunk_document(doc, cfg, tokenizer))
    by_id = {c["chunk_id"]: c for c in chunks}

    recs = []
    with JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    changed = 0
    for rec in recs:
        if "raw_relationships" in rec:
            continue
        chunk = by_id.get(rec["chunk_id"])
        if chunk is None:
            continue
        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
        for pen in PENALTIES:
            raw = cache.get(_make_key(prompt, model, tag(pen)), default=None)
            if raw is None:
                continue
            try:
                obj = _extract_json(raw)
            except Exception:
                continue      # this attempt failed to parse; try the next rung
            rec["raw_entities"] = len(obj.get("entities") or [])
            rec["raw_relationships"] = len(obj.get("relationships") or [])
            rec["raw_penalty"] = pen
            changed += 1
            break

    with JSONL.open("w", encoding="utf-8") as fh:
        for rec in recs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"backfilled {changed} of {len(recs)} records with raw counts")
    raw_e = sum(r.get("raw_entities", 0) for r in recs)
    raw_r = sum(r.get("raw_relationships", 0) for r in recs)
    kept_e = sum(max(r.get("entities", 0), 0) for r in recs)
    kept_r = sum(max(r.get("relationships", 0), 0) for r in recs)
    print(f"  arm C RAW  (what the model emitted) : entities={raw_e} relationships={raw_r}")
    print(f"  arm C KEPT (after the extractor filter): entities={kept_e} relationships={kept_r}")
    if raw_r:
        print(f"  -> the endpoint-consistency filter dropped "
              f"{raw_r - kept_r} of {raw_r} relationships "
              f"({100.0 * (raw_r - kept_r) / raw_r:.1f}%)")


if __name__ == "__main__":
    main()
