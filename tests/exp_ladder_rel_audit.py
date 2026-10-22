"""
exp_ladder_rel_audit.py -- where do the relationships go?

The ladder arm (C) parsed 42/42 chunks -- no failures -- but the corpus
relationship total fell from 161 (arm A) to 118 (arm C), and several chunks
came back with entities and ZERO relationships.  A shape that never appeared
in the other two arms.

This script re-parses each arm's raw responses out of that arm's own cache
and counts, per chunk, exactly where relationships are lost:

  raw_rels      -- relationship objects in the model's JSON
  empty_ep      -- dropped: source or target is an empty string
  unknown_ep    -- dropped: source or target is not in the entity list
  kept          -- what the extractor actually recorded

The point is to tell a "the model emitted nothing" story apart from a
"the model emitted real relationships and the filter threw them away"
story.  They call for different fixes.

READ-ONLY: it opens the experiment's cache and writes nothing.

Run:  python tests/exp_ladder_rel_audit.py [arm]
      arm in {A, B, C}; default C
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARMS = {
    "A": {
        "cache": "data/cache/exp_reextract",
        "records": "data/cache/exp_reextract.jsonl",
        "label": "A  fp=0.25, no retry",
    },
    "B": {
        "cache": "data/cache/exp_reextract_fp0",
        "records": "data/cache/exp_reextract_fp0.jsonl",
        "label": "B  fp=0,    no retry",
    },
    "C": {
        "cache": "data/cache/exp_reextract_ladder",
        "records": "data/cache/exp_reextract_ladder.jsonl",
        "label": "C  fp=0.25 + ladder",
    },
}

arm = (sys.argv[1] if len(sys.argv) > 1 else "C").upper()
if arm not in ARMS:
    print(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    raise SystemExit(1)
SPEC = ARMS[arm]

os.environ["CACHE_DIR"] = SPEC["cache"]
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


def tags_for(penalty: float) -> str:
    return (f"t=0.0;seed=42;np=4096;ctx=8192;fp={penalty};think=False")


def main() -> None:
    cfg = get_config()
    model = cfg.effective_extraction_model()
    cache = diskcache.Cache(SPEC["cache"])

    tokenizer = get_tokenizer(cfg.TOKENIZER)
    chunks = []
    for doc in load_documents("data/raw"):
        chunks.extend(chunk_document(doc, cfg, tokenizer))
    by_id = {c["chunk_id"]: c for c in chunks}

    recs = {}
    with (ROOT / SPEC["records"]).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["chunk_id"]] = r

    rows = []
    for cid, rec in recs.items():
        chunk = by_id.get(cid)
        if chunk is None:
            continue
        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
        # Try every penalty the arm could have used, and take the first
        # response that actually PARSES -- a chunk that failed at fp=0.25
        # and was rescued at fp=0.5 must be audited against the rescue.
        best = None
        for pen in (0.25, 0.5, 1.0):
            raw = cache.get(_make_key(prompt, model, tags_for(pen)), default=None)
            if raw is None:
                continue
            try:
                obj = _extract_json(raw)
            except Exception:
                continue
            best = (pen, obj, raw)
            break
        if best is None:
            continue
        pen, obj, raw = best
        ents = obj.get("entities") or []
        rels = obj.get("relationships") or []
        names = {
            str(e.get("name") or "").strip().lower()
            for e in ents
            if isinstance(e, dict)
        }
        empty_ep = unknown_ep = 0
        for r in rels:
            if not isinstance(r, dict):
                continue
            s = str(r.get("source") or "").strip()
            t = str(r.get("target") or "").strip()
            if not s or not t:
                empty_ep += 1
                continue
            if s.lower() not in names or t.lower() not in names:
                unknown_ep += 1
        rows.append({
            "chunk_id": cid,
            "penalty": pen,
            "raw_ents": len(ents),
            "raw_rels": len(rels),
            "empty_ep": empty_ep,
            "unknown_ep": unknown_ep,
            "kept": rec.get("relationships"),
            "kept_ents": rec.get("entities"),
        })

    total = {
        "raw_rels": sum(r["raw_rels"] for r in rows),
        "empty_ep": sum(r["empty_ep"] for r in rows),
        "unknown_ep": sum(r["unknown_ep"] for r in rows),
        "kept": sum(r["kept"] or 0 for r in rows),
    }
    print("=" * 88)
    print(f"WHERE THE RELATIONSHIPS GO  (arm {arm}: {SPEC['label']})")
    print("=" * 88)
    print(f"chunks audited: {len(rows)}")
    print(f"  relationship objects the model emitted  : {total['raw_rels']}")
    print(f"  dropped: empty source/target            : {total['empty_ep']}")
    print(f"  dropped: endpoint not in the entity list : {total['unknown_ep']}")
    print(f"  KEPT by the extractor                    : {total['kept']}")
    if total["raw_rels"]:
        lost = total["empty_ep"] + total["unknown_ep"]
        print(f"  -> loss rate to the consistency filter  : "
              f"{100.0 * lost / total['raw_rels']:.1f}%")
    print()

    lossy = [r for r in rows if r["raw_rels"] and (r["empty_ep"] or r["unknown_ep"])]
    lossy.sort(key=lambda r: -(r["empty_ep"] + r["unknown_ep"]))
    print(f"chunks that lost AT LEAST ONE relationship: {len(lossy)}")
    print("-" * 88)
    print(f"{'chunk_id':<18}{'fp':>5}{'raw_rels':>9}{'empty':>7}{'unknown':>9}"
          f"{'kept':>6}{'ents':>6}")
    for r in lossy[:20]:
        print(f"{r['chunk_id']:<18}{r['penalty']:>5}{r['raw_rels']:>9}{r['empty_ep']:>7}"
              f"{r['unknown_ep']:>9}{r['kept']:>6}{r['kept_ents']:>6}")
    print()

    zero = [r for r in rows if r["kept"] == 0 and r["kept_ents"]]
    print(f"chunks recorded with entities but ZERO relationships: {len(zero)}")
    print("-" * 88)
    for r in zero:
        why = []
        if r["raw_rels"] == 0:
            why.append("the model emitted NO relationship objects at all")
        if r["empty_ep"]:
            why.append(f"{r['empty_ep']} had an empty source/target")
        if r["unknown_ep"]:
            why.append(f"{r['unknown_ep']} referred to an entity not in the list")
        print(f"  {r['chunk_id']}  ents={r['kept_ents']}  raw_rels={r['raw_rels']}"
              f"  fp={r['penalty']}  -> {'; '.join(why) if why else 'kept 0 for another reason'}")
    print()
    print("=" * 88)


if __name__ == "__main__":
    main()
