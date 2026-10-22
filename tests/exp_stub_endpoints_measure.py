"""
exp_stub_endpoints_measure.py -- how much does the conditional endpoint
filter recover, and what does it recover?

The extractor drops 53 of the 171 relationships the ladder arm emits (31%),
45 of them because the model named an entity in a relationship that it did
not also list as an entity.  This script measures the CONDITIONAL fix
(step 2) on the ladder arm's own raw responses:

    if a relationship endpoint is not in the chunk's entity list BUT its
    name occurs verbatim in the chunk text (casefold, word boundaries),
    add it as an entity with type OTHER and marker stub=True;
    otherwise drop as before.

NO NEW LLM CALLS.  Every raw response is read out of the ladder arm's OWN
cache (data/cache/exp_reextract_ladder), and the chunk texts are rebuilt
from data/raw with the same chunking settings the run used.  The two
arms compared here are the SAME responses, parsed twice: once with the
flag off (today's behaviour) and once with it on.

READ-ONLY: it opens the experiment's cache and writes nothing.

Run:  python tests/exp_stub_endpoints_measure.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ------------------------------------------------------------------ #
# Own CACHE_DIR — the ladder arm's, read-only. Never the canonical one.
# ------------------------------------------------------------------ #
CACHE_DIR = "data/cache/exp_reextract_ladder"
RECORDS = "data/cache/exp_reextract_ladder.jsonl"

os.environ["CACHE_DIR"] = CACHE_DIR
os.environ["ZVEC_ROOT"] = "data/cache"
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
os.environ["EXTRACTION_MODEL"] = "qwen2.5:1.5b"
os.environ["LLM_NUM_PREDICT"] = "4096"
os.environ["LLM_NUM_CTX"] = "8192"
os.environ["LLM_THINK"] = "off"
os.environ["EXTRACTION_STUB_ENDPOINTS"] = "0"

import diskcache  # noqa: E402

from graphrag_lite.config import get_config  # noqa: E402

get_config.cache_clear()

from graphrag_lite.cache.llm_cache import _make_key  # noqa: E402
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer  # noqa: E402
from graphrag_lite.ingestion.extractor import (  # noqa: E402
    _EXTRACTION_PROMPT_TEMPLATE,
    _extract_json,
    _name_occurs_verbatim,
    _parse_extraction,
)
from graphrag_lite.ingestion.loader import load_documents  # noqa: E402

PENALTIES = (0.25, 0.5, 1.0)
FOCUS_CHUNK = "a4a679c427829368"      # the 20-entity / 20-relationship chunk


def tag(penalty: float) -> str:
    return f"t=0.0;seed=42;np=4096;ctx=8192;fp={penalty};think=False"


def main() -> None:
    # The extractor warns once per dropped/recovered relationship; that is
    # the production behaviour but it would bury the measurement. The
    # per-relationship detail is reproduced in the focus-chunk section.
    warnings.simplefilter("ignore", UserWarning)

    cfg = get_config()
    model = cfg.effective_extraction_model()
    cache = diskcache.Cache(CACHE_DIR)

    tokenizer = get_tokenizer(cfg.TOKENIZER)
    chunks = []
    for doc in load_documents("data/raw"):
        chunks.extend(chunk_document(doc, cfg, tokenizer))
    by_id = {c["chunk_id"]: c for c in chunks}
    print(f"[chunks] rebuilt {len(chunks)} chunks from data/raw "
          f"(chunk_id is a hash of doc_id::index, so it is stable)")

    recs = {}
    with (ROOT / RECORDS).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["chunk_id"]] = r

    rows = []
    focus = None
    for cid, rec in recs.items():
        chunk = by_id.get(cid)
        if chunk is None:
            continue
        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
        raw = None
        penalty = None
        for pen in PENALTIES:
            candidate = cache.get(_make_key(prompt, model, tag(pen)), default=None)
            if candidate is None:
                continue
            try:
                _extract_json(candidate)
            except Exception:
                continue      # this attempt failed to parse; try the next rung
            raw, penalty = candidate, pen
            break
        if raw is None:
            continue
        obj = _extract_json(raw)

        off = _parse_extraction(obj, cid, rec["doc_id"],
                                chunk_text=chunk["text"],
                                allow_stub_endpoints=False)
        on = _parse_extraction(obj, cid, rec["doc_id"],
                               chunk_text=chunk["text"],
                               allow_stub_endpoints=True)

        stubs = [e for e in on["entities"] if e.get("stub")]
        # Which names were dropped as unknown AND occur in the text?
        raw_rels = obj.get("relationships") or []
        raw_ents = obj.get("entities") or []
        names = {str(e.get("name") or "").strip().lower()
                 for e in raw_ents if isinstance(e, dict)}
        unknown_present = set()
        unknown_absent = set()
        for r in raw_rels:
            if not isinstance(r, dict):
                continue
            for ep in (str(r.get("source") or "").strip(),
                       str(r.get("target") or "").strip()):
                if not ep or ep.lower() in names:
                    continue
                key = ep.lower()
                if key in unknown_present or key in unknown_absent:
                    continue
                if _name_occurs_verbatim(ep, chunk["text"]):
                    unknown_present.add(key)
                else:
                    unknown_absent.add(key)

        row = {
            "chunk_id": cid,
            "doc_id": rec["doc_id"],
            "penalty": penalty,
            "raw_ents": len(raw_ents),
            "raw_rels": len(raw_rels),
            "off_rels": len(off["relationships"]),
            "on_rels": len(on["relationships"]),
            "on_ents": len(on["entities"]),
            "stubs": len(stubs),
            "stub_names": sorted(e["name"] for e in stubs),
            "unknown_present": len(unknown_present),
            "unknown_absent": len(unknown_absent),
            "recovered": len(on["relationships"]) - len(off["relationships"]),
        }
        rows.append(row)
        if cid == FOCUS_CHUNK:
            focus = (row, obj, chunk, off, on)

    # ---------------------------------------------------------------- #
    # Totals
    # ---------------------------------------------------------------- #
    def tot(key: str) -> int:
        return sum(r[key] for r in rows)

    print()
    print("=" * 92)
    print("CONDITIONAL ENDPOINT FILTER -- measured on the ladder arm's raw responses")
    print("=" * 92)
    print(f"chunks audited                : {len(rows)}")
    print(f"  relationship objects emitted : {tot('raw_rels')}")
    print(f"  kept, flag OFF (today)       : {tot('off_rels')}")
    print(f"  kept, flag ON                : {tot('on_rels')}")
    print(f"  RECOVERED by the flag        : {tot('on_rels') - tot('off_rels')}")
    if tot("raw_rels"):
        loss_off = 100.0 * (tot("raw_rels") - tot("off_rels")) / tot("raw_rels")
        loss_on = 100.0 * (tot("raw_rels") - tot("on_rels")) / tot("raw_rels")
        print(f"  -> loss rate to the filter, flag OFF : {loss_off:.1f}%")
        print(f"  -> loss rate to the filter, flag ON  : {loss_on:.1f}%")
    print(f"  stub entities added          : {tot('stubs')}")
    print(f"  entity records, flag ON      : {tot('on_ents')}")
    print()
    print("  endpoint names that were unknown to the entity list:")
    print(f"    present verbatim in the chunk text (recoverable) : {tot('unknown_present')}")
    print(f"    ABSENT from the chunk text (dropped either way)   : {tot('unknown_absent')}")
    print()

    # ---------------------------------------------------------------- #
    # Per chunk
    # ---------------------------------------------------------------- #
    print("-" * 92)
    print(f"{'chunk_id':<18}{'doc':<32}{'fp':>5}{'rawR':>6}{'off':>6}{'on':>6}"
          f"{'gain':>6}{'stubs':>7}")
    print("-" * 92)
    for r in sorted(rows, key=lambda r: -r["recovered"]):
        if r["recovered"] or r["stubs"]:
            print(f"{r['chunk_id']:<18}{r['doc_id']:<32}{r['penalty']:>5}"
                  f"{r['raw_rels']:>6}{r['off_rels']:>6}{r['on_rels']:>6}"
                  f"{r['recovered']:>6}{r['stubs']:>7}")
    print()
    changed = [r for r in rows if r["recovered"]]
    print(f"chunks whose relationship count changed: {len(changed)} of {len(rows)}")
    print(f"chunks that gained NO relationships    : "
          f"{len(rows) - len(changed)}")
    print()

    # ---------------------------------------------------------------- #
    # The focus chunk
    # ---------------------------------------------------------------- #
    if focus is None:
        print(f"[focus] chunk {FOCUS_CHUNK} not found in the ladder records")
        return

    row, obj, chunk, off, on = focus
    print("=" * 92)
    print(f"FOCUS CHUNK {FOCUS_CHUNK}  (doc={row['doc_id']}, fp={row['penalty']})")
    print("=" * 92)
    print(f"raw entities     : {row['raw_ents']}")
    print(f"raw relationships: {row['raw_rels']}")
    print(f"kept, flag OFF   : {row['off_rels']}")
    print(f"kept, flag ON    : {row['on_rels']}")
    print(f"stubs added      : {row['stubs']}  -> {row['stub_names']}")
    print()
    print("--- THE ENTITY LIST THE MODEL EMITTED ---")
    for e in obj.get("entities") or []:
        print(f"  {e.get('name')!r}  [{e.get('type')}]")
    print()
    print("--- THE 20 RELATIONSHIPS ---")
    raw_rels = obj.get("relationships") or []
    for i, r in enumerate(raw_rels, 1):
        src = str(r.get("source") or "")
        tgt = str(r.get("target") or "")
        known_src = src.lower() in {str(e.get("name") or "").lower()
                                    for e in obj.get("entities") or []}
        known_tgt = tgt.lower() in {str(e.get("name") or "").lower()
                                    for e in obj.get("entities") or []}
        in_src = _name_occurs_verbatim(src, chunk["text"])
        in_tgt = _name_occurs_verbatim(tgt, chunk["text"])
        print(f"  [{i:2d}] {src!r} --{r.get('type')}--> {tgt!r}")
        print(f"       src known={known_src} in_text={in_src} | "
              f"tgt known={known_tgt} in_text={in_tgt}")
        print(f"       desc: {r.get('description')}")
        print(f"       sent: {r.get('source_sentence')}")
    print()
    print("--- CHUNK TEXT (for judging the relationships above) ---")
    print(chunk["text"])
    print()
    print("--- WHAT THE FLAG ON KEEPS ---")
    for r in on["relationships"]:
        print(f"  {r['source']!r} --{r['type']}--> {r['target']!r}")
    print()
    print("--- STUB ENTITIES ADDED ---")
    for e in on["entities"]:
        if e.get("stub"):
            print(f"  {e['name']!r}  type={e['type']}  stub={e['stub']}")
            print(f"     source_sentence: {e['source_sentence']}")


if __name__ == "__main__":
    main()
