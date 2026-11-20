"""
eval/hotpotqa/classify_extraction_diffs.py -- split extraction disagreements into renamed vs truly missing.

An item the reference has but the variant lacks under exact normalised-name match is
  * "renamed/altered" if a variant item's name contains, or is contained in, the reference
    name (e.g. 'rudebox (song)' vs 'rudebox'), and
  * "missing" otherwise.
Relationships are matched the same way on both endpoints. Writes diff_classification.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graphrag_lite.graph.builder import _norm  # noqa: E402

VD = ROOT / "data/benchmarks/hotpotqa/expanded_20261119/extraction_validation"
QD = VD.parent / "extraction_validation_qwen2.5_3b"


def loose(a: str, b: str) -> bool:
    a, b = a.strip("() "), b.strip("() ")
    return bool(a) and bool(b) and (a in b or b in a)


def classify(ref: dict, var: dict) -> dict:
    re_ = {_norm(e["name"]) for e in ref["entities"]}; ve = {_norm(e["name"]) for e in var["entities"]}
    rr = {(_norm(r["source"]), _norm(r["target"])) for r in ref["relationships"]}
    vr = {(_norm(r["source"]), _norm(r["target"])) for r in var["relationships"]}
    e_exact = re_ & ve
    e_ren = {x for x in re_ - ve if any(loose(x, y) for y in ve - re_)}
    r_exact = rr & vr
    r_ren = {(s, t) for (s, t) in rr - vr
             if any((loose(s, s2) and loose(t, t2)) or (loose(s, t2) and loose(t, s2)) for (s2, t2) in vr - rr)}
    return {"ent_ref": len(re_), "ent_exact": len(e_exact), "ent_renamed": len(e_ren),
            "ent_missing": len(re_) - len(e_exact) - len(e_ren),
            "rel_ref": len(rr), "rel_exact": len(r_exact), "rel_renamed_or_reversed": len(r_ren),
            "rel_missing": len(rr) - len(r_exact) - len(r_ren),
            "missing_rel_examples": sorted(f"{s} -> {t}" for (s, t) in (rr - vr) - r_ren)[:4]}


def main() -> int:
    rows = [json.loads(l) for l in (VD / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {(r["arm"], r["chunk_id"]): r for r in rows}
    cids = list(dict.fromkeys(r["chunk_id"] for r in rows))
    variants = {"mistral C": {c: by[("C", c)] for c in cids}, "mistral BC": {c: by[("BC", c)] for c in cids}}
    if (QD / "results.jsonl").exists():
        variants["qwen2.5:3b BC"] = {json.loads(l)["chunk_id"]: json.loads(l)
                                     for l in (QD / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    out = {"reference": "mistral baseline (current prompt, unconstrained)",
           "rule": "renamed = substring containment of normalised names; anything else not exact = missing",
           "variants": {}}
    for name, var in variants.items():
        per = [dict(classify(by[("baseline", c)], var[c]), chunk_id=c) for c in cids if c in var]
        tot = {k: sum(p[k] for p in per) for k in per[0] if isinstance(per[0][k], int)}
        tot["entity_recall_exact"] = round(tot["ent_exact"] / tot["ent_ref"], 4)
        tot["entity_recall_incl_renamed"] = round((tot["ent_exact"] + tot["ent_renamed"]) / tot["ent_ref"], 4)
        tot["relationship_recall_exact"] = round(tot["rel_exact"] / tot["rel_ref"], 4)
        tot["relationship_recall_incl_renamed"] = round((tot["rel_exact"] + tot["rel_renamed_or_reversed"]) / tot["rel_ref"], 4)
        tot["paragraphs_with_truly_missing_relationships"] = sum(p["rel_missing"] > 0 for p in per)
        tot["paragraphs_missing_over_half_relationships_after_renames"] = sum(
            p["rel_ref"] and p["rel_missing"] / p["rel_ref"] > 0.5 for p in per)
        out["variants"][name] = {"totals": tot, "per_paragraph": per}
        print(name, json.dumps(tot))
    (VD / "diff_classification.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
