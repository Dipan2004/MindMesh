"""
exp_compare_arms.py -- compare the three corpus arms side by side.

Three arms, three questions:

  A. fp=0.25, no retry   (data/cache/exp_reextract.jsonl)
     "does the frequency penalty on its own carry the corpus?"
  B. fp=0,    no retry   (data/cache/exp_reextract_fp0.jsonl)
     "what does the corpus look like with no penalty at all?"
  C. fp=0.25 + ladder    (data/cache/exp_reextract_ladder.jsonl)
     "what does the PRODUCTION path (with retry-with-escalation) deliver?"

This script is READ-ONLY. It opens no cache, writes no cache, and touches
no Zvec collection. It only reads the three JSONL files the experiments
wrote and prints a comparison.

Run:  python tests/exp_compare_arms.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARMS = {
    "A fp=0.25 no-retry": "data/cache/exp_reextract.jsonl",
    "B fp=0    no-retry": "data/cache/exp_reextract_fp0.jsonl",
    "C fp=0.25 + ladder": "data/cache/exp_reextract_ladder.jsonl",
}


def load(path: str):
    p = ROOT / path
    if not p.exists():
        return None
    out = {}
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["chunk_id"]] = r
    return out


def arm_stats(recs: dict) -> dict:
    """
    Normalise both record shapes to the same keys.

    IMPORTANT: the counts must be on the SAME basis for the arms to be
    comparable. Arms A and B counted the RAW JSON the model emitted
    (``len(parsed["relationships"])``). Arm C counted the PARSED, FILTERED
    result (``len(result["relationships"])``), which the extractor's
    endpoint-consistency filter has already shrunk. Comparing a raw count
    against a filtered one manufactures a relationship "drop" that never
    happened.

    So: use ``raw_relationships`` when the record has it (arm C after
    ``tests/exp_backfill_raw_counts.py``), and fall back to
    ``relationships`` (arms A and B, whose counts ARE raw). The filtered
    count is reported separately as ``kept_rels``.
    """
    if not recs:
        return {}
    ok = 0
    fail = 0
    ents = 0
    rels = 0
    kept_rels = 0
    secs = 0.0
    for r in recs.values():
        if "parse_status" in r:               # arms A and B
            good = r["parse_status"] == "ok"
            e, rl = r.get("entities") or 0, r.get("relationships") or 0
        else:                                  # arm C (ladder)
            good = r["status"] == "ok"
            e = r.get("raw_entities")
            if e is None:
                e = r.get("entities") or 0
            rl = r.get("raw_relationships")
            if rl is None:
                rl = r.get("relationships") or 0
        ok += int(good)
        fail += int(not good)
        ents += max(e, 0)
        rels += max(rl, 0)
        kept_rels += max(r.get("relationships") or 0, 0)
        secs += float(r.get("seconds") or 0.0)
    return {
        "n": len(recs),
        "ok": ok,
        "fail": fail,
        "ents": ents,
        "rels": rels,
        "kept_rels": kept_rels,
        "secs": round(secs, 1),
    }


def main() -> None:
    print("=" * 92)
    print("CROSS-ARM COMPARISON")
    print("=" * 92)

    arms = {}
    for label, path in ARMS.items():
        recs = load(path)
        arms[label] = recs
        s = arm_stats(recs) if recs else {}
        if not s:
            print(f"{label:<22} : (no file at {path})")
            continue
        rate = 100.0 * s["ok"] / s["n"] if s["n"] else 0.0
        print(f"{label:<22} : {s['ok']:>2}/{s['n']:<2} parsed ({rate:5.1f}%)  "
              f"ent={s['ents']:<4} rel={s['rels']:<4} sum_secs={s['secs']}")
    print("   (ent/rel are RAW counts -- what the model emitted -- so the arms "
          "are comparable)")
    print()

    # ---- arm C's filter loss ------------------------------------------- #
    c = arms.get("C fp=0.25 + ladder")
    if c:
        s = arm_stats(c)
        if s["rels"] and s["kept_rels"] < s["rels"]:
            print(f"ARM C: the extractor's endpoint-consistency filter drops "
                  f"{s['rels'] - s['kept_rels']} of {s['rels']} relationships "
                  f"({100.0 * (s['rels'] - s['kept_rels']) / s['rels']:.1f}%)")
            print(f"        raw {s['rels']}  ->  kept {s['kept_rels']}   "
                  f"(entities are not filtered: {s['ents']} kept)")
            print()

    # ---- per-chunk matrix ------------------------------------------- #
    ids = sorted({cid for recs in arms.values() if recs for cid in recs})
    labels = list(ARMS)
    present = [lb for lb in labels if arms[lb]]

    if len(present) < 2:
        print("(need at least two arms to compare)")
        return

    print("PER-CHUNK MATRIX   . = ok   L = loop/trunc   J = malformed JSON   - = absent")
    print("-" * 92)
    header = f"{'chunk_id':<18}" + "".join(f"{lb.split()[0]:>5}" for lb in present)
    print(header)
    for cid in ids:
        row = f"{cid:<18}"
        for lb in present:
            r = arms[lb].get(cid)
            if r is None:
                row += f"{'-':>5}"
                continue
            if "parse_status" in r:
                good = r["parse_status"] == "ok"
                if good:
                    row += f"{'.':>5}"
                else:
                    # A repetition loop is loud (repeated line / done=length);
                    # anything else is the quiet malformed-JSON mode.
                    looped = (r.get("most_repeated_line_count") or 0) > 5 or \
                             r.get("done_reason") == "length"
                    row += f"{('L' if looped else 'J'):>5}"
            else:
                good = r["status"] == "ok"
                row += ("." if good else "X").rjust(5)
        print(row)
    print()

    # ---- the chunks that matter: where the arms disagree ------------- #
    print("DISAGREEMENTS (arm A vs arm B, i.e. does the penalty change the outcome?)")
    print("-" * 92)
    a, b = arms.get("A fp=0.25 no-retry"), arms.get("B fp=0    no-retry")
    if a and b:
        # Only chunks present in BOTH arms can be compared. A chunk that
        # arm B has not reached yet is "not yet run", not "failed" --
        # conflating the two would inflate the penalty's apparent effect.
        common = sorted(set(a) & set(b))
        only_a, only_b, both_fail = [], [], []
        for cid in common:
            oka = a[cid].get("parse_status") == "ok"
            okb = b[cid].get("parse_status") == "ok"
            if oka and not okb:
                only_a.append(cid)
            elif okb and not oka:
                only_b.append(cid)
            elif not oka and not okb:
                both_fail.append(cid)
        pending = sorted(set(a) - set(b))
        print(f"  chunks compared (present in both arms) : {len(common)}")
        if pending:
            print(f"  not yet run in arm B                  : {len(pending)}")
        print(f"  parsed under fp=0.25 but NOT under fp=0 : {len(only_a)}  {only_a}")
        print(f"  parsed under fp=0    but NOT under fp=0.25: {len(only_b)}  {only_b}")
        print(f"  failed under BOTH                      : {len(both_fail)}  {both_fail}")
        print()

    # ---- ladder rescue ----------------------------------------------- #
    c = arms.get("C fp=0.25 + ladder")
    if c and len(c) < 42:
        # Arm C is still running. Comparing against a partial arm would
        # report every not-yet-run chunk as "rescued", which is nonsense.
        print(f"LADDER")
        print("-" * 92)
        print(f"  arm C is INCOMPLETE ({len(c)}/42 chunks) -- ladder comparison "
              f"skipped until it finishes")
        print()
    elif c:
        retried = [r for r in c.values() if (r.get("retries") or 0) > 0]
        failed = [r for r in c.values() if r["status"] != "ok"]
        print("LADDER")
        print("-" * 92)
        print(f"  chunks that retried        : {len(retried)}  "
              f"{[r['chunk_id'] for r in retried]}")
        print(f"  chunks that still failed   : {len(failed)}  "
              f"{[r['chunk_id'] for r in failed]}")
        if a:
            a_fail = {cid for cid, r in a.items() if r.get("parse_status") != "ok"}
            c_fail = {cid for cid, r in c.items() if r["status"] != "ok"}
            rescued = sorted(a_fail - c_fail)
            still = sorted(a_fail & c_fail)
            print(f"  arm A failures             : {len(a_fail)}  {sorted(a_fail)}")
            print(f"  rescued by the ladder      : {len(rescued)}  {rescued}")
            print(f"  still failing              : {len(still)}  {still}")
            if c_fail - a_fail:
                print(f"  NEW failures (regressions) : {sorted(c_fail - a_fail)}")
        print()

    print("=" * 92)


if __name__ == "__main__":
    main()
