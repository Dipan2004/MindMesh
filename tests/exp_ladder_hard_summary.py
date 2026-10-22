"""
exp_ladder_hard_summary.py -- summarise the retry ladder's rescue rate.

Reads `data/cache/exp_ladder_hard.jsonl` (appended by
`tests/exp_ladder_hard.py`, one record per chunk per repetition) and
answers the two questions the ladder was built to answer:

  1. P(attempt 0 fails)          -- how often the FIRST attempt (fp=0.25)
                                     lands in a bad attractor
  2. P(a later attempt rescues)   -- given attempt 0 failed, how often a
                                     subsequent attempt (fp=0.5, then 1.0)
                                     produced a parse

READ-ONLY. It opens no cache and writes nothing.

Run:  python tests/exp_ladder_hard_summary.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/cache/exp_ladder_hard.jsonl"


def load():
    recs = []
    if not OUT.exists():
        return recs
    with OUT.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main() -> None:
    recs = load()
    print("=" * 88)
    print("RETRY LADDER -- RESCUE RATE ON THE HARD CHUNKS")
    print("=" * 88)
    if not recs:
        print(f"no records at {OUT.relative_to(ROOT)}")
        return

    reps = sorted({r.get("rep", 0) for r in recs})
    print(f"records: {len(recs)}   repetitions: {reps}   chunks: "
          f"{len({r['chunk_id'] for r in recs})}")
    print()

    # ---- per repetition ------------------------------------------------ #
    print("PER REPETITION")
    print("-" * 88)
    print(f"{'rep':<5}{'n':<5}{'a0 failed':<12}{'rescued':<10}{'still failed':<14}")
    tot_a0_fail = tot_rescued = tot_fail = 0
    for rep in reps:
        rr = [r for r in recs if r.get("rep", 0) == rep]
        a0_fail = [r for r in rr if r["per_attempt"][0]["mode"] != "ok"]
        resc = [r for r in rr if r.get("rescued")]
        fail = [r for r in rr if r["status"] != "ok"]
        tot_a0_fail += len(a0_fail)
        tot_rescued += len(resc)
        tot_fail += len(fail)
        print(f"{rep:<5}{len(rr):<5}{len(a0_fail):<12}{len(resc):<10}{len(fail):<14}")
    print()
    print(f"  P(attempt 0 fails)          = {tot_a0_fail}/{len(recs)}"
          f" = {100.0 * tot_a0_fail / len(recs):.1f}%")
    if tot_a0_fail:
        print(f"  P(rescued | attempt 0 failed) = {tot_rescued}/{tot_a0_fail}"
              f" = {100.0 * tot_rescued / tot_a0_fail:.1f}%")
    print(f"  P(chunk ultimately fails)   = {tot_fail}/{len(recs)}"
          f" = {100.0 * tot_fail / len(recs):.1f}%")
    print(f"  P(chunk rescued by ladder)  = {tot_rescued}/{len(recs)}"
          f" = {100.0 * tot_rescued / len(recs):.1f}%")
    print()

    # ---- which attempt did the rescuing -------------------------------- #
    resc_by_attempt = Counter()
    for r in recs:
        if r.get("rescued"):
            resc_by_attempt[r["attempts_used"] - 1] += 1
    print("WHICH ATTEMPT DID THE RESCUE  (attempt index -> number of rescues)")
    print("-" * 88)
    if not resc_by_attempt:
        print("  (no rescues)")
    for a in sorted(resc_by_attempt):
        print(f"  attempt {a} (fp={0.25 if a == 0 else [0.5, 1.0][a - 1]})"
              f" : {resc_by_attempt[a]}")
    print()

    # ---- per chunk ------------------------------------------------------ #
    print("PER CHUNK")
    print("-" * 88)
    by_chunk = defaultdict(list)
    for r in recs:
        by_chunk[r["chunk_id"]].append(r)
    for cid, rr in sorted(by_chunk.items()):
        rr = sorted(rr, key=lambda r: r.get("rep", 0))
        modes = " ".join(
            f"r{r.get('rep', 0)}:" + "->".join(a["mode"] for a in r["per_attempt"])
            for r in rr
        )
        nok = sum(1 for r in rr if r["status"] == "ok")
        nresc = sum(1 for r in rr if r.get("rescued"))
        print(f"  {cid}   ok {nok}/{len(rr)}  rescued {nresc}/{len(rr)}")
        print(f"      {modes}")
    print()

    # ---- the mode of each attempt --------------------------------------- #
    print("MODE OF EACH ATTEMPT (all records)")
    print("-" * 88)
    for a in range(4):
        modes = Counter(
            r["per_attempt"][a]["mode"]
            for r in recs if len(r["per_attempt"]) > a
        )
        if not modes:
            continue
        pen = {0: 0.25, 1: 0.5, 2: 1.0}.get(a, "?")
        total = sum(modes.values())
        print(f"  attempt {a} (fp={pen}): " +
              "  ".join(f"{k}={v}/{total}" for k, v in sorted(modes.items())))
    print()
    print("=" * 88)


if __name__ == "__main__":
    main()
