"""
eval_judge_sheet.py -- score eval/judge_sheet.csv once the human has
labelled the 20 rows marked fill_first=y.

Reports:
  * agent-vs-human agreement (exact label match) with a 95% interval
  * the HUMAN noise rate (share of labelled rows the human calls NOISE)
    with a 95% Wilson interval
  * the agent's noise rate on the same 20 rows, for comparison
  * a per-stratum breakdown (A kept / B recovered / C rescued-chunk)

The 20 labelled rows are a stratified random subsample of the 40-row
sheet (random.Random(20261022), 7/7/6 over A/B/C), so the numbers below
estimate the properties of the SHEET, not of the whole ladder arm.  The
arm-level figures are the ones in DOCUMENT.md.

Label vocabulary (same as DOCUMENT.md "Item 4 -- Relationship quality"):
    REAL   true and supported by the chunk text
    WEAK   true in substance, but the cited sentence does not support the
           edge as stated
    NOISE  unsupported, inverted, or hallucinated

Usage:
    python tests/eval_judge_sheet.py            # score whatever is labelled
    python tests/eval_judge_sheet.py --all      # also score the unlabelled
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "eval" / "judge_sheet.csv"
LABELS = ("REAL", "WEAK", "NOISE")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="score every row, including unlabelled ones")
    args = ap.parse_args()

    if not SHEET.exists():
        print(f"[judge] sheet not found: {SHEET}")
        return 1

    with SHEET.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    labelled = [r for r in rows
                if args.all or (r.get("human_label") or "").strip()]
    n = len(labelled)
    print("=" * 78)
    print("JUDGE SHEET -- agent vs human")
    print("=" * 78)
    print(f"rows in sheet        : {len(rows)}")
    print(f"rows scored          : {n}")
    if n == 0:
        print("\nNo human labels yet.  Fill the 20 rows with fill_first=y in")
        print("eval/judge_sheet.csv (human_label = REAL | WEAK | NOISE), then")
        print("re-run this script.")
        return 0

    human = Counter(r["human_label"].strip().upper() for r in labelled)
    agent = Counter(r["agent_label"].strip().upper() for r in labelled)
    agree = sum(1 for r in labelled
                if r["agent_label"].strip().upper()
                == r["human_label"].strip().upper())
    lo, hi = wilson(agree, n)

    print()
    print("--- label distribution (scored rows) ---")
    print(f"{'label':<8}{'agent':>8}{'human':>8}")
    for lab in LABELS:
        print(f"{lab:<8}{agent.get(lab, 0):>8}{human.get(lab, 0):>8}")
    other = {k: v for k, v in human.items() if k not in LABELS}
    if other:
        print(f"{'other':<8}{sum(agent.get(k, 0) for k in other):>8}{sum(other.values()):>8}"
              f"   {other}")

    print()
    print("--- agreement (exact label match) ---")
    print(f"  agree {agree}/{n} = {pct(agree / n)}   95% CI [{pct(lo)}, {pct(hi)}]")
    print("  confusion (agent -> human):")
    for a in LABELS:
        line = [sum(1 for r in labelled
                    if r["agent_label"].strip().upper() == a
                    and r["human_label"].strip().upper() == h)
                for h in LABELS]
        print(f"    agent {a:<6} -> " + "  ".join(f"{h}:{v}" for h, v in zip(LABELS, line)))

    # ---- noise rates ---------------------------------------------------- #
    def noise_rate(rows_: list[dict], key: str) -> tuple[int, int]:
        k = sum(1 for r in rows_ if r[key].strip().upper() == "NOISE")
        return k, len(rows_)

    hk, hn = noise_rate(labelled, "human_label")
    ak, an = noise_rate(labelled, "agent_label")
    hlo, hhi = wilson(hk, hn)
    alo, ahi = wilson(ak, an)

    print()
    print("--- noise rate (NOISE / scored rows) ---")
    print(f"  human : {hk}/{hn} = {pct(hk / hn)}   95% CI [{pct(hlo)}, {pct(hhi)}]")
    print(f"  agent : {ak}/{an} = {pct(ak / an)}   95% CI [{pct(alo)}, {pct(ahi)}]")
    # Sensitivity: WEAK counted as noise / counted as clean.
    hw = sum(1 for r in labelled if r["human_label"].strip().upper() in ("NOISE", "WEAK"))
    hwn, hwn2 = hw, hn
    wlo, whi = wilson(hwn, hwn2)
    hr = sum(1 for r in labelled if r["human_label"].strip().upper() == "REAL")
    rlo, rhi = wilson(hr, hn)
    print(f"  human, WEAK counted as noise : {pct(hw / hn)}  95% CI [{pct(wlo)}, {pct(whi)}]")
    print(f"  human, only REAL counted clean: {pct(hr / hn)}  95% CI [{pct(rlo)}, {pct(rhi)}]")

    # ---- per stratum ----------------------------------------------------- #
    print()
    print("--- per stratum (scored rows) ---")
    for st in ("A", "B", "C"):
        sub = [r for r in labelled if r["stratum"] == st]
        if not sub:
            continue
        sk = sum(1 for r in sub if r["human_label"].strip().upper() == "NOISE")
        sa = sum(1 for r in sub if r["agent_label"].strip().upper() == r["human_label"].strip().upper())
        names = {"A": "kept normally", "B": "recovered (step 2)",
                 "C": "rescued chunks"}
        print(f"  {st} ({names[st]}): n={len(sub)}  "
              f"human noise {sk}/{len(sub)} = {pct(sk / len(sub))}  "
              f"agreement {sa}/{len(sub)} = {pct(sa / len(sub))}")

    # ---- the rows the human and the agent disagree on --------------------- #
    print()
    print("--- disagreements ---")
    for r in labelled:
        a = r["agent_label"].strip().upper()
        h = r["human_label"].strip().upper()
        if a != h:
            print(f"  [{r['stratum']}] {r['chunk_id'][:8]} "
                  f"{r['source']} --{r['type']}--> {r['target']}")
            print(f"        agent={a}  human={h}")
            print(f"        agent note: {r['agent_note']}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
