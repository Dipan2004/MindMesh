"""
compare_models.py — Side-by-side comparison of all completed model experiments.

Reads evaluation.json from each experiment directory and prints a combined
table. Skips experiments that haven't run yet.

Usage:
    python eval/compare_models.py
    python eval/compare_models.py --out eval/results/model_comparison.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

EXP_BASE = ROOT / "data" / "experiments"

# All experiment arms, in order
EXPERIMENTS = [
    # (directory, label, eval_filename)
    ("qwen2_5_1_5b_validate",             "1.5b gleaning-OFF",    "validation_20261109_223804.json"),
    ("qwen2_5_1_5b_v2_off",               "1.5b stub-OFF",        "evaluation.json"),
    ("qwen2_5_1_5b_v2_on",                "1.5b stub-ON",         "evaluation.json"),
    ("qwen2_5_1_5b_genuine_gleaning_on",  "1.5b glean-ON",        "genuine_gleaning_evaluation.json"),
    ("qwen3_5_2b",                        "3.5b:2b",              "evaluation.json"),
    ("qwen3_4b",                          "3:4b",                 "evaluation.json"),
    ("qwen3_5_9b",                        "3.5b:9b",              "evaluation.json"),
]

PIPELINES = [
    "pipeline_a", "pipeline_b",
    "lightrag_local", "lightrag_global",
    "lightrag_hybrid", "lightrag_mix",
]


def load_agg(exp_dir: Path, eval_file: str) -> dict | None:
    p = exp_dir / eval_file
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("aggregates") or d.get("results", {})


def fmt(v, key="avg_recall_at_k") -> str:
    if v is None:
        return "—"
    if isinstance(v, dict):
        x = v.get(key)
        return f"{x:.4f}" if x is not None else "—"
    return "—"


def main():
    parser = argparse.ArgumentParser(description="Multi-model comparison table")
    parser.add_argument("--out", default=None)
    parser.add_argument("--metric", default="avg_recall_at_k",
                        choices=["avg_recall_at_k", "mrr", "unique_gold_recovered", "avg_latency_s"])
    args = parser.parse_args()

    # Load all available results
    results: list[tuple[str, dict]] = []
    for exp_name, label, eval_file in EXPERIMENTS:
        agg = load_agg(EXP_BASE / exp_name, eval_file)
        if agg:
            results.append((label, agg))

    if not results:
        print("No completed experiments found.")
        return 1

    print(f"\n{'='*120}")
    print(f"MODEL COMPARISON — metric: {args.metric}")
    print(f"{'='*120}")

    # Header
    col_w = 12
    header = f"{'Pipeline':<24}" + "".join(f"{lbl:>{col_w}}" for lbl, _ in results)
    print(header)
    print("-" * (24 + col_w * len(results)))

    for pl in PIPELINES:
        row = f"{pl:<24}"
        vals = []
        for lbl, agg in results:
            a = agg.get(pl, {})
            v = a.get(args.metric)
            if v is not None:
                vals.append(v)
                row += f"{v:>{col_w}.4f}"
            else:
                vals.append(None)
                row += f"{'—':>{col_w}}"
        print(row)

    print("=" * (24 + col_w * len(results)))

    # Save if requested
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metric": args.metric,
            "experiments": [lbl for lbl, _ in results],
            "pipelines": {
                pl: {lbl: agg.get(pl, {}).get(args.metric) for lbl, agg in results}
                for pl in PIPELINES
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
