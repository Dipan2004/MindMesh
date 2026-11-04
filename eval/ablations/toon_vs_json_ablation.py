"""
toon_vs_json_ablation.py — Task 8.5: TOON vs JSON serialization ablation.

Re-runs the Task 8.2 harness with SERIALIZATION_FORMAT=toon vs json,
keeping all other variables (model, pipeline, ranking) identical.

The comparison isolates the serialization variable only, as required by
TASKS.md Task 8.5.  Both arms use the same retrieved evidence chunks —
the only difference is the text representation fed to the generator.

Usage:
    python eval/ablations/toon_vs_json_ablation.py

DoD: paste the 2-way comparison table (tokens, latency, accuracy).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "eval"))

from run_accuracy_eval import main as run_eval, _aggregate, RESULTS_DIR


def run_ablation() -> None:
    print("=" * 80)
    print("TOON vs JSON SERIALIZATION ABLATION (Task 8.5)")
    print("=" * 80)
    print("Comparing Pipeline B with TOON vs JSON evidence serialization.")
    print("All other variables (model, ranking, pipeline) held constant.\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_toon = str(RESULTS_DIR / f"toon_ablation_{ts}.json")
    out_json = str(RESULTS_DIR / f"json_ablation_{ts}.json")

    print("--- Running TOON arm ---")
    run_eval(["--pipeline", "b", "--format", "toon", "--out", out_toon])

    print("\n--- Running JSON arm ---")
    run_eval(["--pipeline", "b", "--format", "json", "--out", out_json])

    # ── Comparison table ──────────────────────────────────────────────────────
    with open(out_toon) as f:
        toon_data = json.load(f)
    with open(out_json) as f:
        json_data = json.load(f)

    agg_toon = toon_data["aggregates"].get("b", {})
    agg_json = json_data["aggregates"].get("b", {})

    print()
    print("=" * 70)
    print("TOON vs JSON COMPARISON TABLE (Pipeline B, dev set)")
    print("=" * 70)
    fmt = "{:<28} {:>10} {:>10} {:>10}"
    print(fmt.format("Metric", "TOON", "JSON", "Delta"))
    print("-" * 62)

    metrics = [
        ("Avg evidence tokens", "avg_evidence_tokens"),
        ("Avg answer tokens", "avg_answer_tokens"),
        ("Avg retrieval latency (s)", "avg_retrieval_latency_s"),
        ("Avg generation latency (s)", "avg_generation_latency_s"),
        ("Avg total latency (s)", "avg_total_latency_s"),
        ("Recall@K", "avg_recall_at_k"),
        ("MRR", "mrr"),
        ("% Grounded", "pct_grounded"),
        ("Term coverage (avg)", "avg_term_coverage"),
        ("% No-answer correct", "pct_no_answer_correct"),
    ]
    for label, key in metrics:
        t = agg_toon.get(key, float("nan"))
        j = agg_json.get(key, float("nan"))
        try:
            delta = t - j
            delta_s = f"{delta:+.4f}"
            t_s = f"{t:.4f}"
            j_s = f"{j:.4f}"
        except Exception:
            t_s = str(t)
            j_s = str(j)
            delta_s = "—"
        print(fmt.format(label, t_s, j_s, delta_s))

    print("=" * 70)
    print("\nToken savings (TOON - JSON, negative = TOON uses fewer):")
    t_tok = agg_toon.get("avg_evidence_tokens", 0)
    j_tok = agg_json.get("avg_evidence_tokens", 0)
    if j_tok > 0:
        pct = 100.0 * (j_tok - t_tok) / j_tok
        print(f"  Evidence tokens: TOON={t_tok:.1f}  JSON={j_tok:.1f}  "
              f"savings={pct:.1f}%")

    ablation_path = RESULTS_DIR / f"toon_vs_json_ablation_{ts}.json"
    with open(ablation_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task": "8.5",
            "toon": toon_data,
            "json": json_data,
            "comparison": {
                "toon_aggregates": agg_toon,
                "json_aggregates": agg_json,
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\nAblation results saved to {ablation_path}")


if __name__ == "__main__":
    run_ablation()
