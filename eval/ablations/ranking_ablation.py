"""
ranking_ablation.py — Task 8.6: PageRank vs degree ranking ablation.

Re-runs the Task 8.2 harness with RANKING_METHOD=pagerank vs degree,
keeping all other variables identical.

Usage:
    python eval/ablations/ranking_ablation.py

DoD: paste the 2-way comparison table.
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

from run_accuracy_eval import main as run_eval, RESULTS_DIR


def run_ablation() -> None:
    print("=" * 80)
    print("RANKING METHOD ABLATION: PageRank vs Degree (Task 8.6)")
    print("=" * 80)
    print("Comparing Pipeline B with RANKING_METHOD=pagerank vs degree.")
    print("All other variables held constant.\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_pr = str(RESULTS_DIR / f"ranking_pagerank_{ts}.json")
    out_deg = str(RESULTS_DIR / f"ranking_degree_{ts}.json")

    print("--- Running PageRank arm ---")
    run_eval(["--pipeline", "b", "--ranking", "pagerank", "--out", out_pr])

    print("\n--- Running Degree arm ---")
    run_eval(["--pipeline", "b", "--ranking", "degree", "--out", out_deg])

    # ── Comparison table ──────────────────────────────────────────────────────
    with open(out_pr) as f:
        pr_data = json.load(f)
    with open(out_deg) as f:
        deg_data = json.load(f)

    agg_pr = pr_data["aggregates"].get("b", {})
    agg_deg = deg_data["aggregates"].get("b", {})

    print()
    print("=" * 70)
    print("PAGERANK vs DEGREE COMPARISON TABLE (Pipeline B, dev set)")
    print("=" * 70)
    fmt = "{:<28} {:>10} {:>10} {:>10}"
    print(fmt.format("Metric", "PageRank", "Degree", "Delta"))
    print("-" * 62)

    metrics = [
        ("Recall@K", "avg_recall_at_k"),
        ("MRR", "mrr"),
        ("% Grounded", "pct_grounded"),
        ("Term coverage (avg)", "avg_term_coverage"),
        ("% No-answer correct", "pct_no_answer_correct"),
        ("Avg retrieval latency (s)", "avg_retrieval_latency_s"),
        ("Avg generation latency (s)", "avg_generation_latency_s"),
        ("Avg total latency (s)", "avg_total_latency_s"),
    ]
    for label, key in metrics:
        p = agg_pr.get(key, float("nan"))
        d = agg_deg.get(key, float("nan"))
        try:
            delta = p - d
            print(fmt.format(label, f"{p:.4f}", f"{d:.4f}", f"{delta:+.4f}"))
        except Exception:
            print(fmt.format(label, str(p), str(d), "—"))

    print("=" * 70)

    ablation_path = RESULTS_DIR / f"ranking_ablation_{ts}.json"
    with open(ablation_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task": "8.6",
            "pagerank": pr_data,
            "degree": deg_data,
            "comparison": {
                "pagerank_aggregates": agg_pr,
                "degree_aggregates": agg_deg,
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\nAblation results saved to {ablation_path}")


if __name__ == "__main__":
    run_ablation()
