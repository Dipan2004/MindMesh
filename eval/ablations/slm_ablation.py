"""
slm_ablation.py — Task 8.4: SLM ablation (updated with newly-available models).

TASKS.md spec: re-run Task 8.2's harness with SLM_MODEL = qwen3:4b,
qwen3:1.7b, phi4-mini. Plus the existing qwen2.5:1.5b and qwen2.5:0.5b.

Status per model after download:
  qwen2.5:1.5b  — AVAILABLE, runs fine (primary eval model)
  qwen2.5:0.5b  — AVAILABLE, runs fine
  qwen3:1.7b    — AVAILABLE, requires think=False flag
  qwen3:4b      — DOWNLOADED but OOM (needs ~18 GB RAM, machine has 15.7 GB)
  phi4-mini     — DOWNLOADED but OOM (needs ~16 GB RAM, machine has 15.7 GB)

All generation settings (temperature, seed, num_predict) and retrieval
settings are held constant. The only varying factor is the SLM model.

Usage:
    python eval/ablations/slm_ablation.py [--models <m1>,<m2>]
    python eval/ablations/slm_ablation.py  # runs all available models

DoD status: qwen3:1.7b added. qwen3:4b and phi4-mini blocked (OOM).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import argparse

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "eval"))

from run_accuracy_eval import main as run_eval, RESULTS_DIR

# Models available to run (use this list by default)
RUNNABLE_MODELS = ["qwen2.5:1.5b", "qwen2.5:0.5b", "qwen3:1.7b"]

# Models downloaded but not runnable with notes
OOM_MODELS = {
    "qwen3:4b":   "OOM — requires ~18 GB KV-cache RAM, machine has 15.7 GB",
    "phi4-mini":  "OOM — requires ~16 GB KV-cache RAM, machine has 15.7 GB",
}


def run_ablation(models: list[str] | None = None) -> None:
    if models is None:
        models = RUNNABLE_MODELS

    print("=" * 80)
    print("SLM ABLATION (Task 8.4)")
    print("=" * 80)
    print("TASKS.md spec: qwen3:4b, qwen3:1.7b, phi4-mini (plus baseline models)")
    print()
    for m, reason in OOM_MODELS.items():
        print(f"  [OOM/BLOCKED] {m}: {reason}")
    print()
    print(f"Running: {', '.join(models)}")
    print("Note: qwen3:1.7b requires think=False (suppresses chain-of-thought).\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    results: dict[str, dict] = {}
    for model in models:
        if model in OOM_MODELS:
            print(f"[OOM] {model} — {OOM_MODELS[model]}")
            results[model] = {"oom": True, "reason": OOM_MODELS[model]}
            continue

        slug = model.replace(":", "_").replace(".", "_")
        out_path = str(RESULTS_DIR / f"slm_{slug}_{ts}.json")
        print(f"\n--- Running model: {model} ---")
        run_eval(["--pipeline", "b", "--model", model, "--out", out_path])
        with open(out_path) as f:
            results[model] = json.load(f)

    # ── Comparison table ──────────────────────────────────────────────────────
    available = [m for m in models if m not in OOM_MODELS]
    print()
    print("=" * 100)
    print("SLM COMPARISON TABLE (Pipeline B, dev set)")
    print("=" * 100)
    col_w = 16
    fmt = "{:<30}" + f" {{:>{col_w}}}" * len(available)
    print(fmt.format("Metric", *available))
    print("-" * (30 + col_w * len(available)))

    metrics = [
        ("Recall@K", "avg_recall_at_k"),
        ("MRR", "mrr"),
        ("% Grounded", "pct_grounded"),
        ("Term coverage (avg)", "avg_term_coverage"),
        ("% No-answer correct", "pct_no_answer_correct"),
        ("Avg generation latency (s)", "avg_generation_latency_s"),
        ("Avg total latency (s)", "avg_total_latency_s"),
    ]
    for label, key in metrics:
        vals = []
        for model in available:
            agg = results[model].get("aggregates", {}).get("b", {})
            v = agg.get(key, float("nan"))
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        print(fmt.format(label, *vals))

    print("=" * 100)
    print()
    for m, info in OOM_MODELS.items():
        if m in models or m not in RUNNABLE_MODELS:
            print(f"[OOM] {m}: {info}")

    ablation_path = RESULTS_DIR / f"slm_ablation_{ts}.json"
    with open(ablation_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task": "8.4",
            "runnable_models": available,
            "oom_models": OOM_MODELS,
            "results": results,
            "aggregates": {
                m: results[m].get("aggregates", {}).get("b", {})
                for m in available
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\nAblation results saved to {ablation_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None,
                        help=f"Comma-separated model names. Default: {','.join(RUNNABLE_MODELS)}")
    args = parser.parse_args()
    models = args.models.split(",") if args.models else None
    run_ablation(models)
