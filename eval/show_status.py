"""
show_status.py — Show the current state of all MindMesh experiments.

Usage:
    python eval/show_status.py
    python eval/show_status.py --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

EXP_BASE = ROOT / "data" / "experiments"


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def show_extraction_progress() -> None:
    """Show genuine gleaning extraction progress."""
    exp = EXP_BASE / "qwen2_5_1_5b_genuine_gleaning_on"
    jsonl = exp / "extractions.jsonl"
    summary = load_json(exp / "extraction_summary.json")

    print_section("GENUINE GLEANING EXTRACTION — qwen2.5:1.5b (G1)")
    if not jsonl.exists():
        print("  Status: NOT STARTED")
        print("  Command: python eval/extract_with_gleaning.py")
        return

    lines = sum(1 for l in open(jsonl, encoding="utf-8") if l.strip())
    done  = lines
    total = 42

    print(f"  Progress:  {done}/{total} chunks  ({100*done//total}%)")
    if summary:
        ts = summary.get("timestamp", "?")
        print(f"  Last save: {ts}")
        print(f"  Parse errors:    {summary.get('parse_errors', '?')}")
        print(f"  Gleaning OK:     {summary.get('gleaning_ok', '?')}")
        print(f"  Entities:        {summary.get('total_entities', '?')} "
              f"(+{summary.get('glean_added_entities', '?')} from gleaning)")
        print(f"  Relationships:   {summary.get('total_rels', '?')} "
              f"(+{summary.get('glean_added_rels', '?')} from gleaning)")
        print(f"  LLM calls:       {summary.get('total_extract_calls', '?')} extract "
              f"+ {summary.get('total_gleaning_calls', '?')} glean")

    if done < total:
        # Estimate remaining time from log
        log_path = exp / "extraction.log"
        if log_path.exists():
            log_lines = open(log_path, encoding="utf-8").readlines()
            times = []
            for l in log_lines:
                if "total  cache:" in l:
                    try:
                        t = float(l.split("=")[0].rsplit(" ", 1)[-1].replace("s", ""))
                        times.append(t)
                    except Exception:
                        pass
            if times:
                avg = sum(times) / len(times)
                est = avg * (total - done)
                print(f"  Avg time/chunk:  {avg:.0f}s")
                print(f"  Est remaining:   {est/3600:.1f}h")
        print(f"\n  To resume: python eval/extract_with_gleaning.py")
    else:
        print(f"\n  COMPLETE. Run: python eval/run_gleaning_evaluation.py")


def show_experiment(name: str, label: str, eval_file: str = "evaluation.json") -> None:
    """Show results for a completed experiment."""
    exp = EXP_BASE / name
    ev_path = exp / eval_file
    ev = load_json(ev_path)
    agg = ev.get("aggregates", {})

    print_section(f"{label}")

    if not exp.exists():
        print("  NOT STARTED")
        return

    if not agg:
        print("  Evaluation not yet run")
        return

    PIPELINES = ["pipeline_a", "pipeline_b",
                 "lightrag_local", "lightrag_global",
                 "lightrag_hybrid", "lightrag_mix"]
    fmt = "  {:<22} {:>10} {:>8} {:>10}"
    print(fmt.format("Pipeline", "Recall@6", "MRR", "UniqueGold"))
    print("  " + "-"*53)
    for pl in PIPELINES:
        a = agg.get(pl, {})
        if a:
            print(fmt.format(pl,
                             f"{a.get('avg_recall_at_k',0):.4f}",
                             f"{a.get('mrr',0):.4f}",
                             str(a.get("unique_gold_recovered",
                                       a.get("unique_gold_chunks_recovered","?")))))


def show_gpu_tasks() -> None:
    """List GPU-deferred tasks and their status."""
    print_section("GPU-DEFERRED TASKS")
    tasks = [
        ("G1", "Genuine gleaning extraction",   "IN PROGRESS — resumable",   True),
        ("G2", "qwen3.5:9b experiment",          "NOT STARTED",               False),
        ("G3", "qwen3:4b experiment",            "NOT STARTED",               False),
        ("G4", "qwen3.5:2b experiment",          "NOT STARTED",               False),
        ("G5", "Multi-model comparison",         "BLOCKED on G1-G4",          False),
        ("G6", "LightRAG Global keyword upgrade","DEFERRED (optional)",        False),
    ]
    for tid, name, status, active in tasks:
        marker = "▶" if active else " "
        print(f"  {marker} [{tid}] {name:<40} {status}")
    print(f"\n  Full details: GPU_DEFERRED_TASKS.md")


def main():
    parser = argparse.ArgumentParser(description="MindMesh experiment status")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  MINDMESH EXPERIMENT STATUS")
    print("="*70)

    # Completed experiments
    show_experiment(
        "qwen2_5_1_5b_validate",
        "PHASE 1 — qwen2.5:1.5b gleaning OFF (baseline)",
        "validation_20261109_223804.json",
    )
    show_experiment(
        "qwen2_5_1_5b_v2_off",
        "v2-OFF — qwen2.5:1.5b stub-filter OFF",
    )
    show_experiment(
        "qwen2_5_1_5b_v2_on",
        "v2-ON  — qwen2.5:1.5b stub-filter ON",
    )

    # In-progress
    show_extraction_progress()

    # Gleaning eval (runs after extraction)
    show_experiment(
        "qwen2_5_1_5b_genuine_gleaning_on",
        "GENUINE GLEANING-ON — evaluation (runs after extraction)",
        "genuine_gleaning_evaluation.json",
    )

    # GPU tasks
    show_gpu_tasks()

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
