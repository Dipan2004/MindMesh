"""
eval/run_lightrag_pipeline.py — Model-selectable experimental retrieval runner for MindMesh.

USAGE:
  python eval/run_lightrag_pipeline.py <model> [options]

EXAMPLES:
  python eval/run_lightrag_pipeline.py qwen2.5:1.5b --limit 2
  python eval/run_lightrag_pipeline.py qwen2.5:3b --pipeline lightrag_hybrid
  python eval/run_lightrag_pipeline.py qwen3.5:4b --dry-run
  python eval/run_lightrag_pipeline.py qwen2.5:7b  # (refuses if not installed, no silent fallback)

RULES:
- Positional model argument is required.
- Strict local Ollama model availability check: never pulls or falls back silently.
- Explicit propagation of model to retrieval, extraction, and answer generation.
- Uses development questions only (eval/gold_dev_set.json). Never touches frozen test questions.
- Separate experiment output directories with full config, evidence, answers, timings, and metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

log = logging.getLogger("run_lightrag_pipeline")


def get_installed_ollama_models() -> list[str]:
    """Return list of locally installed Ollama model tags."""
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return []
        lines = r.stdout.strip().splitlines()
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception as exc:
        log.warning("Could not query Ollama CLI: %s", exc)
        return []


def validate_local_model(model_tag: str) -> None:
    """
    Ensure the requested model is strictly present locally in Ollama.
    Never download or fallback silently.
    """
    installed = get_installed_ollama_models()
    # Case-insensitive or tag-matching check
    installed_lower = {m.lower(): m for m in installed}
    tag_clean = model_tag.strip().lower()

    # Exact or without :latest
    matched = None
    if tag_clean in installed_lower:
        matched = installed_lower[tag_clean]
    elif f"{tag_clean}:latest" in installed_lower:
        matched = installed_lower[f"{tag_clean}:latest"]
    else:
        # Check if model prefix matches
        for m_low, m_orig in installed_lower.items():
            if m_low == tag_clean or m_low.startswith(f"{tag_clean}:"):
                matched = m_orig
                break

    if not matched:
        print(
            f"\n[STRICT CHECK FAILED] Model '{model_tag}' is NOT installed locally in Ollama.\n"
            f"Locally available models:\n  " + "\n  ".join(installed or ["(none)"]) + "\n"
            f"Execution refused without silent model fallback.\n",
            file=sys.stderr,
        )
        sys.exit(1)


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", text).strip("_")


def load_dev_questions(limit: Optional[int] = None) -> list[dict]:
    """
    Load questions exclusively from the development set.
    Under no circumstances is the frozen test set opened.
    """
    dev_path = ROOT / "eval" / "gold_dev_set.json"
    if not dev_path.exists():
        print(f"ERROR: Development set not found at {dev_path}", file=sys.stderr)
        sys.exit(1)

    with open(dev_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data.get("queries", [])
    if limit is not None and limit > 0:
        queries = queries[:limit]
    return queries


def evaluate_retrieval(results: list[dict], queries: list[dict]) -> dict[str, Any]:
    """Calculate Recall@K and MRR for retrieval results against gold dev set."""
    recall_vals = []
    mrr_vals = []

    for q in queries:
        qid = q["query_id"]
        gold = set(q.get("gold_chunk_ids", []))
        res = next((r for r in results if r["query_id"] == qid), None)
        if not res:
            recall_vals.append(0.0)
            mrr_vals.append(0.0)
            continue

        retrieved = res.get("chunk_ids", [])
        if not gold:
            recall_vals.append(1.0)
            mrr_vals.append(1.0)
            continue

        hits = sum(1 for cid in gold if cid in retrieved)
        recall = hits / len(gold)
        recall_vals.append(recall)

        mrr = 0.0
        for rank, cid in enumerate(retrieved, start=1):
            if cid in gold:
                mrr = 1.0 / rank
                break
        mrr_vals.append(mrr)

    n = len(recall_vals)
    return {
        "n_queries": n,
        "avg_recall_at_k": round(sum(recall_vals) / n, 4) if n else 0.0,
        "mrr": round(sum(mrr_vals) / n, 4) if n else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="MindMesh Model-Selectable Experimental Retrieval Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model",
        type=str,
        help="Local Ollama model tag (e.g. qwen2.5:1.5b, qwen2.5:3b, qwen3.5:4b).",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="lightrag_hybrid",
        choices=[
            "lightrag_hybrid",
            "lightrag_local",
            "lightrag_global",
            "lightrag_mix",
            "lightrag_hybrid_opt",
            "lightrag_mix_opt",
            "pipeline_a",
            "pipeline_b",
        ],
        help="Retrieval pipeline to evaluate (default: lightrag_hybrid).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        help="Query mode (default: hybrid).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of chunks/entities to retrieve (default: 6).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of dev questions for smoke testing (e.g. --limit 2).",
    )
    parser.add_argument(
        "--generate-answer",
        action="store_true",
        default=True,
        help="Generate grounded answer for each query using the selected model (default: True).",
    )
    parser.add_argument(
        "--no-generate-answer",
        action="store_false",
        dest="generate_answer",
        help="Skip answer generation (retrieval-only evaluation).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate model and configuration without running inference.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing experiment directory to resume unfinished queries.",
    )

    args = parser.parse_args()

    # 1. Strict model check
    validate_local_model(args.model)
    print(f"[OK] Model '{args.model}' verified available in local Ollama.")

    # 2. Setup experiment output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_slug = slugify(args.model)
    pipe_slug = slugify(args.pipeline)

    if args.resume:
        exp_dir = Path(args.resume)
        if not exp_dir.exists():
            print(f"ERROR: Resume directory '{exp_dir}' does not exist.", file=sys.stderr)
            sys.exit(1)
    else:
        exp_dir = ROOT / "data" / "experiments" / f"lightrag_{model_slug}_{pipe_slug}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)

    # 3. Configure logging
    log_file = exp_dir / "logs" / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    log.info("Starting MindMesh experimental run")
    log.info("Model: %s | Pipeline: %s | Mode: %s | Top-K: %d", args.model, args.pipeline, args.mode, args.top_k)
    log.info("Output directory: %s", exp_dir)

    # 4. Save initial config and metadata
    run_config = {
        "model": args.model,
        "extraction_model": args.model,
        "gleaning_model": args.model,
        "keyword_model": args.model,
        "answer_model": args.model if args.generate_answer else None,
        "pipeline": args.pipeline,
        "mode": args.mode,
        "top_k": args.top_k,
        "generate_answer": args.generate_answer,
        "timestamp": timestamp,
        "dataset": "eval/gold_dev_set.json",
        "dry_run": args.dry_run,
    }
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    if args.dry_run:
        print("\n=== DRY RUN VALIDATION SUCCESSFUL ===")
        print(f"Model: {args.model}")
        print(f"Pipeline: {args.pipeline}")
        print(f"Config written to: {exp_dir / 'config.json'}")
        log.info("Dry run complete. No inference executed.")
        return 0

    # 5. Load development queries only
    queries = load_dev_questions(limit=args.limit)
    log.info("Loaded %d questions from development set", len(queries))

    # 6. Initialize pipeline adapter
    from mindmesh.pipelines.registry import build_adapter
    adapter = build_adapter(
        pipeline_name=args.pipeline,
        model=args.model,
        exp_dir=exp_dir,
        top_k=args.top_k,
    )

    # 7. Execute queries with persistent caching/resumability
    results_path = exp_dir / "retrieval_results.json"
    results = []
    completed_qids = set()

    if results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                results = saved.get("results", [])
                completed_qids = {r["query_id"] for r in results}
                log.info("Resuming run: found %d previously completed queries", len(completed_qids))
        except Exception as exc:
            log.warning("Could not read previous results: %s", exc)

    start_total = time.monotonic()
    for idx, q in enumerate(queries, start=1):
        qid = q["query_id"]
        question = q["query"]

        if qid in completed_qids:
            log.info("[%d/%d] Skipping already completed query %s", idx, len(queries), qid)
            continue

        log.info("[%d/%d] Processing query %s: %s", idx, len(queries), qid, question[:60])
        try:
            q_start = time.monotonic()
            res = adapter.query(
                question=question,
                mode=args.mode,
                generate_answer=args.generate_answer,
            )
            q_elapsed = time.monotonic() - q_start

            entry = {
                "query_id": qid,
                "query": question,
                "gold_chunk_ids": q.get("gold_chunk_ids", []),
                "chunk_ids": res.chunk_ids,
                "retrieved_count": res.retrieved_count,
                "source_docs": res.source_docs,
                "answer": res.answer,
                "retrieval_latency_s": round(res.retrieval_latency_s, 3),
                "generation_latency_s": round(res.generation_latency_s, 3),
                "total_latency_s": round(q_elapsed, 3),
                "errors": res.errors,
            }
            results.append(entry)
            log.info("  Retrieved %d chunks in %.3fs (total: %.3fs)", res.retrieved_count, res.retrieval_latency_s, q_elapsed)

            # Checkpoint results incrementally for fault tolerance
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({"results": results, "count": len(results)}, f, indent=2, ensure_ascii=False)

        except Exception as exc:
            log.exception("Query %s failed: %s", qid, exc)
            results.append({
                "query_id": qid,
                "query": question,
                "gold_chunk_ids": q.get("gold_chunk_ids", []),
                "chunk_ids": [],
                "retrieved_count": 0,
                "source_docs": [],
                "answer": None,
                "retrieval_latency_s": 0.0,
                "generation_latency_s": 0.0,
                "total_latency_s": 0.0,
                "errors": [f"{type(exc).__name__}: {str(exc)}"],
            })

    total_time = time.monotonic() - start_total

    # 8. Compute and save evaluation metrics
    eval_metrics = evaluate_retrieval(results, queries)
    eval_metrics["total_elapsed_s"] = round(total_time, 3)
    eval_metrics["avg_query_latency_s"] = round(total_time / len(queries), 3) if queries else 0.0

    eval_path = exp_dir / "evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    # 9. Final metadata
    metadata = {
        "model": args.model,
        "pipeline": args.pipeline,
        "mode": args.mode,
        "top_k": args.top_k,
        "n_queries": len(queries),
        "total_elapsed_s": round(total_time, 3),
        "recall_at_k": eval_metrics.get("avg_recall_at_k", 0.0),
        "mrr": eval_metrics.get("mrr", 0.0),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(exp_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log.info("=== Run Complete ===")
    log.info("Recall@%d: %.4f | MRR: %.4f | Total Time: %.2fs", args.top_k, eval_metrics["avg_recall_at_k"], eval_metrics["mrr"], total_time)
    print(f"\n[DONE] Results and evaluation saved to:\n  {exp_dir}")
    print(f"  Recall@{args.top_k}: {eval_metrics['avg_recall_at_k']:.4f}")
    print(f"  MRR:        {eval_metrics['mrr']:.4f}")
    print(f"  Total Time: {total_time:.2f}s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
