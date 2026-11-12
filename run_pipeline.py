"""
run_pipeline.py — Unified manual experiment entry point for MindMesh.

USAGE
-----
  python run_pipeline.py --list-pipelines
  python run_pipeline.py --list-models
  python run_pipeline.py --pipeline pipeline_a --model qwen2.5:1.5b
  python run_pipeline.py --pipeline pipeline_b --model qwen2.5:1.5b
  python run_pipeline.py --pipeline pipeline_3_lightrag --model qwen2.5:1.5b
  python run_pipeline.py --pipeline pipeline_a --model qwen2.5:1.5b --dry-run

RULES
-----
1. The --model argument is REQUIRED when running a pipeline. No silent default.
2. The model must be installed locally. Never downloads or pulls automatically.
3. Only the explicitly requested pipeline runs.
4. GPU-intensive tasks (extraction, re-indexing) must be triggered separately.
5. --dry-run validates everything but performs no inference.

OUTPUT
------
Every run writes to data/experiments/<experiment_name>/
  config.json          — run configuration and metadata
  metadata.json        — model info, system state
  retrieval_results.json — retrieved chunks per query
  evaluation.json      — accuracy metrics if --eval
  logs/run.log         — full run log

PIPELINES
---------
  pipeline_a          — flat vector baseline (Pipeline A, unchanged)
  pipeline_b          — structural graph (Pipeline B, unchanged)
  lightrag_local      — MindMesh LightRAG Local (experimental)
  lightrag_global     — MindMesh LightRAG Global (experimental)
  lightrag_hybrid     — MindMesh LightRAG Hybrid (experimental)
  lightrag_mix        — MindMesh LightRAG Mix (experimental)
  pipeline_3_lightrag — Genuine upstream LightRAG v1.5.8 (Pipeline 3)
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
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

log = logging.getLogger("run_pipeline")


# ── Model utilities ───────────────────────────────────────────────────────────

def list_local_models() -> list[dict]:
    """Return info about every locally installed Ollama model. Never downloads."""
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: Ollama not available: {exc}", file=sys.stderr)
        return []

    models = []
    for line in r.stdout.strip().splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3:
            models.append({
                "name":     parts[0],
                "id":       parts[1],
                "size":     parts[2],
                "modified": " ".join(parts[3:]) if len(parts) > 3 else "",
            })
    return models


def validate_model(model: str) -> None:
    """
    Raise SystemExit if ``model`` is not installed locally.
    Never downloads or pulls models.
    """
    models = list_local_models()
    names = [m["name"] for m in models]
    if model not in names:
        print(
            f"\nERROR: Model {model!r} is not installed locally.\n"
            f"Installed models: {', '.join(names) or '(none)'}\n"
            f"Install it manually with: ollama pull {model}\n"
            f"(Never pulled automatically by this runner.)",
            file=sys.stderr,
        )
        sys.exit(1)


# ── Output directory helpers ──────────────────────────────────────────────────

def _slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", s).strip("_")


def make_exp_dir(
    pipeline: str,
    model: str,
    experiment_name: str | None,
    output_dir: str | None,
) -> Path:
    if output_dir:
        base = Path(output_dir)
    else:
        base = ROOT / "data" / "experiments"

    if experiment_name:
        name = _slugify(experiment_name)
    else:
        model_slug = _slugify(model)
        name = f"run_{_slugify(pipeline)}_{model_slug}_{time.strftime('%Y%m%d_%H%M%S')}"

    exp = base / name
    if exp.exists():
        # Never overwrite — append a timestamp suffix
        ts = time.strftime("%H%M%S")
        exp = base / f"{name}_{ts}"

    exp.mkdir(parents=True, exist_ok=True)
    (exp / "logs").mkdir(exist_ok=True)
    return exp


def save_config(exp_dir: Path, args: argparse.Namespace, metadata: dict) -> None:
    cfg = {
        "pipeline":        args.pipeline,
        "model":           args.model,
        "mode":            args.mode,
        "top_k":           args.top_k,
        "experiment_name": args.experiment_name,
        "dataset":         args.dataset,
        "eval":            args.eval,
        "gleaning":        args.gleaning,
        "cache":           args.cache,
        "dry_run":         args.dry_run,
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "exp_dir":         str(exp_dir),
    }
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(exp_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# ── Evaluation helpers ────────────────────────────────────────────────────────

def load_dev_set(dataset: str | None) -> list[dict]:
    """Load the development evaluation set. Never touches the frozen test set."""
    path = ROOT / "eval" / "gold_dev_set.json"
    if dataset and dataset != "dev":
        custom = Path(dataset)
        if custom.exists():
            path = custom
        else:
            print(f"WARNING: Dataset {dataset!r} not found, using default dev set.", file=sys.stderr)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("queries", [])


def evaluate_results(
    results: list[dict],
    queries: list[dict],
) -> dict:
    """Compute recall@K and MRR for a set of retrieval results."""
    if not queries:
        return {}
    recall_vals, mrr_vals = [], []
    for q in queries:
        qid = q["query_id"]
        gold = set(q.get("gold_chunk_ids", []))
        # Find matching result
        r = next((x for x in results if x.get("query_id") == qid), None)
        if r is None:
            recall_vals.append(0.0)
            mrr_vals.append(0.0)
            continue
        retrieved = r.get("chunk_ids", [])
        if not gold:
            recall_vals.append(1.0)
            mrr_vals.append(1.0)
            continue
        recall = sum(1 for g in gold if g in retrieved) / len(gold)
        recall_vals.append(recall)
        mrr = next((1.0 / i for i, cid in enumerate(retrieved, 1) if cid in gold), 0.0)
        mrr_vals.append(mrr)
    n = len(recall_vals)
    return {
        "n":               n,
        "avg_recall_at_k": round(sum(recall_vals) / n, 4) if n else 0.0,
        "mrr":             round(sum(mrr_vals) / n, 4) if n else 0.0,
    }


# ── Main run logic ────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    # Validate model is explicitly provided
    if args.model is None:
        print(
            "ERROR: --model is required. Example: --model qwen2.5:1.5b\n"
            "Use --list-models to see installed models.",
            file=sys.stderr,
        )
        return 1

    # Validate model is installed
    validate_model(args.model)

    # Validate pipeline
    from mindmesh.pipelines.registry import get_adapter_class, build_adapter
    try:
        get_adapter_class(args.pipeline)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Create experiment directory
    exp_dir = make_exp_dir(
        args.pipeline, args.model,
        args.experiment_name, args.output_dir
    )

    # Set up logging
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    fh = logging.FileHandler(str(exp_dir / "logs" / "run.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)

    log.info("=== run_pipeline.py ===")
    log.info("Pipeline: %s  Model: %s  Mode: %s  top_k: %d",
             args.pipeline, args.model, args.mode, args.top_k)
    log.info("Experiment dir: %s", exp_dir)

    # Collect metadata
    models = list_local_models()
    model_info = next((m for m in models if m["name"] == args.model), {})
    metadata: dict[str, Any] = {
        "model":        args.model,
        "model_info":   model_info,
        "pipeline":     args.pipeline,
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "installed_models": [m["name"] for m in models],
    }

    # Build adapter (no inference yet)
    adapter = build_adapter(
        pipeline_name=args.pipeline,
        model=args.model,
        exp_dir=exp_dir,
        top_k=args.top_k,
    )

    # Dry-run: print config and exit
    if args.dry_run:
        dry = adapter.dry_run()
        save_config(exp_dir, args, metadata)
        print("\n=== DRY RUN ===")
        print(json.dumps(dry, indent=2))
        print(f"\nConfig saved to {exp_dir / 'config.json'}")
        log.info("Dry run complete — no inference performed.")
        return 0

    # Load queries
    queries = load_dev_set(args.dataset)
    if not queries:
        print("WARNING: No queries loaded. No evaluation will run.", file=sys.stderr)

    if args.eval and not queries:
        print("ERROR: --eval requires a valid dataset.", file=sys.stderr)
        return 1

    # Run queries
    results = []
    total_start = time.monotonic()

    if queries and args.eval:
        log.info("Running %d queries through %s...", len(queries), args.pipeline)
        for i, q in enumerate(queries, 1):
            qid = q["query_id"]
            question = q["query"]
            log.info("[%02d/%02d] %s ...", i, len(queries), qid)
            try:
                result = adapter.query(
                    question=question,
                    mode=args.mode,
                    generate_answer=False,  # eval-only: no generation
                )
                entry = {
                    "query_id":    qid,
                    "query":       question,
                    "pipeline":    args.pipeline,
                    "model":       args.model,
                    "mode":        args.mode,
                    "chunk_ids":   result.chunk_ids,
                    "source_docs": result.source_docs,
                    "retrieved_count": result.retrieved_count,
                    "retrieval_latency_s": result.retrieval_latency_s,
                    "errors":      result.errors,
                }
                results.append(entry)
                log.info("  -> %d chunks, %.3fs", result.retrieved_count, result.retrieval_latency_s)
            except Exception as exc:
                log.error("  ERROR on %s: %s", qid, exc)
                results.append({
                    "query_id": qid, "query": question, "pipeline": args.pipeline,
                    "model": args.model, "mode": args.mode, "chunk_ids": [],
                    "source_docs": [], "retrieved_count": 0, "errors": [str(exc)],
                    "retrieval_latency_s": 0.0,
                })
    elif args.query:
        # Single query mode
        log.info("Running single query: %r", args.query[:60])
        result = adapter.query(
            question=args.query,
            mode=args.mode,
            generate_answer=not args.no_generate,
        )
        entry = {
            "query_id": "manual",
            "query":    args.query,
            "pipeline": args.pipeline,
            "model":    args.model,
            "mode":     args.mode,
            "chunk_ids":   result.chunk_ids,
            "source_docs": result.source_docs,
            "retrieved_count": result.retrieved_count,
            "answer":          result.answer,
            "retrieval_latency_s": result.retrieval_latency_s,
            "generation_latency_s": result.generation_latency_s,
            "errors": result.errors,
        }
        results.append(entry)
        if result.answer:
            print(f"\nAnswer:\n{result.answer}")
        elif result.retrieved_chunks:
            print(f"\nRetrieved {result.retrieved_count} chunks:")
            for c in result.retrieved_chunks[:3]:
                print(f"  [{c.get('chunk_id','?')[:12]}] {c.get('text','')[:80]}")
        if result.errors:
            for e in result.errors:
                print(f"ERROR: {e}", file=sys.stderr)

    total_s = time.monotonic() - total_start

    # Save retrieval results
    with open(exp_dir / "retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "total_s": round(total_s, 3)},
                  f, indent=2, ensure_ascii=False)

    # Evaluate if requested
    if args.eval and queries and results:
        metrics = evaluate_results(results, queries)
        metrics["total_s"] = round(total_s, 3)
        with open(exp_dir / "evaluation.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        log.info("Evaluation: recall=%.4f  mrr=%.4f  n=%d",
                 metrics["avg_recall_at_k"], metrics["mrr"], metrics["n"])
        print(f"\nEvaluation: Recall@{args.top_k}={metrics['avg_recall_at_k']:.4f}  "
              f"MRR={metrics['mrr']:.4f}  n={metrics['n']}")

    save_config(exp_dir, args, metadata)
    log.info("Complete. Total: %.1fs. Results: %s", total_s, exp_dir)
    print(f"\nResults saved to {exp_dir}")
    return 0


# ── Argument parsing ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MindMesh unified pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Listing commands (no --model required)
    p.add_argument("--list-pipelines", action="store_true",
                   help="List all registered pipelines and exit.")
    p.add_argument("--list-models", action="store_true",
                   help="List locally installed Ollama models and exit.")

    # Core pipeline selection
    p.add_argument("--pipeline", type=str, default=None,
                   help="Pipeline to run. Use --list-pipelines to see options.")
    p.add_argument("--model", type=str, default=None,
                   help="[REQUIRED] Ollama model tag. Must be installed locally.")

    # Query
    p.add_argument("--query", type=str, default=None,
                   help="Run a single query instead of the full eval set.")
    p.add_argument("--no-generate", action="store_true",
                   help="With --query: retrieve only, no answer generation.")

    # Experiment config
    p.add_argument("--experiment-name", type=str, default=None,
                   help="Name for this experiment run. Auto-generated if omitted.")
    p.add_argument("--dataset", type=str, default="dev",
                   help="Dataset to use. 'dev'=gold_dev_set.json (default). "
                        "Path to a custom JSON for other datasets.")
    p.add_argument("--top-k", type=int, default=6,
                   help="Number of evidence items to retrieve (default: 6).")
    p.add_argument("--mode", type=str, default="default",
                   help="Retrieval mode within the pipeline "
                        "(e.g. 'hybrid' for pipeline_3_lightrag).")
    p.add_argument("--eval", action="store_true",
                   help="Run against the full dataset and compute metrics.")
    p.add_argument("--gleaning", type=int, default=None,
                   help="Number of gleaning passes for extraction (GPU task).")
    p.add_argument("--cache", type=str, default=None,
                   help="Custom cache directory for LLM responses.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory for experiment results "
                        "(default: data/experiments/<name>/).")

    # Special modes
    p.add_argument("--dry-run", action="store_true",
                   help="Validate everything but do not perform any inference.")
    p.add_argument("--setup-only", action="store_true",
                   help="[pipeline_3_lightrag only] Index documents and exit (GPU task G7).")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── Listing commands ──────────────────────────────────────────────────────
    if args.list_pipelines:
        from mindmesh.pipelines.registry import list_pipelines
        rows = list_pipelines()
        print(f"\n{'='*70}")
        print("  REGISTERED PIPELINES")
        print(f"{'='*70}")
        for row in rows:
            print(f"  {row['name']:<26} {row['description'][:60]}")
            if row["modes"] and row["modes"] != ["default"]:
                print(f"    {'':24}  modes: {', '.join(row['modes'])}")
        print(f"\nUsage: python run_pipeline.py --pipeline <name> --model <model>")
        return 0

    if args.list_models:
        models = list_local_models()
        print(f"\n{'='*70}")
        print("  LOCALLY INSTALLED OLLAMA MODELS")
        print(f"{'='*70}")
        if not models:
            print("  (none — Ollama not available or no models installed)")
        else:
            fmt = "  {:<35} {:<14} {}"
            print(fmt.format("NAME", "SIZE", "MODIFIED"))
            print("  " + "-" * 60)
            for m in models:
                print(fmt.format(m["name"], m["size"], m["modified"]))
        return 0

    # ── Need a pipeline to run ────────────────────────────────────────────────
    if args.pipeline is None:
        parser.print_help()
        print("\nERROR: --pipeline is required. Use --list-pipelines to see options.",
              file=sys.stderr)
        return 1

    # ── Setup-only (GPU task) ──────────────────────────────────────────────────
    if args.setup_only:
        if args.model is None:
            print("ERROR: --model is required for --setup-only.", file=sys.stderr)
            return 1
        validate_model(args.model)
        from mindmesh.pipelines.registry import build_adapter
        adapter = build_adapter(args.pipeline, args.model, top_k=args.top_k)
        if hasattr(adapter, "index_documents"):
            print(f"Indexing documents for {args.pipeline}...")
            result = adapter.index_documents()
            print(json.dumps(result, indent=2))
            return 0
        else:
            print(f"Pipeline {args.pipeline!r} does not support --setup-only.", file=sys.stderr)
            return 1

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
