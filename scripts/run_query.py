"""
run_query.py -- End-to-end query CLI (Task 6.2).

Wires together:
  client_factory  → LLM client (local Ollama or API stub)
  tracing         → LANGCHAIN_TRACING_V2 toggle (once at startup)
  pipeline_a/b    → flat or structural retrieval
  serialize       → TOON or JSON evidence string for the generator
  generate        → grounded answer with cited evidence IDs

Usage
-----
  python scripts/run_query.py --pipeline b --mode local --query "..."
  python scripts/run_query.py --pipeline a --mode local --query "..."

Options
-------
  --query       Required. The natural language question.
  --pipeline    a | b  (default: b, canonical Pipeline B)
  --mode        local | api  (default: local)
  --api-key     Session-only API key for mode=api. Never written to disk.
  --top-k       Override TOP_N_FINAL_EVIDENCE (default: from config).
  --format      toon | json  (default: from SERIALIZATION_FORMAT config).
  --no-trace    Suppress LANGCHAIN_TRACING_V2 even in local mode.
  --json-out    Emit the final RunRecord as JSON to stdout instead of
                human-readable text.

Exit codes
----------
  0   Success (answer produced, even if it is a "cannot answer" refusal).
  1   Retrieval returned no evidence (empty pipeline result).
  2   Configuration or argument error.
  3   LLM or Zvec runtime error.

Output (human-readable, to stdout)
-----------------------------------
  [run_query] query:     <question>
  [run_query] pipeline:  a | b
  [run_query] mode:      local | api
  [run_query] model:     <SLM_MODEL>
  [run_query] retrieved: N chunks
  [run_query] evidence:
    [0] <chunk_id>  <source_doc>  score=<score>
    ...
  [run_query] answer:
    <answer text>
  [run_query] cited_ids: [<id>, ...]
  [run_query] timings:   retrieval=Xs  generation=Xs  total=Xs
  [run_query] git_commit: <sha or unknown>

A RunRecord JSON is always written to data/cache/run_query_<timestamp>.json
alongside the human-readable output, so results are machine-readable and
time-stamped without requiring the caller to redirect stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Import set_tracing at module level so tests can mock it via
# patch("scripts.run_query.set_tracing").
from graphrag_lite.observability.tracing import set_tracing


# ── helpers ──────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "log", "-1", "--format=%H"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip()[:12] if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        r = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


# ── Run record ────────────────────────────────────────────────────────────────

@dataclass
class RunRecord:
    """
    Immutable record of one run_query execution.  Written as JSON to
    data/cache/run_query_<timestamp>.json after every run.

    Fields that could contain secrets (api_key) are intentionally absent.
    """
    timestamp: str
    git_commit: str
    git_dirty: bool
    query: str
    pipeline: str               # "a" or "b"
    mode: str                   # "local" or "api"
    model: str
    serialization_format: str
    top_k: int
    retrieval_latency_s: float
    generation_latency_s: float
    total_latency_s: float
    retrieved_count: int
    evidence_ids: list[str]
    cited_ids: list[str]
    answer_text: str
    no_answer: bool             # True if the model returned the refusal string
    citation_ids_valid: bool    # cited_ids ⊆ evidence_ids
    hallucinated_ids: list[str] # cited_ids − evidence_ids (empty = clean)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── Core pipeline runner ──────────────────────────────────────────────────────

_NO_ANSWER_MARKER = "i cannot answer this question"


def run_query(
    query: str,
    pipeline: str = "b",
    mode: str = "local",
    api_key: str | None = None,
    top_k: int | None = None,
    serialization_format: str | None = None,
    no_trace: bool = False,
) -> RunRecord:
    """
    Execute a single query through the full Phase 0-5 pipeline.

    Parameters
    ----------
    query : str
        Natural language question.
    pipeline : "a" | "b"
        Retrieval pipeline to use.
    mode : "local" | "api"
        LLM mode ("api" uses the GrokClientStub interface only).
    api_key : str | None
        Session-only API key for mode="api". Never written to disk or logged.
    top_k : int | None
        Override for TOP_N_FINAL_EVIDENCE. None = use config default.
    serialization_format : "toon" | "json" | None
        Override for SERIALIZATION_FORMAT config. None = use config default.
    no_trace : bool
        If True, skip set_tracing() even in local mode.

    Returns
    -------
    RunRecord

    Raises
    ------
    ValueError  — empty query, unknown pipeline/mode
    RuntimeError — Zvec/embedding failure
    """
    if not query.strip():
        raise ValueError("run_query: query must not be empty")
    if pipeline not in ("a", "b"):
        raise ValueError(f"run_query: pipeline must be 'a' or 'b', got {pipeline!r}")
    if mode not in ("local", "api"):
        raise ValueError(f"run_query: mode must be 'local' or 'api', got {mode!r}")

    total_start = time.time()

    # ── 1. Tracing (once at startup per spec) ────────────────────────────
    if not no_trace:
        set_tracing(mode)

    # ── 2. Config ────────────────────────────────────────────────────────
    from graphrag_lite.config import get_config
    cfg = get_config()

    fmt = serialization_format or cfg.SERIALIZATION_FORMAT
    if fmt not in ("toon", "json"):
        fmt = "toon"  # safe fallback

    # ── 3. LLM client ────────────────────────────────────────────────────
    from graphrag_lite.llm.client_factory import get_llm_client
    llm_client = get_llm_client(mode=mode, api_key=api_key)

    # ── 4. Retrieval ─────────────────────────────────────────────────────
    retrieval_start = time.time()

    if pipeline == "a":
        from graphrag_lite.retrieval.pipeline_a import retrieve_flat
        if top_k is not None:
            # Pipeline A doesn't expose top_k as an argument; honour config only
            evidence = retrieve_flat(query)
            evidence = evidence[:top_k]
        else:
            evidence = retrieve_flat(query)
    else:
        from graphrag_lite.retrieval.pipeline_b import retrieve_structural
        evidence = retrieve_structural(query, top_k=top_k)

    retrieval_latency = time.time() - retrieval_start

    # ── 5. Serialization (for the generation prompt) ──────────────────────
    from graphrag_lite.retrieval.serialize import to_toon, to_json, count_tokens

    if fmt == "toon":
        evidence_str = to_toon(evidence)
    else:
        evidence_str = to_json(evidence)

    evidence_dicts = [
        {"text": e.text, "chunk_id": e.chunk_id}
        for e in evidence
    ]
    evidence_ids = [e.chunk_id for e in evidence]

    # ── 6. Generation ────────────────────────────────────────────────────
    generation_start = time.time()

    from graphrag_lite.llm.generator import AnswerGenerator
    gen = AnswerGenerator(client=llm_client)
    answer = gen.generate_answer(query=query, evidence_chunks=evidence_dicts)

    generation_latency = time.time() - generation_start
    total_latency = time.time() - total_start

    # ── 7. Citation validation ───────────────────────────────────────────
    passed_ids_set = set(evidence_ids)
    cited_ids_set = set(answer.cited_evidence_ids)
    hallucinated = sorted(cited_ids_set - passed_ids_set)
    citation_valid = len(hallucinated) == 0

    no_answer = _NO_ANSWER_MARKER in answer.text.lower()

    # ── 8. Build record ──────────────────────────────────────────────────
    return RunRecord(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_commit=_git_commit(),
        git_dirty=_git_dirty(),
        query=query,
        pipeline=pipeline,
        mode=mode,
        model=llm_client.model,
        serialization_format=fmt,
        top_k=len(evidence),
        retrieval_latency_s=round(retrieval_latency, 3),
        generation_latency_s=round(generation_latency, 3),
        total_latency_s=round(total_latency, 3),
        retrieved_count=len(evidence),
        evidence_ids=evidence_ids,
        cited_ids=answer.cited_evidence_ids,
        answer_text=answer.text,
        no_answer=no_answer,
        citation_ids_valid=citation_valid,
        hallucinated_ids=hallucinated,
        error=None,
    )


# ── Output helpers ────────────────────────────────────────────────────────────

def _print_human(rec: RunRecord) -> None:
    """Print a human-readable summary of a RunRecord to stdout."""
    print(f"[run_query] query:     {rec.query}")
    print(f"[run_query] pipeline:  {rec.pipeline}")
    print(f"[run_query] mode:      {rec.mode}")
    print(f"[run_query] model:     {rec.model}")
    print(f"[run_query] retrieved: {rec.retrieved_count} chunks")
    print("[run_query] evidence:")
    for i, eid in enumerate(rec.evidence_ids):
        print(f"    [{i}] {eid}")
    print("[run_query] answer:")
    for line in rec.answer_text.splitlines():
        print(f"    {line}")
    print(f"[run_query] cited_ids:       {rec.cited_ids}")
    print(f"[run_query] citation_valid:  {rec.citation_ids_valid}", end="")
    if rec.hallucinated_ids:
        print(f"  (hallucinated: {rec.hallucinated_ids})", end="")
    print()
    print(f"[run_query] no_answer:       {rec.no_answer}")
    print(
        f"[run_query] timings:    retrieval={rec.retrieval_latency_s}s  "
        f"generation={rec.generation_latency_s}s  "
        f"total={rec.total_latency_s}s"
    )
    print(f"[run_query] git_commit: {rec.git_commit}{'*' if rec.git_dirty else ''}")


def _save_record(rec: RunRecord) -> Path:
    """Write the RunRecord as JSON to data/cache/. Returns the path."""
    out_dir = Path("data") / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = rec.timestamp.replace(":", "-").replace("T", "_")
    out_path = out_dir / f"run_query_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec.to_dict(), f, indent=2, ensure_ascii=False)
    return out_path


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_query",
        description="GraphRAG-Lite end-to-end query runner (Task 6.2)",
    )
    p.add_argument("--query", required=True, help="Natural language question")
    p.add_argument(
        "--pipeline", choices=["a", "b"], default="b",
        help="Retrieval pipeline: a (flat) or b (structural). Default: b",
    )
    p.add_argument(
        "--mode", choices=["local", "api"], default="local",
        help="LLM mode: local (Ollama) or api (stub). Default: local",
    )
    p.add_argument(
        "--api-key", default=None,
        help="Session-only API key for mode=api. Never written to disk.",
    )
    p.add_argument(
        "--top-k", type=int, default=None,
        help="Override TOP_N_FINAL_EVIDENCE. Default: from config.",
    )
    p.add_argument(
        "--format", choices=["toon", "json"], default=None, dest="fmt",
        help="Serialization format for evidence prompt. Default: from config.",
    )
    p.add_argument(
        "--no-trace", action="store_true",
        help="Suppress LANGCHAIN_TRACING_V2 even in local mode.",
    )
    p.add_argument(
        "--json-out", action="store_true",
        help="Emit the RunRecord as JSON to stdout (machine-readable mode).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point.  Returns an exit code (0 = success, 1/2/3 = error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        rec = run_query(
            query=args.query,
            pipeline=args.pipeline,
            mode=args.mode,
            api_key=args.api_key,
            top_k=args.top_k,
            serialization_format=args.fmt,
            no_trace=args.no_trace,
        )
    except ValueError as e:
        print(f"[run_query] ERROR (config/arg): {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[run_query] ERROR (runtime): {e}", file=sys.stderr)
        return 3

    # Save machine-readable record
    saved_path = _save_record(rec)

    # Emit output
    if args.json_out:
        print(json.dumps(rec.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print_human(rec)
        print(f"[run_query] record saved to: {saved_path}")

    # Exit 1 if retrieval returned nothing (generator produced the
    # empty-evidence refusal string)
    if rec.retrieved_count == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
