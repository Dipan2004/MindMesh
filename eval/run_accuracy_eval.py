"""
run_accuracy_eval.py — Task 8.2: Accuracy / efficiency evaluation harness.

Runs every question in the gold development set through Pipeline A and
Pipeline B in local mode (no cloud inference), scores each answer, and
prints a full per-question results table followed by aggregate statistics.

Scoring methods (no LLM-as-judge, no LangSmith required)
---------------------------------------------------------
1. Gold-evidence recall@K
   The fraction of gold chunk IDs that appear in the top-K retrieved results.
   A gold chunk that the retriever returns anywhere in its result list counts
   as retrieved. This is a pure retrieval metric, independent of the answer.

2. Groundedness
   cited_ids ⊆ evidence_ids  (True / False per question).
   Defined in Task 5.2 DoD: a hallucinated citation ID is a real bug.

3. Answer correctness — required-term coverage
   Each gold entry carries a list of ``required_terms`` that must appear in
   the answer.  Coverage = (terms found) / (total required terms).
   This is a lexical proxy for factual correctness; it does not require a
   second LLM call.

4. No-answer accuracy
   For ``type=no-answer`` questions the expected behaviour is the canonical
   refusal string ("i cannot answer this question …").  Correct = 1 if the
   model refuses; 0 if it produces a fabricated answer.

Why fuzzy match instead of LLM-as-judge
-----------------------------------------
The TASKS.md spec says "LLM-as-judge or fuzzy match" (§8.2).  LangSmith
tracing is not configured and the project's approved model list does not
include a separate judge model.  Required-term coverage is the most honest
alternative: the terms were written by a human reading the source chunk, so
their presence in the answer is a meaningful (if conservative) proxy for
accuracy.

Usage
-----
  # canonical run (dev set, Pipelines A and B, pagerank ranking)
  python eval/run_accuracy_eval.py

  # single pipeline
  python eval/run_accuracy_eval.py --pipeline a

  # ranking method override (for Task 8.6)
  python eval/run_accuracy_eval.py --ranking degree

  # serialization format override (for Task 8.5)
  python eval/run_accuracy_eval.py --format json

  # SLM model override (for Task 8.4)
  python eval/run_accuracy_eval.py --model qwen2.5:0.5b

  # write JSON output for downstream scripts
  python eval/run_accuracy_eval.py --out eval/results/run_YYYYMMDD.json

Output files
------------
Results are written to eval/results/<timestamp>.json so each run is
preserved independently (never overwritten).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import importlib
from pathlib import Path
from typing import Literal

# ── path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ── constants ─────────────────────────────────────────────────────────────────
GOLD_DEV_PATH = _REPO_ROOT / "eval" / "gold_dev_set.json"
RESULTS_DIR = _REPO_ROOT / "eval" / "results"
_NO_ANSWER_MARKER = "i cannot answer this question"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_dev_set() -> list[dict]:
    with open(GOLD_DEV_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


def _recall_at_k(
    retrieved_ids: list[str],
    gold_chunk_ids: list[str],
) -> float:
    """Fraction of gold chunks found anywhere in the retrieved list."""
    if not gold_chunk_ids:
        return 1.0  # no-answer questions have no gold chunks
    hit = sum(1 for g in gold_chunk_ids if g in retrieved_ids)
    return hit / len(gold_chunk_ids)


def _gold_evidence_rank(
    retrieved_ids: list[str],
    gold_chunk_ids: list[str],
) -> int | None:
    """1-based rank of the first gold chunk in the retrieved list, or None."""
    for i, cid in enumerate(retrieved_ids, 1):
        if cid in set(gold_chunk_ids):
            return i
    return None


def _required_term_coverage(answer_text: str, required_terms: list[str]) -> float:
    """Fraction of required terms found in the answer (case-insensitive)."""
    if not required_terms:
        return 1.0
    lower = answer_text.lower()
    found = sum(1 for t in required_terms if t.lower() in lower)
    return found / len(required_terms)


def _is_refusal(answer_text: str) -> bool:
    return _NO_ANSWER_MARKER in answer_text.lower()


def _count_tokens(text: str) -> int:
    from graphrag_lite.ingestion.chunker import get_tokenizer
    from graphrag_lite.config import get_config
    tok = get_tokenizer(get_config().TOKENIZER)
    return len(tok.encode(text, disallowed_special=()))


# ── per-question runner ───────────────────────────────────────────────────────

def run_one(
    query_dict: dict,
    pipeline: Literal["a", "b"],
    fmt: str,
    ranking_method: str,
    model: str,
) -> dict:
    """
    Run one gold query through the specified pipeline and return a result dict.

    Ranking method overrides are applied by temporarily resetting the
    pipeline_b graph singleton and restoring it afterward.
    """
    from graphrag_lite.config import get_config
    from graphrag_lite.llm.client_factory import get_llm_client
    from graphrag_lite.llm.generator import AnswerGenerator
    from graphrag_lite.retrieval.serialize import to_toon, to_json, count_tokens

    cfg = get_config()
    query = query_dict["query"]
    gold_chunk_ids: list[str] = query_dict.get("gold_chunk_ids", [])
    q_type: str = query_dict.get("type", "single-chunk")
    gold_entries: list[dict] = query_dict.get("gold", [])

    # Collect required terms from all gold entries
    required_terms: list[str] = []
    for g in gold_entries:
        required_terms.extend(g.get("required_terms", []))
    required_terms = list(dict.fromkeys(required_terms))  # dedup, preserve order

    # ── Ranking method override for Pipeline B ────────────────────────────────
    import graphrag_lite.retrieval.pipeline_b as pb
    _orig_G = pb._G
    _orig_PR = pb._PR
    _orig_PR_MAX = pb._PR_MAX

    need_override = (pipeline == "b" and ranking_method != cfg.RANKING_METHOD)
    if need_override:
        # Reset singleton and temporarily override RANKING_METHOD via env var
        pb._G = None
        pb._PR = None
        pb._PR_MAX = None
        os.environ["RANKING_METHOD"] = ranking_method
        get_config.cache_clear()  # type: ignore[attr-defined]

    # ── Retrieval ─────────────────────────────────────────────────────────────
    t_ret_start = time.monotonic()
    try:
        if pipeline == "a":
            from graphrag_lite.retrieval.pipeline_a import retrieve_flat
            evidence = retrieve_flat(query)
        else:
            from graphrag_lite.retrieval.pipeline_b import retrieve_structural
            evidence = retrieve_structural(query)
    finally:
        if need_override:
            # Restore original state
            pb._G = _orig_G
            pb._PR = _orig_PR
            pb._PR_MAX = _orig_PR_MAX
            os.environ.pop("RANKING_METHOD", None)
            get_config.cache_clear()  # type: ignore[attr-defined]
    retrieval_latency = time.monotonic() - t_ret_start

    evidence_ids = [e.chunk_id for e in evidence]

    # ── Serialization (for token counting) ────────────────────────────────────
    if fmt == "toon":
        evidence_str = to_toon(evidence)
    else:
        evidence_str = to_json(evidence)
    evidence_token_count = count_tokens(evidence_str)

    # ── Generation ────────────────────────────────────────────────────────────
    llm_client = get_llm_client("local")
    # Override model if specified
    if model != cfg.SLM_MODEL:
        from graphrag_lite.llm.ollama_client import OllamaClient
        # qwen3 family requires think=False to suppress chain-of-thought from
        # consuming the entire num_predict budget and leaving an empty response.
        think_off = model.startswith("qwen3")
        llm_client = OllamaClient(
            model=model,
            temperature=cfg.EXTRACTION_TEMPERATURE,
            seed=cfg.EXTRACTION_SEED,
            num_predict=cfg.LLM_NUM_PREDICT,
            think=False if think_off else cfg.effective_think(),
        )

    gen = AnswerGenerator(client=llm_client)
    evidence_dicts = [{"text": e.text, "chunk_id": e.chunk_id} for e in evidence]

    t_gen_start = time.monotonic()
    answer = gen.generate_answer(query, evidence_dicts)
    generation_latency = time.monotonic() - t_gen_start

    # ── Scoring ───────────────────────────────────────────────────────────────
    recall = _recall_at_k(evidence_ids, gold_chunk_ids)
    gold_rank = _gold_evidence_rank(evidence_ids, gold_chunk_ids)

    # Groundedness: every cited ID must be in the evidence set
    cited_ids = answer.cited_evidence_ids
    grounded = set(cited_ids).issubset(set(evidence_ids))
    hallucinated_ids = sorted(set(cited_ids) - set(evidence_ids))

    # Correctness: required-term coverage
    if q_type == "no-answer":
        # No-answer: correct = model refuses
        correct = 1.0 if _is_refusal(answer.text) else 0.0
        term_coverage = 1.0  # not applicable
    else:
        correct = _required_term_coverage(answer.text, required_terms)
        term_coverage = correct

    # No-answer flag
    no_answer_predicted = _is_refusal(answer.text)
    # For no-answer type, this should be True; for others, False
    no_answer_correct = (q_type == "no-answer") == no_answer_predicted

    # Answer token count
    answer_token_count = _count_tokens(answer.text)

    return {
        "query_id": query_dict["query_id"],
        "type": q_type,
        "pipeline": pipeline,
        "model": llm_client.model,
        "format": fmt,
        "ranking_method": ranking_method,
        "retrieved_count": len(evidence),
        "evidence_ids": evidence_ids,
        "gold_chunk_ids": gold_chunk_ids,
        "cited_ids": cited_ids,
        "recall_at_k": round(recall, 4),
        "gold_rank": gold_rank,
        "grounded": grounded,
        "hallucinated_ids": hallucinated_ids,
        "term_coverage": round(term_coverage, 4),
        "no_answer_predicted": no_answer_predicted,
        "no_answer_correct": no_answer_correct,
        "answer_text": answer.text,
        "evidence_token_count": evidence_token_count,
        "answer_token_count": answer_token_count,
        "retrieval_latency_s": round(retrieval_latency, 3),
        "generation_latency_s": round(generation_latency, 3),
        "total_latency_s": round(retrieval_latency + generation_latency, 3),
    }


# ── aggregate statistics ──────────────────────────────────────────────────────

def _aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    def avg(key):
        return round(sum(r[key] for r in results) / n, 4)

    def frac(key):
        return round(sum(1 for r in results if r[key]) / n, 4)

    # MRR: mean reciprocal rank of the first gold chunk
    mrr_values = [1.0 / r["gold_rank"] if r["gold_rank"] else 0.0 for r in results]
    mrr = round(sum(mrr_values) / n, 4)

    return {
        "n": n,
        "avg_recall_at_k": avg("recall_at_k"),
        "mrr": mrr,
        "pct_grounded": frac("grounded"),
        "avg_term_coverage": avg("term_coverage"),
        "pct_no_answer_correct": frac("no_answer_correct"),
        "avg_evidence_tokens": avg("evidence_token_count"),
        "avg_answer_tokens": avg("answer_token_count"),
        "avg_retrieval_latency_s": avg("retrieval_latency_s"),
        "avg_generation_latency_s": avg("generation_latency_s"),
        "avg_total_latency_s": avg("total_latency_s"),
    }


# ── printing ──────────────────────────────────────────────────────────────────

def _print_table(results_a: list[dict], results_b: list[dict] | None) -> None:
    """Print per-question + aggregate table to stdout."""
    agg_a = _aggregate(results_a)
    agg_b = _aggregate(results_b) if results_b else None

    pipelines = ["A"] + (["B"] if results_b else [])
    result_map = {"A": results_a, "B": results_b or []}

    # ── per-question ──────────────────────────────────────────────────────────
    print()
    print("=" * 110)
    print("PIPELINE A vs B — DEVELOPMENT SET EVALUATION")
    print("=" * 110)

    # Header
    hdr_cols = ["QID", "Type", "Ret", "Rank", "Recall", "Grnd", "TermCov", "NoAns", "RetL(s)", "GenL(s)"]
    for pl in pipelines:
        w = 74 if pl == "A" else 60
        print(f"\n{'Pipeline ' + pl + ':':^{w}}")
        print(f"{'QID':<6} {'Type':<12} {'Ret':>3} {'Rank':>4} {'Recall':>6} {'Grnd':>4} {'TermCov':>7} {'NoAns':>5} {'RetL':>6} {'GenL':>6}")
        print("-" * 62)
        for r in result_map[pl]:
            rank_s = str(r["gold_rank"]) if r["gold_rank"] else "—"
            grnd_s = "Y" if r["grounded"] else "N"
            no_ans_s = "Y" if r["no_answer_correct"] else "N"
            recall_s = f"{r['recall_at_k']:.2f}"
            term_s = f"{r['term_coverage']:.2f}"
            print(
                f"{r['query_id']:<6} {r['type']:<12} {r['retrieved_count']:>3} "
                f"{rank_s:>4} {recall_s:>6} {grnd_s:>4} {term_s:>7} {no_ans_s:>5} "
                f"{r['retrieval_latency_s']:>6.2f} {r['generation_latency_s']:>6.1f}"
            )

    # ── aggregates ────────────────────────────────────────────────────────────
    print()
    print("=" * 110)
    print("AGGREGATE RESULTS")
    print("=" * 110)
    fmt_row = "{:<22} {:>10} {:>10}"
    header = fmt_row.format("Metric", "Pipeline A", "Pipeline B" if agg_b else "—")
    print(header)
    print("-" * 45)
    metrics = [
        ("Recall@K (avg)", "avg_recall_at_k"),
        ("MRR", "mrr"),
        ("% Grounded", "pct_grounded"),
        ("Term coverage (avg)", "avg_term_coverage"),
        ("% No-answer correct", "pct_no_answer_correct"),
        ("Avg evidence tokens", "avg_evidence_tokens"),
        ("Avg answer tokens", "avg_answer_tokens"),
        ("Avg retrieval (s)", "avg_retrieval_latency_s"),
        ("Avg generation (s)", "avg_generation_latency_s"),
        ("Avg total (s)", "avg_total_latency_s"),
    ]
    for label, key in metrics:
        val_a = f"{agg_a[key]:.4f}" if isinstance(agg_a.get(key), float) else str(agg_a.get(key, "—"))
        val_b = f"{agg_b[key]:.4f}" if (agg_b and isinstance(agg_b.get(key), float)) else (str(agg_b.get(key, "—")) if agg_b else "—")
        print(fmt_row.format(label, val_a, val_b))
    print("=" * 110)


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GraphRAG-Lite accuracy/efficiency evaluation harness (Task 8.2)"
    )
    parser.add_argument(
        "--pipeline", choices=["a", "b", "both"], default="both",
        help="Which retrieval pipeline(s) to run. Default: both."
    )
    parser.add_argument(
        "--ranking", default=None,
        help="Override RANKING_METHOD (pagerank|degree). Default: from config."
    )
    parser.add_argument(
        "--format", choices=["toon", "json"], default=None, dest="fmt",
        help="Evidence serialization format. Default: from config."
    )
    parser.add_argument(
        "--model", default=None,
        help="Override SLM_MODEL. Default: from config."
    )
    parser.add_argument(
        "--out", default=None,
        help="Path to write JSON results. Default: eval/results/<timestamp>.json."
    )
    parser.add_argument(
        "--queries", default=None,
        help="Comma-separated list of query IDs to run (e.g. D01,D02). Default: all."
    )
    args = parser.parse_args(argv)

    from graphrag_lite.config import get_config
    cfg = get_config()

    ranking = args.ranking or cfg.RANKING_METHOD
    fmt = args.fmt or cfg.SERIALIZATION_FORMAT
    model = args.model or cfg.SLM_MODEL

    print(f"[eval] model={model}  ranking={ranking}  format={fmt}  pipeline={args.pipeline}")
    print(f"[eval] Loading gold dev set from {GOLD_DEV_PATH}")

    queries = _load_dev_set()
    if args.queries:
        target = set(args.queries.split(","))
        queries = [q for q in queries if q["query_id"] in target]
        print(f"[eval] Filtered to {len(queries)} queries: {args.queries}")
    else:
        print(f"[eval] {len(queries)} queries")

    run_pipelines = (
        ["a", "b"] if args.pipeline == "both"
        else [args.pipeline]
    )

    all_results: dict[str, list[dict]] = {}
    for pl in run_pipelines:
        print(f"\n[eval] === Pipeline {pl.upper()} ===")
        results = []
        for i, q in enumerate(queries, 1):
            qid = q["query_id"]
            print(f"  [{i:02d}/{len(queries)}] {qid} ({q['type']}) ...", end="", flush=True)
            try:
                r = run_one(q, pipeline=pl, fmt=fmt, ranking_method=ranking, model=model)
                results.append(r)
                print(
                    f" recall={r['recall_at_k']:.2f}  grnd={'Y' if r['grounded'] else 'N'}"
                    f"  term={r['term_coverage']:.2f}  gen={r['generation_latency_s']:.1f}s"
                )
            except Exception as exc:
                print(f" ERROR: {type(exc).__name__}: {exc}")
                results.append({
                    "query_id": qid,
                    "type": q.get("type"),
                    "pipeline": pl,
                    "model": model,
                    "format": fmt,
                    "ranking_method": ranking,
                    "error": str(exc),
                    "recall_at_k": 0.0,
                    "gold_rank": None,
                    "grounded": False,
                    "hallucinated_ids": [],
                    "term_coverage": 0.0,
                    "no_answer_predicted": False,
                    "no_answer_correct": False,
                    "answer_text": "",
                    "retrieved_count": 0,
                    "evidence_ids": [],
                    "gold_chunk_ids": q.get("gold_chunk_ids", []),
                    "cited_ids": [],
                    "evidence_token_count": 0,
                    "answer_token_count": 0,
                    "retrieval_latency_s": 0.0,
                    "generation_latency_s": 0.0,
                    "total_latency_s": 0.0,
                })
        all_results[pl] = results

    # ── Print table ───────────────────────────────────────────────────────────
    _print_table(
        all_results.get("a", []),
        all_results.get("b") if "b" in all_results else None,
    )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"eval_{ts}.json"
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "ranking_method": ranking,
        "format": fmt,
        "pipeline": args.pipeline,
        "results": all_results,
        "aggregates": {pl: _aggregate(r) for pl, r in all_results.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[eval] Results saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
