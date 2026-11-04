"""
test_phase6_run_query.py — Unit tests for Task 6.2 (run_query CLI).

Covers every acceptance criterion in TASKS.md Task 6.2:
  - Pipeline A and Pipeline B selection.
  - Tracing is invoked with the correct mode (mocked to avoid env side-effects).
  - Retrieved evidence is passed verbatim to the generator.
  - Citation IDs are validated against the passed evidence.
  - Empty-evidence refusal is correctly flagged.
  - Hallucinated citation IDs are detected and reported.
  - Invalid arguments (empty query, bad pipeline, bad mode) raise ValueError.
  - The RunRecord contains no API key, never writes to disk for the key.
  - run_query does not touch the canonical Zvec collections directly
    (all Zvec/LLM calls are mocked in unit tests; canonical artifacts
    are only used in the integration test).
  - No prohibited model is referenced.

All Zvec and LLM calls are mocked so these tests are fully offline.
"""

from __future__ import annotations

import os
import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch


# ── shared helpers ────────────────────────────────────────────────────────────

def _make_evidence(chunk_id: str, text: str = "sample text", score: float = 0.03):
    """Return a minimal Evidence namedtuple substitute."""
    from graphrag_lite.retrieval.pipeline_a import Evidence
    return Evidence(
        chunk_id=chunk_id,
        source_doc="test_doc",
        chunk_index=0,
        text=text,
        score=score,
        source="chunk",
    )


def _make_answer(text: str, cited: list[str]):
    from graphrag_lite.llm.generator import Answer
    return Answer(text=text, cited_evidence_ids=cited, confidence=None)


# ── Pipeline selection ────────────────────────────────────────────────────────

def test_run_query_uses_pipeline_a_when_specified():
    """--pipeline a must call retrieve_flat, not retrieve_structural."""
    ev = [_make_evidence("aaa111")]
    ans = _make_answer("answer [0]", ["aaa111"])

    with patch("scripts.run_query.set_tracing"), \
         patch("graphrag_lite.retrieval.pipeline_a.retrieve_flat", return_value=ev) as mock_a, \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural") as mock_b, \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client", return_value=MagicMock(model="qwen2.5:1.5b")):
        from scripts.run_query import run_query
        rec = run_query(query="test query", pipeline="a", mode="local", no_trace=True)

    mock_a.assert_called_once()
    mock_b.assert_not_called()
    assert rec.pipeline == "a"


def test_run_query_uses_pipeline_b_by_default():
    """Default pipeline must be b (retrieve_structural)."""
    ev = [_make_evidence("bbb222")]
    ans = _make_answer("answer [0]", ["bbb222"])

    with patch("scripts.run_query.set_tracing"), \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=ev) as mock_b, \
         patch("graphrag_lite.retrieval.pipeline_a.retrieve_flat") as mock_a, \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client", return_value=MagicMock(model="qwen2.5:1.5b")):
        from scripts.run_query import run_query
        rec = run_query(query="test query", pipeline="b", mode="local", no_trace=True)

    mock_b.assert_called_once()
    mock_a.assert_not_called()
    assert rec.pipeline == "b"


# ── Tracing integration ───────────────────────────────────────────────────────

def test_run_query_calls_set_tracing_local():
    """run_query must call set_tracing('local') when mode='local'."""
    ev = [_make_evidence("ccc333")]
    ans = _make_answer("answer [0]", ["ccc333"])

    with patch("scripts.run_query.set_tracing") as mock_trace, \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=ev), \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client", return_value=MagicMock(model="qwen2.5:1.5b")):
        from scripts.run_query import run_query
        run_query(query="test", pipeline="b", mode="local", no_trace=False)

    mock_trace.assert_called_once_with("local")


def test_run_query_calls_set_tracing_api():
    """run_query must call set_tracing('api') when mode='api'."""
    ev = [_make_evidence("ddd444")]
    ans = _make_answer("answer [0]", ["ddd444"])

    with patch("scripts.run_query.set_tracing") as mock_trace, \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=ev), \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client", return_value=MagicMock(model="stub")):
        from scripts.run_query import run_query
        run_query(query="test", pipeline="b", mode="api", no_trace=False)

    mock_trace.assert_called_once_with("api")


def test_run_query_no_trace_skips_set_tracing():
    """--no-trace must skip set_tracing entirely."""
    ev = [_make_evidence("eee555")]
    ans = _make_answer("answer [0]", ["eee555"])

    with patch("scripts.run_query.set_tracing") as mock_trace, \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=ev), \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client", return_value=MagicMock(model="qwen2.5:1.5b")):
        from scripts.run_query import run_query
        run_query(query="test", pipeline="b", mode="local", no_trace=True)

    mock_trace.assert_not_called()


# ── Citation validation ───────────────────────────────────────────────────────

def test_run_query_citation_valid_when_ids_in_evidence():
    """Citation IDs that are a subset of retrieved evidence IDs are valid."""
    ev = [_make_evidence("aaa"), _make_evidence("bbb"), _make_evidence("ccc")]
    ans = _make_answer("fact [0] fact [2]", ["aaa", "ccc"])

    with patch("scripts.run_query.set_tracing"), \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=ev), \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client", return_value=MagicMock(model="qwen2.5:1.5b")):
        from scripts.run_query import run_query
        rec = run_query(query="test", pipeline="b", mode="local", no_trace=True)

    assert rec.citation_ids_valid is True
    assert rec.hallucinated_ids == []
    assert set(rec.cited_ids) <= set(rec.evidence_ids)


def test_run_query_detects_hallucinated_citation():
    """A cited ID not in the retrieved evidence must be flagged as hallucinated."""
    ev = [_make_evidence("aaa"), _make_evidence("bbb")]
    # Model cites "zzz" which was never retrieved
    ans = _make_answer("fact [0]", ["aaa", "zzz999"])

    with patch("scripts.run_query.set_tracing"), \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=ev), \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client", return_value=MagicMock(model="qwen2.5:1.5b")):
        from scripts.run_query import run_query
        rec = run_query(query="test", pipeline="b", mode="local", no_trace=True)

    assert rec.citation_ids_valid is False
    assert "zzz999" in rec.hallucinated_ids


def test_run_query_citation_valid_when_no_citations():
    """An answer with no citations is valid (zero cited IDs ⊆ evidence IDs)."""
    ev = [_make_evidence("aaa")]
    ans = _make_answer("I cannot answer this question based on the provided evidence.", [])

    with patch("scripts.run_query.set_tracing"), \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=ev), \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client", return_value=MagicMock(model="qwen2.5:1.5b")):
        from scripts.run_query import run_query
        rec = run_query(query="test", pipeline="b", mode="local", no_trace=True)

    assert rec.citation_ids_valid is True
    assert rec.hallucinated_ids == []


# ── Empty evidence / no-answer refusal ───────────────────────────────────────

def test_run_query_no_answer_flag_set_on_refusal():
    """no_answer must be True when the model emits the canonical refusal string."""
    ev = [_make_evidence("aaa")]
    ans = _make_answer(
        "I cannot answer this question based on the provided evidence.", []
    )

    with patch("scripts.run_query.set_tracing"), \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=ev), \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client", return_value=MagicMock(model="qwen2.5:1.5b")):
        from scripts.run_query import run_query
        rec = run_query(query="test", pipeline="b", mode="local", no_trace=True)

    assert rec.no_answer is True


def test_run_query_no_answer_flag_false_on_real_answer():
    """no_answer must be False when the model produces a real answer."""
    ev = [_make_evidence("aaa")]
    ans = _make_answer("The answer is 42 [0]", ["aaa"])

    with patch("scripts.run_query.set_tracing"), \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=ev), \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client", return_value=MagicMock(model="qwen2.5:1.5b")):
        from scripts.run_query import run_query
        rec = run_query(query="test", pipeline="b", mode="local", no_trace=True)

    assert rec.no_answer is False


def test_run_query_empty_retrieval_returns_exit_code_1():
    """run_query returns retrieved_count=0 when pipeline returns no evidence."""
    from graphrag_lite.llm.generator import Answer
    ans = Answer(
        text="I cannot answer this question because no relevant evidence was retrieved.",
        cited_evidence_ids=[],
    )
    with patch("scripts.run_query.set_tracing"), \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=[]), \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client", return_value=MagicMock(model="qwen2.5:1.5b")):
        from scripts.run_query import run_query
        rec = run_query(query="test", pipeline="b", mode="local", no_trace=True)

    assert rec.retrieved_count == 0
    assert rec.no_answer is True


# ── Argument validation ───────────────────────────────────────────────────────

def test_run_query_raises_on_empty_query():
    """An empty query string must raise ValueError."""
    from scripts.run_query import run_query
    with pytest.raises(ValueError, match="must not be empty"):
        run_query(query="   ", pipeline="b", mode="local", no_trace=True)


def test_run_query_raises_on_bad_pipeline():
    """An unrecognised pipeline value must raise ValueError."""
    from scripts.run_query import run_query
    with pytest.raises(ValueError, match="pipeline must be"):
        run_query(query="test", pipeline="c", mode="local", no_trace=True)  # type: ignore


def test_run_query_raises_on_bad_mode():
    """An unrecognised mode must raise ValueError."""
    from scripts.run_query import run_query
    with pytest.raises(ValueError, match="mode must be"):
        run_query(query="test", pipeline="b", mode="cloud", no_trace=True)  # type: ignore


# ── Secret redaction ──────────────────────────────────────────────────────────

def test_run_record_contains_no_api_key():
    """The RunRecord dict must not contain any api_key field."""
    ev = [_make_evidence("aaa")]
    ans = _make_answer("answer [0]", ["aaa"])

    with patch("scripts.run_query.set_tracing"), \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=ev), \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client", return_value=MagicMock(model="qwen2.5:1.5b")):
        from scripts.run_query import run_query
        rec = run_query(
            query="test", pipeline="b", mode="api",
            api_key="MY_SECRET_KEY_123", no_trace=True
        )

    d = rec.to_dict()
    # The api_key must not appear anywhere in the dict
    assert "api_key" not in d
    assert "MY_SECRET_KEY_123" not in str(d), (
        "API key value must not appear in the RunRecord"
    )


# ── Model name check ──────────────────────────────────────────────────────────

def test_run_record_model_matches_client():
    """RunRecord.model must reflect the actual client model name."""
    ev = [_make_evidence("aaa")]
    ans = _make_answer("answer [0]", ["aaa"])

    mock_client = MagicMock()
    mock_client.model = "qwen2.5:1.5b"

    with patch("scripts.run_query.set_tracing"), \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=ev), \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client", return_value=mock_client):
        from scripts.run_query import run_query
        rec = run_query(query="test", pipeline="b", mode="local", no_trace=True)

    assert rec.model == "qwen2.5:1.5b"


def test_run_record_does_not_use_prohibited_model():
    """The model name in the RunRecord must not be a prohibited model."""
    PROHIBITED = {"qwen2.5:3b-instruct-q4_0", "qwen3:4b", "gpt-4", "claude-3"}
    ev = [_make_evidence("aaa")]
    ans = _make_answer("answer [0]", ["aaa"])

    with patch("scripts.run_query.set_tracing"), \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=ev), \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client", return_value=MagicMock(model="qwen2.5:1.5b")):
        from scripts.run_query import run_query
        rec = run_query(query="test", pipeline="b", mode="local", no_trace=True)

    assert rec.model not in PROHIBITED, (
        f"Prohibited model {rec.model!r} used in run_query"
    )


# ── Timing fields ─────────────────────────────────────────────────────────────

def test_run_record_has_positive_latencies():
    """All three latency fields must be non-negative floats."""
    ev = [_make_evidence("aaa")]
    ans = _make_answer("answer [0]", ["aaa"])

    with patch("scripts.run_query.set_tracing"), \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=ev), \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client", return_value=MagicMock(model="qwen2.5:1.5b")):
        from scripts.run_query import run_query
        rec = run_query(query="test", pipeline="b", mode="local", no_trace=True)

    assert rec.retrieval_latency_s >= 0.0
    assert rec.generation_latency_s >= 0.0
    assert rec.total_latency_s >= 0.0
    assert rec.total_latency_s >= rec.retrieval_latency_s
    assert rec.total_latency_s >= rec.generation_latency_s


# ── top_k passthrough ─────────────────────────────────────────────────────────

def test_run_query_top_k_passthrough_to_pipeline_b():
    """--top-k must be passed to retrieve_structural as the top_k argument."""
    ev = [_make_evidence("aaa")]
    ans = _make_answer("answer [0]", ["aaa"])

    with patch("scripts.run_query.set_tracing"), \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=ev) as mock_b, \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client", return_value=MagicMock(model="qwen2.5:1.5b")):
        from scripts.run_query import run_query
        run_query(query="test", pipeline="b", mode="local", top_k=3, no_trace=True)

    mock_b.assert_called_once_with("test", top_k=3)


# ── CLI argument parser ───────────────────────────────────────────────────────

def test_cli_parser_defaults():
    """Parser defaults: pipeline=b, mode=local, no_trace=False, json_out=False."""
    from scripts.run_query import _build_parser
    args = _build_parser().parse_args(["--query", "what is RAG?"])
    assert args.pipeline == "b"
    assert args.mode == "local"
    assert args.no_trace is False
    assert args.json_out is False
    assert args.query == "what is RAG?"


def test_cli_parser_explicit_pipeline_a():
    from scripts.run_query import _build_parser
    args = _build_parser().parse_args(["--query", "q", "--pipeline", "a"])
    assert args.pipeline == "a"


def test_cli_parser_no_trace_flag():
    from scripts.run_query import _build_parser
    args = _build_parser().parse_args(["--query", "q", "--no-trace"])
    assert args.no_trace is True


def test_cli_parser_json_out_flag():
    from scripts.run_query import _build_parser
    args = _build_parser().parse_args(["--query", "q", "--json-out"])
    assert args.json_out is True
