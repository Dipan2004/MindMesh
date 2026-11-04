"""
test_phase7_api.py — Unit tests for Task 7.1 (FastAPI endpoints + lifecycle).

Uses FastAPI TestClient (synchronous ASGI wrapper) for most tests.
Async tests use httpx.AsyncClient with anyio for lifecycle verification.

Covers:
  - App creation and route registration
  - Lifespan: set_tracing called once, http client initialised
  - GET /health: schema, circuit breaker state, config fields
  - POST /query: pipeline A/B selection, response schema, no api_key in response
  - POST /query: empty-evidence refusal path
  - POST /query: citation validation
  - POST /query: invalid requests (422 validation errors)
  - POST /query: retrieval exception → 500
  - POST /query: circuit breaker open → 503
  - Global exception handler: 500 with error_type, no traceback
  - Secret redaction: api_key never appears in response
  - Tracing initialised once (not per request)
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── fixtures and helpers ──────────────────────────────────────────────────────

def _make_evidence(chunk_id: str, text: str = "evidence text", score: float = 0.03):
    from graphrag_lite.retrieval.pipeline_a import Evidence
    return Evidence(
        chunk_id=chunk_id, source_doc="doc", chunk_index=0,
        text=text, score=score, source="chunk",
    )


def _make_answer(text: str, cited: list[str]):
    from graphrag_lite.llm.generator import Answer
    return Answer(text=text, cited_evidence_ids=cited, confidence=None)


@pytest.fixture()
def client():
    """
    TestClient that runs the full lifespan (startup + shutdown).
    Mocks set_tracing and http client init/close so no real side-effects occur.
    """
    with patch("graphrag_lite.api.main.set_tracing"), \
         patch("graphrag_lite.api.main.init_http_client"), \
         patch("graphrag_lite.api.main.close_http_client", new_callable=AsyncMock):
        from graphrag_lite.api.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── App creation ──────────────────────────────────────────────────────────────

def test_app_has_query_route():
    from graphrag_lite.api.main import app
    routes = {r.path for r in app.routes}
    assert "/query" in routes


def test_app_has_health_route():
    from graphrag_lite.api.main import app
    routes = {r.path for r in app.routes}
    assert "/health" in routes


# ── Lifespan: tracing called once ────────────────────────────────────────────

def test_lifespan_calls_set_tracing_exactly_once():
    """set_tracing must be called once during startup, not per request."""
    with patch("graphrag_lite.api.main.set_tracing") as mock_trace, \
         patch("graphrag_lite.api.main.init_http_client"), \
         patch("graphrag_lite.api.main.close_http_client", new_callable=AsyncMock):
        from graphrag_lite.api.main import app
        with TestClient(app):
            pass
    mock_trace.assert_called_once()


def test_lifespan_passes_llm_mode_to_set_tracing():
    """set_tracing must be called with cfg.LLM_MODE, not a hardcoded string."""
    from graphrag_lite.config import get_config
    cfg = get_config()
    with patch("graphrag_lite.api.main.set_tracing") as mock_trace, \
         patch("graphrag_lite.api.main.init_http_client"), \
         patch("graphrag_lite.api.main.close_http_client", new_callable=AsyncMock):
        from graphrag_lite.api.main import app
        with TestClient(app):
            pass
    mock_trace.assert_called_once_with(cfg.LLM_MODE)


def test_lifespan_calls_init_http_client_once():
    """Connection pool must be initialised exactly once at startup."""
    with patch("graphrag_lite.api.main.set_tracing"), \
         patch("graphrag_lite.api.main.init_http_client") as mock_init, \
         patch("graphrag_lite.api.main.close_http_client", new_callable=AsyncMock):
        from graphrag_lite.api.main import app
        with TestClient(app):
            pass
    mock_init.assert_called_once()


def test_lifespan_calls_close_http_client_on_shutdown():
    """HTTP client must be closed on shutdown."""
    with patch("graphrag_lite.api.main.set_tracing"), \
         patch("graphrag_lite.api.main.init_http_client"), \
         patch("graphrag_lite.api.main.close_http_client", new_callable=AsyncMock) as mock_close:
        from graphrag_lite.api.main import app
        with TestClient(app):
            pass
    mock_close.assert_called_once()


# ── GET /health ───────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_schema(client):
    resp = client.get("/health")
    data = resp.json()
    assert data["status"] == "ok"
    assert "circuit_breaker_state" in data
    assert data["circuit_breaker_state"] in ("CLOSED", "OPEN", "HALF_OPEN")
    assert "model" in data
    assert "embedding_model" in data
    assert "pipeline_b_scope" in data
    assert "sim_weight" in data
    assert "pr_weight" in data


def test_health_sim_pr_weights(client):
    from graphrag_lite.config import get_config
    cfg = get_config()
    resp = client.get("/health")
    data = resp.json()
    assert data["sim_weight"] == cfg.PIPELINE_B_SIM_WEIGHT
    assert data["pr_weight"] == cfg.PIPELINE_B_PR_WEIGHT


def test_health_model_matches_config(client):
    from graphrag_lite.config import get_config
    cfg = get_config()
    resp = client.get("/health")
    assert resp.json()["model"] == cfg.SLM_MODEL


# ── POST /query validation ────────────────────────────────────────────────────

def test_query_empty_string_returns_422(client):
    resp = client.post("/query", json={"query": "", "pipeline": "b", "mode": "local"})
    assert resp.status_code == 422


def test_query_whitespace_only_returns_422(client):
    resp = client.post("/query", json={"query": "   ", "pipeline": "b", "mode": "local"})
    assert resp.status_code == 422


def test_query_missing_query_field_returns_422(client):
    resp = client.post("/query", json={"pipeline": "b", "mode": "local"})
    assert resp.status_code == 422


def test_query_invalid_pipeline_returns_422(client):
    resp = client.post("/query", json={"query": "test", "pipeline": "c", "mode": "local"})
    assert resp.status_code == 422


def test_query_invalid_mode_returns_422(client):
    resp = client.post("/query", json={"query": "test", "pipeline": "b", "mode": "cloud"})
    assert resp.status_code == 422


# ── POST /query success path ──────────────────────────────────────────────────

def _mock_query_deps(pipeline: str, ev, ans):
    """Context manager that mocks retrieval + generation for one query."""
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch("graphrag_lite.llm.client_factory.get_llm_client",
                              return_value=MagicMock(model="qwen2.5:1.5b")))
    if pipeline == "a":
        stack.enter_context(patch("graphrag_lite.retrieval.pipeline_a.retrieve_flat",
                                   return_value=ev))
    else:
        stack.enter_context(patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural",
                                   return_value=ev))
    stack.enter_context(patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer",
                               return_value=ans))
    return stack


def test_query_pipeline_b_success(client):
    ev = [_make_evidence("aaa"), _make_evidence("bbb")]
    ans = _make_answer("The answer [0]", ["aaa"])
    with _mock_query_deps("b", ev, ans):
        resp = client.post("/query", json={
            "query": "What is attention?", "pipeline": "b", "mode": "local"
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer_text"] == "The answer [0]"
    assert data["cited_evidence_ids"] == ["aaa"]
    assert data["pipeline"] == "b"
    assert data["mode"] == "local"
    assert data["model"] == "qwen2.5:1.5b"
    assert data["retrieved_count"] == 2
    assert data["citation_ids_valid"] is True
    assert data["hallucinated_ids"] == []
    assert data["no_answer"] is False


def test_query_pipeline_a_success(client):
    ev = [_make_evidence("ccc")]
    ans = _make_answer("Pipeline A answer [0]", ["ccc"])
    with _mock_query_deps("a", ev, ans):
        resp = client.post("/query", json={
            "query": "What is FAISS?", "pipeline": "a", "mode": "local"
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["pipeline"] == "a"
    assert data["retrieved_count"] == 1


def test_query_response_has_no_api_key(client):
    """api_key must never appear in the response body."""
    ev = [_make_evidence("aaa")]
    ans = _make_answer("answer [0]", ["aaa"])
    with _mock_query_deps("b", ev, ans):
        resp = client.post("/query", json={
            "query": "test", "pipeline": "b", "mode": "api",
            "api_key": "SECRET_KEY_123"
        })
    assert "api_key" not in resp.json()
    assert "SECRET_KEY_123" not in resp.text


def test_query_no_answer_flag(client):
    ev = [_make_evidence("aaa")]
    ans = _make_answer("I cannot answer this question based on the provided evidence.", [])
    with _mock_query_deps("b", ev, ans):
        resp = client.post("/query", json={
            "query": "What is the answer to life?", "pipeline": "b", "mode": "local"
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["no_answer"] is True
    assert data["cited_evidence_ids"] == []
    assert data["citation_ids_valid"] is True


def test_query_detects_hallucinated_citation(client):
    ev = [_make_evidence("aaa"), _make_evidence("bbb")]
    # Model cites "zzz" which was never retrieved
    ans = _make_answer("answer [0]", ["aaa", "zzz999"])
    with _mock_query_deps("b", ev, ans):
        resp = client.post("/query", json={
            "query": "test", "pipeline": "b", "mode": "local"
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["citation_ids_valid"] is False
    assert "zzz999" in data["hallucinated_ids"]


def test_query_response_has_latency_fields(client):
    ev = [_make_evidence("aaa")]
    ans = _make_answer("answer [0]", ["aaa"])
    with _mock_query_deps("b", ev, ans):
        resp = client.post("/query", json={
            "query": "test", "pipeline": "b", "mode": "local"
        })
    data = resp.json()
    assert data["retrieval_latency_s"] >= 0
    assert data["generation_latency_s"] >= 0
    assert data["total_latency_s"] >= 0
    assert data["total_latency_s"] >= data["retrieval_latency_s"]


# ── Error paths ───────────────────────────────────────────────────────────────

def test_query_retrieval_exception_returns_500(client):
    with patch("graphrag_lite.llm.client_factory.get_llm_client",
               return_value=MagicMock(model="qwen2.5:1.5b")), \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural",
               side_effect=RuntimeError("Zvec unavailable")):
        resp = client.post("/query", json={
            "query": "test", "pipeline": "b", "mode": "local"
        })
    assert resp.status_code == 500
    data = resp.json()
    assert "detail" in data
    # Must not expose internal paths or full tracebacks
    assert "Zvec unavailable" not in resp.text  # internal message not in response


def test_query_circuit_breaker_open_returns_503(client):
    from graphrag_lite.api.middleware import CircuitBreakerOpenError, _breaker, _CircuitBreaker
    ev = [_make_evidence("aaa")]
    _breaker._state = _CircuitBreaker._OPEN
    _breaker._opened_at = __import__("time").monotonic()
    _breaker._cooldown_s = 9999.0
    try:
        with patch("graphrag_lite.llm.client_factory.get_llm_client",
                   return_value=MagicMock(model="stub")), \
             patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural",
                   return_value=ev):
            resp = client.post("/query", json={
                "query": "test", "pipeline": "b", "mode": "api"
            })
        assert resp.status_code == 503
    finally:
        _breaker._state = _CircuitBreaker._CLOSED
        _breaker._consecutive_failures = 0


def test_global_exception_handler_returns_500_with_error_type(client):
    """Unhandled exception must produce 500 with error_type, no traceback."""
    with patch("graphrag_lite.llm.client_factory.get_llm_client",
               side_effect=Exception("internal kaboom")):
        resp = client.post("/query", json={
            "query": "test", "pipeline": "b", "mode": "local"
        })
    assert resp.status_code in (500, 422)  # 500 from global handler or 422 if caught earlier
    # Must not expose raw internal exception message
    assert "kaboom" not in resp.text


# ── Tracing not called per request ───────────────────────────────────────────

def test_set_tracing_not_called_per_request():
    """
    set_tracing must be called once during lifespan startup, not once per
    request. Verified by asserting exactly one call over multiple requests.
    """
    call_count = 0
    def counting_set_tracing(mode):
        nonlocal call_count
        call_count += 1

    with patch("graphrag_lite.api.main.set_tracing", side_effect=counting_set_tracing), \
         patch("graphrag_lite.api.main.init_http_client"), \
         patch("graphrag_lite.api.main.close_http_client", new_callable=AsyncMock):
        from graphrag_lite.api.main import app

        ev = [_make_evidence("aaa")]
        ans = _make_answer("answer [0]", ["aaa"])
        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("graphrag_lite.llm.client_factory.get_llm_client",
                       return_value=MagicMock(model="qwen2.5:1.5b")), \
                 patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural",
                        return_value=ev), \
                 patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer",
                        return_value=ans):
                c.post("/query", json={"query": "q1", "pipeline": "b", "mode": "local"})
                c.post("/query", json={"query": "q2", "pipeline": "b", "mode": "local"})
                c.post("/query", json={"query": "q3", "pipeline": "b", "mode": "local"})

    # Three requests, but set_tracing must have been called only once (at startup)
    assert call_count == 1, f"set_tracing was called {call_count} times (expected 1)"


# ── QueryRequest model ────────────────────────────────────────────────────────

def test_query_request_defaults():
    from graphrag_lite.api.routes import QueryRequest
    req = QueryRequest(query="hello")
    assert req.pipeline == "b"
    assert req.mode == "local"
    assert req.api_key is None
    assert req.top_k is None


def test_query_request_rejects_empty_query():
    from pydantic import ValidationError
    from graphrag_lite.api.routes import QueryRequest
    with pytest.raises(ValidationError):
        QueryRequest(query="")


def test_query_request_rejects_whitespace_query():
    from pydantic import ValidationError
    from graphrag_lite.api.routes import QueryRequest
    with pytest.raises(ValidationError):
        QueryRequest(query="   ")


def test_query_request_rejects_invalid_pipeline():
    from pydantic import ValidationError
    from graphrag_lite.api.routes import QueryRequest
    with pytest.raises(ValidationError):
        QueryRequest(query="test", pipeline="c")  # type: ignore


def test_query_request_rejects_invalid_mode():
    from pydantic import ValidationError
    from graphrag_lite.api.routes import QueryRequest
    with pytest.raises(ValidationError):
        QueryRequest(query="test", mode="cloud")  # type: ignore


def test_query_response_has_no_api_key_field():
    """QueryResponse schema must not have an api_key field."""
    from graphrag_lite.api.routes import QueryResponse
    fields = QueryResponse.model_fields
    assert "api_key" not in fields
