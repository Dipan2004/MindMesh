"""
routes.py — FastAPI route handlers for Phase 7 (Task 7.1 + 7.2).

Endpoints
---------
  POST /query   — run a query through Pipeline A or B, return a grounded
                  answer with cited evidence IDs.
  GET  /health  — liveness probe; returns process status and breaker state.

Request / response models
-------------------------
  QueryRequest   : pipeline, mode, api_key (optional), query
  QueryResponse  : answer_text, cited_evidence_ids, no_answer, pipeline,
                   model, retrieved_count, citation_ids_valid,
                   hallucinated_ids, retrieval_latency_s,
                   generation_latency_s, total_latency_s
  HealthResponse : status, circuit_breaker_state, model, embedding_model,
                   pipeline_b_scope, sim_weight, pr_weight

Design notes
------------
- CPU-bound graph work (bfs_expand, rank_pagerank inside retrieve_structural)
  is offloaded to a thread pool via ``run_in_threadpool`` from middleware.py
  so the event loop stays responsive for concurrent requests (Task 7.2).
- Outbound Grok calls for mode="api" go through ``api_call_with_retry``
  from middleware.py, protected by the circuit breaker (Task 7.3).
- For mode="local" the Ollama generate() call is also offloaded to a thread
  because it is a blocking HTTP call to a local server.
- Citation validation is identical to run_query.py — it is not re-invented,
  just inlined (same logic, same _NO_ANSWER_MARKER).
- The api_key from QueryRequest is never written to logs, responses, or the
  RunRecord-equivalent response. It is only forwarded to client_factory.
"""

from __future__ import annotations

import logging
import time
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from graphrag_lite.api.middleware import (
    CircuitBreakerOpenError,
    api_call_with_retry,
    get_circuit_breaker,
    run_in_threadpool,
)
from graphrag_lite.config import get_config

log = logging.getLogger(__name__)

router = APIRouter()

_NO_ANSWER_MARKER = "i cannot answer this question"


# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    Request body for POST /query.

    Fields
    ------
    pipeline : "a" | "b"   Retrieval pipeline.  Default "b".
    mode     : "local" | "api"  LLM mode.  Default "local".
    api_key  : Optional session-only key for mode="api".
               Never echoed back in the response.
    query    : Non-empty natural language question.
    top_k    : Optional override for TOP_N_FINAL_EVIDENCE.
    """
    pipeline: Literal["a", "b"] = Field(
        default="b",
        description="Retrieval pipeline: 'a' (flat) or 'b' (structural).",
    )
    mode: Literal["local", "api"] = Field(
        default="local",
        description="LLM mode: 'local' (Ollama) or 'api' (Grok stub).",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Session-only API key for mode='api'. Not stored or logged.",
    )
    query: str = Field(
        ...,
        min_length=1,
        description="Natural language question (required, non-empty).",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Override for TOP_N_FINAL_EVIDENCE.",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return v


class QueryResponse(BaseModel):
    """
    Response body for POST /query.

    api_key is intentionally absent — it is a secret and must never appear
    in the response.
    """
    answer_text: str
    cited_evidence_ids: list[str]
    no_answer: bool
    pipeline: str
    mode: str
    model: str
    retrieved_count: int
    citation_ids_valid: bool
    hallucinated_ids: list[str]
    retrieval_latency_s: float
    generation_latency_s: float
    total_latency_s: float


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str                   # "ok"
    circuit_breaker_state: str    # "CLOSED" | "OPEN" | "HALF_OPEN"
    model: str
    embedding_model: str
    pipeline_b_scope: str
    sim_weight: float
    pr_weight: float


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_query_response(
    pipeline: str,
    mode: str,
    model: str,
    evidence: list,
    answer_text: str,
    cited_ids: list[str],
    retrieval_latency: float,
    generation_latency: float,
    total_latency: float,
) -> QueryResponse:
    """Validate citations and build the response model."""
    passed_ids_set = {e.chunk_id for e in evidence}
    cited_set = set(cited_ids)
    hallucinated = sorted(cited_set - passed_ids_set)
    return QueryResponse(
        answer_text=answer_text,
        cited_evidence_ids=cited_ids,
        no_answer=_NO_ANSWER_MARKER in answer_text.lower(),
        pipeline=pipeline,
        mode=mode,
        model=model,
        retrieved_count=len(evidence),
        citation_ids_valid=len(hallucinated) == 0,
        hallucinated_ids=hallucinated,
        retrieval_latency_s=round(retrieval_latency, 3),
        generation_latency_s=round(generation_latency, 3),
        total_latency_s=round(total_latency, 3),
    )


# ── POST /query ───────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Run a RAG query through Pipeline A or B",
    responses={
        400: {"description": "Invalid request (empty query, bad parameters)"},
        503: {"description": "Circuit breaker open — API backend unavailable"},
        500: {"description": "Internal server error"},
    },
)
async def query_endpoint(req: QueryRequest, request: Request) -> QueryResponse:
    """
    Execute a retrieval-augmented query and return a grounded answer.

    CPU-bound graph operations are offloaded to a thread pool.
    For mode='api', the LLM call goes through the circuit-breaker wrapper.
    """
    total_start = time.monotonic()
    cfg = get_config()

    # ── 1. LLM client ─────────────────────────────────────────────────────
    from graphrag_lite.llm.client_factory import get_llm_client
    llm_client = get_llm_client(mode=req.mode, api_key=req.api_key)

    # ── 2. Retrieval (CPU-bound graph work → thread pool) ─────────────────
    retrieval_start = time.monotonic()
    try:
        if req.pipeline == "a":
            from graphrag_lite.retrieval.pipeline_a import retrieve_flat
            evidence = await run_in_threadpool(retrieve_flat, req.query)
            if req.top_k is not None:
                evidence = evidence[: req.top_k]
        else:
            from graphrag_lite.retrieval.pipeline_b import retrieve_structural
            evidence = await run_in_threadpool(
                retrieve_structural, req.query, req.top_k
            )
    except Exception as exc:
        log.exception("Retrieval failed for query %r", req.query[:60])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {type(exc).__name__}",
        ) from exc
    retrieval_latency = time.monotonic() - retrieval_start

    # ── 3. Evidence serialisation (for prompt) ────────────────────────────
    evidence_dicts = [
        {"text": e.text, "chunk_id": e.chunk_id}
        for e in evidence
    ]

    # ── 4. Generation ──────────────────────────────────────────────────────
    # For mode="api": wrap in circuit breaker + retry.
    # For mode="local": offload blocking Ollama HTTP call to thread pool.
    from graphrag_lite.llm.generator import AnswerGenerator
    gen = AnswerGenerator(client=llm_client)

    generation_start = time.monotonic()
    try:
        if req.mode == "api":
            answer = await api_call_with_retry(
                lambda: _run_generate_async(gen, req.query, evidence_dicts),
                label=f"generate/{req.query[:40]}",
            )
        else:
            answer = await run_in_threadpool(
                gen.generate_answer, req.query, evidence_dicts
            )
    except CircuitBreakerOpenError as exc:
        log.error("Circuit breaker open: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API backend unavailable (circuit breaker open).",
        ) from exc
    except Exception as exc:
        log.exception("Generation failed for query %r", req.query[:60])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation error: {type(exc).__name__}",
        ) from exc
    generation_latency = time.monotonic() - generation_start
    total_latency = time.monotonic() - total_start

    return _build_query_response(
        pipeline=req.pipeline,
        mode=req.mode,
        model=llm_client.model,
        evidence=evidence,
        answer_text=answer.text,
        cited_ids=answer.cited_evidence_ids,
        retrieval_latency=retrieval_latency,
        generation_latency=generation_latency,
        total_latency=total_latency,
    )


async def _run_generate_async(gen, query: str, evidence_dicts: list[dict]):
    """
    Thin async wrapper used only for the api_call_with_retry path.

    AnswerGenerator.generate_answer is synchronous; we offload it to a
    thread even in the api path so api_call_with_retry stays async.
    """
    return await run_in_threadpool(gen.generate_answer, query, evidence_dicts)


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe — returns process and breaker state",
)
async def health_endpoint() -> HealthResponse:
    """
    Return application health.

    Reports circuit breaker state so callers can distinguish a live but
    degraded server from a fully healthy one.  Does NOT probe external
    dependencies (Ollama, Zvec) — that would add latency and side-effects
    to every health check.  If you need a deep health check, add a separate
    /ready endpoint in Phase 8.
    """
    cfg = get_config()
    breaker = get_circuit_breaker()
    return HealthResponse(
        status="ok",
        circuit_breaker_state=breaker.state,
        model=cfg.SLM_MODEL,
        embedding_model=cfg.EMBEDDING_MODEL,
        pipeline_b_scope=cfg.PIPELINE_B_SCOPE,
        sim_weight=cfg.PIPELINE_B_SIM_WEIGHT,
        pr_weight=cfg.PIPELINE_B_PR_WEIGHT,
    )
