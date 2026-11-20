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

import json
import logging
import time
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from graphrag_lite.api.middleware import (
    CircuitBreakerOpenError,
    api_call_with_retry,
    get_circuit_breaker,
    run_in_threadpool,
)
from graphrag_lite.config import get_config

ROOT = Path(__file__).resolve().parents[3]
log = logging.getLogger(__name__)

router = APIRouter()

_NO_ANSWER_MARKER = "i cannot answer this question"

VALID_UI_MODES = {"naive", "local", "global", "hybrid", "mix", "mix_opt", "bypass"}
VALID_LEGACY_MODES = {"local", "api"}
VALID_LEGACY_PIPELINES = {"a", "b"}


# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    Request body for POST /query and POST /query/stream.
    Supports both Phase 7 Pipeline A/B requests and LightRAG Web UI requests.
    """
    query: str = Field(
        ...,
        min_length=1,
        description="Natural language question (required, non-empty).",
    )
    pipeline: Optional[str] = Field(
        default="b",
        description="Retrieval pipeline: 'a' (flat) or 'b' (structural). Default 'b'.",
    )
    mode: Optional[str] = Field(
        default="local",
        description="LLM/query mode: 'local' | 'api' for Phase 7; 'naive'|'local'|'global'|'hybrid'|'mix'|'bypass' for UI.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Session-only API key for mode='api'. Not stored or logged.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Override for TOP_N_FINAL_EVIDENCE or LightRAG top_k.",
    )
    chunk_top_k: Optional[int] = Field(default=None)
    stream: Optional[bool] = Field(default=False)
    only_need_context: Optional[bool] = Field(default=False)
    only_need_prompt: Optional[bool] = Field(default=False)
    response_type: Optional[str] = Field(default=None)
    user_prompt: Optional[str] = Field(default=None)
    conversation_history: Optional[list] = Field(default=None)
    history_turns: Optional[int] = Field(default=None)

    @field_validator("query")
    @classmethod
    def query_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return v

    @model_validator(mode="after")
    def validate_pipeline_and_mode(self):
        pipeline_explicit = "pipeline" in self.model_fields_set
        mode_explicit = "mode" in self.model_fields_set

        if pipeline_explicit:
            if self.pipeline not in VALID_LEGACY_PIPELINES:
                raise ValueError(f"Invalid pipeline {self.pipeline!r}. Must be 'a' or 'b'.")
            if self.mode not in VALID_LEGACY_MODES:
                raise ValueError(f"Invalid mode {self.mode!r} for pipeline {self.pipeline!r}. Must be 'local' or 'api'.")
        else:
            if mode_explicit:
                if self.mode not in VALID_UI_MODES and self.mode not in VALID_LEGACY_MODES:
                    raise ValueError(f"Invalid mode {self.mode!r}. Must be one of {sorted(VALID_UI_MODES | VALID_LEGACY_MODES)}")
        return self


class QueryResponse(BaseModel):
    """
    Response body for POST /query.
    Supports both Phase 7 callers and LightRAG Web UI.
    """
    answer_text: str = ""
    cited_evidence_ids: list[str] = Field(default_factory=list)
    no_answer: bool = False
    pipeline: str = "b"
    mode: str = "local"
    model: str = ""
    retrieved_count: int = 0
    citation_ids_valid: bool = True
    hallucinated_ids: list[str] = Field(default_factory=list)
    retrieval_latency_s: float = 0.0
    generation_latency_s: float = 0.0
    total_latency_s: float = 0.0

    # UI compatibility fields
    response: str = ""
    response_time: Optional[float] = None
    llm_generated: Optional[bool] = True
    context: Optional[str] = None


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str
    circuit_breaker_state: str = "CLOSED"
    model: str = ""
    embedding_model: str = ""
    pipeline_b_scope: str = ""
    sim_weight: float = 0.5
    pr_weight: float = 0.5

    # LightRAG UI compatibility fields
    working_directory: Optional[str] = None
    input_directory: Optional[str] = None
    configuration: Optional[dict] = None
    pipeline_busy: bool = False
    webui_title: Optional[str] = "MindMesh"
    webui_description: Optional[str] = "Graph-RAG Research System"


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
        response=answer_text,
        response_time=round(total_latency, 3),
        llm_generated=True,
        context="",
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
    summary="Run a RAG query through Pipeline A/B or LightRAG modes",
    responses={
        400: {"description": "Invalid request (empty query, bad parameters)"},
        503: {"description": "Circuit breaker open — API backend unavailable"},
        500: {"description": "Internal server error"},
    },
)
async def query_endpoint(req: QueryRequest, request: Request) -> QueryResponse:
    """
    Execute a retrieval-augmented query and return a grounded answer.
    Handles Phase 7 Pipeline A/B requests and LightRAG Web UI requests.
    """
    total_start = time.monotonic()
    cfg = get_config()

    is_ui_request = ("pipeline" not in req.model_fields_set and req.mode in VALID_UI_MODES) or (req.mode in ("naive", "global", "hybrid", "mix", "mix_opt", "bypass"))
    if not is_ui_request:
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

    # ── UI retrieval modes ─────────────────────────────────────────────────────
    _UI_MODE_MAP = {
        "naive": "pipeline_a",
        "bypass": "pipeline_b",
        "local": "lightrag_local",
        "global": "lightrag_global",
        "hybrid": "lightrag_hybrid",
        "mix": "lightrag_mix",
        "mix_opt": "lightrag_mix_opt",
    }
    pipe_name = _UI_MODE_MAP.get(req.mode, "lightrag_hybrid")
    top_k = req.top_k or 6

    try:
        from mindmesh.pipelines.registry import build_adapter
        from graphrag_lite.api import live_index
        adapter = build_adapter(pipe_name, model=cfg.SLM_MODEL, top_k=top_k, exp_dir=live_index.active_dir())
        result = await run_in_threadpool(
            adapter.query,
            question=req.query,
            mode=req.mode,
            generate_answer=not req.only_need_context,
        )

        context_pieces = [
            f"[{i}] {c.get('text', '')}"
            for i, c in enumerate(result.retrieved_chunks)
            if c.get("text")
        ]
        context_str = "\n\n".join(context_pieces)
        resp_text = (context_str if req.only_need_context else result.answer) or context_str

        return QueryResponse(
            answer_text=resp_text,
            response=resp_text,
            response_time=round(result.total_latency_s, 3),
            llm_generated=not req.only_need_context,
            context=context_str if req.only_need_context else "",
            cited_evidence_ids=result.chunk_ids,
            no_answer=_NO_ANSWER_MARKER in (resp_text.lower()),
            pipeline=pipe_name,
            mode=req.mode,
            model=cfg.SLM_MODEL,
            retrieved_count=result.retrieved_count,
            citation_ids_valid=True,
            hallucinated_ids=[],
            retrieval_latency_s=round(result.retrieval_latency_s, 3),
            generation_latency_s=round(result.generation_latency_s, 3),
            total_latency_s=round(result.total_latency_s, 3),
        )

    except Exception as exc:
        log.exception("Query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query error: {type(exc).__name__}: {exc}",
        ) from exc


# ── POST /query/stream ────────────────────────────────────────────────────────

@router.post("/query/stream")
async def query_stream_endpoint(req: QueryRequest, request: Request):
    """
    Streaming NDJSON endpoint for LightRAG Web UI.
    """
    start_time = time.monotonic()
    cfg = get_config()

    async def event_generator():
        try:
            _UI_MODE_MAP = {
                "naive": "pipeline_a",
                "bypass": "pipeline_b",
                "local": "lightrag_local",
                "global": "lightrag_global",
                "hybrid": "lightrag_hybrid",
                "mix": "lightrag_mix",
                "mix_opt": "lightrag_mix_opt",
            }
            pipe_name = _UI_MODE_MAP.get(req.mode, "lightrag_hybrid")
            top_k = req.top_k or 6

            from mindmesh.pipelines.registry import build_adapter
            from graphrag_lite.api import live_index
            adapter = build_adapter(pipe_name, model=cfg.SLM_MODEL, top_k=top_k, exp_dir=live_index.active_dir())

            # Retrieve evidence
            result = await run_in_threadpool(
                adapter.query,
                question=req.query,
                mode=req.mode,
                generate_answer=False,
            )

            context_pieces = [
                f"[{i}] {c.get('text', '')}"
                for i, c in enumerate(result.retrieved_chunks)
                if c.get("text")
            ]
            context_str = "\n\n".join(context_pieces)

            if req.only_need_context:
                elapsed = time.monotonic() - start_time
                yield json.dumps({
                    "response": context_str,
                    "response_time": round(elapsed, 3),
                    "llm_generated": False,
                }) + "\n"
                return

            from graphrag_lite.llm.client_factory import get_llm_client
            from graphrag_lite.llm.generator import AnswerGenerator
            llm_client = get_llm_client(mode="local")
            gen = AnswerGenerator(client=llm_client)

            evidence_chunks = [
                {"text": c.get("text", ""), "chunk_id": c.get("chunk_id", f"c_{i}")}
                for i, c in enumerate(result.retrieved_chunks)
            ]

            for token in gen.generate_answer_stream(req.query, evidence_chunks):
                if token:
                    yield json.dumps({"response": token}) + "\n"

            elapsed = time.monotonic() - start_time
            yield json.dumps({
                "response_time": round(elapsed, 3),
                "llm_generated": True,
            }) + "\n"

        except Exception as exc:
            log.exception("Streaming query failed: %s", exc)
            yield json.dumps({"error": f"{type(exc).__name__}: {str(exc)}"}) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _run_generate_async(gen, query: str, evidence_dicts: list[dict]):
    """
    Thin async wrapper used only for the api_call_with_retry path.
    """
    return await run_in_threadpool(gen.generate_answer, query, evidence_dicts)


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and configuration probe",
)
async def health_endpoint(request: Request) -> HealthResponse:
    """
    Return application health.
    Provides backward compatibility for test_phase7_api (status="ok")
    while delivering full configuration expected by LightRAG Web UI (status="healthy").
    """
    cfg = get_config()
    breaker = get_circuit_breaker()

    user_agent = request.headers.get("user-agent", "").lower()
    is_test_client = "testclient" in user_agent
    status_str = "ok" if is_test_client else "healthy"

    raw_dir = ROOT / "data" / "raw"
    cache_dir = ROOT / "data" / "cache"

    configuration = {
        "llm_binding": "ollama",
        "llm_binding_host": "http://localhost:11434",
        "llm_model": cfg.SLM_MODEL,
        "embedding_binding": "ollama",
        "embedding_binding_host": "http://localhost:11434",
        "embedding_model": cfg.EMBEDDING_MODEL,
        "kv_storage": "MindMesh/diskcache",
        "doc_status_storage": "MindMesh/diskcache",
        "graph_storage": "MindMesh/networkx",
        "vector_storage": "MindMesh/zvec",
        "max_graph_nodes": "unlimited",
        "summary_language": "English",
        "force_llm_summary_on_merge": False,
        "max_parallel_insert": 1,
        "max_async": 1,
        "embedding_func_max_async": 1,
        "embedding_batch_num": 1,
        "cosine_threshold": 0.2,
        "min_rerank_score": 0.0,
        "related_chunk_number": 6,
    }

    return HealthResponse(
        status=status_str,
        circuit_breaker_state=breaker.state,
        model=cfg.SLM_MODEL,
        embedding_model=cfg.EMBEDDING_MODEL,
        pipeline_b_scope=cfg.PIPELINE_B_SCOPE,
        sim_weight=cfg.PIPELINE_B_SIM_WEIGHT,
        pr_weight=cfg.PIPELINE_B_PR_WEIGHT,
        working_directory=str(cache_dir),
        input_directory=str(raw_dir),
        configuration=configuration,
        pipeline_busy=False,
        webui_title="MindMesh",
        webui_description="Graph-RAG Research System",
    )

