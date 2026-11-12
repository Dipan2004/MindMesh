"""
main.py — FastAPI application entry point (Tasks 7.1 + 7.2 + 7.3).

Application lifecycle (lifespan)
---------------------------------
Startup  (once, before the first request):
  1. Call set_tracing(cfg.LLM_MODE) exactly once so tracing is not
     re-initialized per request (per Phase 6 audit finding and Task 7 spec).
  2. Initialize the shared httpx.AsyncClient and API semaphore.
  3. Log the startup configuration summary (no secrets).

Shutdown (once, after the last request):
  1. Close the shared httpx.AsyncClient cleanly.

Usage
-----
  # development
  uvicorn graphrag_lite.api.main:app --reload --port 8000

  # production (workers read from UVICORN_WORKERS config)
  uvicorn graphrag_lite.api.main:app --workers <N> --port 8000

  # or programmatically (e.g. for tests)
  from fastapi.testclient import TestClient
  from graphrag_lite.api.main import app
  client = TestClient(app)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from graphrag_lite.api.middleware import close_http_client, init_http_client
from graphrag_lite.api.routes import router
from graphrag_lite.api.lightrag_compat import compat_router
from graphrag_lite.config import get_config
from graphrag_lite.observability.tracing import set_tracing

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

_ROOT = Path(__file__).resolve().parents[3]
_UI_BUILD_DIR = _ROOT / "ui" / "lightrag_ui_dist"  # built by: cd ui/lightrag_ui && npm run build


# ── Application lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Everything before ``yield`` runs at startup (once).
    Everything after ``yield`` runs at shutdown (once).
    """
    cfg = get_config()

    # 1. Tracing — called exactly once at startup, not per-request.
    #    Phase 6 audit: set_tracing must not be called inside request handlers.
    set_tracing(cfg.LLM_MODE)
    log.info(
        "Tracing initialised: LANGCHAIN_TRACING_V2=%s  project=%s",
        os.environ.get("LANGCHAIN_TRACING_V2", "(unset)"),
        cfg.LANGCHAIN_PROJECT,
    )

    # 2. HTTP client and API semaphore for outbound Grok calls.
    init_http_client()
    log.info(
        "GraphRAG-Lite API starting — mode=%s  model=%s  pipeline_b_scope=%s",
        cfg.LLM_MODE,
        cfg.SLM_MODEL,
        cfg.PIPELINE_B_SCOPE,
    )

    yield  # ── application is now serving requests ──────────────────────

    # Shutdown
    await close_http_client()
    log.info("GraphRAG-Lite API shut down cleanly.")


# ── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(
    title="MindMesh",
    description=(
        "MindMesh: Lightweight Graph-RAG with LightRAG integration. "
        "Structural filtering + Zvec hybrid retrieval + small local LLM."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# ── CORS (required for the LightRAG UI dev server on a different port) ────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route handlers from routes.py (Pipeline A and B).
app.include_router(router)

# Register LightRAG-compatible routes (all pipelines, documents, graph).
app.include_router(compat_router)

# ── Serve the built LightRAG UI as static files ───────────────────────────────
# The UI is built by running: cd ui/lightrag_ui && bun run build
# It outputs to ui/lightrag_ui_dist/ (adjusted in vite.config.ts).
# When the dist directory doesn't exist the API still works; just no UI served.
if _UI_BUILD_DIR.exists():
    app.mount("/webui", StaticFiles(directory=str(_UI_BUILD_DIR), html=True), name="webui")
    log.info("LightRAG UI served at /webui from %s", _UI_BUILD_DIR)
else:
    log.info(
        "LightRAG UI not built yet. "
        "Run: cd ui/lightrag_ui && bun run build"
        " (adjusting vite.config.ts outDir to ui/lightrag_ui_dist first)"
    )


# ── Global exception handlers ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all for unhandled exceptions.

    Returns a generic 500 with the exception type but NOT the full traceback
    or message, which could expose internal details (file paths, model
    responses containing user data, etc.).
    """
    log.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error.",
            "error_type": type(exc).__name__,
        },
    )
