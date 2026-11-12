"""
lightrag_compat.py — LightRAG UI compatibility API layer for MindMesh.

The LightRAG WebUI calls these endpoints on its backend. This module
implements compatible versions backed by MindMesh's actual infrastructure.

Endpoints implemented:
  GET  /health                — extended health (LightRAG-compatible fields)
  GET  /documents             — list ingested documents from data/raw/
  POST /documents/upload      — accept .txt/.md files, write to data/raw/
  GET  /documents/texts/{id}  — retrieve a document's text
  DELETE /documents/{id}      — not supported (documented clearly)
  POST /query                 — extended query endpoint (all 7 pipelines)
  GET  /graphs                — return entity/relationship graph data
  GET  /graph/label/list      — list entity labels for graph search
  GET  /auth-status           — no-auth stub required by the UI
  POST /login                 — no-auth stub

Limitations documented in the response (never faked):
  - No workspace management (MindMesh is single-workspace)
  - No document processing pipeline through the API (use build_index.py)
  - No pipeline_3_lightrag retrieval until indexed (GPU task G7)
  - Graph data comes from the canonical MindMesh graph, not LightRAG's KV store
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

log = logging.getLogger(__name__)

compat_router = APIRouter(tags=["lightrag-compat"])

_RAW_DIR = ROOT / "data" / "raw"


# ── Auth stubs (LightRAG UI always checks these) ─────────────────────────────

@compat_router.get("/auth-status")
async def auth_status():
    """No-auth stub — MindMesh does not require authentication."""
    return {"authenticated": True, "user": "mindmesh", "auth_enabled": False}


@compat_router.post("/login")
async def login(request: Request):
    """No-auth stub."""
    return {"token": "mindmesh-no-auth", "user": "mindmesh"}


@compat_router.get("/auth/verify")
async def auth_verify():
    return {"valid": True, "user": "mindmesh"}


# ── Health (LightRAG-compatible) ──────────────────────────────────────────────

@compat_router.get("/health")
async def health_extended():
    """
    Extended health response compatible with what the LightRAG UI expects.
    Returns MindMesh-specific configuration values.
    """
    from graphrag_lite.config import get_config
    cfg = get_config()
    return {
        "status": "healthy",
        "working_directory": str(ROOT / "data" / "cache"),
        "input_directory": str(_RAW_DIR),
        "configuration": {
            "llm_binding":        "ollama",
            "llm_binding_host":   "http://localhost:11434",
            "llm_model":          cfg.SLM_MODEL,
            "embedding_binding":  "ollama",
            "embedding_binding_host": "http://localhost:11434",
            "embedding_model":    cfg.EMBEDDING_MODEL,
            "kv_storage":         "MindMesh/diskcache",
            "doc_status_storage": "MindMesh/diskcache",
            "graph_storage":      "MindMesh/networkx",
            "vector_storage":     "MindMesh/zvec",
            "max_graph_nodes":    "unlimited",
            "summary_language":   "English",
            "force_llm_summary_on_merge": False,
            "max_parallel_insert": 1,
            "max_async":          1,
        },
        "mindmesh_note": (
            "This is MindMesh's LightRAG-compatible API. "
            "Some LightRAG UI features are limited — see /docs for details."
        ),
    }


# ── Documents ────────────────────────────────────────────────────────────────

def _doc_id(path: Path) -> str:
    return hashlib.md5(path.name.encode()).hexdigest()[:16]


def _list_raw_docs() -> list[dict]:
    """List documents from data/raw/."""
    if not _RAW_DIR.exists():
        return []
    docs = []
    for f in sorted(_RAW_DIR.iterdir()):
        if f.suffix in (".txt", ".md"):
            stat = f.stat()
            docs.append({
                "id":         _doc_id(f),
                "file_name":  f.name,
                "size":       stat.st_size,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                              time.localtime(stat.st_ctime)),
                "status":     "processed",
                "content_summary": f"Source document: {f.stem}",
            })
    return docs


@compat_router.get("/documents")
async def list_documents(
    page: int = 1,
    page_size: int = 50,
    status_filter: Optional[str] = None,
):
    """List documents from data/raw/."""
    docs = _list_raw_docs()
    total = len(docs)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "statuses": [d["status"] for d in docs],
        "total": total,
        "page": page,
        "page_size": page_size,
        "documents": docs[start:end],
    }


@compat_router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accept .txt or .md files and write to data/raw/."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".txt", ".md"):
        raise HTTPException(
            400,
            f"Unsupported file type {suffix!r}. MindMesh accepts .txt and .md files.",
        )

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = _RAW_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)

    log.info("Document uploaded: %s (%d bytes)", file.filename, len(content))
    return {
        "id":        _doc_id(dest),
        "file_name": file.filename,
        "size":      len(content),
        "status":    "uploaded",
        "note": (
            "File written to data/raw/. "
            "Run 'python scripts/build_index.py --stage extract' to index it."
        ),
    }


@compat_router.get("/documents/texts/{doc_id}")
async def get_document_text(doc_id: str):
    """Return the text of a document by its MindMesh doc ID."""
    for f in _RAW_DIR.iterdir():
        if f.suffix in (".txt", ".md") and _doc_id(f) == doc_id:
            return {"id": doc_id, "file_name": f.name, "text": f.read_text(encoding="utf-8")}
    raise HTTPException(404, f"Document {doc_id!r} not found")


@compat_router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Document deletion is not supported — documented clearly."""
    raise HTTPException(
        405,
        "Document deletion is not supported through the MindMesh API. "
        "Delete the file from data/raw/ manually.",
    )


# ── Extended Query (all pipelines) ───────────────────────────────────────────

class LightRAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str = Field(default="hybrid")
    only_need_context: bool = Field(default=False)
    top_k: int = Field(default=6, ge=1, le=50)
    response_type: Optional[str] = None


@compat_router.post("/query/lightrag")
async def query_extended(req: LightRAGQueryRequest):
    """
    LightRAG UI query endpoint (all pipeline modes).
    Mounted at /query/lightrag to avoid colliding with the original /query
    which only accepts Pipeline A/B.
    
    Maps LightRAG modes to MindMesh pipelines:
      naive   → pipeline_a  (flat vector)
      local   → lightrag_local
      global  → lightrag_global
      hybrid  → lightrag_hybrid (default)
      mix     → lightrag_mix
      bypass  → pipeline_b  (structural graph)
    """
    """
    Extended query endpoint for the LightRAG UI.

    Maps LightRAG modes to MindMesh pipelines:
      naive   → pipeline_a  (flat vector)
      local   → lightrag_local
      global  → lightrag_global
      hybrid  → lightrag_hybrid (default)
      mix     → lightrag_mix
      bypass  → pipeline_b  (structural graph)

    pipeline_3_lightrag is not available through this endpoint until the
    LightRAG working directory has been indexed (GPU task G7).
    """
    _MODE_MAP = {
        "naive":   "pipeline_a",
        "local":   "lightrag_local",
        "global":  "lightrag_global",
        "hybrid":  "lightrag_hybrid",
        "mix":     "lightrag_mix",
        "bypass":  "pipeline_b",
    }

    pipeline_name = _MODE_MAP.get(req.mode, "lightrag_hybrid")

    try:
        from mindmesh.pipelines.registry import build_adapter
        from graphrag_lite.config import get_config
        cfg = get_config()

        adapter = build_adapter(pipeline_name, model=cfg.SLM_MODEL, top_k=req.top_k)
        result = adapter.query(
            question=req.query,
            mode=req.mode,
            generate_answer=not req.only_need_context,
        )

        # Format response to match what LightRAG UI expects
        context_pieces = [
            f"[{i}] {c.get('text', '')}"
            for i, c in enumerate(result.retrieved_chunks)
            if c.get("text")
        ]
        context_str = "\n\n".join(context_pieces)

        return {
            "response":        result.answer or context_str,
            "context":         context_str if req.only_need_context else "",
            "pipeline":        pipeline_name,
            "mode":            req.mode,
            "retrieved_count": result.retrieved_count,
            "source_docs":     result.source_docs,
            "chunk_ids":       result.chunk_ids,
            "latency_s":       result.total_latency_s,
            "errors":          result.errors,
            "mindmesh_note": (
                f"Served by MindMesh pipeline '{pipeline_name}'. "
                "For genuine LightRAG Pipeline 3, index documents first (GPU task G7)."
                if pipeline_name != "pipeline_3_lightrag" else ""
            ),
        }

    except Exception as exc:
        log.exception("Query failed: %s", exc)
        raise HTTPException(500, f"Query error: {type(exc).__name__}: {exc}") from exc


# ── Graph visualization ───────────────────────────────────────────────────────

@compat_router.get("/graphs")
async def get_graph(label: Optional[str] = None, max_depth: int = 2):
    """
    Return the MindMesh canonical knowledge graph in a format
    compatible with the LightRAG UI's sigma.js graph viewer.
    """
    try:
        import io, contextlib
        from graphrag_lite.graph.builder import build_graph, load_extractions

        ext_path = ROOT / "data" / "cache" / "extractions.jsonl"
        if not ext_path.exists():
            return {"nodes": [], "edges": [], "note": "No extraction data available."}

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            records = load_extractions(str(ext_path))
            G = build_graph(records)

        nodes = []
        for node_name, attrs in G.nodes(data=True):
            nodes.append({
                "id":     node_name,
                "labels": [attrs.get("type", "OTHER")],
                "properties": {
                    "name":        node_name,
                    "type":        attrs.get("type", "OTHER"),
                    "source_docs": sorted(attrs.get("source_docs", set())),
                },
            })

        edges = []
        for src, tgt, attrs in G.edges(data=True):
            types = attrs.get("types", ["RELATED_TO"])
            edges.append({
                "id":     f"{src}__{tgt}",
                "source": src,
                "target": tgt,
                "type":   types[0] if types else "RELATED_TO",
                "properties": {
                    "types":       types,
                    "description": attrs.get("description", ""),
                },
            })

        return {"nodes": nodes, "edges": edges}

    except Exception as exc:
        log.exception("Graph endpoint failed: %s", exc)
        raise HTTPException(500, f"Graph error: {type(exc).__name__}") from exc


@compat_router.get("/graph/label/list")
async def list_entity_labels(query: str = "", limit: int = 50):
    """List entity names for the graph search box."""
    try:
        import io, contextlib
        from graphrag_lite.graph.builder import build_graph, load_extractions

        ext_path = ROOT / "data" / "cache" / "extractions.jsonl"
        if not ext_path.exists():
            return {"labels": []}

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            records = load_extractions(str(ext_path))
            G = build_graph(records)

        q = query.lower()
        labels = [n for n in G.nodes() if q in n.lower()] if q else list(G.nodes())
        return {"labels": labels[:limit]}

    except Exception as exc:
        log.exception("Label list failed: %s", exc)
        return {"labels": [], "error": str(exc)}


# ── UI customization (stub) ───────────────────────────────────────────────────

@compat_router.get("/ui/customization")
async def ui_customization():
    """Return MindMesh UI customization settings."""
    return {
        "title":       "MindMesh",
        "subtitle":    "Graph-RAG Research System",
        "logo_url":    None,
        "features":    {
            "document_upload":   True,
            "graph_viewer":      True,
            "pipeline_3":        False,  # requires GPU task G7
            "auth":              False,
            "workspaces":        False,
        },
    }
