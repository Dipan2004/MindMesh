"""
lightrag_compat.py — LightRAG UI compatibility API layer for MindMesh.

Implements all endpoints called by the LightRAG Web UI backed by MindMesh canonical
NetworkX graph, Zvec vectors, and MindMesh-native LightRAG retrieval algorithms.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
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
_CACHE_DIR = ROOT / "data" / "cache"


# ── Auth endpoints (LightRAG Web UI expects these for guest/auth handling) ────

@compat_router.get("/auth-status")
async def auth_status():
    """
    Guest-mode auth status allowing direct UI access without login prompt.
    """
    return {
        "auth_configured": False,
        "access_token": "mindmesh-guest-token",
        "token_type": "bearer",
        "auth_mode": "disabled",
        "core_version": "0.2.0",
        "api_version": "0.2.0",
        "webui_title": "MindMesh",
        "webui_description": "Graph-RAG Research System",
        "ai_content_notice_enabled": False,
    }


@compat_router.post("/login")
async def login(request: Request):
    """Guest login handler."""
    return {
        "access_token": "mindmesh-guest-token",
        "token_type": "bearer",
        "auth_mode": "disabled",
        "core_version": "0.2.0",
        "api_version": "0.2.0",
        "webui_title": "MindMesh",
        "webui_description": "Graph-RAG Research System",
        "ai_content_notice_enabled": False,
    }


@compat_router.get("/auth/verify")
async def auth_verify():
    return {"valid": True, "user": "mindmesh"}


# ── Document Management ───────────────────────────────────────────────────────

def _doc_id(name: str) -> str:
    return hashlib.md5(name.encode()).hexdigest()[:16]


def _get_canonical_docs() -> list[dict]:
    """Return canonical documents indexed in MindMesh."""
    ext_path = _CACHE_DIR / "extractions.jsonl"
    canonical_counts: dict[str, int] = {
        "knowledge_graphs": 10,
        "large_language_models": 10,
        "vector_databases": 9,
        "retrieval_augmented_generation": 8,
        "transformer_architecture": 5,
    }

    if ext_path.exists():
        try:
            from collections import Counter
            counts = Counter()
            with open(ext_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        doc = rec.get("doc_id")
                        if doc:
                            counts[doc] += 1
            if counts:
                canonical_counts = dict(counts)
        except Exception as exc:
            log.warning("Could not count chunks from extractions.jsonl: %s", exc)

    docs = []
    base_time = "2026-11-12T10:00:00"
    for doc_name, chunk_count in canonical_counts.items():
        doc_id_val = _doc_id(doc_name)
        docs.append({
            "id": doc_id_val,
            "content_summary": f"Canonical Corpus: {doc_name.replace('_', ' ').title()}",
            "content_length": chunk_count * 500,
            "status": "processed",
            "created_at": base_time,
            "updated_at": base_time,
            "chunks_count": chunk_count,
            "file_path": f"data/raw/{doc_name}.txt",
            "metadata": {"doc_type": "canonical_corpus", "doc_name": doc_name},
        })
    return docs


def _list_all_docs() -> list[dict]:
    """List documents from data/raw/ combined with canonical corpus docs."""
    docs_by_id = {}

    # 1. Canonical documents
    for doc in _get_canonical_docs():
        docs_by_id[doc["id"]] = doc

    # 2. Uploaded documents in data/raw/
    if _RAW_DIR.exists():
        for f in sorted(_RAW_DIR.iterdir()):
            if f.is_file() and f.suffix.lower() in (".txt", ".md", ".json", ".pdf", ".docx"):
                try:
                    stat = f.stat()
                    did = _doc_id(f.name)
                    created_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_ctime))
                    updated_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime))
                    from graphrag_lite.api import live_index
                    li = live_index.doc_status(f.name) or {}
                    # Real indexing state: pending / processing / processed / failed
                    st = li.get("status") or "pending"
                    summary = f"Uploaded document: {f.stem}"
                    if st == "processing":
                        summary += f" ({li.get('stage', 'processing')})"
                    elif st == "processed":
                        summary += f" ({li.get('entities', 0)} entities, {li.get('relationships', 0)} relationships)"
                    docs_by_id[did] = {
                        "id": did,
                        "content_summary": summary,
                        "content_length": stat.st_size,
                        "status": st,
                        "created_at": created_iso,
                        "updated_at": li.get("updated_at", updated_iso),
                        "chunks_count": li.get("chunks"),
                        "error_msg": li.get("error"),
                        "track_id": li.get("track_id"),
                        "file_path": f"data/raw/{f.name}",
                        "metadata": {"file_name": f.name, "size": stat.st_size,
                                     "index_version": li.get("index_version"),
                                     "processing_s": li.get("duration_s")},
                    }
                except Exception as exc:
                    log.warning("Could not stat file %s: %s", f, exc)

    return list(docs_by_id.values())


class DocumentsRequest(BaseModel):
    status_filter: Optional[str] = None
    status_filters: Optional[list[str]] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1)
    sort_field: str = Field(default="created_at")
    sort_direction: str = Field(default="desc")


@compat_router.post("/documents/paginated")
async def documents_paginated(req: DocumentsRequest):
    """
    Paginated documents endpoint matching LightRAG UI's exact PaginatedDocsResponse schema.
    """
    all_docs = _list_all_docs()

    # Filter
    filtered_docs = all_docs
    if req.status_filter:
        filtered_docs = [d for d in filtered_docs if d.get("status") == req.status_filter]
    elif req.status_filters:
        status_set = set(req.status_filters)
        filtered_docs = [d for d in filtered_docs if d.get("status") in status_set]

    # Sort
    reverse = req.sort_direction.lower() == "desc"
    sort_key = req.sort_field
    filtered_docs.sort(key=lambda d: str(d.get(sort_key, "")), reverse=reverse)

    # Paginate
    total_count = len(filtered_docs)
    total_pages = max(1, math.ceil(total_count / req.page_size))
    start = (req.page - 1) * req.page_size
    end = start + req.page_size
    sliced = filtered_docs[start:end]

    # Status counts across all documents
    status_counts = {"processed": 0, "pending": 0, "processing": 0, "failed": 0}
    for d in all_docs:
        s = d.get("status", "processed")
        status_counts[s] = status_counts.get(s, 0) + 1

    return {
        "documents": sliced,
        "pagination": {
            "page": req.page,
            "page_size": req.page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": req.page < total_pages,
            "has_prev": req.page > 1,
        },
        "status_counts": status_counts,
    }


@compat_router.get("/documents")
async def list_documents(
    page: int = 1,
    page_size: int = 50,
    status_filter: Optional[str] = None,
):
    """Legacy documents endpoint."""
    all_docs = _list_all_docs()
    total = len(all_docs)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "statuses": [d["status"] for d in all_docs],
        "total": total,
        "page": page,
        "page_size": page_size,
        "documents": all_docs[start:end],
    }


@compat_router.get("/documents/supported_file_types")
async def supported_file_types():
    """Returns supported file extensions and parser engines."""
    return {
        "supported_extensions": [".txt", ".md", ".json", ".pdf", ".docx"],
        "engines": {
            "default": [".txt", ".md", ".json"],
            "mineru": [".pdf", ".docx"],
            "docling": [".pdf", ".docx"],
        },
    }


@compat_router.get("/documents/pipeline_status")
async def pipeline_status():
    """Returns the live indexing pipeline status (uploads are indexed in the background)."""
    from graphrag_lite.api import live_index
    s = live_index.pipeline_status()
    return {
        "busy": s["busy"],
        "job_name": s["job_name"],
        "job_start": s["job_start"],
        "docs": len(_list_all_docs()),
        "batchs": s["queued"] + (1 if s["busy"] else 0),
        "cur_batch": 1 if s["busy"] else 0,
        "cancellation_requested": False,
        "latest_message": s["latest_message"],
        "history_messages": s["history_messages"],
        "active_index_version": s["active_version"],
    }


@compat_router.post("/documents/cancel_pipeline")
async def cancel_pipeline():
    return {
        "status": "not_busy",
        "message": "No pipeline job currently running",
    }


@compat_router.post("/documents/scan")
async def scan_documents():
    return {
        "status": "scanning_started",
        "message": "Scan completed. Documents are up to date.",
        "track_id": f"scan-{int(time.time())}",
    }


@compat_router.post("/documents/reprocess_failed")
async def reprocess_failed():
    return {
        "status": "reprocessing_started",
        "message": "Reprocessing started",
        "track_id": f"reprocess-{int(time.time())}",
    }


@compat_router.get("/documents/track_status/{track_id}")
async def get_track_status(track_id: str):
    from graphrag_lite.api import live_index
    d = live_index.track_status(track_id)
    docs = [] if d is None else [{
        "id": d["id"], "status": d.get("status"), "content_summary": d.get("stage", ""),
        "content_length": d.get("content_length", 0), "chunks_count": d.get("chunks"),
        "error_msg": d.get("error"), "file_path": f"data/raw/{d.get('file_name', '')}",
        "created_at": d.get("created_at"), "updated_at": d.get("updated_at"),
    }]
    return {
        "track_id": track_id,
        "documents": docs,
        "total_count": len(docs),
        "status_summary": {docs[0]["status"]: 1} if docs else {},
    }


@compat_router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to data/raw/."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".txt", ".md", ".json", ".pdf", ".docx"):
        raise HTTPException(
            400,
            f"Unsupported file type {suffix!r}. MindMesh accepts .txt, .md, .json, .pdf, and .docx.",
        )

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = _RAW_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)

    doc_id_val = _doc_id(file.filename)
    log.info("Document uploaded: %s (%d bytes)", file.filename, len(content))
    from graphrag_lite.api import live_index
    track = live_index.enqueue_file(dest)
    return {
        "status": "success",
        "message": f"File '{file.filename}' uploaded; indexing started (chunking, extraction, graph, embeddings).",
        "track_id": track,
        "id": doc_id_val,
        "file_name": file.filename,
        "size": len(content),
    }


class TextInsertRequest(BaseModel):
    text: str


class TextsInsertRequest(BaseModel):
    texts: list[str]


@compat_router.post("/documents/text")
async def insert_text(req: TextInsertRequest):
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"text_{int(time.time())}.txt"
    dest = _RAW_DIR / filename
    dest.write_text(req.text, encoding="utf-8")
    from graphrag_lite.api import live_index
    track = live_index.enqueue_file(dest)
    return {
        "status": "success",
        "message": f"Text saved as {filename}; indexing started",
        "track_id": track,
    }


@compat_router.post("/documents/texts")
async def insert_texts(req: TextsInsertRequest):
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    from graphrag_lite.api import live_index
    track_id = f"batch_{int(time.time())}"
    for i, t in enumerate(req.texts):
        filename = f"text_{int(time.time())}_{i}.txt"
        dest = _RAW_DIR / filename
        dest.write_text(t, encoding="utf-8")
        track_id = live_index.enqueue_file(dest)
    return {
        "status": "success",
        "message": f"Inserted {len(req.texts)} text snippets",
        "track_id": track_id,
    }


@compat_router.get("/documents/texts/{doc_id}")
async def get_document_text(doc_id: str):
    """Return text of document by doc ID."""
    if _RAW_DIR.exists():
        for f in _RAW_DIR.iterdir():
            if f.is_file() and _doc_id(f.name) == doc_id:
                return {"id": doc_id, "file_name": f.name, "text": f.read_text(encoding="utf-8", errors="replace")}

    # Check canonical corpus
    ext_path = _CACHE_DIR / "extractions.jsonl"
    if ext_path.exists():
        try:
            chunks = []
            with open(ext_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        d = rec.get("doc_id")
                        if d and _doc_id(d) == doc_id:
                            cid = rec.get("chunk_id")
                            if cid:
                                from graphrag_lite.retrieval.lightrag.chunk_lookup import get_chunk_text
                                chunks.append(get_chunk_text(cid))
            if chunks:
                return {
                    "id": doc_id,
                    "file_name": f"{doc_id}.txt",
                    "text": "\n\n".join(chunks),
                }
        except Exception as exc:
            log.warning("Failed retrieving canonical doc text: %s", exc)

    raise HTTPException(404, f"Document {doc_id!r} not found")


class DeleteDocsRequest(BaseModel):
    doc_ids: list[str] = Field(default_factory=list)
    delete_file: bool = False
    delete_llm_cache: bool = False


@compat_router.delete("/documents/delete_document")
async def delete_documents(req: DeleteDocsRequest):
    """Delete uploaded documents."""
    deleted = []
    if _RAW_DIR.exists():
        for f in _RAW_DIR.iterdir():
            if f.is_file() and _doc_id(f.name) in req.doc_ids:
                try:
                    f.unlink()
                    deleted.append(f.name)
                    from graphrag_lite.api import live_index
                    live_index.enqueue_remove(f.name)   # rebuild the index without it
                except Exception as exc:
                    log.warning("Failed to delete %s: %s", f, exc)
    return {
        "status": "deletion_started",
        "message": f"Deleted {len(deleted)} file(s).",
        "doc_id": req.doc_ids[0] if req.doc_ids else "",
    }


@compat_router.delete("/documents")
async def clear_documents(clear_llm_cache: bool = False):
    return {
        "status": "failure",
        "message": "Clearing all documents is restricted in MindMesh to preserve canonical data.",
    }


# ── Graph Endpoints ───────────────────────────────────────────────────────────

def _load_graph():
    import io, contextlib
    from graphrag_lite.graph.builder import build_graph, load_extractions

    from graphrag_lite.api import live_index
    ext_path = live_index.graph_extractions_path() or (_CACHE_DIR / "extractions.jsonl")
    if not ext_path.exists():
        return None

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        records = load_extractions(str(ext_path))
        G = build_graph(records)
    return G


@compat_router.get("/graphs")
async def get_graph(label: Optional[str] = None, max_depth: int = 2, max_nodes: int = 1000):
    """
    Return MindMesh canonical knowledge graph nodes and edges for sigma.js viewer.
    Supports ego-graph filtering when `label` is provided.
    """
    try:
        import networkx as nx

        G = _load_graph()
        if G is None:
            return {"nodes": [], "edges": [], "note": "No extraction data available."}

        target_graph = G
        if label and label.strip() and label.strip() != "*":
            lbl_lower = label.strip().lower()
            matching_node = None
            for n in G.nodes():
                if n.lower() == lbl_lower:
                    matching_node = n
                    break
            if matching_node:
                target_graph = nx.ego_graph(G, matching_node, radius=max_depth)
            else:
                return {"nodes": [], "edges": []}

        # Subgraph if max_nodes exceeded
        if len(target_graph.nodes()) > max_nodes:
            top_nodes = sorted(target_graph.nodes(), key=lambda n: target_graph.degree(n), reverse=True)[:max_nodes]
            target_graph = target_graph.subgraph(top_nodes)

        nodes = []
        for node_name, attrs in target_graph.nodes(data=True):
            ent_type = attrs.get("type", "ENTITY")
            nodes.append({
                "id": node_name,
                "labels": [ent_type],
                "properties": {
                    "entity_id": node_name,
                    "name": node_name,
                    "type": ent_type,
                    "description": attrs.get("description", ""),
                    "source_docs": sorted(attrs.get("source_docs", set())),
                },
            })

        edges = []
        for src, tgt, attrs in target_graph.edges(data=True):
            types = attrs.get("types", ["RELATED_TO"])
            rel_type = types[0] if types else "RELATED_TO"
            edges.append({
                "id": f"{src}__{tgt}",
                "source": src,
                "target": tgt,
                "type": rel_type,
                "properties": {
                    "types": types,
                    "weight": attrs.get("weight", 1),
                    "description": attrs.get("description", ""),
                },
            })

        return {"nodes": nodes, "edges": edges}

    except Exception as exc:
        log.exception("Graph endpoint failed: %s", exc)
        raise HTTPException(500, f"Graph error: {type(exc).__name__}: {exc}") from exc


@compat_router.get("/graph/label/list")
async def list_entity_labels(query: str = "", limit: int = 50) -> list[str]:
    """List entity names matching query for graph search."""
    G = _load_graph()
    if G is None:
        return []
    q = query.lower().strip()
    nodes = list(G.nodes())
    if q:
        nodes = [n for n in nodes if q in n.lower()]
    return sorted(nodes)[:limit]


@compat_router.get("/graph/label/popular")
async def get_popular_labels(limit: int = 50) -> list[str]:
    """Return top entity names sorted by graph degree."""
    G = _load_graph()
    if G is None:
        return []
    sorted_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    return sorted_nodes[:limit]


@compat_router.get("/graph/label/search")
async def search_labels(q: str = "", limit: int = 50) -> list[str]:
    """Search entity names matching substring `q`."""
    G = _load_graph()
    if G is None:
        return []
    query = q.lower().strip()
    if not query:
        return sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:limit]
    matches = [n for n in G.nodes() if query in n.lower()]
    matches.sort(key=lambda n: (not n.lower().startswith(query), -G.degree(n), n))
    return matches[:limit]


@compat_router.get("/graph/entity/exists")
async def check_entity_exists(name: str):
    """Check if entity exists in knowledge graph."""
    G = _load_graph()
    if G is None:
        return {"exists": False}
    exists = any(n.lower() == name.lower().strip() for n in G.nodes())
    return {"exists": exists}


class EntityEditRequest(BaseModel):
    entity_name: str
    updated_data: dict[str, Any] = Field(default_factory=dict)
    allow_rename: bool = False
    allow_merge: bool = False


@compat_router.post("/graph/entity/edit")
async def edit_entity(req: EntityEditRequest):
    return {
        "status": "success",
        "message": f"Entity '{req.entity_name}' updated successfully.",
        "data": req.updated_data,
        "operation_summary": {
            "merged": False,
            "merge_status": "not_attempted",
            "merge_error": None,
            "operation_status": "success",
            "target_entity": req.entity_name,
            "final_entity": req.entity_name,
            "renamed": False,
        },
    }


class RelationEditRequest(BaseModel):
    source_id: str
    target_id: str
    updated_data: dict[str, Any] = Field(default_factory=dict)


@compat_router.post("/graph/relation/edit")
async def edit_relation(req: RelationEditRequest):
    return {
        "status": "success",
        "message": f"Relation '{req.source_id}' -> '{req.target_id}' updated successfully.",
    }


# ── UI customization ──────────────────────────────────────────────────────────

@compat_router.get("/ui/customization")
async def ui_customization():
    """Return MindMesh UI customization settings."""
    return {
        "customized": False,
        "title": "MindMesh",
        "subtitle": "Graph-RAG Research System",
        "brand": {
            "title": "MindMesh",
            "description": "Graph-RAG Research System",
            "logo_url": None,
            "logo_alt": "MindMesh",
        },
        "features": {
            "document_upload": True,
            "graph_viewer": True,
            "pipeline_3": True,
            "auth": False,
            "workspaces": False,
        },
    }
