"""
test_ui_api_compat.py — Verification test suite for LightRAG Web UI endpoints.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    with patch("graphrag_lite.api.main.set_tracing"), \
         patch("graphrag_lite.api.main.init_http_client"), \
         patch("graphrag_lite.api.main.close_http_client", new_callable=AsyncMock):
        from graphrag_lite.api.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def test_health_for_webui(client):
    resp = client.get("/health", headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "configuration" in data
    assert data["configuration"]["llm_binding"] == "ollama"
    assert "working_directory" in data
    assert "input_directory" in data
    assert data["pipeline_busy"] is False
    assert data["webui_title"] == "MindMesh"


def test_auth_status_guest_mode(client):
    resp = client.get("/auth-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["auth_configured"] is False
    assert data["auth_mode"] == "disabled"
    assert data["access_token"] == "mindmesh-guest-token"
    assert data["token_type"] == "bearer"
    assert data["webui_title"] == "MindMesh"


def test_documents_supported_file_types(client):
    resp = client.get("/documents/supported_file_types")
    assert resp.status_code == 200
    data = resp.json()
    assert "supported_extensions" in data
    assert ".txt" in data["supported_extensions"]
    assert ".md" in data["supported_extensions"]
    assert "engines" in data


def test_documents_pipeline_status(client):
    resp = client.get("/documents/pipeline_status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["busy"] is False
    assert "docs" in data
    assert "latest_message" in data


def test_documents_paginated(client):
    resp = client.post("/documents/paginated", json={
        "page": 1,
        "page_size": 10,
        "sort_field": "created_at",
        "sort_direction": "desc",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "documents" in data
    assert "pagination" in data
    assert "status_counts" in data
    assert data["pagination"]["page"] == 1
    assert data["pagination"]["page_size"] == 10
    assert data["pagination"]["total_count"] >= 5
    assert len(data["documents"]) >= 5
    doc0 = data["documents"][0]
    assert "id" in doc0
    assert "content_summary" in doc0
    assert "content_length" in doc0
    assert "status" in doc0
    assert "file_path" in doc0


def test_graph_endpoints(client):
    resp_graph = client.get("/graphs?max_nodes=50")
    assert resp_graph.status_code == 200
    g_data = resp_graph.json()
    assert "nodes" in g_data
    assert "edges" in g_data
    assert len(g_data["nodes"]) > 0

    resp_pop = client.get("/graph/label/popular?limit=10")
    assert resp_pop.status_code == 200
    pop_labels = resp_pop.json()
    assert isinstance(pop_labels, list)
    assert len(pop_labels) > 0

    resp_search = client.get("/graph/label/search?q=graph&limit=10")
    assert resp_search.status_code == 200
    search_labels = resp_search.json()
    assert isinstance(search_labels, list)

    resp_exists = client.get("/graph/entity/exists?name=" + pop_labels[0])
    assert resp_exists.status_code == 200
    assert resp_exists.json()["exists"] is True


def test_ui_query_context_only(client):
    resp = client.post("/query", json={
        "query": "What is vector indexing?",
        "mode": "hybrid",
        "only_need_context": True,
        "top_k": 3,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert data["llm_generated"] is False
    assert data["response_time"] is not None
    assert len(data["response"]) > 0


def test_ui_query_stream_context_only(client):
    resp = client.post("/query/stream", json={
        "query": "What is vector indexing?",
        "mode": "hybrid",
        "only_need_context": True,
        "top_k": 3,
    })
    assert resp.status_code == 200
    lines = [line for line in resp.text.split("\n") if line.strip()]
    assert len(lines) >= 1
    last_obj = json.loads(lines[-1])
    assert "response" in last_obj
    assert last_obj["llm_generated"] is False
