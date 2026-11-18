"""
test_demo_readiness.py — Comprehensive end-to-end demonstration readiness verification.
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


def test_webui_static_served(client):
    """Verify UI opens at /webui/ with MindMesh branding."""
    resp = client.get("/webui/")
    assert resp.status_code == 200
    assert "<title>MindMesh</title>" in resp.text


def test_api_health_and_model_availability(client):
    """Verify health returns healthy status and model configuration."""
    resp = client.get("/health", headers={"User-Agent": "Mozilla/5.0"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["pipeline_busy"] is False
    assert data["webui_title"] == "MindMesh"
    assert data["configuration"]["llm_binding"] == "ollama"
    assert data["configuration"]["llm_model"] != ""


def test_guest_auth_contract(client):
    """Verify guest-mode auth enables immediate UI entry without login."""
    resp = client.get("/auth-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["auth_configured"] is False
    assert data["access_token"] == "mindmesh-guest-token"
    assert data["webui_title"] == "MindMesh"


def test_documents_management_workflow(client):
    """Verify document pagination, status, and upload."""
    # 1. Pipeline status
    status_resp = client.get("/documents/pipeline_status")
    assert status_resp.status_code == 200
    assert status_resp.json()["busy"] is False

    # 2. Supported file types
    types_resp = client.get("/documents/supported_file_types")
    assert types_resp.status_code == 200
    assert ".txt" in types_resp.json()["supported_extensions"]

    # 3. Document listing and pagination
    list_resp = client.post("/documents/paginated", json={
        "page": 1,
        "page_size": 20,
        "sort_field": "created_at",
        "sort_direction": "desc",
    })
    assert list_resp.status_code == 200
    docs_data = list_resp.json()
    assert len(docs_data["documents"]) >= 5
    assert docs_data["status_counts"]["processed"] >= 5

    # 4. Upload document
    sample_content = b"MindMesh demonstration test document on graph retrieval."
    upload_resp = client.post(
        "/documents/upload",
        files={"file": ("demo_test_doc.txt", sample_content, "text/plain")},
    )
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    assert upload_data["status"] == "success"
    assert "track_id" in upload_data


def test_graph_visualization_endpoints(client):
    """Verify graph nodes, edges, labels, and search."""
    resp = client.get("/graphs?max_nodes=100")
    assert resp.status_code == 200
    g = resp.json()
    assert len(g["nodes"]) > 0
    assert len(g["edges"]) > 0

    popular = client.get("/graph/label/popular?limit=5").json()
    assert len(popular) > 0
    first_label = popular[0]

    ego_resp = client.get(f"/graphs?label={first_label}&max_depth=1")
    assert ego_resp.status_code == 200
    ego_g = ego_resp.json()
    assert len(ego_g["nodes"]) > 0

    search_resp = client.get("/graph/label/search?q=a&limit=10")
    assert search_resp.status_code == 200
    assert isinstance(search_resp.json(), list)


def test_retrieval_modes_context(client):
    """Verify all 6 retrieval modes return grounded context."""
    modes = ["naive", "local", "global", "hybrid", "mix", "bypass"]
    for mode in modes:
        resp = client.post("/query", json={
            "query": "What is an embedding vector?",
            "mode": mode,
            "only_need_context": True,
            "top_k": 2,
        })
        assert resp.status_code == 200, f"Mode {mode} failed with status {resp.status_code}"
        data = resp.json()
        assert data["response"] != "", f"Mode {mode} returned empty response"
        assert data["llm_generated"] is False
        assert data["response_time"] is not None


def test_streaming_query(client):
    """Verify NDJSON streaming returns tokens and final timing metadata."""
    resp = client.post("/query/stream", json={
        "query": "What is self attention?",
        "mode": "hybrid",
        "only_need_context": True,
        "top_k": 2,
    })
    assert resp.status_code == 200
    lines = [json.loads(line) for line in resp.text.split("\n") if line.strip()]
    assert len(lines) >= 1
    assert "response" in lines[0]
    assert "response_time" in lines[-1]
