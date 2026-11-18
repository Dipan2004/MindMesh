"""
test_lightrag_native.py — CPU-safe unit tests for MindMesh-native LightRAG retrievers.

Verifies:
  1. retrieve_local returns valid chunks from the canonical collection.
  2. retrieve_global returns valid chunks from the canonical collection.
  3. retrieve_hybrid merges and deduplicates local and global evidence.
  4. retrieve_mix merges hybrid evidence with flat chunk retrieval.
  5. All returned chunk_ids exist in canonical extractions/chunks.
  6. Empty queries return empty results gracefully.
  7. top_k is respected.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from graphrag_lite.graph.builder import build_graph, load_extractions
from graphrag_lite.graph.ranking import rank_pagerank
from graphrag_lite.retrieval.lightrag import (
    retrieve_local,
    retrieve_global,
    retrieve_hybrid,
    retrieve_mix,
)

ROOT = Path(__file__).resolve().parents[1]
EXTRACTIONS_FILE = ROOT / "data" / "cache" / "extractions.jsonl"


@pytest.fixture(scope="module")
def canonical_chunk_ids() -> set[str]:
    """Set of valid chunk_ids in the canonical extractions."""
    records = load_extractions(str(EXTRACTIONS_FILE))
    cids = {r["chunk_id"] for r in records if "chunk_id" in r}
    assert len(cids) > 0
    return cids


@pytest.fixture(scope="module")
def graph_and_pr():
    records = load_extractions(str(EXTRACTIONS_FILE))
    G = build_graph(records)
    pr = rank_pagerank(G)
    return G, pr


def test_empty_query_returns_empty(graph_and_pr):
    G, pr = graph_and_pr
    assert retrieve_local("", G=G, pr=pr) == []
    assert retrieve_global("", G=G) == []
    assert retrieve_hybrid("", G=G, pr=pr) == []
    assert retrieve_mix("", G=G, pr=pr) == []


def test_retrieve_local_chunk_validity(graph_and_pr, canonical_chunk_ids):
    G, pr = graph_and_pr
    results = retrieve_local("What is a knowledge graph and entities?", G=G, pr=pr, top_k=4)
    assert len(results) > 0
    assert len(results) <= 4
    for r in results:
        assert "chunk_id" in r
        assert "text" in r
        assert "score" in r
        assert r["source"] == "local"
        assert r["chunk_id"] in canonical_chunk_ids


def test_retrieve_global_chunk_validity(graph_and_pr, canonical_chunk_ids):
    G, _ = graph_and_pr
    results = retrieve_global("How are relationships structured between concepts?", G=G, top_k=4)
    assert len(results) > 0
    assert len(results) <= 4
    for r in results:
        assert "chunk_id" in r
        assert "text" in r
        assert "score" in r
        assert r["source"] == "global"
        assert r["chunk_id"] in canonical_chunk_ids


def test_retrieve_hybrid_deduplication(graph_and_pr, canonical_chunk_ids):
    G, pr = graph_and_pr
    results = retrieve_hybrid("Knowledge graph and vector database hybrid search", G=G, pr=pr, top_k=5)
    assert len(results) > 0
    assert len(results) <= 5
    cids = [r["chunk_id"] for r in results]
    assert len(cids) == len(set(cids)), "Hybrid results must be deduplicated by chunk_id"
    for r in results:
        assert r["source"] == "hybrid"
        assert r["chunk_id"] in canonical_chunk_ids


def test_retrieve_mix_integration(graph_and_pr, canonical_chunk_ids):
    G, pr = graph_and_pr
    results = retrieve_mix("What is hybrid search in vector databases?", G=G, pr=pr, top_k=5)
    assert len(results) > 0
    assert len(results) <= 5
    cids = [r["chunk_id"] for r in results]
    assert len(cids) == len(set(cids)), "Mix results must be deduplicated by chunk_id"
    for r in results:
        assert r["source"] == "mix"
        assert r["chunk_id"] in canonical_chunk_ids


def test_adapters_instantiate_and_dry_run():
    from mindmesh.pipelines.registry import build_adapter
    for name in ("lightrag_local", "lightrag_global", "lightrag_hybrid", "lightrag_mix"):
        adapter = build_adapter(name, model="qwen2.5:1.5b", top_k=3)
        dry = adapter.dry_run()
        assert dry["pipeline"] == name
        assert dry["model"] == "qwen2.5:1.5b"

