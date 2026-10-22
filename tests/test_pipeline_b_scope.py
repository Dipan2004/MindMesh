"""Task 4.3 -- the scope of Pipeline B's second (scoped) Zvec query.

Three scopes are implemented:
    'docs'        -- the re-query is restricted to the candidates' source
                     DOCUMENTS (the earlier, weaker implementation)
    'candidates'  -- the re-query is restricted to the CHUNKS in which the
                     ranked candidate nodes were mentioned
    'verbalized'  -- PRD S9 Step 3 verbatim: the candidates are VERBALIZED
                     and that text becomes the DENSE side of the second
                     query; the lexical (BM25) side stays the user's query

These are offline unit tests: the graph, the embedder and the vector store
are all stubbed, so nothing is loaded, embedded or queried for real.
"""
import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.config import get_config
from graphrag_lite.retrieval import pipeline_b


@pytest.fixture()
def stubbed(monkeypatch):
    """A two-node graph + stubbed embedder/vector store, and what was called."""
    G = nx.DiGraph()
    G.add_node(
        "Alpha",
        type="TECHNOLOGY",
        descriptions=[{"text": "A deep learning model.", "chunk_id": "c1", "doc_id": "d1"}],
        source_sentences=[{"text": "Alpha was introduced in 2017.", "chunk_id": "c1", "doc_id": "d1"}],
        source_docs={"d1"},
    )
    G.add_node(
        "Beta",
        type="PERSON",
        descriptions=[{"text": "Wrote the paper.", "chunk_id": "c2", "doc_id": "d2"}],
        source_sentences=[{"text": "Beta wrote it.", "chunk_id": "c2", "doc_id": "d2"}],
        source_docs={"d2"},
    )

    calls = {"embedded": [], "queries": []}

    monkeypatch.setattr(pipeline_b, "_get_graph", lambda: (G, {}, 1.0))
    monkeypatch.setattr(
        pipeline_b,
        "get_candidates",
        lambda q: {
            "entry_points": ["Alpha"],
            "entry_points_filtered": [],
            "bfs_nodes": ["Alpha", "Beta"],
            "ranked_candidates": [("Alpha", 0.9), ("Beta", 0.4)],
            "sim_max": 0.9,
        },
    )

    def fake_embed(texts):
        calls["embedded"].append(list(texts))
        return [[0.0] * 768 for _ in texts]

    class _R:
        id = "c1"
        score = 0.1
        fields = {"source_doc": "d1", "chunk_index": 0, "text": "t"}

    def fake_hybrid_query(**kw):
        calls["queries"].append(kw)
        return [_R()]

    monkeypatch.setattr(pipeline_b, "embed_batch", fake_embed)
    monkeypatch.setattr(pipeline_b, "hybrid_query", fake_hybrid_query)
    return calls


def _scope(monkeypatch, scope):
    cfg = get_config()
    object.__setattr__(cfg, "PIPELINE_B_SCOPE", scope)
    return cfg


# ------------------------------------------------------------------ #
# the verbalization helper
# ------------------------------------------------------------------ #

def test_verbalize_candidates_uses_name_type_description(stubbed):
    G, _, _ = pipeline_b._get_graph()
    out = pipeline_b._verbalize_candidates(G, [("Alpha", 0.9), ("Beta", 0.4)])
    assert out.splitlines() == [
        "Alpha (TECHNOLOGY): A deep learning model.",
        "Beta (PERSON): Wrote the paper.",
    ]


def test_verbalize_candidates_skips_unknown_nodes(stubbed):
    G, _, _ = pipeline_b._get_graph()
    out = pipeline_b._verbalize_candidates(G, [("Alpha", 0.9), ("Ghost", 0.1)])
    assert "Ghost" not in out
    assert "Alpha" in out


def test_verbalize_candidates_handles_node_without_description(stubbed):
    G, _, _ = pipeline_b._get_graph()
    G.add_node("Gamma", type="OTHER", descriptions=[], source_sentences=[], source_docs=set())
    out = pipeline_b._verbalize_candidates(G, [("Gamma", 0.5)])
    assert out == "Gamma (OTHER)"


# ------------------------------------------------------------------ #
# the three scopes
# ------------------------------------------------------------------ #

def test_docs_scope_filters_on_source_doc(stubbed, monkeypatch):
    _scope(monkeypatch, "docs")
    pipeline_b.retrieve_structural("q")
    f = stubbed["queries"][-1]["scalar_filters"]
    assert "source_doc = 'd1'" in f and "source_doc = 'd2'" in f
    assert "chunk_id" not in f


def test_candidates_scope_filters_on_candidate_chunks(stubbed, monkeypatch):
    _scope(monkeypatch, "candidates")
    pipeline_b.retrieve_structural("q")
    f = stubbed["queries"][-1]["scalar_filters"]
    assert "chunk_id = 'c1'" in f and "chunk_id = 'c2'" in f
    assert "source_doc" not in f


def test_candidates_scope_embeds_the_user_query(stubbed, monkeypatch):
    _scope(monkeypatch, "candidates")
    pipeline_b.retrieve_structural("how do I do the thing")
    assert stubbed["embedded"][-1] == ["how do I do the thing"]
    assert stubbed["queries"][-1]["query_text"] == "how do I do the thing"


def test_verbalized_scope_embeds_the_candidate_verbalization(stubbed, monkeypatch):
    _scope(monkeypatch, "verbalized")
    pipeline_b.retrieve_structural("how do I do the thing")
    embedded = stubbed["embedded"][-1][0]
    assert "Alpha (TECHNOLOGY): A deep learning model." in embedded
    assert "how do I do the thing" not in embedded


def test_verbalized_scope_keeps_the_user_query_on_the_lexical_side(stubbed, monkeypatch):
    _scope(monkeypatch, "verbalized")
    pipeline_b.retrieve_structural("how do I do the thing")
    assert stubbed["queries"][-1]["query_text"] == "how do I do the thing"


def test_verbalized_scope_still_filters_on_candidate_chunks(stubbed, monkeypatch):
    _scope(monkeypatch, "verbalized")
    pipeline_b.retrieve_structural("q")
    f = stubbed["queries"][-1]["scalar_filters"]
    assert "chunk_id = 'c1'" in f and "chunk_id = 'c2'" in f


def test_all_three_scopes_return_the_same_number_of_evidence(stubbed, monkeypatch):
    for scope in ("docs", "candidates", "verbalized"):
        _scope(monkeypatch, scope)
        ev = pipeline_b.retrieve_structural("q")
        assert len(ev) == 1
        assert ev[0].source == "structural"


def test_empty_candidate_list_returns_no_evidence(stubbed, monkeypatch):
    monkeypatch.setattr(
        pipeline_b,
        "get_candidates",
        lambda q: {"entry_points": [], "entry_points_filtered": [],
                   "bfs_nodes": [], "ranked_candidates": [], "sim_max": 0.0},
    )
    _scope(monkeypatch, "docs")
    assert pipeline_b.retrieve_structural("q") == []
    assert stubbed["queries"] == []


def test_empty_query_is_rejected(stubbed, monkeypatch):
    _scope(monkeypatch, "docs")
    with pytest.raises(ValueError):
        pipeline_b.retrieve_structural("   ")
