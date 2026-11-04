"""
test_remediation_fixes.py — Tests for the pre-Phase 6 remediation fixes.

Issue A: client_factory uses config values, not hardcoded constants.
Issue B: pipeline_b reads the extractions path from config, not a hardcoded string.
Issue D: the merged BFS loop produces identical ranked_candidates to the
         original two-call version.
Issue F: clear_cache refuses to wipe a directory containing protected artifacts.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ── Issue A: client_factory reads num_predict from config ──────────────

def test_client_factory_uses_config_num_predict():
    """get_llm_client('local') must pass cfg.LLM_NUM_PREDICT to OllamaClient,
    not a hardcoded 2048."""
    from graphrag_lite.llm.client_factory import get_llm_client
    from graphrag_lite.config import get_config

    cfg = get_config()
    client = get_llm_client("local")
    # The OllamaClient stores num_predict as an attribute.
    assert client.num_predict == cfg.LLM_NUM_PREDICT, (
        f"Expected num_predict={cfg.LLM_NUM_PREDICT} from config, "
        f"got {client.num_predict}"
    )


def test_client_factory_uses_config_temperature_and_seed():
    """The client must also use config values for temperature, seed, think,
    num_ctx, and frequency_penalty — not hardcoded constants."""
    from graphrag_lite.llm.client_factory import get_llm_client
    from graphrag_lite.config import get_config

    cfg = get_config()
    client = get_llm_client("local")
    assert client.temperature == cfg.EXTRACTION_TEMPERATURE
    assert client.seed == cfg.EXTRACTION_SEED
    assert client.num_ctx == cfg.LLM_NUM_CTX
    assert client.frequency_penalty == cfg.LLM_FREQUENCY_PENALTY


def test_client_factory_think_matches_config():
    """The think parameter must reflect cfg.effective_think()."""
    from graphrag_lite.llm.client_factory import get_llm_client
    from graphrag_lite.config import get_config

    cfg = get_config()
    client = get_llm_client("local")
    assert client.think == cfg.effective_think()


# ── Issue B: pipeline_b reads extractions path from config ─────────────

def test_pipeline_b_extractions_path_uses_config():
    """_get_graph must build the path from cfg.CACHE_DIR, not a hardcoded
    'data/cache/extractions.jsonl'."""
    import inspect
    from graphrag_lite.retrieval import pipeline_b

    source = inspect.getsource(pipeline_b._get_graph)
    assert '"data/cache/extractions.jsonl"' not in source, (
        "pipeline_b._get_graph still has a hardcoded extractions path"
    )
    assert "cfg.CACHE_DIR" in source or "cfg.cache_dir" in source.lower(), (
        "pipeline_b._get_graph does not reference cfg.CACHE_DIR"
    )


# ── Issue D: BFS merge produces identical ranking ───────────────────────

def test_bfs_merge_preserves_ranking_order():
    """The merged BFS loop must produce the same ranked_candidates as the
    original two-call version for the same inputs.

    We simulate both approaches on a small synthetic graph and compare
    the resulting ranked lists.
    """
    import networkx as nx
    from graphrag_lite.graph.traversal import bfs_expand

    # Build a small test graph
    G = nx.DiGraph()
    G.add_edges_from([
        ("A", "B"), ("B", "C"), ("C", "D"),
        ("E", "F"), ("F", "G"),
    ])
    # Undirected for BFS
    G_undirected = G.to_undirected()

    entry_points = ["A", "E"]
    entry_sim = {"A": 0.032, "E": 0.025}
    pr = {"A": 0.05, "B": 0.03, "C": 0.02, "D": 0.01, "E": 0.04, "F": 0.03, "G": 0.02}
    pr_max = max(pr.values())
    SIM_WEIGHT = 0.85
    PR_WEIGHT = 0.15
    hops = 2

    # --- ORIGINAL approach (two calls) ---
    bfs_nodes_original = bfs_expand(G, entry_points, hops=hops)

    node_sim_original = {}
    for ep, sim in entry_sim.items():
        reachable = bfs_expand(G, [ep], hops=hops)
        for n in reachable:
            if n not in node_sim_original or sim > node_sim_original[n]:
                node_sim_original[n] = sim

    sim_max = max(entry_sim.values()) if entry_sim else 0.0
    ranked_original = []
    for n in bfs_nodes_original:
        sim_score = node_sim_original.get(n, 0.0)
        sim_norm = (sim_score / sim_max) if sim_max > 0 else 0.0
        pr_norm = pr.get(n, 0.0) / pr_max if pr_max > 0 else 0.0
        combined = SIM_WEIGHT * sim_norm + PR_WEIGHT * pr_norm
        ranked_original.append((n, combined))
    ranked_original.sort(key=lambda x: (-x[1], x[0]))

    # --- MERGED approach (single loop) ---
    node_sim_merged = {}
    bfs_nodes_set = set()
    for ep, sim in entry_sim.items():
        reachable = bfs_expand(G, [ep], hops=hops)
        bfs_nodes_set.update(reachable)
        for n in reachable:
            if n not in node_sim_merged or sim > node_sim_merged[n]:
                node_sim_merged[n] = sim
    for ep in entry_points:
        bfs_nodes_set.add(ep)
    bfs_nodes_merged = sorted(bfs_nodes_set)

    ranked_merged = []
    for n in bfs_nodes_merged:
        sim_score = node_sim_merged.get(n, 0.0)
        sim_norm = (sim_score / sim_max) if sim_max > 0 else 0.0
        pr_norm = pr.get(n, 0.0) / pr_max if pr_max > 0 else 0.0
        combined = SIM_WEIGHT * sim_norm + PR_WEIGHT * pr_norm
        ranked_merged.append((n, combined))
    ranked_merged.sort(key=lambda x: (-x[1], x[0]))

    # --- Compare ---
    assert bfs_nodes_original == bfs_nodes_merged, (
        f"BFS node sets differ:\n  original: {bfs_nodes_original}\n"
        f"  merged:   {bfs_nodes_merged}"
    )
    assert ranked_original == ranked_merged, (
        f"Ranked lists differ:\n  original: {ranked_original}\n"
        f"  merged:   {ranked_merged}"
    )


def test_bfs_merge_handles_empty_entry_points():
    """With no entry points, both approaches must produce empty results."""
    from graphrag_lite.graph.traversal import bfs_expand
    import networkx as nx

    G = nx.DiGraph()
    G.add_edge("A", "B")
    entry_sim: dict[str, float] = {}
    entry_points: list[str] = []

    # Original
    bfs_nodes_original = bfs_expand(G, entry_points, hops=2)

    # Merged
    bfs_nodes_set: set[str] = set()
    for ep, sim in entry_sim.items():
        reachable = bfs_expand(G, [ep], hops=2)
        bfs_nodes_set.update(reachable)
    for ep in entry_points:
        bfs_nodes_set.add(ep)
    bfs_nodes_merged = sorted(bfs_nodes_set)

    assert bfs_nodes_original == bfs_nodes_merged == []


# ── Issue F: clear_cache guard regression test ──────────────────────────

def test_clear_cache_refuses_when_protected_artifacts_present():
    """clear_cache must raise RuntimeError when the cache directory contains
    Zvec collections (manifest.* files) or extractions*.jsonl files."""
    from graphrag_lite.cache import llm_cache

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake cache dir with a protected extraction file
        cache_dir = os.path.join(tmpdir, "test_cache")
        os.makedirs(cache_dir)
        Path(cache_dir, "extractions.jsonl").write_text("{}\n")

        # Patch _get_cache to point at our temp dir
        fake_cache = MagicMock()
        fake_cache.directory = cache_dir

        with patch.object(llm_cache, "_get_cache", return_value=fake_cache):
            with pytest.raises(RuntimeError, match="protected artefacts"):
                llm_cache.clear_cache(force=False)


def test_clear_cache_refuses_when_zvec_collection_present():
    """clear_cache must raise when a Zvec collection (manifest.* file) exists."""
    from graphrag_lite.cache import llm_cache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "test_cache")
        os.makedirs(cache_dir)
        # Create a fake Zvec collection directory with a manifest file
        col_dir = os.path.join(cache_dir, "entities_v1")
        os.makedirs(col_dir)
        Path(col_dir, "manifest.1").write_text("{}")

        fake_cache = MagicMock()
        fake_cache.directory = cache_dir

        with patch.object(llm_cache, "_get_cache", return_value=fake_cache):
            with pytest.raises(RuntimeError, match="protected artefacts"):
                llm_cache.clear_cache(force=False)


def test_clear_cache_succeeds_with_force_on_protected():
    """clear_cache(force=True) must proceed even with protected artifacts."""
    from graphrag_lite.cache import llm_cache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "test_cache")
        os.makedirs(cache_dir)
        Path(cache_dir, "extractions.jsonl").write_text("{}\n")

        fake_cache = MagicMock()
        fake_cache.directory = cache_dir

        with patch.object(llm_cache, "_get_cache", return_value=fake_cache):
            # Should not raise
            llm_cache.clear_cache(force=True)
            fake_cache.clear.assert_called_once()


def test_clear_cache_succeeds_on_clean_cache():
    """clear_cache must proceed when no protected artifacts are present."""
    from graphrag_lite.cache import llm_cache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "test_cache")
        os.makedirs(cache_dir)
        # Only a regular diskcache file, no extractions or manifests
        Path(cache_dir, "some_cache_key").write_text("data")

        fake_cache = MagicMock()
        fake_cache.directory = cache_dir

        with patch.object(llm_cache, "_get_cache", return_value=fake_cache):
            llm_cache.clear_cache(force=False)
            fake_cache.clear.assert_called_once()
