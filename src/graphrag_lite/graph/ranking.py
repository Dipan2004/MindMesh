"""
ranking.py — Node ranking functions for the knowledge graph.

Two pure ranking functions over a NetworkX DiGraph:
  - rank_pagerank: weighted importance by link structure (damping from config)
  - rank_degree:   raw in+out degree count

Both are pure functions with no side effects and no config coupling beyond
the damping factor, which is read from get_config() inside rank_pagerank
so it can be overridden per ablation without code changes.

Usage:
    from graphrag_lite.graph.ranking import rank_pagerank, rank_degree
    pr = rank_pagerank(G)
    dg = rank_degree(G)
    top5_pr = sorted(pr, key=pr.get, reverse=True)[:5]
    top5_dg = sorted(dg, key=dg.get, reverse=True)[:5]
"""

from __future__ import annotations

import networkx as nx

from graphrag_lite.config import get_config


def rank_pagerank(graph: nx.DiGraph, damping: float | None = None) -> dict[str, float]:
    """
    Compute PageRank scores for all nodes.

    Parameters
    ----------
    graph:
        The knowledge graph from ``build_graph()``.
    damping:
        Damping factor. If None, reads ``PAGERANK_DAMPING`` from config
        (default 0.85 per CONFIG.md §5). Pass explicitly to override for
        testing or ablation without touching .env.

    Returns
    -------
    dict[node_name, score]  — scores sum to 1.0 across all nodes.
    """
    if damping is None:
        damping = get_config().PAGERANK_DAMPING
    return nx.pagerank(graph, alpha=damping)


def rank_degree(graph: nx.DiGraph) -> dict[str, int]:
    """
    Compute total degree (in + out) for all nodes.

    Pure function: no config coupling, no side effects.

    Returns
    -------
    dict[node_name, total_degree]
    """
    return dict(graph.degree())
