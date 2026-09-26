"""
traversal.py — Bounded BFS expansion from seed nodes.

Returns all nodes reachable within ``hops`` steps from any seed node,
on the undirected projection of the knowledge graph (both in- and out-edges
are traversable, matching typical KG retrieval semantics where a 2-hop
path A->B->C is relevant regardless of edge direction).

This function is deliberately UNRANKED — it returns a flat list of reachable
node names. Ranking is a separate step (ranking.py / Pipeline B step 2) so
the two concerns are not conflated, per PRD §9.

Usage:
    from graphrag_lite.graph.traversal import bfs_expand
    from graphrag_lite.config import get_config

    cfg = get_config()
    neighbours = bfs_expand(G, seed_nodes=["Wikidata"], hops=cfg.BFS_HOP_DEPTH)
"""

from __future__ import annotations

import networkx as nx


def bfs_expand(
    graph: nx.DiGraph,
    seed_nodes: list[str],
    hops: int,
) -> list[str]:
    """
    Return all nodes reachable within ``hops`` steps from any seed node.

    Traversal uses the UNDIRECTED projection of the graph so both in- and
    out-edges are traversable — a node 2 hops away via a reverse edge is
    still relevant for retrieval.

    The seed nodes themselves are included in the result (hop 0). Seeds
    that do not exist in the graph are logged with a warning and skipped —
    the function never raises on a missing seed, because query-time entity
    linking may produce names not perfectly matching a graph node.

    Parameters
    ----------
    graph:
        The knowledge graph (nx.DiGraph) from ``build_graph()``.
    seed_nodes:
        List of canonical node names to start from.
    hops:
        Maximum number of edges to traverse. Must be >= 0.
        0 returns only the seeds themselves (that exist in the graph).

    Returns
    -------
    list[str]
        Unique node names reachable within ``hops`` steps, sorted
        alphabetically for deterministic output.

    Raises
    ------
    ValueError
        If ``hops`` < 0.
    """
    if hops < 0:
        raise ValueError(f"hops must be >= 0, got {hops}")

    undirected = graph.to_undirected()
    visited: set[str] = set()

    for seed in seed_nodes:
        if seed not in graph:
            print(
                f"  [WARN] bfs_expand: seed node {seed!r} not in graph — skipped"
            )
            continue

        # BFS up to depth `hops` from this seed
        reachable = nx.single_source_shortest_path_length(
            undirected, seed, cutoff=hops
        )
        visited.update(reachable.keys())

    return sorted(visited)
