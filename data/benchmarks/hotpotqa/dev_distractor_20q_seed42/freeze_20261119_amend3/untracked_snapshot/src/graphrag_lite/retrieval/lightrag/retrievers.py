"""
retrievers.py -- MindMesh-native LightRAG-style retrieval (Local / Global /
Hybrid / Mix), adapted from the algorithm description in arXiv:2410.05779.

All four read MindMesh storage only: Zvec entity / relationship / chunk
collections in an experiment (or benchmark) directory, plus the NetworkX
graph built by graphrag_lite.graph.builder.

  Local   : query -> entity search -> entity provenance chunks
            + one-hop edge provenance chunks (cross-document via shared nodes)
  Global  : query -> relationship search -> relationship provenance chunks
            + endpoint entity provenance chunks
  Hybrid  : RRF fusion of Local and Global chunk rankings
  Mix     : RRF fusion of Local, Global and direct chunk vector search

``optimized=True`` (the *_opt pipelines) changes four things, matching the
adapter descriptions:
  * single-query embedding (vector computed once, reused for every search)
  * degree normalisation of one-hop expansion (hub nodes contribute less)
  * primary-provenance weighting (chunks that mention the matched entity /
    relationship directly outweigh chunks reached through expansion)
  * dense-anchored Mix: dense keeps the top slots, graph evidence fills the
    last ~20% (see retrieve_mix)
Answer generation latency protection lives in the adapter (num_predict cap).

See context.py for the reconstruction note.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from graphrag_lite.retrieval.lightrag.context import (
    collection_path,
    edge_chunk_ids,
    embed_query,
    materialize,
    node_chunk_ids,
    query_collection_direct,
    rrf_fuse,
)
from graphrag_lite.retrieval.lightrag.keywords import (
    extract_high_level_keywords,
    extract_low_level_keywords,
)

# Expansion weights (non-optimized / optimized)
_EDGE_WEIGHT = 0.5
_ENDPOINT_WEIGHT = 0.3
_OPT_PRIMARY = 1.0
_OPT_EDGE = 0.35
_OPT_ENDPOINT = 0.2


def _pool(top_k: int) -> int:
    return max(top_k * 3, 20)


def _sorted(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


# ── Local ────────────────────────────────────────────────────────────────────

def _local_scores(query, G, exp_dir, top_k, optimized, qvec=None) -> dict[str, float]:
    path = collection_path(exp_dir, "entities")
    if path is None or G is None:
        return {}
    qvec = qvec if qvec is not None else embed_query(query)
    hits = query_collection_direct(path, qvec, extract_low_level_keywords(query), top_k=_pool(top_k))
    if not hits:
        return {}
    max_s = max(h.score for h in hits) or 1.0
    scores: dict[str, float] = {}
    for rank, h in enumerate(hits):
        name = h.fields.get("name") or h.id
        if name not in G:
            continue
        w = (h.score / max_s) if max_s > 0 else 1.0 / (rank + 1)
        primary = _OPT_PRIMARY if optimized else 1.0
        for cid in node_chunk_ids(G, name):
            scores[cid] = scores.get(cid, 0.0) + w * primary
        # one-hop expansion over in- and out-edges
        edges = [(name, v, d) for _, v, d in G.out_edges(name, data=True)] + \
                [(u, name, d) for u, _, d in G.in_edges(name, data=True)]
        for u, v, d in edges:
            nbr = v if u == name else u
            if optimized:
                ew = _OPT_EDGE / math.log2(2 + G.degree(nbr))
            else:
                ew = _EDGE_WEIGHT
            for cid in edge_chunk_ids(d):
                scores[cid] = scores.get(cid, 0.0) + w * ew
    return scores


def retrieve_local(query: str, G: Any = None, pr: dict | None = None,
                   exp_dir: Path | str | None = None, top_k: int = 6,
                   optimized: bool = False, rel_store=None) -> list[dict]:
    if not query or not query.strip():
        return []
    return materialize(_sorted(_local_scores(query, G, exp_dir, top_k, optimized)),
                       exp_dir, "local", top_k)


# ── Global ───────────────────────────────────────────────────────────────────

def _global_scores(query, G, exp_dir, top_k, optimized, rel_store=None, qvec=None) -> dict[str, float]:
    path = collection_path(exp_dir, "relationships")
    if path is None:
        return {}
    if rel_store is None:
        from graphrag_lite.retrieval.lightrag.relationship_store import get_relationship_store
        rel_store = get_relationship_store(exp_dir)
    qvec = qvec if qvec is not None else embed_query(query)
    hits = query_collection_direct(path, qvec, extract_high_level_keywords(query), top_k=_pool(top_k))
    if not hits:
        return {}
    max_s = max(h.score for h in hits) or 1.0
    scores: dict[str, float] = {}
    for rank, h in enumerate(hits):
        w = (h.score / max_s) if max_s > 0 else 1.0 / (rank + 1)
        rec = rel_store.get(h.id) if rel_store is not None else None
        src = h.fields.get("source_entity", "")
        tgt = h.fields.get("target_entity", "")
        chunk_ids: list[str] = list(rec.get("chunk_ids", [])) if rec else []
        if not chunk_ids and G is not None and G.has_edge(src, tgt):
            chunk_ids = edge_chunk_ids(G[src][tgt])
        primary = _OPT_PRIMARY if optimized else 1.0
        for cid in chunk_ids:
            scores[cid] = scores.get(cid, 0.0) + w * primary
        if G is None:
            continue
        for ent in (src, tgt):
            if ent not in G:
                continue
            ew = (_OPT_ENDPOINT / math.log2(2 + G.degree(ent))) if optimized else _ENDPOINT_WEIGHT
            for cid in node_chunk_ids(G, ent):
                scores[cid] = scores.get(cid, 0.0) + w * ew
    return scores


def retrieve_global(query: str, G: Any = None, exp_dir: Path | str | None = None,
                    top_k: int = 6, optimized: bool = False, rel_store=None,
                    pr: dict | None = None) -> list[dict]:
    if not query or not query.strip():
        return []
    return materialize(_sorted(_global_scores(query, G, exp_dir, top_k, optimized, rel_store)),
                       exp_dir, "global", top_k)


# ── Hybrid ───────────────────────────────────────────────────────────────────

def retrieve_hybrid(query: str, G: Any = None, pr: dict | None = None,
                    exp_dir: Path | str | None = None, top_k: int = 6,
                    optimized: bool = False, rel_store=None) -> list[dict]:
    if not query or not query.strip():
        return []
    qvec = embed_query(query)
    loc = [c for c, _ in _sorted(_local_scores(query, G, exp_dir, top_k, optimized, qvec))]
    glo = [c for c, _ in _sorted(_global_scores(query, G, exp_dir, top_k, optimized, rel_store, qvec))]
    weights = [1.0, 0.6] if optimized else [1.0, 1.0]
    return materialize(rrf_fuse([loc, glo], weights), exp_dir, "hybrid", top_k)


# ── Mix ──────────────────────────────────────────────────────────────────────

def _chunk_vector_ranking(query, exp_dir, top_k, qvec) -> list[str]:
    """
    The dense side of Mix, built exactly like Pipeline A's query: one hybrid
    (HNSW + BM25, RRF) search with topk = max(top_k, ZVEC_TOP_N_FLAT).
    The topk matters -- Zvec's RRF reranker fuses each sub-query's top-k, so
    a wider pool changes the order (measured on the dev set: topk 8 -> R@6
    0.925, topk 20 -> 0.850).
    """
    path = collection_path(exp_dir, "chunks")
    if path is None:
        return []
    from graphrag_lite.config import get_config
    pool = max(top_k, get_config().ZVEC_TOP_N_FLAT)
    hits = query_collection_direct(path, qvec, query, top_k=pool)
    return [h.id for h in hits]


def _round_robin(lists: list[list[str]]) -> list[str]:
    """Interleave ranked lists (first of each, then second of each ...), deduplicated."""
    out: list[str] = []
    seen: set[str] = set()
    depth = max((len(l) for l in lists), default=0)
    for i in range(depth):
        for l in lists:
            if i < len(l) and l[i] not in seen:
                seen.add(l[i])
                out.append(l[i])
    return out


def _graph_slots(top_k: int) -> int:
    """Slots the optimized Mix gives to graph-only evidence (1 at k=6, 2 at k=10)."""
    return max(1, round(top_k * 0.2))


def retrieve_mix(query: str, G: Any = None, pr: dict | None = None,
                 exp_dir: Path | str | None = None, top_k: int = 6,
                 optimized: bool = False, rel_store=None) -> list[dict]:
    """
    Mix = dense chunk search + graph (Local entity, Global relationship) evidence.

    optimized=False  (MindMesh Mix) -- upstream LightRAG ``mix`` merge: the
        vector, entity and relationship chunk lists are round-robin
        interleaved (vector first), deduplicated.
    optimized=True   (MindMesh Mix Optimized) -- dense-anchored: the dense
        ranking keeps the first top_k - g slots; the last g slots
        (g = _graph_slots(top_k)) go to the best graph-supported chunks that
        dense did not already place, ranked by RRF of the optimized Local
        and Global lists.  Graph evidence can add a bridge chunk dense search
        missed without displacing dense's strongest hits.

    Both designs were chosen after a per-list diagnosis on the 20 internal
    DEV questions (eval/results/internal_dev_4pipelines_qwen2.5_1.5b_20261119.md);
    HotpotQA is the held-out check.  Equal-weight RRF of all three lists (the
    first rebuild) scored R@6 0.650 there because the weak graph lists
    displaced dense hits.
    """
    if not query or not query.strip():
        return []
    qvec = embed_query(query)
    loc = [c for c, _ in _sorted(_local_scores(query, G, exp_dir, top_k, optimized, qvec))]
    glo = [c for c, _ in _sorted(_global_scores(query, G, exp_dir, top_k, optimized, rel_store, qvec))]
    vec = _chunk_vector_ranking(query, exp_dir, top_k, qvec)

    if not optimized:
        merged = _round_robin([vec, loc, glo])
        n = len(merged)
        return materialize([(c, round(1.0 - i / max(n, 1), 6)) for i, c in enumerate(merged)],
                           exp_dir, "mix", top_k)

    g = _graph_slots(top_k)
    head = vec[: max(top_k - g, 0)]
    graph_rank = [c for c, _ in rrf_fuse([loc, glo], [1.0, 1.0])]
    fill = [c for c in graph_rank if c not in head][:g]
    tail = [c for c in vec if c not in head and c not in fill]
    merged = head + fill + tail
    n = len(merged)
    return materialize([(c, round(1.0 - i / max(n, 1), 6)) for i, c in enumerate(merged)],
                       exp_dir, "mix", top_k)
