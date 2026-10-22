"""
pipeline_b.py -- Pipeline B: structural graph filter + Zvec hybrid retrieval.

Steps per PRD S9:
  Step 1 -- Entry points: hybrid_query on entities_v1 (top ZVEC_TOP_K_ENTRY),
             with type-label entry-point filtering.
  Step 2 -- BFS expansion: bfs_expand from entry points (BFS_HOP_DEPTH hops)
  Step 3 -- Rank by combined score: sim_norm * PIPELINE_B_SIM_WEIGHT
             + pagerank_norm * PIPELINE_B_PR_WEIGHT, where sim_norm is the
             entry-point similarity normalised to [0,1]. Keep
             TOP_N_CANDIDATES.
  Step 4 -- Scoped re-query: hybrid_query on chunks_v1 scoped to candidate docs,
             keep TOP_N_FINAL_EVIDENCE

Evidence type is the same as Pipeline A so serialisation/generation are shared.

Design notes:
  - Entity similarity: the Zvec score from step 1 for each entry-point entity.
    Propagated to BFS neighbours so the ranking depends on the query, not
    just global PageRank. The propagation is a MAX, not a decay: a node
    inherits the highest entry-point similarity of any entry point that can
    reach it within BFS_HOP_DEPTH hops. There is no distance discount, so
    a node two hops from a very strong entry point scores as highly as the
    entry point itself. This is intentional -- it keeps the signal
    query-dependent -- but it is not a weighted graph distance.
  - Similarity is NORMALISED before weighting. Raw Zvec RRF scores are
    ~0.016-0.033 while normalised PageRank is ~0-1; mixing them raw lets
    PageRank dominate ~30:1 and the same high-PageRank node ('Temporal
    knowledge graphs') wins 5 of 10 gold queries. After normalisation the
    query-dependent term dominates and PageRank is a tiebreak.
  - Type-label filter: entry points whose name matches a generic taxonomy term
    (Person, Organization, Location, etc.) are excluded -- they dominate
    PageRank but are not meaningful retrieval targets. The same filter is
    applied to the ranked candidates, because BFS can reintroduce them.
  - retrieve_structural(query, top_k=None) accepts an explicit k so A and B
    can be compared at the same k without mutating global config.
"""

from __future__ import annotations

from graphrag_lite.config import get_config
from graphrag_lite.embeddings.embedder import embed_batch
from graphrag_lite.graph.builder import build_graph, load_extractions, _norm
from graphrag_lite.graph.ranking import rank_pagerank, rank_degree
from graphrag_lite.graph.traversal import bfs_expand
from graphrag_lite.retrieval.pipeline_a import Evidence
from graphrag_lite.vectorstore.zvec_client import hybrid_query

# Nodes whose name is a generic taxonomy type label -- not meaningful entry points
_TYPE_LABEL_NAMES: frozenset[str] = frozenset({
    "person", "organization", "location", "event", "technology",
    "other", "entity", "node", "edge", "concept", "relationship",
    "type", "class", "object", "thing",
})

# Module-level graph + PageRank singleton (built once, reused across queries)
_G = None
_PR = None
_PR_MAX = None


def _get_graph():
    global _G, _PR, _PR_MAX
    if _G is None:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            records = load_extractions("data/cache/extractions.jsonl")
            _G = build_graph(records)
        # RANKING_METHOD selects the node-ranking signal (CONFIG.md §5).
        # The default 'pagerank' is what the measured A@6/B@6 numbers were
        # produced with; 'degree' is the documented ablation alternative.
        if get_config().RANKING_METHOD == "degree":
            _PR = {k: float(v) for k, v in rank_degree(_G).items()}
        else:
            _PR = rank_pagerank(_G)
        _PR_MAX = max(_PR.values()) if _PR else 1.0
    return _G, _PR, _PR_MAX


def get_candidates(query: str) -> dict:
    """
    Pipeline B steps 1-3: return ranked candidate nodes.

    Returns
    -------
    dict with keys:
      entry_points         : list[str] -- entity names found by Zvec (after type-label filter)
      entry_points_filtered: list[str] -- names removed by type-label filter
      bfs_nodes            : list[str] -- all nodes reachable within BFS_HOP_DEPTH hops
      ranked_candidates    : list[tuple[str, float]] -- (node, combined_score) truncated
                             to TOP_N_CANDIDATES_AFTER_RANKING
    """
    cfg = get_config()
    G, pr, pr_max = _get_graph()

    # Step 1 -- embed query and find entry-point entities
    q_vec = embed_batch([query])[0]
    raw_results = hybrid_query(
        collection_name=cfg.ZVEC_COLLECTION_ENTITIES,
        query_vector=q_vec,
        query_text=query,
        top_k=cfg.ZVEC_TOP_K_ENTRY,
    )

    # Map Zvec doc IDs back to canonical node names; apply type-label filter
    entry_points = []
    filtered_out = []
    entry_sim: dict[str, float] = {}   # entity name -> Zvec similarity score

    for r in raw_results:
        name = r.fields.get("name") or r.id.replace("_", " ")
        if name not in G:
            continue
        if _norm(name) in _TYPE_LABEL_NAMES:
            filtered_out.append(name)
            continue
        entry_points.append(name)
        entry_sim[name] = r.score

    # Step 2 -- BFS expansion
    bfs_nodes = bfs_expand(G, entry_points, hops=cfg.BFS_HOP_DEPTH)

    # Step 3 -- Combined score, query-DEPENDENT first.
    #
    # The similarity term is the query-dependent signal; the PageRank term
    # is query-independent. Raw Zvec RRF scores are ~0.016-0.033 while
    # normalised PageRank is ~0-1, so an unnormalised mix lets PageRank
    # dominate by ~30x and the SAME high-PageRank node wins every query.
    # Normalising the similarity against the best entry-point score puts
    # both terms on a 0-1 scale, so the weights mean what they say and
    # PageRank is reduced to a tiebreak.
    #
    # Propagate similarity to BFS neighbours: a neighbour inherits the max
    # similarity of any entry point that can reach it. This is a MAX
    # propagation, not a distance decay -- see the module docstring.
    node_sim: dict[str, float] = {}
    for ep, sim in entry_sim.items():
        reachable = bfs_expand(G, [ep], hops=cfg.BFS_HOP_DEPTH)
        for n in reachable:
            if n not in node_sim or sim > node_sim[n]:
                node_sim[n] = sim

    sim_max = max(entry_sim.values()) if entry_sim else 0.0

    # Compute combined score for each BFS node
    ranked = []
    for n in bfs_nodes:
        sim_score = node_sim.get(n, 0.0)
        sim_norm = (sim_score / sim_max) if sim_max > 0 else 0.0
        pr_norm = pr.get(n, 0.0) / pr_max if pr_max > 0 else 0.0
        combined = (cfg.PIPELINE_B_SIM_WEIGHT * sim_norm
                    + cfg.PIPELINE_B_PR_WEIGHT * pr_norm)
        ranked.append((n, combined))

    ranked.sort(key=lambda x: (-x[1], x[0]))
    ranked = ranked[: cfg.TOP_N_CANDIDATES_AFTER_RANKING]

    if cfg.PIPELINE_B_DROP_TYPE_LABELS:
        # Type labels are the highest-PageRank nodes in the graph and are
        # never meaningful retrieval targets. They can reach the candidate
        # list through BFS even though the entry-point filter removed them.
        ranked = [(n, s) for n, s in ranked if _norm(n) not in _TYPE_LABEL_NAMES]

    return {
        "entry_points": entry_points,
        "entry_points_filtered": filtered_out,
        "bfs_nodes": bfs_nodes,
        "ranked_candidates": ranked,
        "sim_max": sim_max,
    }


def retrieve_structural(query: str, top_k: int | None = None) -> list[Evidence]:
    """
    Pipeline B steps 1-4: return final evidence.

    Parameters
    ----------
    query : str
        Natural language query string.
    top_k : int | None
        How many evidence pieces to return. Defaults to
        ``TOP_N_FINAL_EVIDENCE`` from config. Overriding it lets a caller
        compare Pipeline A and Pipeline B at the SAME k without mutating
        global config.
    """
    if not query.strip():
        raise ValueError("retrieve_structural: query must not be empty")

    cfg = get_config()
    if top_k is None:
        top_k = cfg.TOP_N_FINAL_EVIDENCE
    G, _, _ = _get_graph()
    candidates_info = get_candidates(query)
    ranked = candidates_info["ranked_candidates"]

    if not ranked:
        return []

    # Collect source_docs from candidates
    candidate_docs: set[str] = set()
    for node_name, _ in ranked:
        if node_name in G:
            candidate_docs.update(G.nodes[node_name].get("source_docs", set()))

    if not candidate_docs:
        return []

    q_vec = embed_batch([query])[0]
    doc_filter = " OR ".join(f"source_doc = '{d}'" for d in sorted(candidate_docs))

    raw_results = hybrid_query(
        collection_name=cfg.ZVEC_COLLECTION_CHUNKS,
        query_vector=q_vec,
        query_text=query,
        top_k=top_k,
        scalar_filters=doc_filter,
    )

    return [
        Evidence(
            chunk_id=r.id,
            source_doc=r.fields.get("source_doc", ""),
            chunk_index=int(r.fields.get("chunk_index", 0)),
            text=r.fields.get("text", ""),
            score=r.score,
            source="structural",
        )
        for r in raw_results
    ]
