"""
query_focused.py -- query-focused graph retrieval (prototype "query_focused_c5").

A NEW retrieval variant. The four existing pipelines are not changed.

Per question:
  1. Dense seeds: the Pipeline A query (one hybrid HNSW+BM25 search on chunks,
     topk = max(top_k, ZVEC_TOP_N_FLAT)); the first SEED_K results are seeds.
  2. Expansion (one cheap hop, no LLM needed to decide):
       a. paragraphs whose title (without a trailing "(...)" qualifier) appears in
          the question, and
       b. paragraphs whose title matches an entity extracted from a seed paragraph.
  3. Extraction on demand: per-paragraph records are taken from the extraction
     cache (``extractions.jsonl``). Extraction is per paragraph and context-free,
     so a cached record is identical to what on-demand extraction would produce.
     Every paragraph used is recorded in ``touched`` for cold-cost accounting.
  4. A temporary graph is built from only the selected paragraphs' records
     (MindMesh's own builder).
  5. Anchors: graph nodes whose normalised name occurs in the question as a
     whole phrase (>= 3 characters). BFS on the undirected graph, <= HOPS hops.
  6. Ranking: each reached node adds 1/(1+d) to its provenance chunks; each edge
     between reached nodes adds 0.5/(1+min(d_u, d_v)) to its provenance chunks.
     Chunks with support are ranked by support (ties by dense rank), then the
     remaining dense results in dense order, then any unsupported expansion
     paragraphs. With no anchors, the dense order is returned unchanged.

Design fixed on 2026-11-19 before any 100-question result for this variant existed.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from pathlib import Path

import networkx as nx

from graphrag_lite.graph.builder import _norm, build_graph

SEED_K = 5
HOPS = 2
EDGE_WEIGHT = 0.5
MIN_ANCHOR_CHARS = 3
MIN_TITLE_CHARS = 4

_INDEX_CACHE: dict[str, "QueryFocusedIndex"] = {}


def _phrase_in(phrase: str, text: str) -> bool:
    return bool(phrase) and re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def _base_title(title: str) -> str:
    return _norm(re.sub(r"\s*\([^)]*\)\s*$", "", title))


class QueryFocusedIndex:
    """Extraction cache + title index for one experiment directory."""

    def __init__(self, exp_dir: Path | str):
        self.exp_dir = Path(exp_dir)
        self.records: dict[str, dict] = {}
        self.titles: dict[str, str] = {}
        p = self.exp_dir / "extractions.jsonl"
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cid = r["chunk_id"]
            self.titles[cid] = r.get("title") or (r.get("text", "").split("\n", 1)[0])
            if "_parse_error" not in r:
                self.records[cid] = r
        self.by_full_title: dict[str, list[str]] = {}
        self.by_base_title: dict[str, list[str]] = {}
        for cid, t in self.titles.items():
            self.by_full_title.setdefault(_norm(t), []).append(cid)
            self.by_base_title.setdefault(_base_title(t), []).append(cid)
        self.touched: set[str] = set()

    def expand(self, query: str, seeds: list[str]) -> tuple[set[str], dict]:
        qn = _norm(query)
        S = set(seeds)
        from_question = {cid for bt, cids in self.by_base_title.items()
                         if len(bt) >= MIN_TITLE_CHARS and _phrase_in(bt, qn) for cid in cids}
        ents = {_norm(e["name"]) for c in seeds if c in self.records for e in self.records[c]["entities"]}
        from_entities = set()
        for e in ents:
            from_entities.update(self.by_full_title.get(e, []))
            if len(e) >= MIN_TITLE_CHARS:
                from_entities.update(self.by_base_title.get(e, []))
        S |= from_question | from_entities
        return S, {"from_question": sorted(from_question - set(seeds)),
                   "from_entities": sorted(from_entities - set(seeds) - from_question)}


def get_index(exp_dir: Path | str) -> QueryFocusedIndex:
    key = str(Path(exp_dir).resolve())
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = QueryFocusedIndex(exp_dir)
    return _INDEX_CACHE[key]


def dense_ranking(query: str, top_k: int) -> list[str]:
    """Exactly the Pipeline A query (hybrid HNSW + BM25 on the chunks collection)."""
    from graphrag_lite.config import get_config
    from graphrag_lite.embeddings.embedder import embed_batch
    from graphrag_lite.vectorstore.zvec_client import hybrid_query
    cfg = get_config()
    qv = embed_batch([query])[0]
    hits = hybrid_query(collection_name=cfg.ZVEC_COLLECTION_CHUNKS, query_vector=qv, query_text=query,
                        top_k=max(top_k, cfg.ZVEC_TOP_N_FLAT))
    return [h.id for h in hits]


def rank_query_focused(query: str, idx: QueryFocusedIndex, dense: list[str], top_k: int) -> tuple[list[str], dict]:
    seeds = dense[:SEED_K]
    S, exp_info = idx.expand(query, seeds)
    idx.touched |= S
    recs = [idx.records[c] for c in S if c in idx.records]
    with contextlib.redirect_stdout(io.StringIO()):
        G = build_graph(recs)
    qn = _norm(query)
    anchors = [n for n in G.nodes if len(_norm(n)) >= MIN_ANCHOR_CHARS and _phrase_in(_norm(n), qn)]
    info = {"selected_paragraphs": sorted(S), "n_selected": len(S), "anchors": sorted(anchors),
            "graph_nodes": G.number_of_nodes(), "graph_edges": G.number_of_edges(), **exp_info}
    dense_rank = {c: i for i, c in enumerate(dense)}
    if not anchors:
        info["fallback"] = "no question entity found in the query graph; dense order"
        return dense[:top_k], info
    U = G.to_undirected(as_view=True)
    dist: dict[str, int] = {}
    for a in anchors:
        for n, d in nx.single_source_shortest_path_length(U, a, cutoff=HOPS).items():
            dist[n] = min(d, dist.get(n, d))
    score: dict[str, float] = {}
    for n, d in dist.items():
        nd = G.nodes[n]
        cids = {m.get("chunk_id") for m in nd.get("source_sentences", []) + nd.get("descriptions", []) if m.get("chunk_id")}
        for c in cids:
            if c in S:
                score[c] = score.get(c, 0.0) + 1.0 / (1 + d)
    for u, v, ed in G.edges(data=True):
        if u in dist and v in dist:
            w = EDGE_WEIGHT / (1 + min(dist[u], dist[v]))
            for c in {ss.get("chunk_id") for ss in ed.get("source_sentences", []) if ss.get("chunk_id")}:
                if c in S:
                    score[c] = score.get(c, 0.0) + w
    supported = sorted(score, key=lambda c: (-score[c], dense_rank.get(c, 10**6), c))
    ranked = supported + [c for c in dense if c not in score]
    ranked += sorted(c for c in S if c not in ranked)
    info["supported_paragraphs"] = len(supported)
    return ranked[:top_k], info


def retrieve_query_focused(query: str, exp_dir: Path | str, top_k: int = 10,
                           dense_ids: list[str] | None = None) -> tuple[list[dict], dict]:
    """Return (evidence dicts in rank order, diagnostics)."""
    if not query or not query.strip():
        return [], {}
    idx = get_index(exp_dir)
    dense = dense_ids if dense_ids is not None else dense_ranking(query, top_k)
    ranked, info = rank_query_focused(query, idx, dense, top_k)
    from graphrag_lite.retrieval.lightrag.chunk_lookup import fetch_chunks_by_ids
    meta = fetch_chunks_by_ids(ranked, exp_dir=exp_dir)
    ev = [{"chunk_id": c, "source_doc": meta.get(c, {}).get("source_doc", ""),
           "chunk_index": int(meta.get(c, {}).get("chunk_index", 0) or 0),
           "text": meta.get(c, {}).get("text", ""), "score": round(1.0 - i / max(len(ranked), 1), 6),
           "source": "query_focused"} for i, c in enumerate(ranked) if c in meta]
    info["touched_total"] = len(idx.touched)
    return ev, info
