"""
context.py -- shared storage helpers for the MindMesh-native LightRAG retrievers.

RECONSTRUCTION NOTE (2026-11-19)
--------------------------------
The original ``graphrag_lite/retrieval/lightrag/`` package was never
committed: the repository's ``.gitignore`` line ``LightRAG/`` (meant for the
vendored upstream checkout) also matches ``src/graphrag_lite/retrieval/lightrag/``
on case-insensitive file systems, so git silently skipped it.  Every caller
(adapters, eval scripts, tests, API) imported a module that only existed on
the original author's machine.

This package re-implements the same public API from the documented design
(LIGHTRAG_INTEGRATION.md, adapter docstrings, eval/run_model_experiment.py
Stage E collection layout).  Numbers produced by earlier runs (for example
data/experiments/lightrag_*_mix_opt_*) were made with the lost code and are
NOT guaranteed to be reproduced bit-for-bit by this implementation.

Storage used -- MindMesh's own, never upstream LightRAG's:
  <exp_dir>/entities        Zvec  (name, entity_type, source_doc, pagerank_score, description)
  <exp_dir>/relationships   Zvec  (source_entity, target_entity, rel_type, description, ...)
  <exp_dir>/chunks          Zvec  (chunk_id, source_doc, chunk_index, text)
  <exp_dir>/relationship_records.json   rel_id -> chunk_ids / endpoints
plus the NetworkX DiGraph built by graphrag_lite.graph.builder.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import zvec

from graphrag_lite.vectorstore.zvec_client import ScoredResult

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[4]
CANONICAL_DIR = ROOT / "data" / "cache"
DEFAULT_EXP_DIR = ROOT / "data" / "experiments" / "qwen2_5_1_5b_validate"

_FTS_FIELD = {"entities": "description", "relationships": "description", "chunks": "text"}

# kind -> directory names accepted, in priority order.  "<kind>_v1" lets the
# canonical data/cache layout (entities_v1 / chunks_v1) be used directly.
_DIR_NAMES = {
    "entities": ("entities", "entities_v1"),
    "relationships": ("relationships", "relationships_v1"),
    "chunks": ("chunks", "chunks_v1"),
}


def resolve_exp_dir(exp_dir: Path | str | None) -> Path:
    """Return the storage directory the retrievers should read from."""
    if exp_dir is not None:
        return Path(exp_dir)
    if DEFAULT_EXP_DIR.exists():
        return DEFAULT_EXP_DIR
    return CANONICAL_DIR


# Fallback stores used when an experiment directory holds no collection of a
# kind (eval/run_lightrag_pipeline.py passes its fresh OUTPUT directory as
# exp_dir).  Entities and chunks fall back to the canonical store, which is
# consistent with the canonical graph the adapters load from
# data/cache/extractions.jsonl.  The canonical store has no relationship
# collection, so relationships fall back to the Phase-1 validation
# experiment, whose relationship provenance uses the canonical chunk IDs.
_FALLBACKS = {
    "entities": [CANONICAL_DIR / "entities_v1"],
    "chunks": [CANONICAL_DIR / "chunks_v1"],
    "relationships": [DEFAULT_EXP_DIR / "relationships"],
}


def _has_collection(p: Path) -> bool:
    return p.is_dir() and any(p.iterdir())


def collection_path(exp_dir: Path | str | None, kind: str) -> Path | None:
    """Locate the Zvec collection of ``kind`` for an experiment directory."""
    if exp_dir is not None:
        base = Path(exp_dir)
        for name in _DIR_NAMES[kind]:
            if _has_collection(base / name):
                return base / name
    for p in _FALLBACKS[kind]:
        if _has_collection(p):
            return p
    return None


# Single-query embedding cache: the optimized variants embed a query once and
# reuse the vector for entity, relationship and chunk searches.
_EMBED_CACHE: dict[str, list[float]] = {}


def embed_query(query: str) -> list[float]:
    vec = _EMBED_CACHE.get(query)
    if vec is None:
        from graphrag_lite.embeddings.embedder import embed_batch
        vec = embed_batch([query])[0]
        if len(_EMBED_CACHE) > 512:
            _EMBED_CACHE.clear()
        _EMBED_CACHE[query] = vec
    return vec


def _clean_fts(text: str) -> str:
    import re
    t = re.sub(r"--+", " ", text)
    t = re.sub(r"\s+-\s+", " ", t)
    # FTS parser rejects quotes / brackets / colons in some builds.
    t = re.sub(r"[\"'():\[\]{}?!*^~\\/]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def query_collection_direct(
    path: Path | str,
    query_vec: list[float],
    query_text: str,
    top_k: int,
    fts_field: str | None = None,
    scalar_filter: str | None = None,
) -> list[ScoredResult]:
    """
    Hybrid (dense HNSW + BM25) query against an arbitrary Zvec collection
    path, fused with RRF.  Falls back to dense-only if the FTS side fails.
    """
    path = str(path)
    kind = Path(path).name.replace("_v1", "")
    fts_field = fts_field or _FTS_FIELD.get(kind, "text")
    col = zvec.open(path)
    try:
        fts_text = _clean_fts(query_text)
        try:
            if not fts_text:
                raise ValueError("empty fts text")
            res = col.query(
                queries=[
                    zvec.Query("text_vec", vector=query_vec,
                               param=zvec.HnswQueryParam(ef=max(top_k * 4, 64))),
                    zvec.Query(fts_field, fts=zvec.Fts(fts_text)),
                ],
                topk=top_k,
                filter=scalar_filter,
                reranker=zvec.RrfReRanker(),
            )
        except Exception:
            res = col.query(
                queries=[zvec.Query("text_vec", vector=query_vec,
                                    param=zvec.HnswQueryParam(ef=max(top_k * 4, 64)))],
                topk=top_k,
                filter=scalar_filter,
            )
        return [
            ScoredResult(id=d.id, score=d.score if d.score is not None else 0.0,
                         fields=d.fields or {})
            for d in res
        ]
    finally:
        col.close()


def node_chunk_ids(G: Any, node: str) -> list[str]:
    """Provenance chunk IDs of a graph node, in extraction order, deduplicated."""
    nd = G.nodes[node]
    seen: dict[str, None] = {}
    for m in list(nd.get("descriptions", [])) + list(nd.get("source_sentences", [])):
        cid = m.get("chunk_id") if isinstance(m, dict) else None
        if cid:
            seen.setdefault(cid, None)
    return list(seen)


def edge_chunk_ids(edge_data: dict) -> list[str]:
    seen: dict[str, None] = {}
    for ss in edge_data.get("source_sentences", []):
        cid = ss.get("chunk_id")
        if cid:
            seen.setdefault(cid, None)
    return list(seen)


def rrf_fuse(
    ranked_lists: list[list[str]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Weighted reciprocal-rank fusion of ranked ID lists."""
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, cid in enumerate(lst):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def materialize(
    scored: list[tuple[str, float]],
    exp_dir: Path | str | None,
    source: str,
    top_k: int,
) -> list[dict]:
    """Turn (chunk_id, score) pairs into evidence dicts with chunk text."""
    from graphrag_lite.retrieval.lightrag.chunk_lookup import fetch_chunks_by_ids
    picked = scored[:top_k]
    if not picked:
        return []
    meta = fetch_chunks_by_ids([cid for cid, _ in picked], exp_dir=exp_dir)
    out = []
    for cid, sc in picked:
        m = meta.get(cid)
        if m is None:
            log.warning("chunk %s referenced by graph but missing from chunk store", cid)
            continue
        out.append({
            "chunk_id": cid,
            "source_doc": m.get("source_doc", ""),
            "chunk_index": int(m.get("chunk_index", 0) or 0),
            "text": m.get("text", ""),
            "score": round(float(sc), 6),
            "source": source,
        })
    return out
