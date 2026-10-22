"""
pipeline_a.py -- Pipeline A: flat vector-only baseline retrieval.

No graph structure involved. Single Zvec hybrid query against the chunks
collection, top_k = ZVEC_TOP_N_FLAT.

This is the baseline that Pipeline B must beat on accuracy at comparable
or lower cost. The two pipelines share the same Evidence type and the same
serialization layer so the paper comparison isolates retrieval method only.

Usage:
    from graphrag_lite.retrieval.pipeline_a import retrieve_flat
    evidence = retrieve_flat("what is HNSW?")
"""

from __future__ import annotations

from dataclasses import dataclass, field

from graphrag_lite.config import get_config
from graphrag_lite.embeddings.embedder import embed_batch
from graphrag_lite.vectorstore.zvec_client import hybrid_query


@dataclass
class Evidence:
    """
    One piece of retrieved evidence, shared by Pipeline A and Pipeline B.

    Fields
    ------
    chunk_id    : str   -- stable chunk identifier from chunker.py
    source_doc  : str   -- doc_id of the source document
    chunk_index : int   -- 0-based position within the document
    text        : str   -- raw chunk text (Pipeline A) or entity description (Pipeline B)
    score       : float -- Zvec hybrid RRF score
    source      : str   -- 'chunk' or 'entity' — distinguishes A vs B evidence
    """
    chunk_id: str
    source_doc: str
    chunk_index: int
    text: str
    score: float
    source: str = "chunk"


def retrieve_flat(query: str) -> list[Evidence]:
    """
    Pipeline A: embed query, hybrid search against chunks_v1, return Evidence.

    Parameters
    ----------
    query : str
        Natural language query string.

    Returns
    -------
    list[Evidence]
        Up to ZVEC_TOP_N_FLAT results, sorted by descending hybrid score.

    Raises
    ------
    RuntimeError
        If EMBEDDING_DIM is 0 (embed_batch never called) or Zvec collection
        does not exist (insert_chunks not yet run).
    ValueError
        If query is empty.
    """
    if not query.strip():
        raise ValueError("retrieve_flat: query must not be empty")

    cfg = get_config()

    # Embed the query
    query_vec = embed_batch([query])[0]

    # Hybrid query against the chunks collection
    raw_results = hybrid_query(
        collection_name=cfg.ZVEC_COLLECTION_CHUNKS,
        query_vector=query_vec,
        query_text=query,
        top_k=cfg.ZVEC_TOP_N_FLAT,
    )

    evidence = []
    for r in raw_results:
        evidence.append(Evidence(
            chunk_id=r.id,
            source_doc=r.fields.get("source_doc", ""),
            chunk_index=int(r.fields.get("chunk_index", 0)),
            text=r.fields.get("text", ""),
            score=r.score,
            source="chunk",
        ))

    return evidence
