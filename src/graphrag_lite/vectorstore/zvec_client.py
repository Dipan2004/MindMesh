"""
zvec_client.py -- Zvec collection management for entities and chunks.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zvec

from graphrag_lite.config import get_config


def _to_doc_id(name: str) -> str:
    """Convert entity name to a valid Zvec doc ID (spaces and / -> _)."""
    return name.replace(" ", "_").replace("/", "_")


@dataclass
class ScoredResult:
    id: str
    score: float
    fields: dict[str, Any]


def _zvec_root() -> str:
    """
    Return the directory holding the Zvec collections.

    ZVEC_ROOT wins when set; otherwise the collections live under CACHE_DIR.
    Splitting the two lets an experiment use a private LLM-response cache
    (CACHE_DIR) without losing access to the canonical collections.
    """
    cfg = get_config()
    return cfg.ZVEC_ROOT or cfg.CACHE_DIR


def _entities_path() -> str:
    cfg = get_config()
    return str(Path(_zvec_root()) / cfg.ZVEC_COLLECTION_ENTITIES)


def _chunks_path() -> str:
    cfg = get_config()
    return str(Path(_zvec_root()) / cfg.ZVEC_COLLECTION_CHUNKS)


def _entities_schema(dim: int) -> zvec.CollectionSchema:
    return zvec.CollectionSchema(
        name="entities_v1",
        fields=[
            zvec.FieldSchema("name",           zvec.DataType.STRING),
            zvec.FieldSchema("entity_type",    zvec.DataType.STRING),
            zvec.FieldSchema("source_doc",     zvec.DataType.STRING),
            zvec.FieldSchema("pagerank_score", zvec.DataType.FLOAT),
            zvec.FieldSchema("description",    zvec.DataType.STRING,
                             index_param=zvec.FtsIndexParam()),
        ],
        vectors=[
            zvec.VectorSchema("text_vec", zvec.DataType.VECTOR_FP32, dimension=dim,
                              index_param=zvec.HnswIndexParam())
        ],
    )


def _chunks_schema(dim: int) -> zvec.CollectionSchema:
    return zvec.CollectionSchema(
        name="chunks_v1",
        fields=[
            zvec.FieldSchema("chunk_id",    zvec.DataType.STRING),
            zvec.FieldSchema("source_doc",  zvec.DataType.STRING),
            zvec.FieldSchema("chunk_index", zvec.DataType.INT32),
            zvec.FieldSchema("text",        zvec.DataType.STRING,
                             index_param=zvec.FtsIndexParam()),
        ],
        vectors=[
            zvec.VectorSchema("text_vec", zvec.DataType.VECTOR_FP32, dimension=dim,
                              index_param=zvec.HnswIndexParam())
        ],
    )


def reset_collection(path: str) -> None:
    """
    Explicitly delete a collection directory so it can be rebuilt.
    The ONLY function in this module that calls shutil.rmtree.
    Refuses to delete directories containing non-Zvec files.
    """
    p = Path(path)
    if not p.exists():
        return
    known = {"LOCK", "chunking_strategy.txt", "0", "del.0", "idmap.0"}
    files = {f.name for f in p.iterdir()}
    zvec_files = {f for f in files if f in known or
                  f.startswith("manifest.") or f.startswith("idmap.") or f.startswith("del.")}
    unexpected = files - zvec_files
    if unexpected:
        raise RuntimeError(
            f"reset_collection: {path!r} contains unexpected files {unexpected}. "
            f"Remove manually if intentional."
        )
    shutil.rmtree(path)


def _check_strategy_tag(path: str, cfg) -> None:
    """
    Read-only strategy tag check used by both open/create and query paths.
    Raises RuntimeError on mismatch. Handles missing tag via ZVEC_ADOPT_UNTAGGED.
    Never modifies the directory.
    """
    strategy_tag_path = Path(path) / "chunking_strategy.txt"
    if strategy_tag_path.exists():
        stored = strategy_tag_path.read_text().strip()
        current = cfg.CHUNKING_STRATEGY
        if stored != current:
            raise RuntimeError(
                f"Collection at {path!r} was built with "
                f"CHUNKING_STRATEGY={stored!r} but current config is "
                f"{current!r}. Change ZVEC_COLLECTION_* names in .env "
                f"to point to a new collection, or call reset_collection()."
            )
    else:
        adopt = cfg.ZVEC_ADOPT_UNTAGGED
        if not adopt:
            raise RuntimeError(
                f"Collection at {path!r} has no chunking_strategy.txt tag. "
                f"Set ZVEC_ADOPT_UNTAGGED=1 in .env to adopt it as "
                f"{cfg.CHUNKING_STRATEGY!r}, or call reset_collection() and rebuild."
            )
        print(f"  [ZVEC_ADOPT_UNTAGGED] Tagging {path!r} as "
              f"CHUNKING_STRATEGY={cfg.CHUNKING_STRATEGY!r}")
        strategy_tag_path.write_text(cfg.CHUNKING_STRATEGY)


def _open_or_create(path: str, schema_fn) -> zvec.Collection:
    """
    Open an existing Zvec collection, or create one if none exists.
    Never calls shutil.rmtree. Use reset_collection(path) to rebuild.
    """
    cfg = get_config()
    dim = cfg.EMBEDDING_DIM
    if dim == 0:
        raise RuntimeError(
            "EMBEDDING_DIM is 0 -- call embed_batch() before opening a Zvec collection."
        )

    try:
        _probe = zvec.open(path)
        _probe.close()
        collection_exists = True
        open_error = None
    except Exception as exc:
        collection_exists = False
        open_error = exc

    if collection_exists:
        _check_strategy_tag(path, cfg)
        return zvec.open(path)

    p = Path(path)
    if p.exists():
        contents = list(p.iterdir())
        if contents:
            raise RuntimeError(
                f"Cannot open or create collection at {path!r}: "
                f"directory is non-empty but zvec.open failed "
                f"({type(open_error).__name__}: {open_error}). "
                f"Inspect manually or call reset_collection({path!r})."
            ) from open_error
        p.rmdir()

    col = zvec.create_and_open(path, schema_fn(dim))
    (Path(path) / "chunking_strategy.txt").write_text(cfg.CHUNKING_STRATEGY)
    return col


def get_entities_collection() -> zvec.Collection:
    return _open_or_create(_entities_path(), _entities_schema)


def get_chunks_collection() -> zvec.Collection:
    return _open_or_create(_chunks_path(), _chunks_schema)


def insert_entities(
    entities: list[dict],
    embeddings: list[list[float]],
    pagerank_scores: dict[str, float] | None = None,
) -> None:
    """Insert entity records into entities_v1. Closes on exit."""
    if len(entities) != len(embeddings):
        raise ValueError(f"insert_entities: {len(entities)} entities vs {len(embeddings)} embeddings")
    col = get_entities_collection()
    pr = pagerank_scores or {}
    docs = []
    for entity, vec in zip(entities, embeddings):
        name = entity.get("name", "")
        descs = entity.get("descriptions", [])
        first_desc = descs[0]["text"] if descs else ""
        src_docs = entity.get("source_docs", set())
        primary = next(iter(src_docs)) if src_docs else ""
        docs.append(zvec.Doc(
            id=_to_doc_id(name),
            vectors={"text_vec": vec},
            fields={
                "name": name, "entity_type": entity.get("type", "OTHER"),
                "source_doc": primary, "pagerank_score": float(pr.get(name, 0.0)),
                "description": first_desc,
            },
        ))
    col.insert(docs)
    col.flush()
    col.close()


def insert_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """Insert chunk records into chunks_v1. Closes on exit."""
    if len(chunks) != len(embeddings):
        raise ValueError(f"insert_chunks: {len(chunks)} chunks vs {len(embeddings)} embeddings")
    col = get_chunks_collection()
    docs = []
    for chunk, vec in zip(chunks, embeddings):
        docs.append(zvec.Doc(
            id=chunk["chunk_id"],
            vectors={"text_vec": vec},
            fields={
                "chunk_id": chunk["chunk_id"], "source_doc": chunk["doc_id"],
                "chunk_index": chunk["chunk_index"], "text": chunk["text"],
            },
        ))
    col.insert(docs)
    col.flush()
    col.close()


def hybrid_query(
    collection_name: str,
    query_vector: list[float],
    query_text: str,
    top_k: int,
    scalar_filters: str | None = None,
) -> list[ScoredResult]:
    """
    Hybrid (dense HNSW + BM25 FTS) query fused with RRF.

    Guards the query path with the same strategy-tag check as inserts.
    Read-only: never writes to the directory (unless ZVEC_ADOPT_UNTAGGED=1
    and tag is missing, in which case the tag is written -- same behaviour
    as the open path).

    collection_name: 'entities_v1' or 'chunks_v1'
    Returns list[ScoredResult] sorted by fused RRF score descending.
    """
    cfg = get_config()
    if collection_name == cfg.ZVEC_COLLECTION_ENTITIES:
        path = _entities_path()
        fts_field = "description"
    elif collection_name == cfg.ZVEC_COLLECTION_CHUNKS:
        path = _chunks_path()
        fts_field = "text"
    else:
        raise ValueError(f"Unknown collection: {collection_name!r}")

    # Strategy tag check before opening for query (same guard as insert path)
    _check_strategy_tag(path, cfg)

    col = zvec.open(path)
    try:
        result: zvec.DocList = col.query(
            queries=[
                zvec.Query("text_vec", vector=query_vector,
                           param=zvec.HnswQueryParam(ef=64)),
                zvec.Query(fts_field, fts=zvec.Fts(query_text)),
            ],
            topk=top_k,
            filter=scalar_filters,
            reranker=zvec.RrfReRanker(),
        )
        return [
            ScoredResult(
                id=doc.id,
                score=doc.score if doc.score is not None else 0.0,
                fields=doc.fields or {},
            )
            for doc in result
        ]
    finally:
        col.close()
