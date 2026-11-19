"""chunk_lookup.py -- fetch chunk text/metadata by chunk_id from MindMesh Zvec storage."""

from __future__ import annotations

from pathlib import Path

import zvec

from graphrag_lite.retrieval.lightrag.context import CANONICAL_DIR, collection_path


def fetch_chunks_by_ids(
    chunk_ids: list[str],
    exp_dir: Path | str | None = None,
) -> dict[str, dict]:
    """
    Return {chunk_id: {"text", "source_doc", "chunk_index"}} for the IDs that
    exist.  Looks in the experiment's chunk collection first, then the
    canonical chunks_v1 for any IDs not found there.
    """
    ids = list(dict.fromkeys(c for c in chunk_ids if c))
    out: dict[str, dict] = {}
    if not ids:
        return out
    paths = []
    p = collection_path(exp_dir, "chunks")
    if p is not None:
        paths.append(p)
    canon = CANONICAL_DIR / "chunks_v1"
    if canon.exists() and canon not in paths:
        paths.append(canon)
    for path in paths:
        missing = [c for c in ids if c not in out]
        if not missing:
            break
        col = zvec.open(str(path))
        try:
            docs = col.fetch(missing, include_vector=False)
        finally:
            col.close()
        for cid, doc in docs.items():
            f = doc.fields or {}
            out[cid] = {
                "text": f.get("text", ""),
                "source_doc": f.get("source_doc", ""),
                "chunk_index": f.get("chunk_index", 0),
            }
    return out


def get_chunk_text(chunk_id: str, exp_dir: Path | str | None = None) -> str:
    return fetch_chunks_by_ids([chunk_id], exp_dir=exp_dir).get(chunk_id, {}).get("text", "")
