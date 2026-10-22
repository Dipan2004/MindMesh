"""
probe_v2_collections.py -- prove the v2 collections are usable by the
retrieval layer, without touching the canonical ones.

Read-only against data/cache_v2.  It sets ZVEC_ROOT=data/cache_v2 and the
collection names to the v2 ones, runs the real hybrid_query (dense HNSW +
BM25 FTS, RRF fusion) on both arms, and prints the top hits.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ["ZVEC_ROOT"] = "data/cache_v2"
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"

from graphrag_lite.config import get_config  # noqa: E402

get_config.cache_clear()

from graphrag_lite.embeddings.embedder import embed_batch  # noqa: E402
from graphrag_lite.vectorstore.zvec_client import hybrid_query  # noqa: E402

QUESTIONS = [
    "Who developed the Transformer architecture?",
    "What is retrieval augmented generation?",
    "Which vector databases support hybrid search?",
]


def run(collection: str, field: str) -> None:
    cfg = get_config()
    # hybrid_query maps the collection NAME through the config, so both
    # names have to be set for the branch we want to be taken.
    object.__setattr__(cfg, "ZVEC_COLLECTION_ENTITIES",
                       collection if collection.startswith("entities") else "entities_v2")
    object.__setattr__(cfg, "ZVEC_COLLECTION_CHUNKS", "chunks_v2")
    print(f"\n=== {collection} ===")
    for q in QUESTIONS:
        vec = embed_batch([q])[0]
        hits = hybrid_query(collection, vec, q, top_k=5)
        print(f"  Q: {q}")
        for h in hits:
            name = h.fields.get("name") or h.fields.get("chunk_id")
            desc = (h.fields.get("description") or h.fields.get("text") or "")[:90]
            print(f"     {h.score:.4f}  {name!r}  {desc!r}")
        if not hits:
            print("     (no hits)")


def main() -> None:
    for coll, field in (("entities_v2", "name"), ("entities_v2_stub", "name"),
                        ("chunks_v2", "chunk_id")):
        run(coll, field)


if __name__ == "__main__":
    main()
