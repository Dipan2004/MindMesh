"""
eval/hotpotqa/build_dense_index.py -- Dense-only Zvec index for a HotpotQA corpus (no LLM extraction).

    python eval/hotpotqa/build_dense_index.py --corpus <corpus.jsonl> --out <dir>

Embeds every paragraph with the configured embedding model (nomic-embed-text) into a
`chunks` Zvec collection with the same schema, index types and chunk IDs as the full
MindMesh index (HNSW text_vec + BM25 on text, RRF at query time). Writes nothing else;
existing indexes are only read. Refuses to overwrite an existing collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = args.out.resolve()
    if (out / "chunks").exists():
        print(f"[REFUSED] {out / 'chunks'} already exists", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    from mindmesh.benchmarks.hotpotqa.env import CHUNKING_TAG, activate_index
    activate_index(out)
    import zvec
    from graphrag_lite.config import get_config
    from graphrag_lite.embeddings.embedder import embed_batch
    cfg = get_config()

    paras = [json.loads(l) for l in args.corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    t0 = time.monotonic()
    vecs = embed_batch([p["text"] for p in paras])
    embed_s = time.monotonic() - t0
    dim = len(vecs[0])
    schema = zvec.CollectionSchema(
        name="chunks_dense_only",
        fields=[zvec.FieldSchema("chunk_id", zvec.DataType.STRING),
                zvec.FieldSchema("source_doc", zvec.DataType.STRING),
                zvec.FieldSchema("chunk_index", zvec.DataType.INT32),
                zvec.FieldSchema("text", zvec.DataType.STRING, index_param=zvec.FtsIndexParam())],
        vectors=[zvec.VectorSchema("text_vec", zvec.DataType.VECTOR_FP32, dimension=dim,
                                   index_param=zvec.HnswIndexParam())])
    col = zvec.create_and_open(str(out / "chunks"), schema)
    docs = [zvec.Doc(id=p["chunk_id"], vectors={"text_vec": v},
                     fields={"chunk_id": p["chunk_id"], "source_doc": p["doc_id"], "chunk_index": 0, "text": p["text"]})
            for p, v in zip(paras, vecs)]
    for i in range(0, len(docs), 1000):
        col.insert(docs[i:i + 1000])
    col.flush()
    col.close()
    (out / "chunks" / "chunking_strategy.txt").write_text(CHUNKING_TAG)
    meta = {"corpus": str(args.corpus.resolve()), "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
            "paragraphs": len(paras), "embedding_model": cfg.EMBEDDING_MODEL, "embedding_dim": dim,
            "collection": "chunks (HNSW text_vec + BM25 text)", "llm_extraction": False,
            "embed_s": round(embed_s, 1), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "dense_index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
