"""
dump_chunk_inventory.py -- print the chunk inventory of the corpus.

Deterministic, no LLM calls: it chunks data/raw/ with the configured
strategy and prints, for every chunk, the id, the document, the index and
the first lines of the text.  This is the material the 40-question gold set
is written from -- the questions are written by reading the corpus, and the
chunk ids are looked up here AFTER the question is written, so no retrieval
output can influence the questions.

Run:  python tests/dump_chunk_inventory.py > data/cache/chunk_inventory.txt
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CACHE_DIR", "data/cache/exp_inventory")
os.environ.setdefault("CHUNKING_STRATEGY", "markdown_aware")
os.environ.setdefault("CHUNK_SIZE_EXTRACTION", "450")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.config import get_config  # noqa: E402

get_config.cache_clear()

from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer  # noqa: E402
from graphrag_lite.ingestion.loader import load_documents  # noqa: E402

cfg = get_config()
tok = get_tokenizer(cfg.TOKENIZER)

docs = load_documents("data/raw")
print(f"documents: {len(docs)}  strategy: {cfg.CHUNKING_STRATEGY}  "
      f"size: {cfg.CHUNK_SIZE_EXTRACTION}  overlap: {cfg.CHUNK_OVERLAP_PCT}")

total = 0
for doc in docs:
    chunks = chunk_document(doc, cfg, tok)
    print("\n" + "=" * 100)
    print(f"DOC {doc['doc_id']}  ({len(chunks)} chunks)")
    print("=" * 100)
    for c in chunks:
        total += 1
        body = c["text"].replace("\n", " ⏎ ")
        print(f"\n[{c['chunk_id']}]  idx={c['chunk_index']}  "
              f"tokens={c.get('token_count')}  chars={len(c['text'])}")
        print(f"   {body}")
print(f"\nTOTAL CHUNKS: {total}")
