"""Inspect chunk boundaries to find structural difference causing runaway generation."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.config import get_config

tokenizer = get_tokenizer("qwen")
docs = load_documents("data/raw")
kg_doc = next(d for d in docs if d["doc_id"] == "knowledge_graphs")

for strategy, size in [("sliding_window", "450"), ("recursive_char", "450")]:
    os.environ["CHUNKING_STRATEGY"] = strategy
    os.environ["CHUNK_SIZE_EXTRACTION"] = size
    get_config.cache_clear()
    cfg = get_config()
    chunks = chunk_document(kg_doc, cfg, tokenizer)

    print(f"=== {strategy} ===")
    for i, c in enumerate(chunks):
        text = c["text"]
        # Show last 80 chars of each chunk (the boundary that matters for generation)
        last_80 = repr(text[-80:])
        first_40 = repr(text[:40])
        print(f"  [{i}] tokens={c['token_count']:4d}  "
              f"starts={first_40}  ...ends={last_80}")
    print()
