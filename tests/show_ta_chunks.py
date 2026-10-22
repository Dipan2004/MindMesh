import sys, os
sys.path.insert(0, "src")
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
from graphrag_lite.config import get_config; get_config.cache_clear()
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
cfg = get_config()
tok = get_tokenizer(cfg.TOKENIZER)
docs = load_documents("data/raw")
ta = next(d for d in docs if d["doc_id"] == "transformer_architecture")
chunks = chunk_document(ta, cfg, tok)
for c in chunks:
    idx = c["chunk_index"]
    cid = c["chunk_id"][:12]
    tok_count = c["token_count"]
    first60 = c["text"][:60].replace("\n", " ")
    print(f"idx={idx}  chunk_id={cid}  tokens={tok_count}  '{first60}'")
