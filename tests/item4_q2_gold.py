"""Item 4: Find the FAISS and Zvec chunks by reading corpus text."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
from graphrag_lite.config import get_config; get_config.cache_clear()
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.retrieval.pipeline_a import retrieve_flat
from graphrag_lite.embeddings.embedder import embed_batch

cfg = get_config()
tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = load_documents("data/raw")
all_chunks = []
for d in docs:
    all_chunks.extend(chunk_document(d, cfg, tokenizer))

# Find chunks by heading content
print("=== Chunks in vector_databases doc ===")
vd_chunks = [c for c in all_chunks if c["doc_id"] == "vector_databases"]
for c in vd_chunks:
    first_100 = c["text"][:120].replace("\n", " ")
    print(f"  chunk_id={c['chunk_id'][:12]}  idx={c['chunk_index']}  tokens={c['token_count']}")
    print(f"    {first_100}")

print()
print("=== Specific chunk texts for gold identification ===")
for c in vd_chunks:
    if "### FAISS" in c["text"] or "### Zvec" in c["text"] or "Hybrid Search" in c["text"]:
        print(f"\nchunk_id={c['chunk_id'][:12]}  idx={c['chunk_index']}")
        print("--- FULL TEXT (first 600 chars) ---")
        print(c["text"][:600])

print()
print("=== Q2 retrieval (for rank check) ===")
_ = embed_batch(["seed"])  # ensure EMBEDDING_DIM populated
q2 = "What makes Zvec different from FAISS for hybrid search?"
evidence = retrieve_flat(q2)
print(f"Query: {q2!r}")
top_ids = {ev.chunk_id[:12]: i+1 for i, ev in enumerate(evidence)}
gold_ids = []
for c in vd_chunks:
    if "### FAISS" in c["text"] or "### Zvec" in c["text"] or "Hybrid Search" in c["text"]:
        cid = c["chunk_id"][:12]
        rank = top_ids.get(cid, "NOT IN TOP-8")
        gold_ids.append(cid)
        print(f"  chunk_id={cid}  heading={'FAISS' if '### FAISS' in c['text'] else 'Zvec' if '### Zvec' in c['text'] else 'Hybrid Search'}  rank={rank}")
