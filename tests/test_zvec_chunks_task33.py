"""Task 3.3 DoD -- Zvec chunks collection: insert + hybrid query."""
import sys, os, io, contextlib, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.embeddings.embedder import embed_batch
from graphrag_lite.vectorstore.zvec_client import (
    insert_chunks, hybrid_query, _chunks_path, reset_collection,
)
from graphrag_lite.config import get_config

cfg = get_config()

print("=" * 60)
print("TASK 3.3 -- Zvec Chunks Collection DoD")
print("=" * 60)

# ── Build chunks from the 5-doc corpus ───────────────────────────────────────
tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = load_documents("data/raw")
all_chunks = []
for d in docs:
    all_chunks.extend(chunk_document(d, cfg, tokenizer))

print(f"Strategy : {cfg.CHUNKING_STRATEGY}")
print(f"Chunks   : {len(all_chunks)}")
print()

# ── Embed chunk texts ─────────────────────────────────────────────────────────
print("Embedding chunks...", flush=True)
chunk_texts = [c["text"] for c in all_chunks]
chunk_vecs = embed_batch(chunk_texts)
print(f"Embeddings: {len(chunk_vecs)} vectors, dim={len(chunk_vecs[0])}")
print()

# ── Alignment spot-check BEFORE insert ───────────────────────────────────────
import math
print("Alignment spot-check (3 indices):")
for idx in [0, 10, 25]:
    if idx >= len(all_chunks):
        continue
    solo_vec = embed_batch([chunk_texts[idx]])[0]
    cosine = sum(a*b for a,b in zip(chunk_vecs[idx], solo_vec)) / (
        math.sqrt(sum(x*x for x in chunk_vecs[idx])) *
        math.sqrt(sum(x*x for x in solo_vec))
    )
    print(f"  idx={idx:3d}  chunk_id={all_chunks[idx]['chunk_id'][:8]}"
          f"  doc={all_chunks[idx]['doc_id']}"
          f"  cosine={cosine:.8f}")
    assert cosine > 0.9999, f"Alignment failed at idx={idx}: cosine={cosine}"
print("  Alignment: PASS")
print()

# ── Remove stale collection ───────────────────────────────────────────────────
cpath = _chunks_path()
reset_collection(cpath)
print(f"Reset collection at {cpath}")

# ── Insert ────────────────────────────────────────────────────────────────────
print("Inserting chunks...", flush=True)
insert_chunks(all_chunks, chunk_vecs)
print(f"Insert complete: {len(all_chunks)} chunks inserted and flushed.")
print()

# ── Hybrid query ──────────────────────────────────────────────────────────────
QUERY = "attention mechanism transformer self-attention"
print(f"Hybrid query: {QUERY!r}  top_k=5")
print()

query_vec = embed_batch([QUERY])[0]
results = hybrid_query(
    collection_name=cfg.ZVEC_COLLECTION_CHUNKS,
    query_vector=query_vec,
    query_text=QUERY,
    top_k=5,
)

print(f"{'Rank':<5} {'chunk_id':<20} {'Score':>10}  {'doc':<40}  chunk_idx")
print("-" * 90)
for i, r in enumerate(results):
    cid   = r.id[:16]
    src   = r.fields.get("source_doc", "?")
    cidx  = r.fields.get("chunk_index", "?")
    text  = r.fields.get("text", "")[:60]
    print(f"  {i+1}.  {cid:<20} {r.score:>10.6f}  {src:<40}  {cidx}")
    print(f"       text: {text!r}")

print()

# ── Assertions ────────────────────────────────────────────────────────────────
assert len(results) == 5, f"Expected 5 results, got {len(results)}"
scores = [r.score for r in results]
assert scores == sorted(scores, reverse=True), "Results not sorted by score"
unique_scores = len(set(round(s, 6) for s in scores))
if unique_scores == 1:
    print("WARNING: all 5 scores identical -- fusion may not be working.")
else:
    print(f"Score diversity: {unique_scores}/5 unique scores -- fusion working.")

# Confirm all results reference real chunk_ids from the inserted set
inserted_ids = {c["chunk_id"] for c in all_chunks}
for r in results:
    # chunk_id stored verbatim (no _to_doc_id needed -- hex strings have no spaces)
    assert r.id in inserted_ids, f"Result id {r.id!r} not in inserted chunk IDs"
print("All result IDs match inserted chunk IDs: PASS")
print()
print("ALL ASSERTIONS PASSED")
