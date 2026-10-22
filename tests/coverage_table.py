"""Per-document graph coverage table for Phase 3 closure."""
import sys, os, io, json, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.graph.builder import build_graph, load_extractions
from graphrag_lite.config import get_config

cfg = get_config()
tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = load_documents("data/raw")

# Total source tokens per doc
doc_tokens = {d["doc_id"]: len(tokenizer.encode(d["text"])) for d in docs}
total_src_tokens = sum(doc_tokens.values())

# Chunks per doc
all_chunks = []
chunks_by_doc = {}
for d in docs:
    c = chunk_document(d, cfg, tokenizer)
    chunks_by_doc[d["doc_id"]] = c
    all_chunks.extend(c)

# Extraction results from JSONL
with open("data/cache/extractions.jsonl") as f:
    extraction_records = [json.loads(l) for l in f if l.strip()]

# Per-chunk: entities, failed
by_chunk = {r["chunk_id"]: r for r in extraction_records}

# Build graph silently
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)

# Count graph nodes/edges per doc
nodes_by_doc = {}
for node in G.nodes():
    src_docs = G.nodes[node].get("source_docs", set())
    for d in src_docs:
        nodes_by_doc.setdefault(d, set()).add(node)

edges_by_doc = {}
for u, v, data in G.edges(data=True):
    d = data.get("source_doc", "")
    edges_by_doc.setdefault(d, set()).add((u, v))

# Print table
print(f"{'Doc':<45} {'Tokens':>7} {'%src':>5} {'Chunks':>7} {'0-ent':>6} {'Failed':>7} "
      f"{'RawEnt':>7} {'GrNodes':>8} {'GrEdges':>8}")
print("-" * 120)

total_chunks = total_zero = total_failed = total_raw_ent = total_nodes = total_edges = 0
for doc in docs:
    doc_id = doc["doc_id"]
    chunks = chunks_by_doc[doc_id]
    tokens = doc_tokens[doc_id]
    pct = tokens / total_src_tokens * 100

    zero_ent = 0
    failed = 0
    raw_ent = 0
    for c in chunks:
        rec = by_chunk.get(c["chunk_id"])
        if rec is None:
            continue
        if "_parse_error" in rec:
            failed += 1
        elif len(rec.get("entities", [])) == 0:
            zero_ent += 1
        raw_ent += len(rec.get("entities", []))

    gr_nodes = len(nodes_by_doc.get(doc_id, set()))
    gr_edges = len(edges_by_doc.get(doc_id, set()))

    print(f"  {doc_id:<43} {tokens:>7} {pct:>5.1f} {len(chunks):>7} {zero_ent:>6} {failed:>7} "
          f"{raw_ent:>7} {gr_nodes:>8} {gr_edges:>8}")
    total_chunks += len(chunks)
    total_zero += zero_ent
    total_failed += failed
    total_raw_ent += raw_ent
    total_nodes += gr_nodes
    total_edges += gr_edges

print("-" * 120)
print(f"  {'TOTAL':<43} {total_src_tokens:>7} {'100%':>5} {total_chunks:>7} "
      f"{total_zero:>6} {total_failed:>7} {total_raw_ent:>7} {total_nodes:>8} {total_edges:>8}")
print()
print("Notes:")
print("  Tokens   = raw source tokens (no overlap, from get_tokenizer)")
print("  %src     = fraction of total corpus tokens")
print("  0-ent    = chunks that parsed OK but the LLM extracted 0 entities")
print("  Failed   = chunks with _parse_error (LLM output truncated/unparseable)")
print("  RawEnt   = total entity instances extracted (before dedup/merge)")
print("  GrNodes  = unique graph nodes with source_doc containing this doc")
print("  GrEdges  = unique graph edges with source_doc == this doc")
