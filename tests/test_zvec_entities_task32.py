"""Task 3.2 DoD — Zvec entities collection: insert + hybrid query."""
import sys, os, io, contextlib, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.graph.builder import build_graph, load_extractions
from graphrag_lite.graph.ranking import rank_pagerank
from graphrag_lite.embeddings.embedder import embed_batch
from graphrag_lite.vectorstore.zvec_client import (
    get_entities_collection, insert_entities, hybrid_query,
    _entities_path, reset_collection,
)
from graphrag_lite.config import get_config

cfg = get_config()

print("=" * 60)
print("TASK 3.2 -- Zvec Entities Collection DoD")
print("=" * 60)

# ── Build graph ───────────────────────────────────────────────────────────────
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)

# ── Compute PageRank scores ───────────────────────────────────────────────────
buf2 = io.StringIO()
with contextlib.redirect_stdout(buf2):
    pr_scores = rank_pagerank(G)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print()

# ── Prepare entity dicts for insertion ───────────────────────────────────────
entity_dicts = []
for node in G.nodes():
    nd = dict(G.nodes[node])
    nd["name"] = node
    # source_docs is a set — may need to stay as-is for insert_entities
    entity_dicts.append(nd)

print(f"Entities to insert: {len(entity_dicts)}")

# ── Embed entity texts: "name: first_description" ────────────────────────────
print("Embedding entities...", flush=True)
entity_texts = []
for ed in entity_dicts:
    desc = ed.get("descriptions", [])
    first_desc = desc[0]["text"] if desc else ""
    entity_texts.append(f"{ed['name']}: {first_desc}" if first_desc else ed["name"])

entity_vecs = embed_batch(entity_texts)
print(f"Embeddings: {len(entity_vecs)} vectors, dim={len(entity_vecs[0])}")
print(f"EMBEDDING_DIM in config: {cfg.EMBEDDING_DIM}")
print()

# ── Remove stale collection if it exists ─────────────────────────────────────
epath = _entities_path()
reset_collection(epath)
print(f"Reset collection at {epath}")

# ── Insert ────────────────────────────────────────────────────────────────────
print("Inserting entities...", flush=True)
insert_entities(entity_dicts, entity_vecs, pagerank_scores=pr_scores)
print(f"Insert complete: {len(entity_dicts)} entities inserted and flushed.")
print()

# ── Hybrid query ──────────────────────────────────────────────────────────────
QUERY = "graph database knowledge representation"
print(f"Hybrid query: {QUERY!r}  top_k=5")
print()

query_vec = embed_batch([QUERY])[0]
results = hybrid_query(
    collection_name=cfg.ZVEC_COLLECTION_ENTITIES,
    query_vector=query_vec,
    query_text=QUERY,
    top_k=5,
)

print(f"{'Rank':<5} {'ID':<35} {'Score':>10}  Fields")
print("-" * 80)
for i, r in enumerate(results):
    etype = r.fields.get("entity_type", "?")
    src   = r.fields.get("source_doc", "?")
    pr    = r.fields.get("pagerank_score", 0.0)
    print(f"  {i+1}.  {r.id:<35} {r.score:>10.6f}  type={etype}  src={src}  pr={pr:.4f}")

print()

# ── Assertions ────────────────────────────────────────────────────────────────
assert len(results) == 5, f"Expected 5 results, got {len(results)}"
scores = [r.score for r in results]
assert scores == sorted(scores, reverse=True) or all(s == scores[0] for s in scores), \
    "Results not sorted by score"

# Check scores are differentiated (not all identical)
unique_scores = len(set(round(s, 6) for s in scores))
if unique_scores == 1:
    print("WARNING: all 5 scores are identical — RRF fusion may not be working.")
    print("         Mark task [!] blocked if this persists.")
else:
    print(f"Score diversity: {unique_scores}/5 unique scores — fusion is working.")

assert len(results) > 0
print()
print("ALL ASSERTIONS PASSED")
