"""
Task 3.2 follow-up: points 2 (alignment check) and 3 (schema-artifact query).
"""
import sys, os, io, contextlib, math, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.graph.builder import build_graph, load_extractions
from graphrag_lite.graph.ranking import rank_pagerank
from graphrag_lite.embeddings.embedder import embed_batch
from graphrag_lite.vectorstore.zvec_client import (
    insert_entities, hybrid_query, _entities_path, reset_collection,
)
from graphrag_lite.config import get_config

cfg = get_config()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)
    pr_scores = rank_pagerank(G)

entity_dicts = []
for node in G.nodes():
    nd = dict(G.nodes[node])
    nd["name"] = node
    entity_dicts.append(nd)

entity_texts = []
for ed in entity_dicts:
    descs = ed.get("descriptions", [])
    first_desc = descs[0]["text"] if descs else ""
    entity_texts.append(f"{ed['name']}: {first_desc}" if first_desc else ed["name"])

entity_vecs = embed_batch(entity_texts)

# ── POINT 2: embedding-entity alignment check ────────────────────────────────
print("=" * 60)
print("POINT 2: Embedding-entity alignment at 3 spot-check indices")
print("=" * 60)

CHECK_INDICES = [0, 10, 37]   # spread across the list
for idx in CHECK_INDICES:
    name = entity_dicts[idx]["name"]
    text = entity_texts[idx]
    batch_vec = entity_vecs[idx]

    # Independently re-embed the same text
    solo_vec = embed_batch([text])[0]

    cosine = sum(a*b for a,b in zip(batch_vec, solo_vec)) / (
        math.sqrt(sum(x*x for x in batch_vec)) *
        math.sqrt(sum(x*x for x in solo_vec))
    )
    exact = all(abs(a-b) < 1e-9 for a,b in zip(batch_vec, solo_vec))

    print(f"  idx={idx:3d}  name={name!r:35s}")
    print(f"           text[:60]={text[:60]!r}")
    print(f"           batch_vec[:3]={[round(x,6) for x in batch_vec[:3]]}")
    print(f"           solo_vec[:3] ={[round(x,6) for x in solo_vec[:3]]}")
    print(f"           cosine={cosine:.8f}  exact_match={exact}")
    # Within-process: exact match expected
    # Cross-process: cosine ~1.0 acceptable (BLAS non-determinism noted in Task 3.1)
    assert cosine > 0.9999, f"cosine {cosine:.6f} too low at idx={idx}"
    print()

print("POINT 2: PASS — embeddings[i] corresponds to entities[i] for all 3 checked indices")
print()

# ── Rebuild collection fresh for the schema-artifact test ────────────────────
epath = _entities_path()
reset_collection(epath)
with contextlib.redirect_stdout(io.StringIO()):
    insert_entities(entity_dicts, entity_vecs, pagerank_scores=pr_scores)

# ── POINT 3: Schema-artifact query test ──────────────────────────────────────
print("=" * 60)
print("POINT 3: Schema-artifact retrieval test")
print("=" * 60)

ARTIFACT_QUERY = "what types of entities exist in a knowledge graph"
query_vec = embed_batch([ARTIFACT_QUERY])[0]

results = hybrid_query(
    collection_name=cfg.ZVEC_COLLECTION_ENTITIES,
    query_vector=query_vec,
    query_text=ARTIFACT_QUERY,
    top_k=8,
)

SCHEMA_ARTIFACTS = {"person", "organization", "location", "entity", "edge", "node"}

print(f"Query: {ARTIFACT_QUERY!r}  top_k=8")
print()
print(f"{'Rank':<5} {'ID':<38} {'Score':>9}  {'PR':>8}  {'artifact?'}")
print("-" * 75)
artifacts_in_results = []
for i, r in enumerate(results):
    is_artifact = r.id.replace("_", " ").lower() in SCHEMA_ARTIFACTS
    if is_artifact:
        artifacts_in_results.append(r)
    pr_val = r.fields.get("pagerank_score", 0.0)
    flag = "*** SCHEMA ARTIFACT" if is_artifact else ""
    print(f"  {i+1}   {r.id:<38} {r.score:>9.6f}  {pr_val:>8.4f}  {flag}")

print()
if artifacts_in_results:
    print(f"Schema artifacts found: {len(artifacts_in_results)}/8")
    for r in artifacts_in_results:
        pr_val = r.fields.get("pagerank_score", 0.0)
        print(f"  {r.id!r}  score={r.score:.6f}  pr={pr_val:.6f}")
    print()
    # Check whether high PR score pushed them to top
    if artifacts_in_results[0] == results[0]:
        print("WARNING: schema artifact ranked #1 — its high PR is distorting retrieval.")
    else:
        top_artifact_rank = next(i+1 for i, r in enumerate(results)
                                 if r in artifacts_in_results)
        print(f"Schema artifacts appear at rank {top_artifact_rank}+.")
        print("They appear in results but are not dominating the top position.")
else:
    print("No schema artifacts (Person/Organization/Location/Edge) in top-8 results.")
    print("Schema-artifact risk is LOW for this query type.")
