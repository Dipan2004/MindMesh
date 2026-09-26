"""Task 2.1 DoD test — graph builder."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import networkx as nx
from graphrag_lite.graph.builder import build_graph, load_extractions, _norm

JSONL = "data/cache/extractions.jsonl"

print("=" * 60)
print("TASK 2.1 — Graph Builder DoD")
print("=" * 60)
print()

# ── Load ──────────────────────────────────────────────────────────────────────
records = load_extractions(JSONL)
print(f"Records loaded: {len(records)}")
print()

# ── Build ─────────────────────────────────────────────────────────────────────
G = build_graph(records)
print()

# ── Basic stats ───────────────────────────────────────────────────────────────
print(f"Nodes : {G.number_of_nodes()}")
print(f"Edges : {G.number_of_edges()}")
print()

# ── One full node attribute dict ──────────────────────────────────────────────
# Pick a node known to have merged from multiple chunks
sample_node = None
for n in G.nodes():
    nd = G.nodes[n]
    if len(nd.get("descriptions", [])) > 1:
        sample_node = n
        break
if sample_node is None:
    sample_node = list(G.nodes())[0]

print(f"Sample node: {sample_node!r}")
nd = G.nodes[sample_node]
print(f"  type        : {nd['type']}")
print(f"  source_docs : {nd['source_docs']}")
print(f"  descriptions ({len(nd['descriptions'])} entries):")
for d in nd["descriptions"][:3]:
    print(f"    [{d['doc_id']}] {d['text'][:80]}")
print(f"  source_sentences ({len(nd['source_sentences'])} entries):")
for s in nd["source_sentences"][:2]:
    print(f"    [{s['doc_id']}] {s['text'][:80]}")
print()

# ── One full edge attribute dict ──────────────────────────────────────────────
sample_edge = list(G.edges())[0]
print(f"Sample edge: {sample_edge[0]!r} --> {sample_edge[1]!r}")
ed = G[sample_edge[0]][sample_edge[1]]
print(f"  types      : {ed['types']}")
print(f"  description: {ed.get('description','')[:80]}")
print(f"  source_doc : {ed.get('source_doc','')}")
print(f"  source_sentences: {len(ed.get('source_sentences', []))} entries")
print()

# ── Merge verification — known duplicates from Phase 1 dedup ─────────────────
print("MERGE VERIFICATION (known duplicates from Phase 1 dedup analysis):")
known_dupes = [
    "edge", "rafael rafailov", "knowledge graph",
    "chroma", "lancedb", "zvec",
]
all_node_keys = {_norm(n): n for n in G.nodes()}
for name in known_dupes:
    key = _norm(name)
    count = sum(1 for n in G.nodes() if _norm(n) == key)
    canon = all_node_keys.get(key, "NOT FOUND")
    print(f"  {name!r:25s}  node_count={count}  canonical={canon!r}")

assert all(
    sum(1 for n in G.nodes() if _norm(n) == _norm(name)) <= 1
    for name in known_dupes
), "Merge failed — duplicate nodes found!"
print("  All collapse to ≤ 1 node: PASS")
print()

# ── Dangling edge check ───────────────────────────────────────────────────────
dangling = 0
for u, v in G.edges():
    if u not in G.nodes or v not in G.nodes:
        dangling += 1
print(f"Dangling edge check: {dangling}/{G.number_of_edges()} relationships reference a missing node")
assert dangling == 0, f"Dangling edges found: {dangling}"
print("  PASS")
print()

# ── Connected components ──────────────────────────────────────────────────────
undirected = G.to_undirected()
components = list(nx.connected_components(undirected))
print(f"Connected components: {len(components)}")
sizes = sorted((len(c) for c in components), reverse=True)
print(f"  Largest component size: {sizes[0]}")
print(f"  Component size distribution: {sizes[:10]}{'...' if len(sizes) > 10 else ''}")
print()
print("ALL ASSERTIONS PASSED")
