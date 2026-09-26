"""
Pre-Task-2.3 graph structure diagnostics.
Covers all three points from the review:
  1. Full component size distribution + singleton count + average degree
  2. Schema-artifact nodes identification
  3. Organization/Location PageRank tie — structural symmetry check
"""
import sys, os, io, contextlib
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import networkx as nx
from graphrag_lite.graph.builder import build_graph, load_extractions
from graphrag_lite.graph.ranking import rank_pagerank

# Build graph silently
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)

undirected = G.to_undirected()
components = sorted(nx.connected_components(undirected), key=len, reverse=True)

# ── Point 1: Component size distribution ─────────────────────────────────────
print("=" * 60)
print("POINT 1: Connected component structure")
print("=" * 60)
sizes = [len(c) for c in components]
size_counts = Counter(sizes)

print(f"Total nodes : {G.number_of_nodes()}")
print(f"Total edges : {G.number_of_edges()}")
print(f"Components  : {len(components)}")
print()
print("Size distribution (size: count):")
for size in sorted(size_counts, reverse=True):
    bar = "#" * size_counts[size]
    print(f"  size {size:3d}: {size_counts[size]:3d} component(s)  {bar}")
print()

singletons = size_counts[1]
print(f"Singleton nodes (0 edges, isolated): {singletons}")
avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
print(f"Average degree (2*edges/nodes)     : {avg_degree:.3f}")
print()

# Max BFS hop depth achievable from any node
max_hops_possible = max(sizes) - 1
print(f"Largest component: {max(sizes)} nodes")
print(f"Max BFS hops possible (anywhere in graph): {max_hops_possible}")
print()

# What fraction of nodes are in the largest component?
pct_largest = max(sizes) / G.number_of_nodes() * 100
print(f"Nodes in largest component: {max(sizes)}/{G.number_of_nodes()} ({pct_largest:.1f}%)")
nodes_in_components_of_size_1 = singletons
nodes_in_components_of_size_2 = size_counts.get(2, 0) * 2
nodes_in_components_of_size_3 = size_counts.get(3, 0) * 3
print(f"Nodes reachable by 0-hop BFS (singletons)  : {nodes_in_components_of_size_1}")
print(f"Nodes in size-2 components (max 1 hop)     : {nodes_in_components_of_size_2}")
print(f"Nodes in size-3 components (max 2 hops)    : {nodes_in_components_of_size_3}")
print()

print("All component sizes (sorted desc):")
print(" ", sizes)
print()

# ── Point 3: Organization/Location tie ────────────────────────────────────────
print("=" * 60)
print("POINT 3: Organization / Location PageRank tie")
print("=" * 60)
pr = rank_pagerank(G)
print(f"Organization PR: {pr.get('Organization', 'NOT FOUND'):.6f}")
print(f"Location PR    : {pr.get('Location', 'NOT FOUND'):.6f}")
print()

for node in ["Organization", "Location"]:
    if node not in G:
        print(f"  {node}: NOT IN GRAPH")
        continue
    preds = list(G.predecessors(node))
    succs = list(G.successors(node))
    print(f"{node}:")
    print(f"  predecessors ({len(preds)}): {preds}")
    print(f"  successors   ({len(succs)}): {succs}")
    # component
    for comp in components:
        if node in comp:
            print(f"  component members: {sorted(comp)}")
            break
    print()

# ── Point 2: Schema-artifact identification ───────────────────────────────────
print("=" * 60)
print("POINT 2: Schema-artifact nodes in ranking")
print("=" * 60)
SCHEMA_TERMS = {
    "person", "organization", "location", "entity", "edge", "node",
    "relationship", "concept", "object", "thing", "type", "class",
}
schema_artifacts = [n for n in G.nodes() if n.lower().strip() in SCHEMA_TERMS]
print(f"Schema-artifact nodes found in graph ({len(schema_artifacts)}):")
for n in schema_artifacts:
    print(f"  {n!r:20s}  PR={pr.get(n, 0):.6f}  degree={G.degree(n)}")
print()
top5_pr = sorted(pr, key=pr.get, reverse=True)[:5]
schema_in_top5 = [n for n in top5_pr if n.lower().strip() in SCHEMA_TERMS]
print(f"Schema artifacts in PageRank top-5: {schema_in_top5} ({len(schema_in_top5)}/5)")
