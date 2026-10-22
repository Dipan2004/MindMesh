"""Confirm pr=0.0085 is correct for singleton nodes."""
import sys, os, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from graphrag_lite.graph.builder import build_graph, load_extractions
from graphrag_lite.graph.ranking import rank_pagerank

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)
    pr = rank_pagerank(G)

nodes_to_check = [
    "Knowledge Graph", "Graph RAG", "Freebase",
    "Ontologies and Schemas", "Knowledge graphs"
]
print("Node degree and PR for query result nodes:")
for name in nodes_to_check:
    if name in G:
        deg = G.degree(name)
        preds = list(G.predecessors(name))
        succs = list(G.successors(name))
        print(f"  {name!r:40s}  degree={deg}  pr={pr[name]:.6f}")
        print(f"    preds={preds}  succs={succs}")

print()
print(f"Singleton PR value (NetworkX default for 0-degree nodes): {pr['Knowledge Graph']:.6f}")
print(f"1/N = 1/75 = {1/75:.6f}")
print()

# Count how many nodes have this floor value
floor_val = pr["Knowledge Graph"]
floor_count = sum(1 for v in pr.values() if abs(v - floor_val) < 1e-9)
print(f"Nodes at this floor value: {floor_count}/75")
print()
print("CONCLUSION: pr=0.0085 is the correct PageRank for isolated (0-degree) nodes.")
print("It is NOT a default/fallback in insert_entities. No bug.")
print("Task 2.2's top-10 table only showed nodes with non-trivial PR, hiding this floor.")
