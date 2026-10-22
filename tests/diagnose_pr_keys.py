"""
Diagnose PageRank key mismatch in Task 3.2.
Checks whether pagerank_scores keys align with entity names going into insert_entities.
"""
import sys, os, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.graph.builder import build_graph, load_extractions
from graphrag_lite.graph.ranking import rank_pagerank

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)
    pr_scores = rank_pagerank(G)

# Build entity dicts exactly as the test does
entity_dicts = []
for node in G.nodes():
    nd = dict(G.nodes[node])
    nd["name"] = node
    entity_dicts.append(nd)

print("=== PageRank key format (top 10) ===")
top10_pr = sorted(pr_scores, key=pr_scores.get, reverse=True)[:10]
for k in top10_pr:
    print(f"  key={k!r:40s}  pr={pr_scores[k]:.6f}")

print()
print("=== Entity name format (first 10) ===")
for ed in entity_dicts[:10]:
    name = ed["name"]
    val = pr_scores.get(name)
    val_str = f"{val:.6f}" if val is not None else "MISSING"
    print(f"  name={name!r:40s}  pr_lookup={val_str}")

print()
print("=== Checking 5 specific nodes from query results ===")
for name in ["Knowledge Graph", "Graph RAG", "Freebase", "Ontologies and Schemas", "Knowledge graphs"]:
    direct = pr_scores.get(name, "MISSING")
    print(f"  {name!r:40s}  direct_lookup={direct}")

print()
print("=== All nodes where pr_scores.get(name) is MISSING ===")
missing = [ed["name"] for ed in entity_dicts if ed["name"] not in pr_scores]
print(f"  Missing count: {len(missing)}/{len(entity_dicts)}")
for m in missing[:10]:
    print(f"    {m!r}")

print()
print("=== Checking: are PR keys the same object references as graph nodes? ===")
graph_nodes = set(G.nodes())
pr_keys = set(pr_scores.keys())
print(f"  Graph nodes: {len(graph_nodes)}")
print(f"  PR keys:     {len(pr_keys)}")
print(f"  Intersection: {len(graph_nodes & pr_keys)}")
print(f"  Only in graph (not in PR): {len(graph_nodes - pr_keys)}")
print(f"  Only in PR (not in graph): {len(pr_keys - graph_nodes)}")

# What are the actual PR values for the 4 nodes with pr=0.0085?
print()
print("=== Actual PR for the 4 nodes from query result ===")
for name in ["Knowledge Graph", "Graph RAG", "Freebase", "Ontologies and Schemas"]:
    # Try to find this node in PR keys with case-insensitive search
    exact = pr_scores.get(name)
    lower_match = {k: v for k, v in pr_scores.items() if k.lower() == name.lower()}
    print(f"  {name!r}")
    print(f"    exact match:  {exact}")
    print(f"    lower match:  {lower_match}")

print()
# What is 1/75 (uniform distribution, the default PageRank value)?
print(f"1/75 (uniform PR) = {1/75:.6f}")
print(f"Nodes in graph:      {G.number_of_nodes()}")
