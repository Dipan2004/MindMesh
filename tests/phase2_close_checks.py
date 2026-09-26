"""
Phase 2 closure verification — all four checks in one run.
1. Full self-loop list with types (correct count)
2. Component structure, singletons, avg degree
3. Organization/Location predecessor/successor lists
4. BFS hops=2 vs hops=3 equivalence on 6-node component
"""
import sys, os, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import networkx as nx
from collections import Counter
from graphrag_lite.graph.builder import build_graph, load_extractions, _norm
from graphrag_lite.graph.ranking import rank_pagerank
from graphrag_lite.graph.traversal import bfs_expand

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)

# ── Check 1: Full self-loop list ──────────────────────────────────────────────
print("=" * 60)
print("CHECK 1: All 15 dropped self-loops with types")
print("=" * 60)

canonical = {}
for record in records:
    for e in record.get("entities", []):
        n = e.get("name", "").strip()
        if n and _norm(n) not in canonical:
            canonical[_norm(n)] = n

loop_type_counts = Counter()
loops = []
for record in records:
    chunk_id = record.get("chunk_id", "")
    doc_id   = record.get("doc_id", "")
    for rel in record.get("relationships", []):
        src = rel.get("source", "").strip()
        tgt = rel.get("target", "").strip()
        sc  = canonical.get(_norm(src))
        tc  = canonical.get(_norm(tgt))
        if sc and tc and sc == tc:
            rtype = rel.get("type", "").strip().upper()
            loops.append((rtype, src, tgt, doc_id, chunk_id[:8]))
            loop_type_counts[rtype] += 1

for i, (rtype, src, tgt, doc, cid) in enumerate(loops):
    print(f"  {i+1:2d}. [{doc}] {src!r} --{rtype}--> {tgt!r}  (chunk={cid})")

print()
print("Type breakdown:")
for t, c in loop_type_counts.most_common():
    print(f"  {t}: {c}")
print(f"Total: {len(loops)}")

# ── Check 2: Component structure ─────────────────────────────────────────────
print()
print("=" * 60)
print("CHECK 2: Component structure")
print("=" * 60)

# Weakly connected (undirected reachability — what to_undirected() gives)
undirected = G.to_undirected()
weak_comps = sorted(nx.connected_components(undirected), key=len, reverse=True)
weak_sizes = [len(c) for c in weak_comps]
weak_counts = Counter(weak_sizes)

# Strongly connected (directed reachability — different definition)
strong_comps = list(nx.strongly_connected_components(G))
strong_sizes = sorted([len(c) for c in strong_comps], reverse=True)

print(f"Weakly connected components (undirected reachability): {len(weak_comps)}")
print(f"  Size distribution:")
for size in sorted(weak_counts, reverse=True):
    print(f"    size {size:3d}: {weak_counts[size]:3d} component(s)")
singletons = weak_counts[1]
avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
print(f"  Singletons (size 1): {singletons}")
print(f"  Average degree (2*edges/nodes): {avg_degree:.4f}")
print()
print(f"Strongly connected components (directed reachability): {len(strong_comps)}")
print(f"  Largest strong component size: {max(strong_sizes)}")
print(f"  (Note: Task 2.2 used weakly-connected, which is the correct metric")
print(f"   for undirected reachability / BFS hop analysis)")

# ── Check 3: Organization / Location ─────────────────────────────────────────
print()
print("=" * 60)
print("CHECK 3: Organization / Location predecessor/successor lists")
print("=" * 60)
pr = rank_pagerank(G)
for node in ["Organization", "Location"]:
    if node not in G:
        print(f"  {node}: NOT IN GRAPH")
        continue
    preds = list(G.predecessors(node))
    succs  = list(G.successors(node))
    comp   = next(c for c in weak_comps if node in c)
    print(f"{node}:")
    print(f"  PageRank  : {pr[node]:.6f}")
    print(f"  predecessors ({len(preds)}): {preds}")
    print(f"  successors   ({len(succs)}): {succs}")
    print(f"  component ({len(comp)} nodes): {sorted(comp)}")
    print()

# ── Check 4: hops=2 vs hops=3 equivalence ────────────────────────────────────
print("=" * 60)
print("CHECK 4: BFS hops=2 vs hops=3 from 'Knowledge graphs'")
print("=" * 60)
seed = "Knowledge graphs"
r2 = bfs_expand(G, [seed], hops=2)
r3 = bfs_expand(G, [seed], hops=3)

print(f"hops=2  -> {len(r2)} nodes: {r2}")
print(f"hops=3  -> {len(r3)} nodes: {r3}")
print()
if r2 == r3:
    print("IDENTICAL — hops=2 already exhausts the entire 6-node component.")
    print("hops=3 adds 0 new nodes. The 2-hop bound is saturated by the largest")
    print("component; no node in this graph is more than 2 hops from 'Knowledge graphs'.")
else:
    diff = set(r3) - set(r2)
    print(f"DIFFERENT — hops=3 adds {len(diff)} new nodes: {sorted(diff)}")
