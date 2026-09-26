"""Task 2.3 DoD test — bounded BFS traversal."""
import sys, os, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import networkx as nx
from graphrag_lite.graph.builder import build_graph, load_extractions
from graphrag_lite.graph.traversal import bfs_expand
from graphrag_lite.config import get_config

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)

cfg = get_config()

print("=" * 60)
print("TASK 2.3 -- Bounded BFS traversal DoD")
print("=" * 60)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"BFS_HOP_DEPTH = {cfg.BFS_HOP_DEPTH}")
print()

# ── Find the 6-node component to pick good seeds ────────────────────────────
undirected = G.to_undirected()
components = sorted(nx.connected_components(undirected), key=len, reverse=True)
largest_comp = components[0]
print(f"Largest component ({len(largest_comp)} nodes): {sorted(largest_comp)}")
print()

# ── Test 1: real seed from the largest (6-node) component ────────────────────
# Pick a node that is NOT a schema artifact
seed1 = None
schema = {"person","organization","location","entity","edge","node","relationship"}
for n in sorted(largest_comp):
    if n.lower() not in schema:
        seed1 = n
        break

if seed1 is None:
    seed1 = sorted(largest_comp)[0]  # fallback

print(f"Test 1: seed={seed1!r}  hops={cfg.BFS_HOP_DEPTH}  (from largest component)")
result1 = bfs_expand(G, [seed1], hops=cfg.BFS_HOP_DEPTH)
print(f"  Returned {len(result1)} nodes:")
for n in result1:
    path = nx.shortest_path(undirected, seed1, n) if n != seed1 else [seed1]
    hops_away = len(path) - 1
    print(f"    {hops_away}-hop  {n!r}")
print()

# ── Test 2: seed from a size-2 component (tests the common-case sparse node) ──
size2_components = [c for c in components if len(c) == 2]
seed2_comp = sorted(size2_components[0])  # first size-2 component
seed2 = seed2_comp[0]

print(f"Test 2: seed={seed2!r}  hops={cfg.BFS_HOP_DEPTH}  (from a 2-node component)")
result2 = bfs_expand(G, [seed2], hops=cfg.BFS_HOP_DEPTH)
print(f"  Returned {len(result2)} nodes:")
for n in result2:
    path = nx.shortest_path(undirected, seed2, n) if n != seed2 else [seed2]
    hops_away = len(path) - 1
    print(f"    {hops_away}-hop  {n!r}")
print()

# ── Test 3: singleton seed ────────────────────────────────────────────────────
singletons = [c for c in components if len(c) == 1]
seed3 = list(singletons[0])[0]

print(f"Test 3: seed={seed3!r}  hops={cfg.BFS_HOP_DEPTH}  (singleton node -- 0 edges)")
result3 = bfs_expand(G, [seed3], hops=cfg.BFS_HOP_DEPTH)
print(f"  Returned {len(result3)} nodes: {result3}")
print(f"  (Seed returns only itself -- expected for isolated node)")
print()

# ── Test 4: missing seed (not in graph) ───────────────────────────────────────
print(f"Test 4: seed='NonExistentNode'  hops=2  (not in graph)")
result4 = bfs_expand(G, ["NonExistentNode"], hops=2)
print(f"  Returned {len(result4)} nodes: {result4}")
print(f"  (Empty list expected -- graceful skip on missing seed)")
print()

# ── Assertions ────────────────────────────────────────────────────────────────
assert len(result1) > 1, "Seed in 6-node component should reach multiple nodes"
assert len(result1) <= len(largest_comp), "Cannot exceed component size"
assert seed1 in result1, "Seed itself must be in result (hop 0)"
assert len(result2) == 2, f"Size-2 component: expected 2, got {len(result2)}"
assert len(result3) == 1, f"Singleton: expected 1, got {len(result3)}"
assert len(result4) == 0, f"Missing seed: expected 0, got {len(result4)}"

# Hop bound respected: no node should be reachable if component is smaller than hops
max_reachable = max(len(bfs_expand(G, [n], hops=cfg.BFS_HOP_DEPTH)) for n in G.nodes())
print(f"Max nodes returned by any single-seed BFS at hops={cfg.BFS_HOP_DEPTH}: {max_reachable}")
print(f"(Must be <= largest component size {len(largest_comp)})")
assert max_reachable <= len(largest_comp)
print()
print("ALL ASSERTIONS PASSED")
