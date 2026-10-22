"""Items 2a/2b: BFS with 'Knowledge Graph' and 'Knowledge graphs'; edge reconciliation."""
import sys, os, io, contextlib, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.graph.builder import build_graph, load_extractions, _norm, _norm_alias
from graphrag_lite.graph.traversal import bfs_expand

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)

print("=" * 60)
print("Item 2a: BFS from 'Knowledge Graph' (canonical) and 'Knowledge graphs' (merged-away)")
print("=" * 60)

# 'Knowledge graphs' was merged into 'Knowledge Graph' -- should warn and skip
for seed_name in ["Knowledge Graph", "Knowledge graphs"]:
    print(f"\nseed={seed_name!r}  hops=2")
    result = bfs_expand(G, [seed_name], hops=2)
    print(f"  Returned {len(result)} nodes: {result}")

print()
print("=" * 60)
print("Item 2b: Edge reconciliation against rewritten builder.py")
print("=" * 60)

# Rebuild canonical map from records
canonical = {}
for record in records:
    for entity in record.get("entities", []):
        name = entity.get("name", "").strip()
        if name:
            key = _norm(name)
            if key not in canonical:
                canonical[key] = name

# Apply alias merge (same logic as builder)
from collections import defaultdict
alias_groups = defaultdict(list)
for n in canonical.values():
    alias_groups[_norm_alias(n)].append(n)
alias_merge_map = {}
for nodes in alias_groups.values():
    if len(nodes) > 1:
        canon = sorted(nodes)[0]
        for n in nodes:
            if n != canon:
                alias_merge_map[n] = canon

# Trace all relationships
total = 0
self_loops_pre  = 0   # pre-alias self-loops (X -> X from LLM)
self_loops_post = 0   # new self-loops CREATED by alias merge
dangling        = 0
parallel        = seen = 0

edge_set = {}  # (src, tgt) -> count

for record in records:
    for rel in record.get("relationships", []):
        src_raw = rel.get("source","").strip()
        tgt_raw = rel.get("target","").strip()
        if not src_raw or not tgt_raw: continue
        total += 1

        src_c = canonical.get(_norm(src_raw))
        tgt_c = canonical.get(_norm(tgt_raw))
        if src_c is None or tgt_c is None:
            dangling += 1; continue

        # Pre-alias self-loop?
        if src_c == tgt_c:
            self_loops_pre += 1; continue

        # Apply alias merge
        src_a = alias_merge_map.get(src_c, src_c)
        tgt_a = alias_merge_map.get(tgt_c, tgt_c)

        # New self-loop from alias merge?
        if src_a == tgt_a:
            self_loops_post += 1; continue

        pair = (src_a, tgt_a)
        if pair in edge_set:
            parallel += 1
        else:
            edge_set[pair] = 1
            seen += 1

print(f"Total relationships in JSONL:              {total}")
print(f"  Dangling (endpoint not in entity list):  {dangling}")
print(f"  Self-loops pre-alias-merge (LLM noise):  {self_loops_pre}")
print(f"  Self-loops post-alias-merge (NEW):       {self_loops_post}")
print(f"  Normal edges after all merges:           {self_loops_pre + dangling + self_loops_post + seen + parallel}")
print(f"    Parallel merged into existing:         {parallel}")
print(f"    Final unique (src, tgt) edges:         {seen}")
print(f"  CHECK: {dangling}+{self_loops_pre}+{self_loops_post}+{seen}+{parallel} = {dangling+self_loops_pre+self_loops_post+seen+parallel} (expect {total})")
print()
print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
assert seen == G.number_of_edges(), f"Edge count mismatch: calculated {seen} vs graph {G.number_of_edges()}"
print("Edge count CONSISTENT with graph")
