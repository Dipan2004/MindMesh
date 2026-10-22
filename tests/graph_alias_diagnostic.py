"""
Items 1a/1b/1c: Cross-doc structure + alias analysis + copy-merge simulation.
Read-only on the canonical graph. Operates on a copy for merge simulation.
"""
import sys, os, io, re, contextlib, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import networkx as nx
from collections import defaultdict
from graphrag_lite.graph.builder import build_graph, load_extractions

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)

print("=" * 65)
print("ITEM 1a: Cross-document node/edge structure")
print("=" * 65)

# Nodes with >1 source doc
multi_doc_nodes = [(n, G.nodes[n]["source_docs"])
                   for n in G.nodes()
                   if len(G.nodes[n].get("source_docs", set())) > 1]
print(f"Nodes with >1 source_doc: {len(multi_doc_nodes)}")
for name, docs in sorted(multi_doc_nodes, key=lambda x: len(x[1]), reverse=True)[:10]:
    print(f"  {name!r:40s}  docs={sorted(docs)}")
print()

# Edges connecting nodes from different docs
cross_doc_edges = []
for u, v, data in G.edges(data=True):
    docs_u = G.nodes[u].get("source_docs", set())
    docs_v = G.nodes[v].get("source_docs", set())
    if not docs_u.isdisjoint(docs_v):
        # share at least one doc — same-doc edge
        pass
    else:
        cross_doc_edges.append((u, v, sorted(docs_u), sorted(docs_v)))
print(f"Edges where u and v share NO source_doc: {len(cross_doc_edges)}")
for u, v, du, dv in cross_doc_edges[:10]:
    print(f"  {u!r:30s}({du}) --> {v!r:30s}({dv})")
print()

print("=" * 65)
print("ITEM 1b: Near-duplicate node pairs (conservative normalisation)")
print("=" * 65)

def _norm_alias(name: str) -> str:
    """Casefold, strip punctuation, collapse whitespace, strip trailing 's'."""
    s = name.casefold()
    s = re.sub(r"[^\w\s]", "", s)   # strip punctuation
    s = re.sub(r"\s+", " ", s).strip()
    if s.endswith("s"):
        s = s[:-1]                   # conservative plural strip
    return s

# Group nodes by normalised key
groups: dict[str, list[str]] = defaultdict(list)
for node in G.nodes():
    groups[_norm_alias(node)].append(node)

dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
print(f"Duplicate groups (≥2 nodes same norm key): {len(dup_groups)}")
print(f"Nodes involved in a duplicate group: {sum(len(v) for v in dup_groups.values())}")
print()

# Categorise: "safe" auto-merge vs "needs judgment" (X vs X-pipeline style)
safe_merges = {}
judgment_needed = {}

for key, nodes in dup_groups.items():
    names_set = set(n.lower() for n in nodes)
    # Flag "needs judgment" if any pair is X vs "X pipeline" / "X system" / "X model"
    pipeline_suffixes = ("pipeline", "system", "model", "framework", "approach",
                         "method", "technique", "algorithm")
    has_qualified = any(
        any(f"{base} {suf}" in n.lower() for suf in pipeline_suffixes
            for base in names_set)
        for n in nodes
    )
    if has_qualified:
        judgment_needed[key] = nodes
    else:
        safe_merges[key] = nodes

print(f"Safe auto-merges: {len(safe_merges)}")
print(f"Needs judgment  : {len(judgment_needed)}")
print()

# Print 15 examples, prioritising the named ones
named_first = [
    ("Knowledge Graph", "Knowledge graphs"),
    ("vector database", "Vector databases"),
]
printed = set()
print("15 example duplicate pairs:")

# Print named ones first
for a, b in named_first:
    ka, kb = _norm_alias(a), _norm_alias(b)
    if ka == kb and ka in dup_groups:
        bucket = "JUDGMENT" if ka in judgment_needed else "SAFE"
        print(f"  [{bucket}] {dup_groups[ka]}")
        printed.add(ka)

for key, nodes in list(dup_groups.items()):
    if len(printed) >= 15:
        break
    if key in printed:
        continue
    bucket = "JUDGMENT" if key in judgment_needed else "SAFE"
    print(f"  [{bucket}] {nodes}")
    printed.add(key)

print()
if judgment_needed:
    print("JUDGMENT pairs (X vs X-pipeline):")
    for key, nodes in judgment_needed.items():
        print(f"  key={key!r}  nodes={nodes}")
    print()

print("=" * 65)
print("ITEM 1c: Simulate safe merges on a COPY of the graph")
print("=" * 65)

def report_graph(g, label):
    undirected = g.to_undirected()
    comps = sorted(nx.connected_components(undirected), key=len, reverse=True)
    sizes = [len(c) for c in comps]
    singletons = sizes.count(1)
    multi_doc = sum(1 for n in g.nodes()
                    if len(g.nodes[n].get("source_docs", set())) > 1)
    xdoc_edges = sum(
        1 for u, v in g.edges()
        if g.nodes[u].get("source_docs", set()).isdisjoint(
           g.nodes[v].get("source_docs", set()))
    )
    print(f"\n  [{label}]")
    print(f"    nodes={g.number_of_nodes()}  edges={g.number_of_edges()}")
    print(f"    components={len(comps)}  largest={sizes[0] if sizes else 0}  singletons={singletons}")
    print(f"    multi-doc nodes={multi_doc}  cross-doc edges={xdoc_edges}")

report_graph(G, "BEFORE merge")

# Build merge map: norm_key -> canonical node (first in sorted order)
merge_map: dict[str, str] = {}  # old_name -> canonical_name
for key, nodes in safe_merges.items():
    canonical = sorted(nodes)[0]   # alphabetically first as canonical
    for n in nodes:
        if n != canonical:
            merge_map[n] = canonical

print(f"\n  Safe merges to apply: {len(merge_map)} node renames")
for old, new in sorted(merge_map.items())[:10]:
    print(f"    {old!r} -> {new!r}")
if len(merge_map) > 10:
    print(f"    ... and {len(merge_map)-10} more")

# Apply on a deep copy
G2 = G.copy()

# Step 1: for each merged-away node, redirect its edges to the canonical
for old_name, canonical in merge_map.items():
    if old_name not in G2:
        continue

    # Merge node attributes into canonical
    if canonical not in G2:
        G2.add_node(canonical)
    old_data = G2.nodes[old_name]
    can_data = G2.nodes[canonical]
    can_data["descriptions"].extend(old_data.get("descriptions", []))
    can_data["source_sentences"].extend(old_data.get("source_sentences", []))
    can_data["source_docs"].update(old_data.get("source_docs", set()))
    if can_data.get("type") == "OTHER" and old_data.get("type") != "OTHER":
        can_data["type"] = old_data["type"]

    # Redirect outgoing edges
    for _, tgt, edata in list(G2.out_edges(old_name, data=True)):
        real_tgt = merge_map.get(tgt, tgt)
        if real_tgt == canonical:
            continue  # would be a self-loop after merge — skip
        if not G2.has_edge(canonical, real_tgt):
            G2.add_edge(canonical, real_tgt, **edata)
        else:
            G2[canonical][real_tgt]["source_sentences"].extend(
                edata.get("source_sentences", []))

    # Redirect incoming edges
    for src, _, edata in list(G2.in_edges(old_name, data=True)):
        real_src = merge_map.get(src, src)
        if real_src == canonical:
            continue
        if not G2.has_edge(real_src, canonical):
            G2.add_edge(real_src, canonical, **edata)
        else:
            G2[real_src][canonical]["source_sentences"].extend(
                edata.get("source_sentences", []))

    G2.remove_node(old_name)

report_graph(G2, "AFTER safe merge")
print()

# Summary delta
before_nodes = G.number_of_nodes()
before_edges = G.number_of_edges()
after_nodes  = G2.number_of_nodes()
after_edges  = G2.number_of_edges()
print(f"\n  Delta: nodes {before_nodes} -> {after_nodes} ({before_nodes - after_nodes} removed)")
print(f"         edges {before_edges} -> {after_edges} ({after_edges - before_edges:+d})")

comps_before = len(list(nx.connected_components(G.to_undirected())))
comps_after  = len(list(nx.connected_components(G2.to_undirected())))
print(f"         components {comps_before} -> {comps_after} ({comps_before - comps_after} reduced)")
largest_before = max(len(c) for c in nx.connected_components(G.to_undirected()))
largest_after  = max(len(c) for c in nx.connected_components(G2.to_undirected()))
print(f"         largest component {largest_before} -> {largest_after}")
