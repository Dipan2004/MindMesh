"""
Item 4: Rebuild graph with alias merge applied in build_graph.
Adds _norm_alias to builder, merges 3 pairs, logs before/after.
"""
import sys, os, re, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import networkx as nx
from collections import defaultdict
from graphrag_lite.graph.builder import build_graph, load_extractions, _norm
from graphrag_lite.graph.ranking import rank_pagerank

def _norm_alias(name: str) -> str:
    s = name.casefold()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.endswith("s"):
        s = s[:-1]
    return s

def build_graph_with_aliases(records):
    """Build graph then apply alias merges."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        G = build_graph(records)

    # Find safe alias merges
    groups = defaultdict(list)
    for node in G.nodes():
        groups[_norm_alias(node)].append(node)
    safe_merges = {k: v for k, v in groups.items() if len(v) > 1}

    merge_map = {}
    for key, nodes in safe_merges.items():
        canonical = sorted(nodes)[0]
        for n in nodes:
            if n != canonical:
                merge_map[n] = canonical

    print(f"  Alias merges: {len(merge_map)}")
    for old, new in sorted(merge_map.items()):
        print(f"    {old!r} -> {new!r}")

    # Apply merges
    for old_name, canonical in merge_map.items():
        if old_name not in G:
            continue
        old_data = G.nodes[old_name]
        can_data = G.nodes[canonical]
        can_data["descriptions"].extend(old_data.get("descriptions", []))
        can_data["source_sentences"].extend(old_data.get("source_sentences", []))
        can_data["source_docs"].update(old_data.get("source_docs", set()))
        if can_data.get("type") == "OTHER" and old_data.get("type") != "OTHER":
            can_data["type"] = old_data["type"]
        for _, tgt, edata in list(G.out_edges(old_name, data=True)):
            real_tgt = merge_map.get(tgt, tgt)
            if real_tgt == canonical:
                continue
            if not G.has_edge(canonical, real_tgt):
                G.add_edge(canonical, real_tgt, **edata)
        for src, _, edata in list(G.in_edges(old_name, data=True)):
            real_src = merge_map.get(src, src)
            if real_src == canonical:
                continue
            if not G.has_edge(real_src, canonical):
                G.add_edge(real_src, canonical, **edata)
        G.remove_node(old_name)

    return G

def report(G, label):
    undirected = G.to_undirected()
    comps = sorted(nx.connected_components(undirected), key=len, reverse=True)
    sizes = [len(c) for c in comps]
    singletons = sizes.count(1)
    print(f"\n  [{label}]")
    print(f"    nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")
    print(f"    components={len(comps)}  largest={sizes[0] if sizes else 0}  singletons={singletons}")
    pr = rank_pagerank(G)
    top5 = sorted(pr, key=pr.get, reverse=True)[:5]
    print("    PageRank top-5:")
    for n in top5:
        print(f"      {n!r:40s}  {pr[n]:.6f}  degree={G.degree(n)}")
    return G

print("Loading extractions...")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")

print(f"Records: {len(records)}")
print()

# BEFORE
print("Building graph WITHOUT alias merge...")
buf2 = io.StringIO()
with contextlib.redirect_stdout(buf2):
    G_before = build_graph(records)
report(G_before, "BEFORE alias merge")

# AFTER
print()
print("Building graph WITH alias merge...")
G_after = build_graph_with_aliases(records)
report(G_after, "AFTER alias merge")
