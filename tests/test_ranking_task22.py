"""Task 2.2 DoD test — PageRank vs degree-centrality ranking."""
import sys, os, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.graph.builder import build_graph, load_extractions
from graphrag_lite.graph.ranking import rank_pagerank, rank_degree
from graphrag_lite.config import get_config

# Build graph silently
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)

cfg = get_config()
print("=" * 60)
print("TASK 2.2 — Ranking DoD")
print("=" * 60)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"PAGERANK_DAMPING = {cfg.PAGERANK_DAMPING}")
print()

# ── Compute both rankings ─────────────────────────────────────────────────────
pr = rank_pagerank(G)
dg = rank_degree(G)

top5_pr = sorted(pr, key=pr.get, reverse=True)[:5]
top5_dg = sorted(dg, key=dg.get, reverse=True)[:5]

# ── Print side by side ────────────────────────────────────────────────────────
print(f"{'Rank':<5}  {'PageRank (score)':45}  {'Degree (count)'}")
print("-" * 80)
for i in range(5):
    pr_node = top5_pr[i]
    dg_node = top5_dg[i]
    pr_score = pr[pr_node]
    dg_count = dg[dg_node]
    print(f"  {i+1}.   {pr_node:<35} {pr_score:.6f}   {dg_node:<35} {dg_count}")

print()

# ── Comparison ────────────────────────────────────────────────────────────────
pr_set = set(top5_pr)
dg_set = set(top5_dg)
overlap = pr_set & dg_set
diff = pr_set.symmetric_difference(dg_set)

print(f"Top-5 overlap (nodes in both lists): {len(overlap)}/5")
if overlap:
    print(f"  Shared: {sorted(overlap)}")
if diff:
    print(f"  Only in PageRank:  {sorted(pr_set - dg_set)}")
    print(f"  Only in Degree:    {sorted(dg_set - pr_set)}")
print()

if pr_set == dg_set:
    print("VERDICT: The two top-5 lists are IDENTICAL (same 5 nodes, order may differ).")
    print("         On a sparse 30-edge graph this is a legitimate result — both methods")
    print("         converge on the same high-connectivity hubs.")
else:
    print("VERDICT: The two top-5 lists DIFFER.")
    print(f"         {len(overlap)}/5 nodes shared; {5 - len(overlap)} nodes differ between methods.")
    print("         PageRank and degree diverge on this graph's structure.")
print()

# ── Also print full score/degree for the union of top-10 (for paper table) ───
print("Extended view — top-10 by each method:")
print(f"{'#':<4}  {'PageRank node':<35} {'PR score':>10}   {'Degree node':<35} {'Deg':>5}")
print("-" * 95)
top10_pr = sorted(pr, key=pr.get, reverse=True)[:10]
top10_dg = sorted(dg, key=dg.get, reverse=True)[:10]
for i in range(10):
    pn = top10_pr[i] if i < len(top10_pr) else ""
    dn = top10_dg[i] if i < len(top10_dg) else ""
    ps = f"{pr[pn]:.6f}" if pn else ""
    ds = str(dg[dn]) if dn else ""
    print(f"  {i+1:<3}  {pn:<35} {ps:>10}   {dn:<35} {ds:>5}")

print()
print("ALL OUTPUT PRINTED — Task 2.2 DoD complete.")
