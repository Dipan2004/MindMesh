"""
Diagnose self-loops and edge count reconciliation for Task 2.1.
Traces every relationship from extractions.jsonl to its final fate.
"""
import json, sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.graph.builder import load_extractions, _norm
import networkx as nx

JSONL = "data/cache/extractions.jsonl"
records = load_extractions(JSONL)

# ── Reproduce the canonical name lookup (Pass 1) ──────────────────────────────
canonical: dict[str, str] = {}   # norm_key -> first canonical name
for record in records:
    for entity in record.get("entities", []):
        name = entity.get("name", "").strip()
        if not name:
            continue
        key = _norm(name)
        if key not in canonical:
            canonical[key] = name

# ── Collect every relationship with pre-merge AND post-merge names ────────────
all_rels = []
for record in records:
    chunk_id = record.get("chunk_id", "")
    doc_id = record.get("doc_id", "")
    for rel in record.get("relationships", []):
        src_raw = rel.get("source", "").strip()
        tgt_raw = rel.get("target", "").strip()
        rel_type = rel.get("type", "RELATED_TO").strip().upper()
        desc = rel.get("description", "").strip()
        ssent = rel.get("source_sentence", "").strip()

        src_canon = canonical.get(_norm(src_raw))
        tgt_canon = canonical.get(_norm(tgt_raw))

        fate = None
        if src_canon is None or tgt_canon is None:
            fate = "DANGLING"
        elif src_canon == tgt_canon:
            # Determine if pre-merge or post-merge self-loop
            if _norm(src_raw) == _norm(tgt_raw):
                fate = "SELF_LOOP_PRE_MERGE"   # LLM produced X ---> X directly
            else:
                fate = "SELF_LOOP_POST_MERGE"  # two different names merged
        else:
            fate = "EDGE"

        all_rels.append({
            "src_raw": src_raw, "tgt_raw": tgt_raw,
            "src_canon": src_canon, "tgt_canon": tgt_canon,
            "type": rel_type, "desc": desc, "ssent": ssent,
            "chunk_id": chunk_id, "doc_id": doc_id,
            "fate": fate,
        })

total = len(all_rels)
print(f"Total relationships in extractions.jsonl: {total}")
print()

# ── Self-loop analysis ────────────────────────────────────────────────────────
pre_merge_loops  = [r for r in all_rels if r["fate"] == "SELF_LOOP_PRE_MERGE"]
post_merge_loops = [r for r in all_rels if r["fate"] == "SELF_LOOP_POST_MERGE"]
dangling         = [r for r in all_rels if r["fate"] == "DANGLING"]
normal_edges     = [r for r in all_rels if r["fate"] == "EDGE"]

print(f"Self-loops (pre-merge, LLM produced X-->X):  {len(pre_merge_loops)}")
for r in pre_merge_loops:
    print(f"  [{r['doc_id']}] {r['src_raw']!r} --{r['type']}--> {r['tgt_raw']!r}")
    print(f"    chunk_id={r['chunk_id'][:8]}  desc={r['desc'][:60]}")
print()

print(f"Self-loops (post-merge, two names collapsed): {len(post_merge_loops)}")
for r in post_merge_loops:
    print(f"  [{r['doc_id']}] {r['src_raw']!r} --{r['type']}--> {r['tgt_raw']!r}")
    print(f"    -> both merge to canonical: {r['src_canon']!r}")
    print(f"    chunk_id={r['chunk_id'][:8]}")
print()

print(f"Dangling (endpoint not a node):              {len(dangling)}")
for r in dangling:
    missing = []
    if r['src_canon'] is None: missing.append(f"src={r['src_raw']!r}")
    if r['tgt_canon'] is None: missing.append(f"tgt={r['tgt_raw']!r}")
    print(f"  {', '.join(missing)}")
print()

# ── Edge count reconciliation — count parallel (same src+tgt+type) ─────────────
from collections import Counter
edge_key_counter = Counter()
for r in normal_edges:
    edge_key_counter[(r["src_canon"], r["tgt_canon"], r["type"])] += 1

parallel_absorbed = sum(v - 1 for v in edge_key_counter.values() if v > 1)
unique_edges = len(edge_key_counter)

print(f"Normal edges (non-self-loop, non-dangling):  {len(normal_edges)}")
print(f"  Unique (src, tgt, type) combos:            {unique_edges}")
print(f"  Parallel edges absorbed into existing:     {parallel_absorbed}")
print()

# Same-src+tgt different type (would create different DiGraph edge slots):
same_pair_diff_type = Counter()
for r in normal_edges:
    same_pair_diff_type[(r["src_canon"], r["tgt_canon"])] += 1
multi_type_pairs = {k: v for k, v in same_pair_diff_type.items() if v > 1}
print(f"  Pairs with >1 relationship type:           {len(multi_type_pairs)}")
for pair, count in list(multi_type_pairs.items())[:5]:
    print(f"    {pair[0]!r} --> {pair[1]!r}: {count} rels")
print()

# ── Full reconciliation ────────────────────────────────────────────────────────
# DiGraph only keeps ONE edge per (src, tgt) pair — latest write wins
# So final edge count = number of unique (src_canon, tgt_canon) pairs
unique_pairs = len(Counter((r["src_canon"], r["tgt_canon"]) for r in normal_edges))
all_type_absorbed = len(normal_edges) - unique_pairs

print("=" * 60)
print("RECONCILIATION")
print("=" * 60)
print(f"  Total relationships in JSONL   : {total}")
print(f"  Self-loops (pre-merge)         : {len(pre_merge_loops)}")
print(f"  Self-loops (post-merge)        : {len(post_merge_loops)}")
print(f"  Dangling (no node endpoint)    : {len(dangling)}")
print(f"  Normal edges (non-loop, found) : {len(normal_edges)}")
print(f"    → absorbed as parallel       : {all_type_absorbed} (same src+tgt, diff type OR dup type)")
print(f"    → final unique (src,tgt) pairs: {unique_pairs}")
check = len(pre_merge_loops) + len(post_merge_loops) + len(dangling) + unique_pairs + all_type_absorbed
print(f"  CHECK: {len(pre_merge_loops)} + {len(post_merge_loops)} + {len(dangling)} + {unique_pairs} + {all_type_absorbed} = {check}  (should == {total})")
assert check == total, f"Reconciliation doesn't balance: {check} != {total}"
print("  BALANCED ✓")
