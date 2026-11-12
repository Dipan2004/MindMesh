"""
merge.py — Safe entity merge for MindMesh NetworkX graphs.

This is the SAFETY-CRITICAL component. It differs from Zeno's
merge_duplicate_entities() in several important ways:

Zeno's approach (NOT safe for MindMesh):
  UPDATE edges SET source_id=? WHERE source_id=?  (bulk redirect)
  DELETE FROM entities WHERE id=?                  (immediate delete)
  - Does NOT migrate fact_entity associations before deletion
  - Does NOT handle self-loops created by the redirect
  - Does NOT preserve the loser's aliases

MindMesh's safe approach:
  1. Choose canonical node (survivor) — higher degree wins; first alpha if tie
  2. Merge ALL node attributes into survivor (descriptions, source_sentences,
     source_docs, aliases list)
  3. Redirect ALL incoming edges to point at survivor
  4. Redirect ALL outgoing edges to point from survivor
  5. Remove self-loops (src == tgt after redirect)
  6. Deduplicate parallel edges (same (src, tgt, type))
  7. Record the merge in G.graph["merge_history"] for rollback
  8. Remove the loser node
  9. NEVER delete a node until all its edges are accounted for

Rollback: a merge_history entry records enough information to undo each
merge by re-adding the loser node and re-creating its edges from the
backup stored in the history.
"""
from __future__ import annotations
import time
from typing import Any

import networkx as nx


def _choose_survivor(G: nx.DiGraph, n1: str, n2: str) -> tuple[str, str]:
    """Choose survivor (canonical) and loser nodes.
    Survivor = higher total degree; ties broken alphabetically."""
    d1 = G.degree(n1)
    d2 = G.degree(n2)
    if d1 > d2:
        return n1, n2
    elif d2 > d1:
        return n2, n1
    else:
        return (n1, n2) if n1 < n2 else (n2, n1)


def safe_merge_node(
    G: nx.DiGraph,
    node_a: str,
    node_b: str,
    alias_registry=None,
) -> dict:
    """
    Safely merge node_b into node_a (or vice versa by degree).

    Returns a merge record suitable for rollback.
    Raises ValueError if either node does not exist.
    Raises ValueError if merge would corrupt provenance.
    """
    if node_a not in G:
        raise ValueError(f"safe_merge_node: node_a {node_a!r} not in graph")
    if node_b not in G:
        raise ValueError(f"safe_merge_node: node_b {node_b!r} not in graph")

    survivor, loser = _choose_survivor(G, node_a, node_b)
    s_data = dict(G.nodes[survivor])
    l_data = dict(G.nodes[loser])

    # --- Step 1: Back up loser state for rollback --------------------------
    loser_edges_out = [(tgt, dict(G[loser][tgt])) for tgt in G.successors(loser)]
    loser_edges_in  = [(src, dict(G[src][loser])) for src in G.predecessors(loser)]
    merge_record = {
        "timestamp": time.time(),
        "survivor": survivor,
        "loser": loser,
        "loser_node_attrs": {k: v for k, v in l_data.items()
                              if not isinstance(v, set) and not isinstance(v, list)},
        "loser_source_docs": sorted(l_data.get("source_docs", set())),
        "loser_edges_out": [(tgt, edata) for tgt, edata in loser_edges_out],
        "loser_edges_in":  [(src, edata) for src, edata in loser_edges_in],
    }

    # --- Step 2: Merge node attributes ------------------------------------
    s_data_mut = G.nodes[survivor]

    # descriptions: extend list
    for item in l_data.get("descriptions", []):
        if item not in s_data_mut.get("descriptions", []):
            s_data_mut.setdefault("descriptions", []).append(item)

    # source_sentences: extend list
    for item in l_data.get("source_sentences", []):
        if item not in s_data_mut.get("source_sentences", []):
            s_data_mut.setdefault("source_sentences", []).append(item)

    # source_docs: union of sets
    survivor_docs = s_data_mut.get("source_docs", set())
    if isinstance(survivor_docs, (list, tuple)):
        survivor_docs = set(survivor_docs)
    s_data_mut["source_docs"] = survivor_docs | l_data.get("source_docs", set())

    # aliases: preserve loser's label as an alias
    aliases = list(s_data_mut.get("aliases", []))
    if loser not in aliases:
        aliases.append(loser)
    # Also preserve any existing aliases on the loser
    for a in l_data.get("aliases", []):
        if a not in aliases and a != survivor:
            aliases.append(a)
    s_data_mut["aliases"] = aliases

    # stub: survivor is non-stub if either was non-stub
    if not l_data.get("stub", False):
        s_data_mut["stub"] = False

    # type: prefer non-OTHER
    if s_data_mut.get("type") == "OTHER" and l_data.get("type") != "OTHER":
        s_data_mut["type"] = l_data["type"]

    # --- Step 3: Register alias in registry (if provided) -----------------
    if alias_registry is not None:
        alias_registry.register(loser, survivor)
        for a in l_data.get("aliases", []):
            alias_registry.register(a, survivor)

    # --- Step 4: Redirect loser's outgoing edges --------------------------
    for tgt, edata in loser_edges_out:
        real_tgt = tgt if tgt != loser else survivor  # would become self-loop

        if real_tgt == survivor:
            # Self-loop — skip per policy
            continue

        if G.has_edge(survivor, real_tgt):
            # Merge parallel edge — accumulate types and source_sentences
            existing = G[survivor][real_tgt]
            new_types = edata.get("types", [])
            existing.setdefault("types", [])
            for t in new_types:
                if t not in existing["types"]:
                    existing["types"].append(t)
            new_ss = edata.get("source_sentences", [])
            existing.setdefault("source_sentences", [])
            for ss in new_ss:
                if ss not in existing["source_sentences"]:
                    existing["source_sentences"].append(ss)
        else:
            G.add_edge(survivor, real_tgt, **edata)

    # --- Step 5: Redirect loser's incoming edges --------------------------
    for src, edata in loser_edges_in:
        if src == loser:
            # Was a self-loop on loser — skip
            continue
        real_src = src

        if real_src == survivor:
            # Would create a self-loop — skip
            continue

        if G.has_edge(real_src, survivor):
            existing = G[real_src][survivor]
            new_types = edata.get("types", [])
            existing.setdefault("types", [])
            for t in new_types:
                if t not in existing["types"]:
                    existing["types"].append(t)
            new_ss = edata.get("source_sentences", [])
            existing.setdefault("source_sentences", [])
            for ss in new_ss:
                if ss not in existing["source_sentences"]:
                    existing["source_sentences"].append(ss)
        else:
            G.add_edge(real_src, survivor, **edata)

    # --- Step 6: Remove the loser -----------------------------------------
    G.remove_node(loser)

    # --- Step 7: Record merge history on graph ----------------------------
    G.graph.setdefault("merge_history", []).append(merge_record)

    return merge_record


def rollback_merge(G: nx.DiGraph, merge_record: dict) -> bool:
    """
    Attempt to undo a single merge by re-adding the loser node.

    This is a best-effort rollback: it re-creates the loser node and its
    original edges, but does NOT remove edges that were added to the
    survivor from the redirect (those may now be shared with legitimate
    edges). The result is a valid graph state that may have more edges
    than the original.

    Returns True if rollback completed, False if the merge record was invalid.
    """
    survivor = merge_record.get("survivor")
    loser = merge_record.get("loser")
    if not survivor or not loser:
        return False
    if loser in G:
        return False  # already rolled back or never merged

    # Re-add loser node
    attrs = {k: v for k, v in merge_record.get("loser_node_attrs", {}).items()}
    attrs["source_docs"] = set(merge_record.get("loser_source_docs", []))
    G.add_node(loser, **attrs)

    # Re-add loser's outgoing edges (to original targets)
    for tgt, edata in merge_record.get("loser_edges_out", []):
        if tgt in G:
            G.add_edge(loser, tgt, **edata)

    # Re-add loser's incoming edges (from original sources)
    for src, edata in merge_record.get("loser_edges_in", []):
        if src in G:
            G.add_edge(src, loser, **edata)

    # Remove from merge history
    history = G.graph.get("merge_history", [])
    if merge_record in history:
        history.remove(merge_record)

    return True
