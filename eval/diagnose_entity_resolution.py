"""
diagnose_entity_resolution.py — Read-only diagnostic for cross-document entity connectivity.

Answers:
  1. Why does MindMesh have zero cross-document edges?
  2. What is the root cause in the graph builder?
  3. Which entities are cross-doc candidates?
  4. D18-specific diagnosis.

SAFETY: This script NEVER modifies the graph. It reads existing extractions
and builds an in-memory analysis graph only.

Usage:
    python eval/diagnose_entity_resolution.py
    python eval/diagnose_entity_resolution.py --extractions data/cache_v2/extractions_v2_off.jsonl
    python eval/diagnose_entity_resolution.py --with-embeddings   # adds cosine similarity (slower)
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import contextlib
import warnings
warnings.filterwarnings("ignore")

from graphrag_lite.graph.builder import build_graph, load_extractions, _norm, _norm_alias


# ── Zeno-style normalisation for diagnosis ----------------------------------

def zeno_normalize(text: str) -> str:
    """NFKC + casefold + whitespace collapse (Zeno ids.py style)."""
    s = unicodedata.normalize("NFKC", text).strip().lower()
    return re.sub(r"\s+", " ", s)


def zeno_entity_id(label: str, kind: str) -> str:
    import hashlib
    payload = f"{zeno_normalize(label)}|{kind}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def compatible_types(t1: str, t2: str) -> bool:
    """Return True if two entity types are compatible for merging."""
    if t1 == t2:
        return True
    # OTHER is compatible with anything
    if "OTHER" in (t1, t2):
        return True
    # Broad type groups
    tech = {"TECHNOLOGY", "CONCEPT", "SYSTEM"}
    org  = {"ORGANIZATION", "PERSON"}
    return (t1 in tech and t2 in tech) or (t1 in org and t2 in org)


def main():
    parser = argparse.ArgumentParser(description="Entity resolution diagnostic (read-only)")
    parser.add_argument("--extractions", default=None,
                        help="Path to extractions JSONL. Defaults to best available.")
    parser.add_argument("--with-embeddings", action="store_true",
                        help="Compute embedding similarity for candidate pairs.")
    args = parser.parse_args()

    # Find best extractions
    candidates = [
        ROOT / "data/experiments/qwen2_5_1_5b_genuine_gleaning_on/extractions.jsonl",
        ROOT / "data/cache_v2/extractions_v2_off.jsonl",
        ROOT / "data/cache/extractions.jsonl",
    ]
    if args.extractions:
        ext_path = Path(args.extractions)
    else:
        ext_path = next((p for p in candidates if p.exists()), None)
    if not ext_path or not ext_path.exists():
        print(f"ERROR: No extractions found. Provide --extractions <path>")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  ENTITY RESOLUTION DIAGNOSTIC")
    print(f"  Source: {ext_path.name}")
    print(f"{'='*70}\n")

    # Load + build graph (suppressing print output)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        records = load_extractions(str(ext_path))
        G = build_graph(records)
    build_log = buf.getvalue()

    # ── Section 1: Graph overview ─────────────────────────────────────────
    print("1. GRAPH OVERVIEW")
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")

    import networkx as nx
    comps = list(nx.weakly_connected_components(G))
    print(f"   Components: {len(comps)}")
    print(f"   Largest component: {max(len(c) for c in comps)} nodes")
    isolated = sum(1 for n in G.nodes() if G.degree(n) == 0)
    print(f"   Isolated nodes: {isolated}")

    cross_doc_nodes = [(n, sorted(d.get("source_docs", set())))
                       for n, d in G.nodes(data=True)
                       if len(d.get("source_docs", set())) >= 2]
    cross_doc_edges = [(u, v) for u, v in G.edges()
                       if G.nodes[u].get("source_docs", set())
                       and G.nodes[v].get("source_docs", set())
                       and not G.nodes[u].get("source_docs", set())
                        .intersection(G.nodes[v].get("source_docs", set()))]
    print(f"   Cross-document nodes: {len(cross_doc_nodes)}")
    print(f"   Cross-document edges: {len(cross_doc_edges)}  <- THE KEY FINDING")

    # ── Section 2: Root cause analysis ────────────────────────────────────
    print("\n2. ROOT CAUSE ANALYSIS — WHY ZERO CROSS-DOC EDGES")
    print()
    print("   builder.py _norm(): casefold + whitespace collapse only")
    print("   builder.py _norm_alias(): casefold + strip punct + strip trailing s")
    print()
    print("   Critical finding:")
    print("   Both normalization functions operate PER-CHUNK on a single chunk's")
    print("   entity list. There is NO cross-chunk or cross-document resolution.")
    print()
    print("   For entity X to have source_docs = {doc_A, doc_B}, BOTH must be true:")
    print("   a) doc_A chunk extracts entity with EXACTLY the same _norm() key as")
    print("      doc_B chunk. E.g. 'FAISS' in doc_A must normalize to the SAME")
    print("      key as 'FAISS' in doc_B. This CAN happen (same exact name).")
    print("   b) BUT for a cross-doc EDGE, we need a RELATIONSHIP where source is")
    print("      from doc_A and target from doc_B — this requires the LLM to extract")
    print("      a relationship spanning two documents IN A SINGLE CHUNK. Since each")
    print("      chunk comes from exactly ONE document, this is structurally impossible.")
    print()
    print("   DIAGNOSIS: The fundamental constraint is per-chunk extraction. Each chunk")
    print("   belongs to exactly one doc_id. Relationships can only exist between")
    print("   entities co-extracted from the same chunk. No relationship can bridge")
    print("   two documents unless a single chunk spans both — which never happens.")

    # ── Section 3: Entity name overlaps across documents ──────────────────
    print("\n3. ENTITY NAMES SHARED ACROSS DOCUMENTS")
    # Build entity->docs mapping from raw extractions
    name_to_docs: dict[str, set] = collections.defaultdict(set)
    name_to_types: dict[str, set] = collections.defaultdict(set)
    for rec in records:
        doc = rec.get("doc_id", "")
        for e in rec.get("entities", []):
            name = e.get("name", "").strip()
            etype = e.get("type", "OTHER")
            if name:
                key = _norm(name)
                name_to_docs[key].add(doc)
                name_to_types[key].add(etype)

    shared = {k: (v, name_to_types[k]) for k, v in name_to_docs.items() if len(v) >= 2}
    print(f"   Entity names (normalized) appearing in 2+ documents: {len(shared)}")
    print()
    if shared:
        print(f"   {'Normalized Name':<35} {'Docs':^30} {'Types'}")
        print(f"   {'-'*35} {'-'*30} {'-'*20}")
        for key, (docs, types) in sorted(shared.items(), key=lambda x: -len(x[1][0])):
            print(f"   {key:<35} {str(sorted(docs)):<30} {str(sorted(types))}")

    # ── Section 4: Why shared names don't become cross-doc nodes ──────────
    print("\n4. WHY SHARED NAMES DON'T ALWAYS CREATE CROSS-DOC NODES")
    shared_in_graph = [(n, sorted(d.get("source_docs", set())))
                       for n, d in G.nodes(data=True)
                       if len(d.get("source_docs", set())) >= 2]
    not_in_graph = []
    for norm_key, (docs, types) in shared.items():
        # Check if this normalized name has a cross-doc node
        found = any(_norm(n) == norm_key for n in [nd for nd in G.nodes()])
        if not found:
            not_in_graph.append((norm_key, docs, types))
    print(f"   Names shared across docs but MERGED in graph: {len(shared_in_graph)}")
    print(f"   Names shared across docs but NOT merged: {len(not_in_graph)}")
    if not_in_graph:
        print("   Not merged (possible alias mismatch):")
        for key, docs, types in not_in_graph[:5]:
            print(f"     {key!r:35} docs={sorted(docs)}")
            # Find actual node names for this key
            matching = [(n, G.nodes[n].get("source_docs",set())) for n in G.nodes()
                        if _norm_alias(n) == _norm_alias(key) or _norm(n) == key]
            for n, ndocs in matching:
                print(f"       graph node: {n!r} in docs {sorted(ndocs)}")

    # ── Section 5: Cross-doc candidate pairs via exact normalization ───────
    print("\n5. CROSS-DOC ENTITY CANDIDATE PAIRS (exact normalization)")
    # For each node, find nodes in OTHER docs with same normalized key
    candidate_pairs = []
    nodes_list = list(G.nodes(data=True))
    seen_pairs: set = set()
    for n1, d1 in nodes_list:
        docs1 = d1.get("source_docs", set())
        type1 = d1.get("type", "OTHER")
        for n2, d2 in nodes_list:
            if n1 >= n2:
                continue
            docs2 = d2.get("source_docs", set())
            if not docs1.intersection(docs2):  # different docs
                type2 = d2.get("type", "OTHER")
                # Match criteria
                exact = _norm(n1) == _norm(n2)
                alias = _norm_alias(n1) == _norm_alias(n2)
                zeno_match = zeno_normalize(n1) == zeno_normalize(n2)
                compatible = compatible_types(type1, type2)
                pair_key = (min(n1,n2), max(n1,n2))
                if pair_key not in seen_pairs and (exact or alias or zeno_match) and compatible:
                    seen_pairs.add(pair_key)
                    candidate_pairs.append({
                        "entity_1": n1, "type_1": type1, "docs_1": sorted(docs1),
                        "entity_2": n2, "type_2": type2, "docs_2": sorted(docs2),
                        "exact_norm_match": exact,
                        "alias_norm_match": alias,
                        "zeno_norm_match": zeno_match,
                        "type_compatible": compatible,
                        "classification": (
                            "AUTOMATIC" if (exact or zeno_match) and compatible
                            else "REVIEW" if alias and compatible
                            else "KEEP_SEPARATE"
                        ),
                    })

    auto = [p for p in candidate_pairs if p["classification"] == "AUTOMATIC"]
    review = [p for p in candidate_pairs if p["classification"] == "REVIEW"]
    print(f"   Candidate pairs (exact/alias match, compatible types): {len(candidate_pairs)}")
    print(f"   -> AUTOMATIC resolution (exact name match): {len(auto)}")
    print(f"   -> REVIEW (alias match, needs validation): {len(review)}")
    print()
    if candidate_pairs:
        print(f"   {'Entity 1':<30} {'Entity 2':<30} {'Docs':<40} Classification")
        print(f"   {'-'*30} {'-'*30} {'-'*40} {'-'*15}")
        for p in candidate_pairs[:20]:
            d1 = str(p['docs_1'])[:18]
            d2 = str(p['docs_2'])[:18]
            print(f"   {p['entity_1']:<30} {p['entity_2']:<30} {d1+'->'+d2:<40} {p['classification']}")
    if len(candidate_pairs) > 20:
        print(f"   ... and {len(candidate_pairs)-20} more")

    # ── Section 6: Embedding similarity (optional) ─────────────────────────
    if args.with_embeddings and candidate_pairs:
        print("\n6. EMBEDDING SIMILARITY FOR CANDIDATE PAIRS")
        from graphrag_lite.embeddings.embedder import embed_batch
        all_names = list({p["entity_1"] for p in candidate_pairs} |
                         {p["entity_2"] for p in candidate_pairs})
        print(f"   Embedding {len(all_names)} entity names...")
        vecs = embed_batch(all_names)
        vec_map = dict(zip(all_names, vecs))

        def cosine(a, b):
            dot = sum(x*y for x,y in zip(a,b))
            na = sum(x*x for x in a)**0.5
            nb = sum(x*x for x in b)**0.5
            return dot / (na * nb) if na * nb > 0 else 0.0

        for p in candidate_pairs[:10]:
            v1 = vec_map.get(p["entity_1"])
            v2 = vec_map.get(p["entity_2"])
            if v1 and v2:
                sim = cosine(v1, v2)
                p["embedding_similarity"] = round(sim, 4)
                print(f"   {p['entity_1'][:25]} ↔ {p['entity_2'][:25]} sim={sim:.4f}")
    else:
        print("\n6. EMBEDDING SIMILARITY — skipped (use --with-embeddings to enable)")

    # ── Section 7: D18 specific diagnosis ─────────────────────────────────
    print("\n7. D18 DIAGNOSTIC — Cross-document paraphrase query")
    print()

    # Load D18
    dev_path = ROOT / "eval" / "gold_dev_set.json"
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    d18 = next((q for q in dev["queries"] if q["query_id"] == "D18"), None)
    if d18:
        print(f"   Query: {d18['query']}")
        print(f"   Gold chunk: {d18['gold_chunk_ids']}")
        print()

        # Get the gold chunk's doc
        import zvec
        col = zvec.open(str(ROOT / "data/cache/chunks_v1"))
        res = col.query(queries=[zvec.Query("text_vec", vector=[0.0]*768,
                        param=zvec.HnswQueryParam(ef=100))], topk=100)
        cmap = {r.id: ((r.fields or {}).get("source_doc",""), (r.fields or {}).get("text","")[:200])
                for r in res}
        col.close()

        for cid in d18["gold_chunk_ids"]:
            doc, text_preview = cmap.get(cid, ("UNKNOWN", ""))
            print(f"   Gold chunk {cid[:12]} -> doc: {doc}")
            print(f"   Text preview: {text_preview[:120]}")
            print()

        # Find D18 gold chunk entities from extraction records
        gold_cid = d18["gold_chunk_ids"][0] if d18["gold_chunk_ids"] else None
        gold_entities = []
        if gold_cid:
            for rec in records:
                if rec.get("chunk_id") == gold_cid:
                    gold_entities = [(e.get("name",""), e.get("type","")) for e in rec.get("entities",[])]
                    break

        print(f"   Entities extracted from gold chunk: {len(gold_entities)}")
        for name, etype in gold_entities:
            in_graph = name in G.nodes() or any(_norm(n)==_norm(name) for n in G.nodes())
            docs = sorted(G.nodes.get(name,{}).get("source_docs",set())) if in_graph else []
            print(f"     {name!r:<35} ({etype}) in_graph={in_graph} docs={docs}")

        print()
        print("   The D18 query concerns: merging keyword/lexical search with semantic search")
        print("   The gold chunk is in 'vector_databases' (Hybrid Search section)")
        print("   The query's framing comes from 'retrieval_augmented_generation' perspective")
        print()
        print("   Why D18 fails:")
        print("   a) The gold chunk entities (if any) are in vector_databases graph component")
        print("   b) Query entry points from entity search land in RAG/transformer components")
        print("   c) Zero cross-doc edges means no path from query entry-points to gold chunk")
        print("   d) This is NOT primarily an entity-resolution failure — it's a structural gap")
        print("      caused by per-chunk extraction never producing cross-doc relationships")

    # ── Section 8: Nodes per document ─────────────────────────────────────
    print("\n8. NODE COUNT PER SOURCE DOCUMENT")
    doc_nodes = collections.defaultdict(list)
    for n, d in G.nodes(data=True):
        for doc in d.get("source_docs", set()):
            doc_nodes[doc].append(n)
    for doc in sorted(doc_nodes):
        print(f"   {doc:<45} {len(doc_nodes[doc])} nodes")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"  Total nodes:              {G.number_of_nodes()}")
    print(f"  Total edges:              {G.number_of_edges()}")
    print(f"  Cross-doc nodes:          {len(cross_doc_nodes)}")
    print(f"  Cross-doc edges:          {len(cross_doc_edges)}  <- always 0 (structural)")
    print(f"  Shared names (raw):       {len(shared)}")
    print(f"  Candidate merge pairs:    {len(candidate_pairs)}")
    print(f"  -> Automatic resolution:   {len(auto)}")
    print(f"  -> Needs review:           {len(review)}")
    print()
    print("  ROOT CAUSE: Per-chunk extraction + no cross-chunk entity resolution.")
    print("  The graph builder has no mechanism to link same-entity nodes from")
    print("  different documents BEYOND exact name normalization.")
    print()
    print("  WHAT WOULD FIX IT:")
    print("  1. Deterministic canonical IDs (Zeno-style) so 'FAISS' and 'FAISS library'")
    print("     map to the same node ID across all documents.")
    print("  2. Alias registry so abbreviations like 'BM25' and 'BM25 retrieval' merge.")
    print("  3. Conservative embedding-based candidate discovery for near-duplicates.")
    print("  4. Cross-doc relationship injection from shared entity anchors.")
    print()

    # Save candidates JSON for the resolver
    out_path = ROOT / "data" / "entity_resolution_candidates.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": str(ext_path),
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "cross_doc_edges": len(cross_doc_edges),
            "shared_name_count": len(shared),
            "candidate_pairs": candidate_pairs,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Candidate pairs saved to: {out_path}")


if __name__ == "__main__":
    main()
