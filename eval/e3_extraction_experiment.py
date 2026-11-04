"""
E3 — Targeted extraction experiment.

Reads canonical chunks from Zvec, re-extracts entities/relationships
with the specified model, builds a graph, and runs Pipeline B retrieval
for D18 to check if vector_databases enters the candidate scope.

Usage:
    python eval/e3_extraction_experiment.py --model qwen3:1.7b
    python eval/e3_extraction_experiment.py --model qwen2.5:3b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

EXPERIMENT_DIR = ROOT / "data" / "cache_e3"

# ── Read all 42 canonical chunks from Zvec ─────────────────────────────────

def read_canonical_chunks() -> list[dict]:
    import zvec
    col = zvec.open(str(ROOT / "data" / "cache" / "chunks_v1"))
    res = col.query(
        queries=[zvec.Query("text_vec", vector=[0.0] * 768,
                            param=zvec.HnswQueryParam(ef=200))],
        topk=100,
    )
    chunks = []
    for doc in res:
        f = doc.fields or {}
        chunks.append({
            "chunk_id": doc.id,
            "doc_id": f.get("source_doc", ""),
            "chunk_index": f.get("chunk_index", 0),
            "text": f.get("text", ""),
        })
    col.close()
    chunks.sort(key=lambda c: (c["doc_id"], c["chunk_index"]))
    return chunks


# ── Run extraction with a given model ──────────────────────────────────────

def run_extraction(model: str, chunks: list[dict]) -> list[dict]:
    from graphrag_lite.config import get_config
    from graphrag_lite.llm.ollama_client import OllamaClient
    from graphrag_lite.ingestion.extractor import extract_entities_relationships
    from graphrag_lite.ingestion.gleaning import glean

    cfg = get_config()

    # Configure client for this model
    think = False if model.startswith("qwen3") else cfg.effective_think()
    client = OllamaClient(
        model=model,
        temperature=cfg.EXTRACTION_TEMPERATURE,
        seed=cfg.EXTRACTION_SEED,
        num_predict=cfg.LLM_NUM_PREDICT,
        num_ctx=cfg.LLM_NUM_CTX,
        think=think,
        frequency_penalty=cfg.LLM_FREQUENCY_PENALTY,
    )

    results = []
    total_start = time.monotonic()

    for i, chunk in enumerate(chunks):
        cid = chunk["chunk_id"]
        doc = chunk["doc_id"]
        t0 = time.monotonic()
        print(f"  [{i+1:02d}/{len(chunks)}] {cid[:8]} ({doc[:12]}) ...", end="", flush=True)
        try:
            extraction = extract_entities_relationships(chunk, client)
            # Run gleaning if configured
            if cfg.MAX_GLEANINGS > 0:
                extraction = glean(chunk, extraction, client, max_gleanings=cfg.MAX_GLEANINGS)

            n_ent = len(extraction.get("entities", []))
            n_rel = len(extraction.get("relationships", []))
            dt = time.monotonic() - t0
            print(f" {n_ent} ent / {n_rel} rel ({dt:.1f}s)")

            results.append({
                "chunk_id": cid,
                "doc_id": doc,
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "entities": extraction.get("entities", []),
                "relationships": extraction.get("relationships", []),
            })
        except Exception as exc:
            dt = time.monotonic() - t0
            print(f" ERROR: {type(exc).__name__}: {str(exc)[:60]} ({dt:.1f}s)")
            results.append({
                "chunk_id": cid,
                "doc_id": doc,
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "entities": [],
                "relationships": [],
                "_parse_error": str(exc),
            })

    total_dt = time.monotonic() - total_start
    print(f"\n  Total extraction time: {total_dt:.1f}s ({total_dt/len(chunks):.1f}s/chunk avg)")
    return results


# ── Build graph and check cross-document edges ─────────────────────────────

def build_and_analyze_graph(extractions: list[dict], model_name: str) -> dict:
    import contextlib, io
    from graphrag_lite.graph.builder import build_graph
    import networkx as nx
    import collections

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        G = build_graph(extractions)

    # Cross-document analysis
    cross_doc_nodes = []
    for node, attrs in G.nodes(data=True):
        src_docs = attrs.get("source_docs", set())
        if len(src_docs) >= 2:
            cross_doc_nodes.append((node, sorted(src_docs)))

    cross_doc_edges = 0
    for u, v, data in G.edges(data=True):
        u_docs = G.nodes[u].get("source_docs", set())
        v_docs = G.nodes[v].get("source_docs", set())
        if u_docs and v_docs and not u_docs.intersection(v_docs):
            cross_doc_edges += 1

    # Entity names per doc
    entities_by_doc = collections.defaultdict(set)
    for n, d in G.nodes(data=True):
        for doc in d.get("source_docs", set()):
            entities_by_doc[doc].add(n)

    # Check if vector_databases entities appear in RAG/transformer extractions
    vdb_entities = entities_by_doc.get("vector_databases", set())
    rag_entities = entities_by_doc.get("retrieval_augmented_generation", set())
    trans_entities = entities_by_doc.get("transformer_architecture", set())

    # Cross-doc overlap
    rag_vdb_overlap = vdb_entities & rag_entities
    trans_vdb_overlap = vdb_entities & trans_entities

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": nx.number_weakly_connected_components(G),
        "cross_doc_nodes": len(cross_doc_nodes),
        "cross_doc_edges": cross_doc_edges,
        "cross_doc_node_details": cross_doc_nodes,
        "entities_per_doc": {d: len(v) for d, v in entities_by_doc.items()},
        "rag_vdb_shared_entities": sorted(rag_vdb_overlap),
        "trans_vdb_shared_entities": sorted(trans_vdb_overlap),
        "graph": G,
    }


# ── Run D18 retrieval with a given graph ────────────────────────────────────

def run_d18_retrieval(graph_analysis: dict, model_name: str) -> dict:
    import graphrag_lite.retrieval.pipeline_b as pb
    from graphrag_lite.config import get_config

    # Replace the pipeline_b graph singleton with our experimental graph
    G = graph_analysis["graph"]
    cfg = get_config()

    # Compute PageRank for the experimental graph
    from graphrag_lite.graph.ranking import rank_pagerank
    pr = rank_pagerank(G)
    pr_max = max(pr.values()) if pr else 1.0

    # Save and replace singleton
    orig_G, orig_PR, orig_PR_MAX = pb._G, pb._PR, pb._PR_MAX
    pb._G = G
    pb._PR = pr
    pb._PR_MAX = pr_max

    query = ("When one technique copes with codes and the second copes with meaning, "
             "how can the two be merged so that neither blind spot dominates?")
    gold_chunk = "895ee8b439ed7d7b"

    try:
        from graphrag_lite.retrieval.pipeline_b import retrieve_structural
        from graphrag_lite.retrieval.pipeline_a import retrieve_flat

        # Pipeline A (unchanged - uses canonical Zvec)
        t0 = time.monotonic()
        ev_a = retrieve_flat(query)
        a_latency = time.monotonic() - t0

        # Pipeline B (experimental graph)
        t0 = time.monotonic()
        ev_b = retrieve_structural(query)
        b_latency = time.monotonic() - t0

        a_ids = [e.chunk_id for e in ev_a]
        b_ids = [e.chunk_id for e in ev_b]

        # Check candidate docs from Pipeline B
        from graphrag_lite.retrieval.pipeline_b import get_candidates
        cand = get_candidates(query)
        candidate_docs = set()
        for node_name, _ in cand["ranked_candidates"]:
            if node_name in G:
                candidate_docs.update(G.nodes[node_name].get("source_docs", set()))

        return {
            "model": model_name,
            "a_retrieved_ids": a_ids,
            "a_retrieved_count": len(ev_a),
            "a_gold_rank": next((i+1 for i, e in enumerate(ev_a) if e.chunk_id == gold_chunk), None),
            "a_latency_s": round(a_latency, 3),
            "b_retrieved_ids": b_ids,
            "b_retrieved_count": len(ev_b),
            "b_gold_rank": next((i+1 for i, e in enumerate(ev_b) if e.chunk_id == gold_chunk), None),
            "b_latency_s": round(b_latency, 3),
            "b_candidate_docs": sorted(candidate_docs),
            "vdb_in_scope": "vector_databases" in candidate_docs,
            "gold_retrieved_by_b": gold_chunk in b_ids,
            "gold_retrieved_by_a": gold_chunk in a_ids,
        }
    finally:
        # Restore original singleton
        pb._G = orig_G
        pb._PR = orig_PR
        pb._PR_MAX = orig_PR_MAX


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E3 extraction experiment")
    parser.add_argument("--model", required=True, help="Ollama model tag")
    parser.add_argument("--full", action="store_true",
                        help="Run all 42 chunks (default: priority 13 only)")
    args = parser.parse_args()

    model = args.model
    model_slug = model.replace(":", "_").replace(".", "_")
    exp_dir = EXPERIMENT_DIR / model_slug
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"E3 EXPERIMENT: {model}")
    print(f"{'='*70}\n")

    # 1. Read canonical chunks
    chunks = read_canonical_chunks()
    print(f"Read {len(chunks)} canonical chunks from Zvec")

    if not args.full:
        # Priority chunks: RAG (8) + transformer (5) = 13
        priority_docs = {"retrieval_augmented_generation", "transformer_architecture"}
        priority = [c for c in chunks if c["doc_id"] in priority_docs]
        other = [c for c in chunks if c["doc_id"] not in priority_docs]
        print(f"Priority mode: {len(priority)} chunks (RAG+transformer)")
        run_chunks = priority
    else:
        run_chunks = chunks
        print(f"Full mode: {len(run_chunks)} chunks (all 42)")

    # 2. Run extraction
    print(f"\n--- Extracting with {model} ---")
    extractions = run_extraction(model, run_chunks)

    # If priority mode, merge with canonical extractions for non-priority chunks
    if not args.full:
        print(f"\nMerging with canonical extractions for {len(other)} non-priority chunks...")
        with open(ROOT / "data" / "cache" / "extractions.jsonl", "r", encoding="utf-8") as f:
            canonical = [json.loads(l) for l in f if l.strip()]
        # Keep canonical for non-priority docs, new for priority docs
        merged = []
        for rec in canonical:
            doc = rec.get("doc_id", "")
            if doc in priority_docs:
                # Replace with new extraction (if we have one)
                new = [e for e in extractions if e["doc_id"] == doc]
                merged.extend(new)
            else:
                merged.append(rec)
        # Remove duplicates by chunk_id
        seen = set()
        deduped = []
        for rec in merged:
            cid = rec.get("chunk_id", "")
            if cid not in seen:
                seen.add(cid)
                deduped.append(rec)
        extractions = deduped
        print(f"Merged: {len(extractions)} total extraction records")

    # Save extractions
    ext_path = exp_dir / "extractions.jsonl"
    with open(ext_path, "w", encoding="utf-8") as f:
        for rec in extractions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved extractions to {ext_path}")

    # 3. Build and analyze graph
    print(f"\n--- Building graph ---")
    analysis = build_and_analyze_graph(extractions, model)
    print(f"  Nodes: {analysis['nodes']}")
    print(f"  Edges: {analysis['edges']}")
    print(f"  Components: {analysis['components']}")
    print(f"  Cross-doc nodes: {analysis['cross_doc_nodes']}")
    print(f"  Cross-doc edges: {analysis['cross_doc_edges']}")
    print(f"  Entities per doc: {analysis['entities_per_doc']}")
    if analysis["rag_vdb_shared_entities"]:
        print(f"  RAG-VDB shared entities: {analysis['rag_vdb_shared_entities']}")
    if analysis["trans_vdb_shared_entities"]:
        print(f"  TRANS-VDB shared entities: {analysis['trans_vdb_shared_entities']}")
    if analysis["cross_doc_nodes"]:
        for n, docs in analysis["cross_doc_node_details"][:10]:
            print(f"  Cross-doc node: {n!r} in {docs}")

    # 4. Run D18 retrieval
    print(f"\n--- D18 retrieval test ---")
    result = run_d18_retrieval(analysis, model)
    print(f"  Pipeline A: {result['a_retrieved_count']} chunks, gold rank={result['a_gold_rank']}, latency={result['a_latency_s']}s")
    print(f"  Pipeline B: {result['b_retrieved_count']} chunks, gold rank={result['b_gold_rank']}, latency={result['b_latency_s']}s")
    print(f"  B candidate docs: {result['b_candidate_docs']}")
    print(f"  vector_databases in scope: {result['vdb_in_scope']}")
    print(f"  Gold retrieved by B: {result['gold_retrieved_by_b']}")
    print(f"  Gold retrieved by A: {result['gold_retrieved_by_a']}")

    # Save results
    result_path = exp_dir / "d18_result.json"
    # Don't save the graph object
    save_result = {k: v for k, v in result.items()}
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(save_result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
