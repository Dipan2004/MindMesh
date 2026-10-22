"""Task 4.2 DoD -- Pipeline B structural retrieval, 10 gold queries.
Gold chunk IDs defined from corpus text BEFORE running retrieval.
A@6 vs B@6 at equal k.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.retrieval.pipeline_b import get_candidates, retrieve_structural
from graphrag_lite.retrieval.pipeline_a import retrieve_flat
from graphrag_lite.config import get_config

cfg = get_config()

# ── Gold chunk IDs (defined from reading corpus text, before retrieval) ───────
# Each entry: (query_id, query, set of gold chunk_id prefixes)
GOLD_QUERIES = [
    # transformer_architecture
    ("Q01", "How does the self-attention mechanism compute attention scores?",
     {"c81c2a06"}),   # chunk idx=1: Self-Attention section
    ("Q02", "What is the Encoder-Decoder structure of the original Transformer?",
     {"745e90bb", "c81c2a06"}),  # idx=0 (overview) + idx=1 (core components)
    # knowledge_graphs
    ("Q03", "How are entities and relationships extracted from text in GraphRAG?",
     {"6d878b7f"}),   # idx=6: Automated Extraction section
    ("Q04", "What is Wikidata and who maintains it?",
     {"ba844007"}),   # idx=1: Major KGs section (Wikidata)
    ("Q05", "What are knowledge graph embedding methods like TransE and RotatE?",
     {"4e802a79"}),   # idx=7: KG Embeddings section
    # retrieval_augmented_generation
    ("Q06", "What makes Zvec different from FAISS for hybrid search?",
     {"895ee8b4", "5a5f4b8a", "d49b99f9"}),  # hybrid search, Zvec section, FAISS section
    ("Q07", "What is Retrieval-Augmented Generation and why does it matter?",
     {"33d32892"}),   # idx=0: RAG overview
    ("Q08", "How does HyDE improve retrieval quality in RAG systems?",
     {"b744eeeb"}),   # idx=4: Advanced RAG section (HyDE)
    # vector_databases
    ("Q09", "What is HNSW and how does it speed up approximate nearest neighbor search?",
     {"417905a7"}),   # idx=2: Core Concepts (HNSW section)
    # large_language_models
    ("Q10", "What is Direct Preference Optimization and how does it differ from RLHF?",
     {"32c85d7e"}),   # idx=2: DPO section
]

K = 6   # equal k for both pipelines

print("=" * 70)
print(f"TASK 4.2 -- Pipeline A@{K} vs Pipeline B@{K}  (10 gold queries)")
print("=" * 70)
print(f"BFS_HOP_DEPTH={cfg.BFS_HOP_DEPTH}  ZVEC_TOP_K_ENTRY={cfg.ZVEC_TOP_K_ENTRY}")
print(f"TOP_N_FINAL_EVIDENCE={K} (enforced for both pipelines)")
print()

results = []
for qid, query, gold_prefixes in GOLD_QUERIES:
    ev_a = retrieve_flat(query)[:K]
    ev_b = retrieve_structural(query)

    cands = get_candidates(query)
    ep = cands["entry_points"]
    ep_filtered = cands["entry_points_filtered"]
    bfs = cands["bfs_nodes"]
    ranked = cands["ranked_candidates"]

    a_ids = {ev.chunk_id[:8] for ev in ev_a}
    b_ids = {ev.chunk_id[:8] for ev in ev_b}
    gold  = {g[:8] for g in gold_prefixes}

    rec_a = len(gold & a_ids) / len(gold)
    rec_b = len(gold & b_ids) / len(gold)

    # B-only chunks (not in A top-K)
    b_only = b_ids - a_ids

    results.append((qid, rec_a, rec_b, b_only))

    print(f"{qid}: {query!r}")
    print(f"  Gold: {sorted(gold)}")
    print(f"  Entry points ({len(ep)}): {ep[:4]}")
    if ep_filtered:
        print(f"  Filtered type-labels: {ep_filtered}")
    print(f"  BFS nodes: {len(bfs)}  Ranked candidates: {len(ranked)}")
    print(f"  Top-5 B candidates: {[(n[:20], round(s,4)) for n,s in ranked[:5]]}")
    print(f"  A@{K}: {sorted(a_ids)}  recall={rec_a:.2f}")
    print(f"  B@{K}: {sorted(b_ids)}  recall={rec_b:.2f}")
    if b_only:
        print(f"  B-only chunks (not in A top-{K}): {sorted(b_only)}")
    print()

# Summary table
print("=" * 70)
print(f"SUMMARY: A@{K} vs B@{K}")
print("=" * 70)
print(f"{'QID':<5} {'Recall_A':>10} {'Recall_B':>10} {'B_better':>9} {'B_only_chunks':>14}")
print("-" * 52)
for qid, ra, rb, b_only in results:
    better = "YES" if rb > ra else ("TIE" if rb == ra else "NO")
    print(f"{qid:<5} {ra:>10.2f} {rb:>10.2f} {better:>9} {str(sorted(b_only)):>14}")

a_wins = sum(1 for _,ra,rb,_ in results if ra > rb)
b_wins = sum(1 for _,ra,rb,_ in results if rb > ra)
ties   = sum(1 for _,ra,rb,_ in results if ra == rb)
avg_a  = sum(ra for _,ra,_,_ in results) / len(results)
avg_b  = sum(rb for _,_,rb,_ in results) / len(results)
print("-" * 52)
print(f"{'AVG':<5} {avg_a:>10.2f} {avg_b:>10.2f}")
print(f"A wins={a_wins}  B wins={b_wins}  Ties={ties}")
print()
print("NOTE: A~B on sparse graph expected. See Task 4.2 analysis.")
print("ALL OUTPUT PRINTED -- Task 4.2 DoD complete.")
