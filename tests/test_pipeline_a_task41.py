"""Task 4.1 DoD -- Pipeline A flat baseline retrieval, 3 sample queries.
Each query has an expected top chunk (verified against corpus text).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.retrieval.pipeline_a import retrieve_flat, Evidence
from graphrag_lite.config import get_config

cfg = get_config()

# Each entry: (query, expected_chunk_id_prefix, expected_doc, notes)
QUERIES = [
    (
        "How does the self-attention mechanism compute attention scores?",
        "c81c2a06",  # transformer_architecture chunk_idx=1: "## Core Components ### Self-Attention Mechanism"
        "transformer_architecture",
        "Expected: Self-Attention section (chunk_idx=1)",
    ),
    (
        "What makes Zvec different from FAISS for hybrid search?",
        "895ee8b4",  # vector_databases chunk_idx=7: "## Hybrid Search -- Pure vector search ... FAISS ... Zvec"
        "vector_databases",
        "Expected: Hybrid Search section (chunk_idx=7). Also note FAISS section (chunk_idx>=3).",
    ),
    (
        "How are entities and relationships extracted from text in GraphRAG?",
        "6d878b7f",  # knowledge_graphs chunk_idx=6: "### Automated Extraction"
        "knowledge_graphs",
        "Expected: Automated Extraction section (chunk_idx=6)",
    ),
]

print("=" * 65)
print("TASK 4.1 -- Pipeline A (flat baseline)")
print("=" * 65)
print(f"ZVEC_TOP_N_FLAT = {cfg.ZVEC_TOP_N_FLAT}")
print()

all_results = []
for qi, (query, expected_id_prefix, expected_doc, note) in enumerate(QUERIES):
    print(f"Query {qi+1}: {query!r}")
    print(f"  {note}")
    evidence = retrieve_flat(query)
    all_results.append(evidence)

    # Find rank of expected chunk
    expected_rank = None
    faiss_rank = None
    zvec_section_rank = None
    for i, ev in enumerate(evidence):
        if ev.chunk_id.startswith(expected_id_prefix):
            expected_rank = i + 1
        # For Q2: track FAISS mentions and Zvec/hybrid section
        if qi == 1:
            text_lower = ev.text.lower()
            if "faiss" in text_lower and faiss_rank is None:
                faiss_rank = i + 1
            if "zvec" in text_lower and zvec_section_rank is None:
                zvec_section_rank = i + 1

    print(f"  Retrieved {len(evidence)} chunks:")
    for i, ev in enumerate(evidence):
        marker = " <-- EXPECTED" if ev.chunk_id.startswith(expected_id_prefix) else ""
        print(f"  [{i+1}] score={ev.score:.5f}  doc={ev.source_doc:<38}  "
              f"chunk_idx={ev.chunk_index}  id={ev.chunk_id[:12]}{marker}")
        print(f"       text: {ev.text[:100]!r}")

    if expected_rank is not None:
        print(f"  Rank of expected chunk ({expected_id_prefix}): {expected_rank}/{len(evidence)}")
    else:
        print(f"  WARN: expected chunk ({expected_id_prefix}) not in top-{len(evidence)}")

    if qi == 1:
        if faiss_rank:
            print(f"  First chunk mentioning 'FAISS' at rank: {faiss_rank}")
        if zvec_section_rank:
            print(f"  First chunk mentioning 'Zvec' at rank: {zvec_section_rank}")
    print()

# Assertions
for qi, (query, expected_id_prefix, expected_doc, note) in enumerate(QUERIES):
    evidence = all_results[qi]
    assert len(evidence) > 0, f"Query {qi+1} returned 0 results"
    assert len(evidence) <= cfg.ZVEC_TOP_N_FLAT
    scores = [ev.score for ev in evidence]
    assert scores == sorted(scores, reverse=True), f"Query {qi+1} not sorted"
    for ev in evidence:
        assert ev.source == "chunk"
        assert ev.text
        assert ev.chunk_id

print("ALL ASSERTIONS PASSED")
