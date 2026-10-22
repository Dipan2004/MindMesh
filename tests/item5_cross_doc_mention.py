"""Item 5: Cross-doc mention check for 72 graph nodes."""
import sys, os, re, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from graphrag_lite.graph.builder import build_graph, load_extractions
from graphrag_lite.ingestion.loader import load_documents

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)

docs = load_documents("data/raw")
doc_texts = {d["doc_id"]: d["text"].casefold() for d in docs}

# Count how many docs each node name appears in (casefold, word boundaries)
multi_doc_nodes = []
for node in G.nodes():
    name_cf = node.casefold()
    # Try word-boundary match; fall back to substring for multi-word names
    pattern = r'\b' + re.escape(name_cf) + r'\b'
    doc_hits = [doc_id for doc_id, text in doc_texts.items()
                if re.search(pattern, text)]
    if len(doc_hits) > 1:
        multi_doc_nodes.append((node, doc_hits))

print(f"Nodes mentioned in 2+ docs (casefold, word boundaries): {len(multi_doc_nodes)}")
for node, doc_hits in sorted(multi_doc_nodes, key=lambda x: -len(x[1])):
    source_docs = G.nodes[node].get("source_docs", set())
    print(f"  {node!r:40s}  extracted_in={sorted(source_docs)}  mentioned_in={sorted(doc_hits)}")

# Count how many cross-doc edges a mention-based link would add
print()
print("Cross-doc edges if mention-based linking were implemented:")
cross_doc_potential = 0
seen_pairs = set()
for node, doc_hits in multi_doc_nodes:
    for i, da in enumerate(doc_hits):
        for db in doc_hits[i+1:]:
            pair = tuple(sorted([da, db]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                cross_doc_potential += 1
print(f"  Unique doc-pair connections that would be added: {cross_doc_potential}")
print(f"  (This is an upper bound -- actual edges depend on which pairs have relationships)")
print()
print("Recommendation: Not implemented. Cross-doc links via text mention would add")
print("noise without provenance (no source sentence, no relationship type). The")
print("correct fix is better extraction (qwen3:4b) which would produce cross-chunk")
print("relationships naturally through the pipeline.")
