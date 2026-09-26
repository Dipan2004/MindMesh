"""Print the full Organization --> Person edge data for Task 2.1 follow-up 1."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from graphrag_lite.graph.builder import build_graph, load_extractions
import io, contextlib

# suppress the per-chunk builder chatter
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)

ed = G["Organization"]["Person"]

print("Edge: Organization --> Person")
print()
print(f"types ({len(ed['types'])} entries): {ed['types']}")
print()
print(f"description: {ed['description']}")
print(f"source_doc:  {ed['source_doc']}")
print()
print(f"source_sentences ({len(ed['source_sentences'])} entries):")
for i, s in enumerate(ed["source_sentences"]):
    print(f"  [{i}] chunk_id={s['chunk_id'][:16]}  type={s['type']}")
    print(f"       text: {s['text'][:120]}")
print()

# Verify counts
assert len(ed["types"]) == 4, f"Expected 4 types, got {len(ed['types'])}"
assert len(ed["source_sentences"]) == 4, f"Expected 4 source_sentences, got {len(ed['source_sentences'])}"
print("ASSERTIONS PASSED: 4 types, 4 source_sentences")
