"""Task 3.1 DoD — embedder using 3 real sentences from the corpus."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.embeddings.embedder import embed_batch, _EXPECTED_DIM
from graphrag_lite.config import get_config

cfg = get_config()
model = cfg.EMBEDDING_MODEL

print("=" * 60)
print("TASK 3.1 -- Embedder DoD")
print("=" * 60)
print(f"Model         : {model}")
print(f"Expected dim  : {_EXPECTED_DIM.get(model, 'unknown')}")
print()

# 3 real sentences from the actual corpus (from entity source_sentences in
# data/cache/extractions.jsonl / the built graph):
#   1. Knowledge Graph entity — knowledge_graphs.md
#   2. Person entity (schema artifact, expected per option-b policy) — knowledge_graphs.md
#   3. Organization entity (schema artifact) — knowledge_graphs.md
SENTENCES = [
    "A knowledge graph is a structured representation of facts about entities "
    "and the relationships between them.",
    "A person is an entity that has a unique identifier and a type "
    "(such as Person, Organization, Location, or Concept).",
    "An organization is a group of people or things that work together "
    "to achieve a common goal.",
]

print("Input sentences:")
for i, s in enumerate(SENTENCES):
    print(f"  [{i+1}] {s[:90]}")
print()

vectors = embed_batch(SENTENCES, model=model)

print(f"EMBEDDING_DIM (measured from model output): {cfg.EMBEDDING_DIM}")
print()
print("Results:")
for i, (sentence, vec) in enumerate(zip(SENTENCES, vectors)):
    print(f"  [{i+1}] length={len(vec)}  first_5_floats={[round(x, 6) for x in vec[:5]]}")

print()

# Assertions
assert len(vectors) == 3, f"Expected 3 vectors, got {len(vectors)}"
dims = [len(v) for v in vectors]
assert len(set(dims)) == 1, f"Inconsistent dimensions: {dims}"
expected = _EXPECTED_DIM.get(model)
if expected:
    assert dims[0] == expected, f"Dimension mismatch: got {dims[0]}, expected {expected}"
assert cfg.EMBEDDING_DIM == dims[0], "Config EMBEDDING_DIM not updated"
assert cfg.EMBEDDING_DIM > 0, "EMBEDDING_DIM must be > 0 after embed call"

print(f"Vector lengths: {dims}  (all identical: {len(set(dims)) == 1})")
print(f"Config EMBEDDING_DIM populated: {cfg.EMBEDDING_DIM}")
print()
print("ALL ASSERTIONS PASSED")
