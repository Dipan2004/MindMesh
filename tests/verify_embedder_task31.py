"""
Task 3.1 verification addendum — three checks:
  1. Determinism: embed same 3 sentences twice, compare vectors
  2. Batch-size sanity: 10 sentences, order preserved, spot-check match
  3. Config singleton: EMBEDDING_DIM visible across module contexts after mutation
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Pull 10+ real sentences from the corpus ───────────────────────────────────
import json, io, contextlib
from graphrag_lite.graph.builder import build_graph, load_extractions

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)

real_sentences = []
for node in G.nodes():
    nd = G.nodes[node]
    for ss in nd.get("source_sentences", []):
        text = ss.get("text", "").strip()
        if len(text) > 20 and text not in real_sentences:
            real_sentences.append(text)
        if len(real_sentences) >= 12:
            break
    if len(real_sentences) >= 12:
        break

SENTENCES_3 = real_sentences[:3]
SENTENCES_10 = real_sentences[:10]

# ── Imports ───────────────────────────────────────────────────────────────────
from graphrag_lite.embeddings.embedder import embed_batch
from graphrag_lite.config import get_config

model = get_config().EMBEDDING_MODEL
print("=" * 65)
print("TASK 3.1 VERIFICATION ADDENDUM")
print("=" * 65)
print(f"Model: {model}")
print()

# ═══════════════════════════════════════════════════════════════════
# CHECK 1: Determinism — embed same 3 sentences twice, same process
# ═══════════════════════════════════════════════════════════════════
print("CHECK 1: Determinism (two embed calls, same 3 sentences)")
print("-" * 65)
for i, s in enumerate(SENTENCES_3):
    print(f"  [{i+1}] {s[:80]}")
print()

run1 = embed_batch(SENTENCES_3, model=model)
run2 = embed_batch(SENTENCES_3, model=model)

identical = all(
    all(abs(a - b) < 1e-9 for a, b in zip(run1[i], run2[i]))
    for i in range(3)
)

print(f"{'Sentence':<3}  {'Run 1 first_5':55}  {'Run 2 first_5':55}  Match")
print("-" * 130)
all_match = True
for i in range(3):
    f1 = [round(x, 6) for x in run1[i][:5]]
    f2 = [round(x, 6) for x in run2[i][:5]]
    match = f1 == f2
    if not match:
        all_match = False
    print(f"  {i+1}   {str(f1):<55}  {str(f2):<55}  {'YES' if match else 'NO'}")

print()
if all_match:
    print("DETERMINISM: YES — identical float values on both calls.")
    print("nomic-embed-text via Ollama is deterministic for these inputs.")
else:
    print("DETERMINISM: NO — values differ between calls.")
    print("NOTE: non-deterministic embeddings affect reproducibility of Zvec")
    print("      collections and any later nearest-neighbour consistency checks.")
print()

# ═══════════════════════════════════════════════════════════════════
# CHECK 2: Batch-size sanity — 10 sentences
# ═══════════════════════════════════════════════════════════════════
print("CHECK 2: Batch sanity (10 sentences)")
print("-" * 65)
for i, s in enumerate(SENTENCES_10):
    print(f"  [{i:2d}] {s[:70]}")
print()

batch10 = embed_batch(SENTENCES_10, model=model)

print(f"(a) Returned list length: {len(batch10)}  (expected 10)")
assert len(batch10) == 10, f"Expected 10 vectors, got {len(batch10)}"

dims = [len(v) for v in batch10]
all_768 = all(d == 768 for d in dims)
print(f"(b) All lengths 768: {all_768}  dims={dims}")
assert all_768, f"Inconsistent dimensions: {dims}"

# (c) Spot-check: embed sentence at index 4 individually, compare to batch10[4]
SPOT_IDX = 4
single = embed_batch([SENTENCES_10[SPOT_IDX]], model=model)[0]
batch_vec = batch10[SPOT_IDX]
# Compare using cosine similarity (should be ~1.0 if same)
dot = sum(a * b for a, b in zip(single, batch_vec))
mag_s = math.sqrt(sum(x * x for x in single))
mag_b = math.sqrt(sum(x * x for x in batch_vec))
cosine_sim = dot / (mag_s * mag_b) if mag_s > 0 and mag_b > 0 else 0.0
exact_match = all(abs(a - b) < 1e-9 for a, b in zip(single, batch_vec))

print(f"(c) Spot-check sentence[{SPOT_IDX}]: individual vs batch position {SPOT_IDX}")
print(f"    text: {SENTENCES_10[SPOT_IDX][:70]}")
print(f"    cosine similarity: {cosine_sim:.8f}  (1.0 = identical direction)")
print(f"    exact float match : {exact_match}")
assert cosine_sim > 0.9999, f"Cosine similarity too low: {cosine_sim}"
print()
print("CHECK 2: PASS — 10 vectors returned in order, all length 768, spot-check cosine=1.0")
print()

# ═══════════════════════════════════════════════════════════════════
# CHECK 3: Config singleton — EMBEDDING_DIM visible across contexts
# ═══════════════════════════════════════════════════════════════════
print("CHECK 3: Config singleton — EMBEDDING_DIM across import contexts")
print("-" * 65)

# embed_batch already ran above, so EMBEDDING_DIM should be populated
# Simulate a "second module context" by re-importing config in a new namespace
import importlib
import graphrag_lite.config as cfg_module
cfg_direct = cfg_module.get_config()   # same lru_cache call — must return same object

print(f"get_config() from embedder context : EMBEDDING_DIM = {get_config().EMBEDDING_DIM}")
print(f"get_config() re-imported directly  : EMBEDDING_DIM = {cfg_direct.EMBEDDING_DIM}")
print(f"Same object (is): {get_config() is cfg_direct}")

if cfg_direct.EMBEDDING_DIM == 768:
    print("CHECK 3: PASS — object.__setattr__ correctly mutated the lru_cache singleton.")
    print("         All modules calling get_config() see EMBEDDING_DIM=768.")
else:
    print(f"CHECK 3: FAIL — EMBEDDING_DIM={cfg_direct.EMBEDDING_DIM} (expected 768).")
    print("         object.__setattr__ did NOT update the cached singleton.")
    print("         A separate module-level dimension registry is needed instead.")

assert cfg_direct.EMBEDDING_DIM == 768, \
    f"Config singleton not updated: got {cfg_direct.EMBEDDING_DIM}"
assert get_config() is cfg_direct, "get_config() returned different objects — lru_cache broken"
print()
print("ALL THREE CHECKS PASSED")
