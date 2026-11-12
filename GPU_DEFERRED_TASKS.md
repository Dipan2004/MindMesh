# GPU-Intensive Tasks — Deferred to High-Memory Device

These tasks require sustained LLM inference and cannot complete within
the available timeout window on this machine (i7-1255U, ~15.7 GB RAM,
no dedicated GPU). Transfer this repository to a device with a GPU and
sufficient RAM before running them.

---

## Task G1 — Genuine gleaning-ON full extraction (IN PROGRESS, resumable)

**Script:** `eval/extract_with_gleaning.py`
**Command:** `python eval/extract_with_gleaning.py`
**Status:** 11/42 chunks complete. Fully resumable — restarts from where it stopped.
**Output dir:** `data/experiments/qwen2_5_1_5b_genuine_gleaning_on/`
**What it does:** Extracts entities and relationships from each of the 42 canonical
chunks using `qwen2.5:1.5b`, then runs one genuine gleaning pass per chunk.
**Estimated time:** ~193s/chunk × 31 remaining ≈ 100 min on a device with
a mid-range GPU. Up to 5-6 hours on CPU-only.
**Prerequisites:**
- `qwen2.5:1.5b` installed in Ollama
- `nomic-embed-text` installed in Ollama
- Canonical `data/cache/chunks_v1` present (read-only)
**After completion:** Run `python eval/run_phase2_evaluation_gleaning.py` to
build the graph, Zvec collections, and evaluate all 6 pipelines.

---

## Task G2 — qwen3.5:9b extraction experiment

**Script:** `eval/run_model_experiment.py`
**Command:** `python eval/run_model_experiment.py --model qwen3.5:9b`
**Status:** Not started.
**What it does:** Full extraction + graph + collections + 6-pipeline evaluation
using `qwen3.5:9b` (6.6 GB model). Same 42 canonical chunks.
**Prerequisites:**
- `qwen3.5:9b` installed in Ollama (confirmed: `6488c96fa5fa  6.6 GB`)
- At least 12 GB free RAM for KV cache
**Estimated time:** 3-6 hours on GPU.
**Output dir:** `data/experiments/qwen3_5_9b/` (auto-created)

---

## Task G3 — qwen3:4b extraction experiment

**Script:** `eval/run_model_experiment.py`
**Command:** `python eval/run_model_experiment.py --model qwen3:4b --think off`
**Status:** Not started. (Note: `think=off` required for qwen3 family to suppress chain-of-thought.)
**What it does:** Full extraction + graph + collections + 6-pipeline evaluation
using `qwen3:4b` (2.5 GB model).
**Prerequisites:**
- `qwen3:4b` installed in Ollama (confirmed: `359d7dd4bcda  2.5 GB`)
- Requires `--think off` flag or `LLM_THINK=off` in `.env`
**Estimated time:** 1-3 hours on GPU.
**Output dir:** `data/experiments/qwen3_4b/` (auto-created; previous aborted run exists)

---

## Task G4 — qwen3.5:2b extraction experiment

**Script:** `eval/run_model_experiment.py`
**Command:** `python eval/run_model_experiment.py --model qwen3.5:2b --think off`
**Status:** Not started.
**What it does:** Full extraction with `qwen3.5:2b` (2.7 GB, reasoning model).
**Prerequisites:**
- `qwen3.5:2b` installed in Ollama (confirmed: `53a3cb30c58c  2.7 GB`)
- `think=False` required (qwen3.5 is a reasoning model; thinking consumes
  the entire `num_predict` budget and leaves `response` empty)
**Estimated time:** 2-4 hours on GPU.
**Output dir:** `data/experiments/qwen3_5_2b/` (auto-created)

---

## Task G5 — Multi-model comparison (after G1-G4 complete)

**Script:** To be written as `eval/compare_models.py`
**Status:** Blocked on G1-G4.
**What it does:** Side-by-side comparison of all models on the same 20 dev questions.
Answers the core research question: does a larger/better extraction model produce
graphs that improve retrieval recall?
**Input:** evaluation.json from each model experiment directory.

---

## Task G6 — LightRAG Global keyword-extraction upgrade (optional)

**Status:** Deferred. The current `extract_high_level_keywords()` returns the
query as-is. A proper implementation calls the LLM to extract high-level concepts.
**Cost:** 1 LLM call per query × 20 queries = 20 inference calls.
**Expected benefit:** Modest improvement to LightRAG Global precision; recall may
not change significantly since the HNSW index already handles semantic similarity.
**Script:** Modify `src/graphrag_lite/retrieval/lightrag/keywords.py`

---

## Running on the GPU device

1. Copy this repository to the GPU device.
2. Install Ollama and pull required models:
   ```
   ollama pull qwen2.5:1.5b
   ollama pull nomic-embed-text
   ollama pull qwen3:4b
   ollama pull qwen3.5:9b
   ollama pull qwen3.5:2b
   ```
3. Install Python dependencies:
   ```
   pip install -e ".[dev]"
   ```
4. Run tasks in order: G1 first (resumable), then G2-G4 sequentially.
5. Never run two extraction experiments simultaneously on the same GPU.
6. After G1 completes, run:
   ```
   python eval/extract_with_gleaning.py --status   # verify 42/42 complete
   python eval/run_phase2_evaluation_gleaning.py   # build graph + evaluate
   ```

## What is already complete and does NOT need GPU

- Phase 1 baseline (qwen2.5:1.5b, gleaning OFF) — `data/experiments/qwen2_5_1_5b_validate/`
- v2-OFF arm (stub filter OFF) — `data/experiments/qwen2_5_1_5b_v2_off/`
- v2-ON arm (stub filter ON) — `data/experiments/qwen2_5_1_5b_v2_on/`
- LightRAG retrieval implementation — `src/graphrag_lite/retrieval/lightrag/`
- Universal experiment runner — `eval/run_model_experiment.py`
- Resumable gleaning extractor — `eval/extract_with_gleaning.py`
- All regression tests — `tests/test_model_cache_isolation.py`
- Comparison reports — `data/experiments/phase2_comparison_20261110_002916.json`

---

## Task G7 — Pipeline 3 LightRAG document indexing

**Script:** `run_pipeline.py`
**Command:** `python run_pipeline.py --pipeline pipeline_3_lightrag --model qwen2.5:1.5b --setup-only`
**Status:** NOT STARTED. Pipeline 3 is fully implemented and importable. Indexing is the only remaining step.
**What it does:** Calls `LightRAG.ainsert(docs)` on the 5 documents in `data/raw/`. LightRAG uses its own entity extraction, relationship extraction, and knowledge graph building. This is independent from the MindMesh canonical graph.
**Output dir:** `data/pipeline3_workdir/`
**Prerequisites:**
- `qwen2.5:1.5b` installed in Ollama
- `nomic-embed-text` installed in Ollama
- `data/raw/` contains the 5 source documents
**Estimated time:** 30-90 min on GPU (LightRAG extraction is faster than MindMesh's because it uses async parallel extraction)
**After completion:** `python run_pipeline.py --pipeline pipeline_3_lightrag --model qwen2.5:1.5b --query "what is HNSW?" --mode hybrid`

---

## Updated Status (2026-11-11)

### Completed (no GPU needed):
- Phase 1 baseline, v2-OFF, v2-ON experiments — all evaluated
- LightRAG v1.5.8 fully importable from `LightRAG/`
- Universal pipeline runner `run_pipeline.py` — CPU tests 22/22 PASS
- Pipeline 3 adapter — importable, dry-run verified
- LightRAG-compat API layer — all routes registered and responding
- LightRAG UI copied to `ui/lightrag_ui/` — ready for `bun run dev`
- Pipeline registry — 7 pipelines registered

### Still deferred to GPU device:
- G1: Genuine gleaning extraction (11/42 done, resumable)
- G2: qwen3.5:9b experiment
- G3: qwen3:4b experiment
- G4: qwen3.5:2b experiment
- G5: Multi-model comparison
- G6: LightRAG Global keyword upgrade (optional)
- G7: Pipeline 3 document indexing (NEW)
