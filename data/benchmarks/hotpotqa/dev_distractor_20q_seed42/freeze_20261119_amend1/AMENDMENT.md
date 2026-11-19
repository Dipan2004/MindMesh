# Freeze amendment 1 (2026-11-19)

Original freeze: `freeze_20261119/` (kept unmodified).

**File changed:** `eval/run_model_experiment.py` (`build_experiment_collections`)

**Failure:** `ValueError: Too many docs: 1311 exceeds max write batch size of 1024` at `ent_col.insert(ent_docs)`.
The HotpotQA graph has 1,311 entities; Zvec accepts at most 1,024 documents per insert call.

**Change:** the entity, relationship and chunk `insert()` calls now write in batches of 1,000.
Write path only. The stored documents, retrieval code, retrieval/fusion settings and generation
settings are unchanged. Approved by the user before the full benchmark began.

All other hashes are identical to the original manifest.
