# Dense vs query-focused C5 with on-demand extraction: 100 HotpotQA questions (2026-11-20)

**Exploratory evaluation: one run, 100 questions, one machine.** Run `dense_vs_c5_6ca3b11e`, completed 2026-11-20 02:07.

## Setup

| Item | Value |
|---|---|
| Questions | 100 HotpotQA dev-distractor questions (50 bridge + 50 comparison), seed 20261119. The same IDs as the expanded experiment. |
| Corpus | The same 985 deduplicated paragraphs (shared candidate corpus, not FullWiki) |
| Dense index | New Dense-only Zvec index: HNSW + BM25, `nomic-embed-text` 768-d. Built offline in **27.9 s**, embedding only, no LLM. |
| C5 extraction | **qwen2.5:3b** (digest `357c53fb…`), original MindMesh extraction prompt, temperature 0, seed 42, retry ladder. **Extracted on demand** into a persistent cache (version `644b068b5b6c`). The 104 Mistral records were never read. |
| Generator (both pipelines) | qwen2.5:3b, the same JSON-schema prompt, temperature 0, seed 42, `num_ctx` 8192, `num_predict` 256, top-5 paragraphs as context |
| Hardware | RTX 4060 Laptop 8 GB, i7-13700HX, 15.6 GB RAM. During the run: GPU mean 87.7 °C (max 93), 88% of samples ≥ 87 °C, median 2,055 MHz, peak GPU memory 2.88 GB. |
| Completeness | **Verified** (`verification.json`): 100/100 retrieval and 100/100 generation rows per pipeline, no duplicates, no missing IDs, 0 rows with errors |

## 1. C5 retrieval evaluation [measured]

C5 ran with the design fixed in advance:
- Dense top 5;
- title/entity expansion hop;
- temporary per-question graph, built from valid cached or newly extracted qwen2.5:3b records only;
- BFS of 2 hops from question entities;
- path-support ranking.

Nothing was tuned on these results.

| | Recall@5 | Recall@10 | MRR | Gold docs all in top 10 |
|---|---|---|---|---|
| MindMesh Dense | 0.925 | 0.990 | **0.907** | 0.98 |
| C5 on-demand | **0.950** | **0.995** | 0.750 | **0.99** |
| C5 − Dense (95% paired bootstrap CI) | +0.025 [−0.005, +0.055] | +0.005 [0.000, +0.015] | **−0.157 [−0.220, −0.094]** | +0.01 [0.00, +0.03] |
| C5 better / worse, by question | 8 / 3 | 1 / 0 | 6 / 33 | 1 / 0 |

**By question type:**

| | Dense bridge | C5 bridge | Dense comparison | C5 comparison |
|---|---|---|---|---|
| Recall@5 | 0.91 | 0.95 | 0.94 | 0.95 |
| Recall@10 | 0.98 | 0.99 | 1.00 | 1.00 |
| MRR | 0.960 | 0.823 | 0.853 | 0.676 |

**C5 diagnostics:**

| Measure | Value |
|---|---|
| Paragraphs selected per question | mean 5.43, median 5, max 9 |
| Selected by the expansion hop | 0.18 per question from question titles, 0.25 from extracted entities |
| Gold-paragraph coverage of the selection | **0.98** (both gold paragraphs selected for 96/100) |
| Temporary graph size | mean 18.0 nodes / 9.8 edges (median 16 / 8) |
| Questions anchored on an entity from the question | **92**; 8 fell back to Dense order |
| Selected paragraphs whose extraction had failed | 6 in total (0.06 per question), excluded from graphs |

**Reading:**
- **C5 finds slightly more of the gold evidence.** Recall@5 is +0.025, but the CI includes 0.
- **C5 ranks gold evidence clearly lower.** MRR is −0.157, the CI excludes 0, and C5 was worse on 33 questions versus better on 6. Path-support ranking promotes graph-connected paragraphs above the top dense hit, across both question types.
- **No retrieval improvement is established.** This is one run of 100 questions.

## 2. Answer quality [measured; same generator for both]

| | Answer EM | Answer F1 | SP precision | SP recall | SP F1 | Joint EM | Joint F1 | Invalid generator output |
|---|---|---|---|---|---|---|---|---|
| MindMesh Dense | **0.55** | **0.616** | 0.427 | 0.286 | 0.320 | 0.05 | 0.217 | 2 |
| C5 on-demand | 0.50 | 0.569 | 0.403 | 0.280 | 0.310 | 0.04 | 0.197 | 3 |
| C5 − Dense (95% CI) | −0.05 [−0.13, +0.03] | −0.047 [−0.131, +0.036] | | | −0.010 [−0.080, +0.062] | | −0.020 [−0.084, +0.044] | |

- **Neither answer nor supporting-fact quality differs measurably.** All CIs include 0.
- **The point estimates favour Dense.** This is consistent with C5's lower MRR: the generator sees only the top 5.
- **These are string-match metrics.** They do not establish semantic correctness.

## 3. Latency: online retrieval separated from extraction [measured]

| Per question (s) | Mean | Median | p90 | Max |
|---|---|---|---|---|
| Dense retrieval | 0.486 | 0.468 | 0.547 | — |
| **C5 online retrieval** (Dense step + graph build + search/ranking, excluding extraction) | **1.165** | **0.479** | 0.703 | 12.2 |
| — of which C5's Dense step | 0.458 | 0.414 | | 4.67 |
| — of which graph construction | 0.0014 | 0.0002 | | 0.022 |
| — of which search, ranking and cache bookkeeping | 0.705 | 0.058 | | 11.7 |
| **C5 on-demand extraction** (graph-construction work done at query time) | **48.8** | **35.4** | | 363.6 |
| C5 recorded retrieval latency (online + extraction, cold cache) | 50.0 | 35.8 | | 375.8 |
| Answer generation: Dense / C5 | 0.846 / 0.816 | | | |

**Comparability:**
- **C5's recorded `retrieval_latency_s` mixes in build work.** It includes on-demand extraction, which is graph-construction work Dense did offline (in 27.9 s of embedding). That is why its "total latency" (50.8 s versus 1.33 s) is not a like-for-like online comparison.
- **Like-for-like online cost is close.** C5's online retrieval is 1.17 s versus 0.49 s (median 0.48 versus 0.47).
- **Extraction is the cost of building C5's graph,** reported in the next section.

## 4. Selective-extraction cost

**4a. Measured: on-demand extraction actually performed in this run**

| | Value |
|---|---|
| Unique paragraphs extracted | **507 of 985 (51.5%)**; each extracted once (cache verified, no duplicates) |
| Outcome | 501 ok, **6 failed_model** (malformed JSON after the retry ladder), 0 failed_infra |
| Total extraction time | **4,905.5 s = 1.36 h** |
| Time per paragraph | mean 9.68 s, median 5.84 s, max 243.7 s |
| Output tokens per paragraph | mean 764, median 524, max 12,288 (runaway outputs hitting 3 × 4,096) |
| Decode rate | 81.0 tok/s |
| Extraction per question | mean 48.8 s, median 35.4 s, max 363.6 s; **every question needed at least one extraction** |
| Cache | 543 lookups: **38 hits (7.0%)**, 505 misses. The cumulative hit rate was 0% through question 25 and 7% at question 100. |
| Graph density | 3.81 entities and 1.93 relationships per valid paragraph |

The cache barely helped here. In the HotpotQA distractor setting, each question mostly brings its own paragraphs, so this run is close to the cold-cache worst case for C5.

**4b. Estimate (not measured): full-corpus extraction with the same model**

| | Value |
|---|---|
| Estimated full-corpus extraction | 985 × 9.68 s mean ≈ **2.65 h**. This is an estimate, not measured runtime. |
| Measured C5 extraction | 1.36 h, about 51% of the estimate, done as on-demand work |

This does **not** show that C5 extracts faster overall:
- C5 performed and measured 507 extractions;
- the full-corpus alternative with qwen2.5:3b was not run.

C5's saving here comes only from touching about half the corpus. Its cost moves from indexing time to the first time each paragraph is queried.

## 5. Zvec keyword-parser fallbacks [measured from raw log; nothing re-queried]

| | Dense | C5 first-stage search |
|---|---|---|
| First-stage queries | 100 | 100 |
| Normal hybrid (HNSW + BM25) | 96 | 96 |
| Fell back to vector-only | **4 (4%)** | **4 (4%)** |

- **Affected questions:** `5a7602d8554299109176e5fb`, `5a8220db5542995ce29dccc1`, `5a837121554299123d8c2128`, `5a84c5545542991dd0999dee`.
- **Warnings:** 8 lines in total, of two kinds:
  - `fts syntax error … extraneous input 'and'`
  - `field-prefixed queries are not supported` (a colon in the question)
- **Identical behaviour in both pipelines:** fallback happened in **both** pipelines, on the **same** questions.
  - Both pipelines call the same function (`zvec_client.hybrid_query`) with the same query text and top-k.
  - Verified directly: C5's stored first-stage results equal Dense's retrieved IDs on **100/100** questions.
- **Effect on comparability:** none between the two pipelines, because the same 4 questions lost the keyword signal in both. In absolute terms, those 4 questions (4% of the sample) were retrieved by vector similarity alone.

## 6. Failures and incidents

- **Retrieval and generation:** 0 failed retrievals and 0 failed generation calls. Invalid generator output (JSON without a usable answer): Dense 2, C5 3. They are scored as answers that failed, not counted as correct.
- **Extraction:** 6 failed_model (listed in `analysis.json`), excluded from the C5 graphs.
- **Thermal hard reboot at 23:04:21** during question 38. The run was resumed from a verified checkpoint, with nothing rerun.
  - 2 extractions saved before the reboot count as cache hits for question 38, so that question's per-question extraction understates by 23.75 s.
  - The total in 4a is computed from the cache entries and includes them: 4,881.8 + 23.75 = 4,905.5 s, reconciled exactly.
  - Details in `incidents_log.md`.
- The 2-question smoke test used a separate throwaway cache, not counted here.

## 7. Limitations

- **Small sample:** 100 questions and one run. CIs come from a paired bootstrap; no formal significance tests, and no multiple-comparison correction.
- **Narrow comparison:** only Dense and C5. Graph, Mix and Mix Optimized need a full graph and were not part of this run.
- **A different generator from the four-pipeline design:** the generator is qwen2.5:3b. Answer quality isn't comparable with the pilot (mistral) or the proposed qwen2.5:1.5b setup.
- **Sparse extraction:** qwen2.5:3b produces sparse graphs (3.8 entities and 1.9 relationships per paragraph). This limits C5's expansion: only 0.43 extra paragraphs per question.
- **Throttled timings:** extraction timings were measured under thermal throttling and include long runaway outputs.
- **The distractor setting limits C5's savings:** about 51% of the corpus is touched, and cache reuse is very low. Savings in a FullWiki-scale corpus are not measured here.

## Files (all in this folder)

- `runs/dense_vs_c5_6ca3b11e/`:
  - raw data: `retrieval.jsonl`, `results.jsonl`, `run_config.json`, `gpu_telemetry.csv`;
  - checks: `verification.json`, `fts_fallback_report.json`;
  - analysis: `summary.json`, `analysis.json`, `c5_ondemand_comparison.csv`, `c5_ondemand_per_question.csv`.
- `extraction_cache/644b068b5b6c/`: `cache.jsonl` (507 entries) and `cache_meta.json`.
- `dense_index/`: Dense-only index and `dense_index_meta.json`.
- Run records:
  - `run_full.log`: the raw log, including the keyword-parser warnings;
  - `incidents_log.md`, `code_freeze.json`.
- Scripts in `eval/hotpotqa/`:
  - `build_dense_index.py`, `run_c5_ondemand.py`, `analyze_c5_ondemand.py`;
  - `fts_fallback_report.py`, `verify_c5_run.py`.
- Modules in `src/graphrag_lite/retrieval/`: `extraction_cache.py`, `query_focused_ondemand.py`.
