# Expanded HotpotQA experiments: methodology

Written 2026-11-19, **before** any expanded-experiment results existed.

## Research question

Does adding graph-derived evidence to dense retrieval improve document retrieval, answer quality,
or supporting-evidence retrieval compared with conventional dense RAG?

No direction is assumed. Negative and neutral results are reported as they come out.

## Data

- **Dataset:** HotpotQA development set, distractor setting (7,405 questions: 5,918 bridge,
  1,487 comparison). The local file is
  `data/benchmarks/hotpotqa/raw/hotpot_qa_distractor_validation.parquet` (HotpotQA organisation's
  Hugging Face mirror; SHA-256 recorded in each `benchmark_config.json`). No other dataset was
  downloaded for this experiment.
- **No frozen test data is used.** HotpotQA's test set has no public answers and was not touched.
  MindMesh's internal frozen test set (`eval/gold_test_set_frozen.json`) was not opened.
- **Sampling (nested, stratified, seed 20261119):**
  - Each type's pool is sorted by `_id`. One `random.Random(20261119)` draws 100 bridge questions,
    then 100 comparison questions.
  - **100q** = the first 50 of each draw. **200q** = all 100 of each draw.
  - The 100q set is therefore a strict subset of the 200q set.
  - Exact IDs are in `dev_distractor_{100,200}q_seed20261119/sample_ids.json`.
  - 1 question overlaps the earlier 20-question pilot by chance.
- **Corpus (shared candidate corpus, NOT FullWiki):** the union of the sampled questions' supplied
  gold and distractor paragraphs, deduplicated by (title, sentences). One chunk per paragraph.

| | 100q | 200q |
|---|---|---|
| Bridge / comparison | 50 / 50 | 100 / 100 |
| Paragraphs supplied → unique | 993 → 985 | 1,986 → 1,960 |
| Sentences | 3,888 | 7,816 |
| Supporting facts (all mapped to chunks) | 239 | 477 |

## Index (one per experiment, shared by all four pipelines)

- **Graph extraction:** MindMesh's own extractor with `mistral:latest`, temperature 0, seed 42,
  frequency penalty 0.25 with the retry ladder [0.5, 1.0], gleanings 0. A paragraph whose extraction
  fails to parse after the retries is recorded as a failure. It stays in the dense index but
  contributes nothing to the graph.
- **Graph build:** MindMesh's NetworkX builder (entity merge by normalised name and alias pass).
- **Zvec collections:** entities, relationships and chunks, each HNSW (768-d `nomic-embed-text`)
  plus BM25, fused with RRF.
- **Reuse of identical work:** an extraction is reused by identical `chunk_id` (a content hash) when
  it came from the same model and settings. This applies to 14 pilot paragraphs in 100q, and the
  100q paragraphs in 200q. Reused IDs are listed in `index/mistral_latest/seeded_extractions.json`.
- **Transient failures:** extraction or generation records that failed only because Ollama was
  unreachable (the machine shutting down) are moved to `*.quarantine.jsonl` and retried. Real model
  failures are kept and counted.

## Pipelines (existing implementations, retrieval logic unchanged)

Full parameters, and SHA-256 hashes of every code file, are recorded in `experiment_environment.json`.

| Pipeline | Retrieval |
|---|---|
| MindMesh Dense (`pipeline_a`) | One hybrid HNSW+BM25 query on chunks, top 10 |
| MindMesh Graph (`pipeline_b`) | 8 entry entities by hybrid search → 2-hop BFS → rank by 0.85·similarity + 0.15·PageRank → top 20 candidates → scoped chunk re-query (document scope), top 10 |
| MindMesh Mix (`lightrag_mix`) | Round-robin merge of dense chunks, entity-local chunks and relationship-global chunks (upstream LightRAG `mix` merge) |
| MindMesh Mix Optimized (`lightrag_mix_opt`) | Dense top 8, then 2 slots for the best graph-supported chunks (RRF of degree-normalised local and global lists) |

The Mix designs were fixed on 2026-11-19 after a diagnosis on MindMesh's internal dev set (not on
HotpotQA). They are not changed for these experiments.

## Controls (identical for all four pipelines)

- Same questions, corpus, index and embedding model.
- Same generator: `mistral:latest` (digest `6577803a…`). Same prompt (`hotpotqa-json-v2-schema`)
  and JSON-schema output.
- Generation settings: temperature 0, seed 42, `num_ctx` 8192, `num_predict` 256.
- The top 5 retrieved paragraphs go to the generator. Retrieval is evaluated at top 10.
- Same machine: RTX 4060 Laptop 8 GB, i7-13700HX, Windows 11, Ollama 0.40.16. Runs are sequential,
  with a cross-process GPU lock, so there is never concurrent inference.

## Metrics

- **Retrieval:**
  - Recall@5 and Recall@10: the fraction of the question's supporting-fact paragraphs in the top k.
  - MRR: the reciprocal rank of the first supporting paragraph.
  - Gold-document retrieval rate: the fraction of questions with **all** supporting paragraphs in
    the top 10.
- **Answer:** Exact Match and token F1 (official HotpotQA normalisation and yes/no rule).
- **Supporting facts:** precision, recall and F1 over (title, sentence) pairs (official).
- **Joint:** EM and F1 (official).
- **Efficiency and failures:** mean retrieval, generation and total latency per question; failed
  retrieval; failed generation; invalid or truncated output (JSON that does not parse into
  answer + supporting facts). A failed generation scores 0. It is never counted as an answer.

Retrieval metrics measure evidence retrieval, not answer correctness. String-match answer metrics do
not establish semantic correctness.

## Quality checks (Stage C)

Each run must satisfy all of these:
- 4 × n rows, with no duplicate (pipeline, question) pairs.
- Question IDs exactly match the sample.
- Every per-question metric recomputes from the stored raw prediction.
- `summary.json` matches the raw rows.
- At most 5% failed queries per pipeline.

The 200q run starts only if the 100q run passes. Results: `qc_100q.json`, `qc_200q.json`.

## Statistics

- **Paired bootstrap over questions**, 10,000 resamples, seed 12345. Each statistic is the mean
  difference (pipeline − Dense) on the same questions, with a 95% percentile interval.
- An interval that excludes 0 is reported as "CI excludes 0". It is not presented as a formal
  significance test, and there is no multiple-comparison correction.
- Per-question wins, losses and ties are reported alongside.

## Execution

`eval/hotpotqa/run_expanded.py` runs, and resumes, the stages in order: prepare → index100 → smoke
(2 questions, diagnostic only) → run100 → qc100 → index200 → run200 → qc200.

A per-user Startup launcher restarts it after a reboot. This was needed because the laptop
hard-reboots under sustained GPU load. The launcher deletes itself on completion. Logs are in
`driver.log` and `logs/`.
