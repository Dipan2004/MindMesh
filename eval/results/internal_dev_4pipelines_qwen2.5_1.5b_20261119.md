# Internal dev evaluation — 4 pipelines, qwen2.5:1.5b (2026-11-19)

Commands run exactly as requested, sequentially (one GPU job at a time):

```
python eval/run_lightrag_pipeline.py qwen2.5:1.5b --pipeline pipeline_a       --limit 20 --top-k 6 --generate-answer
python eval/run_lightrag_pipeline.py qwen2.5:1.5b --pipeline pipeline_b       --limit 20 --top-k 6 --generate-answer
python eval/run_lightrag_pipeline.py qwen2.5:1.5b --pipeline lightrag_mix     --limit 20 --top-k 6 --generate-answer
python eval/run_lightrag_pipeline.py qwen2.5:1.5b --pipeline lightrag_mix_opt --limit 20 --top-k 6 --generate-answer
```

- Dataset: `eval/gold_dev_set.json` (20 dev questions D01–D20). The frozen test set was not opened.
- Corpus/storage: canonical MindMesh store `data/cache` (42 chunks), restored byte-exact from `backups/canonical`.
- Generator: `qwen2.5:1.5b` (pulled with approval on 2026-11-19, 986 MB). Embeddings: `nomic-embed-text`.
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, driver 592.82). `ollama ps` during the run:
  qwen2.5:1.5b 1.4 GB **100% GPU** (ctx 8192), nomic-embed-text 335 MB **100% GPU**.
- All four runs exited 0; 0 per-question errors; every question produced an answer.

## Results

| Pipeline | Label | Recall@6 | MRR | Qs with ≥1 gold chunk | Mean retrieval s | Mean generation s | Mean total s | Wall s | "Cannot answer" |
|---|---|---|---|---|---|---|---|---|---|
| `pipeline_a` | MindMesh Dense | **0.925** | **0.845** | 17/20 | 0.47 | 2.25 | 2.75 | 55.1 | 2 |
| `pipeline_b` | MindMesh Graph | 0.875 | 0.835 | 16/20 | 0.58 | 0.90 | 1.54 | 30.8 | 3 |
| `lightrag_mix` | MindMesh Mix | 0.650 | 0.433 | 13/20 | 0.68 | 2.55 | 3.29 | 65.8 | 2 |
| `lightrag_mix_opt` | MindMesh Mix Optimized | 0.675 | 0.473 | 13/20 | 0.64 | 0.89 | 1.69 | 33.8 | 3 |

Recall@6 here is the runner's metric: fraction of a question's gold chunk IDs found in the top 6, averaged over 20 questions. It measures retrieval only; it says nothing about whether the generated answers are correct.

## Comparison with the previous recorded runs (2026-11-18)

| Pipeline | Previous R@6 / MRR | Today R@6 / MRR | |
|---|---|---|---|
| MindMesh Dense | 0.925 / 0.845 (×2 runs) | 0.925 / 0.845 | identical |
| MindMesh Graph | 0.875 / 0.835 (×2 runs) | 0.875 / 0.835 | identical |
| MindMesh Mix | 0.800 / 0.483 | 0.650 / 0.433 | lower |
| MindMesh Mix Optimized | 0.925 / 0.543 (×2 runs) | 0.675 / 0.473 | lower |

Dense and Graph reproduce the earlier numbers exactly, which confirms the canonical store and the
pipelines are intact on this machine.

**Mix and Mix Optimized do not reproduce.** The code behind them
(`src/graphrag_lite/retrieval/lightrag/`) was never committed — `.gitignore`'s `LightRAG/` rule
also matched that folder on Windows — so the clone had no Mix implementation at all. It was rebuilt
from the documented design on 2026-11-19. The rebuilt version is a different implementation, and on
this set it ranks gold chunks lower than both the lost version and plain Dense.

## Per-question misses (gold chunk not in top 6)

- Dense: D15 (1 of 2 gold), D18
- Graph: D15 (1 of 2), D16, D18
- Mix: D03, D09, D10 (1 of 2), D12 (1 of 2), D13 (1 of 2), D15 (1 of 2), D16, D17, D18
- Mix Optimized: D02, D10 (1 of 2), D12 (1 of 2), D13, D15 (1 of 2), D16, D17, D18

D18 is missed by every pipeline (documented graph limitation: needs cross-document edges).

## Mix fix (same day) — diagnosis and re-run

### Diagnosis (per-list Recall@6 / MRR on the same 20 dev questions)

| Ranked list | R@6 | MRR |
|---|---|---|
| Local (entity → provenance + one-hop) | 0.375 | 0.259 |
| Global (relationship → provenance + endpoints) | 0.550 | 0.385 |
| Dense chunk search, pool 20 (first rebuild) | 0.850 | 0.841 |
| Dense chunk search, pool 8 (= Pipeline A's query) | 0.925 | 0.845 |
| Equal-weight RRF of all three (first rebuild of Mix) | 0.650 | 0.456 |
| Round-robin vector → entity → relationship (upstream LightRAG `mix` merge) | 0.775 | 0.825 |
| Dense top 5 + 1 graph slot | 0.900 | 0.845 |
| Dense top 4 + 2 graph slots | 0.775 | 0.839 |
| Dense with graph score boost (any weight > 0.02) | ≤ 0.925 | falls to 0.53–0.77 |

The canonical graph is tiny (72 nodes, 30 edges for 42 chunks), so the graph lists are weak on
their own. Equal-weight RRF let them displace dense's correct hits. A second, smaller cause: the
first rebuild queried 20 dense candidates, and Zvec's RRF reranker orders differently with a wider
pool. **On this internal set no fusion beats Dense alone**; graph evidence never added a gold chunk
that Dense missed.

### What changed

- Dense side of Mix is now built exactly like Pipeline A's query (pool = max(top_k, ZVEC_TOP_N_FLAT)).
- **MindMesh Mix**: upstream LightRAG `mix` merge — round-robin interleave of vector, entity and
  relationship chunk lists (vector first), deduplicated.
- **MindMesh Mix Optimized**: dense-anchored — dense keeps the first top_k − g slots, the best
  graph-supported chunks not already placed fill the last g = max(1, round(0.2·top_k)) slots
  (1 at k=6, 2 at k=10).

Caveat: these designs were chosen by looking at this same 20-question dev set, so the dev numbers
below are optimistic. HotpotQA is the held-out check.

### Re-run results (same commands, qwen2.5:1.5b, top-k 6)

| Pipeline | Before fix R@6 / MRR | After fix R@6 / MRR | Mean total s | Misses |
|---|---|---|---|---|
| MindMesh Mix | 0.650 / 0.433 | **0.775 / 0.813** | 2.13 | D10, D13, D14, D15, D16, D18 |
| MindMesh Mix Optimized | 0.675 / 0.473 | **0.900 / 0.845** | 1.98 | D13, D15, D18 |
| (MindMesh Dense, unchanged) | 0.925 / 0.845 | — | 2.75 | D15, D18 |

Output: `data/experiments/lightrag_qwen2.5_1.5b_lightrag_mix_20261119_031755/` and
`data/experiments/lightrag_qwen2.5_1.5b_lightrag_mix_opt_20261119_031838/`. 0 errors, all 20 answered.
The earlier recorded Mix Optimized result (0.925 / 0.543) came from the lost implementation and is
still not reproduced.

## Output directories

- `data/experiments/lightrag_qwen2.5_1.5b_pipeline_a_20261119_030500/`
- `data/experiments/lightrag_qwen2.5_1.5b_pipeline_b_20261119_030556/`
- `data/experiments/lightrag_qwen2.5_1.5b_lightrag_mix_20261119_030627/`
- `data/experiments/lightrag_qwen2.5_1.5b_lightrag_mix_opt_20261119_030733/`
- Combined log: `data/experiments/internal_dev_qwen2.5_1.5b_4pipelines.log`

Each directory holds `config.json`, `retrieval_results.json` (per-question chunk IDs, answers, latencies), `evaluation.json`, `metadata.json` and `logs/run.log`.
