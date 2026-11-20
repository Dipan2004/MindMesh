# MindMesh: paper preparation (outline approved by the user, 2026-11-20)

All numbers below come from existing, immutable artifacts. Nothing was re-run.

## A. Experiment inventory (completed vs incomplete)

| ID | Experiment | Status | n | Artifact |
|---|---|---|---|---|
| E1 | Internal dev set, 4 pipelines, qwen2.5:1.5b, top-6 (before and after the Mix fix) | **Completed** | 20 | `eval/results/internal_dev_4pipelines_qwen2.5_1.5b_20261119.md`; `data/experiments/lightrag_qwen2.5_1.5b_*_20261119_*` |
| E2 | HotpotQA pilot, 4 pipelines, mistral:latest extraction + generation, shared 200-paragraph corpus | **Completed** (pilot; three documented pre-run amendments) | 20 | `eval/results/hotpotqa_pilot_report_20261119.md`; `dev_distractor_20q_seed42/runs/gen-mistral_latest__idx-mistral_latest__a3b14137` |
| E3 | Extraction-variant validation (Mistral baseline / C / B+C) + qwen2.5:3b B+C | **Completed** (agreement study, no gold annotations) | 30 paragraphs | `expanded_20261119/extraction_validation_report.md` |
| E4 | Query-focused graph simulation (coverage / paths), offline | **Completed, exploratory** | 20 | `query_focused_graph_investigation.md`; `query_focused_simulation_pilot20_mistral.json` |
| E5 | C5 prototype functional test on the pilot (stored Dense lists) | **Exploratory only.** The pilot was used to design C5, so it is not held out | 20 | conversation record / `paper_prep` notes only |
| E6 | **Dense vs C5 with on-demand qwen2.5:3b extraction** | **Completed, verified** | **100** | `c5_ondemand_qwen3b_20261119/report.md`; `runs/dense_vs_c5_6ca3b11e` |
| E7 | Mistral full-corpus 4-pipeline experiment | **INCOMPLETE: paused by user at 104/985 extractions. No results exist.** | — | `expanded_20261119/` (checkpoint only) |
| E8 | 200-question expansion | **Not started** (held) | — | — |
| E9 | qwen3:4b multi-model architecture | **Not run.** Code and config prepared; model downloaded; Stage 1 not run | — | `config/models.json`, `src/mindmesh/model_config.py` |

Smoke tests (2 questions) are diagnostic only and are not reported as results.

## B. Proposed research questions
- **RQ1.** Does adding graph-derived evidence to dense retrieval improve gold-evidence retrieval (Recall@k, MRR)
  over dense RAG on multi-hop QA?
- **RQ2.** Does it improve answer quality (EM/F1) or supporting-fact F1, under an identical generator?
- **RQ3.** Can query-focused, on-demand graph construction (C5) reduce extraction cost relative to full-corpus
  construction while keeping evidence coverage?
- **RQ4.** What is the cost structure of graph-RAG on commodity hardware? Covers offline construction vs online
  latency, extraction-model choice, and thermal constraints.

## C. Paper outline
1. **Introduction:** graph-RAG claims vs cost; a small local-model setting (8 GB laptop GPU).
2. **Related work:** GraphRAG, LightRAG (MindMesh adapts its retrieval modes and UI; MIT licence attributed),
   HotpotQA.
3. **System:** MindMesh.
   - Zvec hybrid HNSW + BM25 index; NetworkX entity graph.
   - The four pipelines (Dense, Graph, Mix, Mix Optimized) and C5 (query-focused, on-demand, cached).
   - Disclose that the Mix/Mix Optimized code was reconstructed (the original was never committed) and redesigned
     once, on internal dev data.
4. **Experimental setup:**
   - HotpotQA dev-distractor; seeded stratified samples (bridge/comparison); shared candidate corpus (not FullWiki).
   - Official HotpotQA metrics plus Recall@k/MRR; paired bootstrap 95% CIs.
   - Freeze manifests with documented amendments; hardware and thermal reporting.
5. **Results:**
   - 5.1 Four-pipeline pilot (E2, n = 20), labelled a pilot.
   - 5.2 Dense vs C5 (E6, n = 100): retrieval, answers, latency split, extraction cost.
   - 5.3 Extraction-model and prompt study (E3).
   - 5.4 C5 loss analysis (33 MRR losses; post-hoc, clearly marked).
6. **Discussion:**
   - Graph evidence reorders rather than discovers.
   - Comparison questions.
   - Extraction sparsity of small models.
   - When on-demand construction pays off (cache reuse; FullWiki).
7. **Limitations and threats to validity** (Section F).
8. **Conclusion.**

**Tables:**
- T1: dataset and sample statistics (20q and 100q).
- T2: models, digests and settings.
- T3: pilot four-pipeline results (E2).
- T4: Dense vs C5 retrieval and answer metrics with CIs (E6).
- T5: latency split, online vs extraction (E6).
- T6: measured extraction cost vs the labelled full-corpus estimate (E6).
- T7: extraction-variant agreement (E3).
- T8: C5 loss causes (post-hoc).

**Figures:**
- F1: architecture (offline vs online paths).
- F2: per-question MRR difference histogram (E6).
- F3: C5 cumulative cache-hit rate vs question index (E6).
- F4: extraction time per paragraph distribution, with GPU temperature trace.
- F5: recall by question type (bridge vs comparison).

## D. Claim–evidence table

| # | Proposed claim | Evidence (exact result) | Artifact | Strength |
|---|---|---|---|---|
| C1 | On 100 HotpotQA questions, C5 does not significantly improve gold-evidence recall over Dense | R@5 0.950 vs 0.925, diff +0.025, CI [−0.005, +0.055]; R@10 +0.005, CI [0, 0.015] | E6 `analysis.json` | Supported (n = 100, one run) |
| C2 | C5 ranks gold evidence lower than Dense | MRR 0.750 vs 0.907, diff −0.157, CI [−0.220, −0.094]; worse on 33, better on 6 | E6 `analysis.json` | Supported |
| C3 | No measurable difference in answer or supporting-fact quality between Dense and C5 under an identical generator | EM −0.05, CI [−0.13, 0.03]; F1 −0.047, CI [−0.131, 0.036]; SP-F1 −0.010, CI [−0.080, 0.062] | E6 | Supported (CIs include 0) |
| C4 | C5's online retrieval cost is close to Dense once extraction is separated | Mean 1.17 vs 0.49 s; median 0.48 vs 0.47 s | E6 `report.md` §3 | Supported |
| C5 | On-demand construction touched about half the corpus | 507/985 paragraphs extracted (51.5%), 1.36 h measured | E6 cache + analysis | Supported (measured) |
| C6 | Full-corpus extraction with the same model would take ~2.65 h | 985 × measured mean 9.68 s | E6 | **Estimate only**: must be labelled as such |
| C7 | In the HotpotQA distractor setting, extraction caching gives little reuse | 7.0% hit rate; 0% through question 25 | E6 | Supported for this setting only |
| C8 | C5's MRR loss comes mainly from reordering Dense's own candidates at rank 1 | 30/33 losses had gold at Dense rank 1; 57/60 promoted paragraphs were in Dense's top 10; path-support ranking is the primary cause in 19/33 | `paper_prep/c5_mrr_loss_analysis.json` | Supported (post-hoc, descriptive) |
| C9 | Keeping Dense's top-1 would recover Dense-level MRR | Post-hoc 0.908 | same | **Not claimable**: hypothesis for a held-out test |
| C10 | In the 20q pilot, the Graph pipeline retrieved less gold evidence than Dense | R@10 0.825 vs 0.975; Graph worse on 5 questions | E2 report | Pilot only (n = 20) |
| C11 | Small models produce sparse graphs; qwen2.5:3b agrees poorly with Mistral extractions | Relationship recall vs Mistral 0.29 (B+C, renames counted); 3.8 entities/paragraph in E6 | E3, E6 | Agreement, not correctness |
| C12 | Sentence-ID output (B+C) cuts extraction time ~45% but changes the extracted relationships substantially | −44.6% time; relationship recall 0.65 vs baseline | E3 | Supported (30 paragraphs, agreement) |
| C13 | Sustained extraction on an 8 GB laptop GPU is thermally limited | Mean 87.7–88.2 °C, ≥ 87 °C in 88–94% of samples; multiple thermal reboots | E3, E6 telemetry; incident logs | Supported (engineering observation) |
| — | **Do not claim:** four-pipeline results at 100q, graph benefit, faster extraction overall, or any 200q result | E7 incomplete, E8 not run | — | — |

## E. Completed vs exploratory: wording rules for the paper
- Only E6 is an n = 100, verified comparison. E2 and E1 are pilots (n = 20).
- E4 and E5 are exploratory. E5 must not be presented as a held-out evaluation.
- **E7 (Mistral 100q) must be described, if at all, as an incomplete run stopped at 104/985 extractions.** No E7
  metrics exist.
- The post-hoc ranking counterfactual (C9) appears only in the analysis section, labelled post hoc.

## F. Limitations and missing information for a credible submission
1. **No 100q four-pipeline results.** Graph, Mix and Mix Optimized are only evaluated at n = 20 (E2) and on the
   internal set (E1). RQ1 for the graph pipelines needs E7 completed, or a re-run.
2. **One run per configuration.** No seed or run variance; no multiple-comparison correction.
3. **Held-out evaluation for any C5 change** (Section 4 of the loss analysis), on a new disjoint sample.
4. **No gold extraction annotations.** Extraction quality is agreement-based only. A small human-annotated
   paragraph set would make extraction claims credible.
5. **Generator inconsistency across experiments:** E2 mistral; E6 qwen2.5:3b; E9 plans qwen2.5:1.5b. Answer
   metrics are not comparable across experiments.
6. **Corpus scope:** shared candidate corpus, not FullWiki. No evidence yet on C5's savings at scale.
7. **Baselines:** no BM25-only baseline; the official LightRAG (Pipeline 3) is untested because the vendored
   `LightRAG/` package is absent from the repo; no GraphRAG baseline.
8. **Reproducibility:**
   - Most of this work is **uncommitted** (no commit or push was authorised). A tagged commit is needed.
   - Exact-tie nondeterminism in C5.
   - Thermal throttling affects latencies.
9. **Implementation provenance:** Mix/Mix Optimized were reconstructed and redesigned (disclose; E1 shows the
   change).
10. **Statistics:** paired bootstrap CIs only. A pre-registered primary metric and test would strengthen claims.
11. **Ethics and licence:** HotpotQA (CC BY-SA 4.0) and LightRAG (MIT) attributions; model licences
    (Qwen, Mistral) to be stated.
