# Why C5 loses MRR against Dense: analysis of the 33 losses

Source: the completed run `c5_ondemand_qwen3b_20261119/runs/dense_vs_c5_6ca3b11e`, 100 questions. Read-only; no code
changed and nothing re-run. Machine-readable detail, one entry per question: `c5_mrr_loss_analysis.json`.

## Method
- **Replaying the scores.** The run stored C5's selections, Dense lists and anchors, but not per-paragraph support
  scores. The scores were recomputed on the CPU from the run's own saved qwen2.5:3b extraction records, using the
  same C5 functions, with no LLM calls.
- **Validation of the replay:** it reproduces **99/100** stored C5 rankings exactly.
  - The single difference (question `5ac44b47…`) is two paragraphs with an exact score tie swapping ranks 4 and 5.
  - No metric changes: Recall@5 = 1.0 and MRR = 1.0 either way.
  - Cause: C5 sums support while iterating Python sets, whose order varies per process, so tied scores can break
    differently between runs. This is a **reproducibility caveat** of the C5 implementation.
- **Primary-cause rules**, applied in this order to the gold paragraph Dense ranked highest ("demoted gold"):
  1. it has no valid extraction record → *failed extraction*;
  2. it has no path support and ≤ 1 extracted entity → *sparse extraction*;
  3. it has no path support but has entities → *graph path* (its entities are not reachable from the question's anchors);
  4. an expansion-hop paragraph outscored it → *expansion*;
  5. otherwise it was supported, but a non-gold paragraph scored higher → *path-support ranking*.

## Findings

| | Count |
|---|---|
| Questions where C5 MRR < Dense / > Dense / equal | **33** / 6 / 61 |
| Losses by type | bridge 13, **comparison 20** |
| Losses where Dense had a gold paragraph at **rank 1** | **30 / 33** |
| Losses where C5 still has gold in its top 10 | **33 / 33** |
| C5 rank of the first gold paragraph in losses | rank 2: 15, rank 3: 11, rank 4: 5, rank 5: 2 |
| Non-gold paragraphs promoted above the demoted gold | 60 |
| — of which were already in Dense's own top 10 | **57 / 60** |
| — of which came from the expansion hop | 8 / 60 |
| Demoted gold had zero path support | 9 / 33 |
| Median entities in the demoted gold paragraph's record | **1** (qwen2.5:3b extraction is sparse) |
| Losses whose anchors include a generic or very short phrase | 4 / 33 |

**Primary causes:**

| Cause | Losses | Share |
|---|---|---|
| **Path-support ranking**: gold supported, non-gold scored higher | **19** | 58% |
| Expansion paragraph outscored gold | 5 | 15% |
| Graph path: gold's entities not reachable from the question's anchors | 4 | 12% |
| Sparse extraction: gold record has ≤ 1 entity and no support | 4 | 12% |
| Failed extraction: gold has no valid record | 1 | 3% |

**Examples** (full list in the JSON):
- `5ab53ef1…` (bridge): Dense ranked 'Río Hondo Department' first. C5 promoted 'Rio Hondo (Southern Nevada)' and
  'Rio Hondo (Northern…)' (support 11.0 / 4.6) because the anchor 'Rio Hondo' matches many paragraphs. The gold's
  entities were unreachable from the anchor.
- `5a7602d8…` (bridge): the expansion paragraph 'Absolutely Fabulous' (support 3.75) and 'French and Saunders'
  outscored the gold 'Dawn French' (0.5).
- `5a8c8828…` (bridge): the gold 'Edward Fitzalan-Howard, 18th Duke of Norfolk' had a 1-entity record and no
  support. The expansion paragraph 'Duke of Norfolk' (4.5) was promoted.

## Interpretation
- **The loss is almost entirely a top-1 reordering problem.** In 30 of 33 losses Dense already had a gold paragraph at
  rank 1; C5 never drops it out of the top 10, mostly to rank 2 or 3. 57 of the 60 promoted paragraphs were already
  in Dense's top 10. **The graph is reordering Dense's own candidates, not adding new noise.**
- **Path support measures entity overlap, not answerability.** A paragraph mentioning a question entity (or a
  neighbour) more often gets more support, even when Dense's top hit is the relevant one. This is especially true
  for **comparison** questions (20 of 33 losses), where the gold paragraphs are about two separate entities and
  paths between them rarely help.
- **Sparse extraction contributes but is secondary.** The median demoted gold record has a single entity, so it
  often cannot gather support.

## Is a ranking change justified?

**Yes, as a hypothesis. Not as a result on these questions.**

**Post-hoc diagnostic only** (the same 100 questions; **not reportable as a result**):

| Ranking rule | MRR |
|---|---|
| Dense | 0.907 |
| C5 as run | 0.750 |
| C5 with Dense's rank-1 kept fixed | 0.908 |
| Equal RRF of Dense and support | 0.831 |

**Smallest proposed change (not implemented): "Dense-anchored C5 ranking".**
- Keep Dense's rank-1 paragraph at rank 1.
- Apply path-support ordering only from rank 2 onward.

This mirrors the already-registered Mix Optimized design (Dense keeps the top slots), so it introduces no new
mechanism.

**Required safeguards:**
1. **Pre-register the rule** before testing it.
2. **Evaluate it on a new held-out question sample:** a new seed, disjoint from these 100, the 20-question pilot
   and the expanded draw. **Never on these 100**, which motivated it.
3. **Report both** the registered C5 and the variant, with paired bootstrap CIs.
4. Optionally fix the **tie-ordering nondeterminism** (sort score contributions deterministically) at the same
   time, and document it.
