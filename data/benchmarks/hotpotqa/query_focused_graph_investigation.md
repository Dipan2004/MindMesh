# Query-focused, on-demand graph construction: investigation (2026-11-19)

Read-only investigation. The running Mistral baseline was not modified or interrupted, HOLD_200Q stays
in place, and no retrieval code was changed. No LLM calls or GPU work were used for the measurements
below.

Each statement is labelled **[measured]** (from code, logs or existing data) or **[estimate]** (projected).

## 1. How the current system works

**Why all 985 paragraphs are extracted [measured, code].**
- `eval/hotpotqa/build_index.py` extracts every paragraph in `corpus.jsonl` before any question is
  asked. This is the offline GraphRAG/LightRAG design MindMesh follows.
- The 100q corpus is the union of the 100 questions' supplied paragraphs (10 each, 985 after
  deduplication), so every candidate paragraph is extracted.

**Where graph construction happens [measured, code]. Both:**
- **Offline:**
  - LLM extraction writes `extractions.jsonl`;
  - entities, relationships and chunks are embedded into Zvec collections;
  - `graph.graphml` is saved.
- **At query time:**
  - Graph (`pipeline_b._get_graph`) and Mix (`lightrag_experimental._load_graph`) rebuild the
    NetworkX graph from `extractions.jsonl` once per process and compute **global PageRank**;
  - queries then hit the prebuilt entity, relationship and chunk collections.
- **No LLM extraction happens per query.**

**Can the extractor run selectively? Yes [measured, code].**
- `extract_entities_relationships(chunk, client)` works on one paragraph, and its prompt contains
  only that paragraph. So extraction is **context-free**: a paragraph's record is the same whatever
  else is selected.
- It is already cached in two ways: an LLM response cache (`cached_call`, keyed by prompt + model +
  settings), and per-paragraph records in `extractions.jsonl` keyed by a content-hash `chunk_id`.
  `run_expanded.py` already reuses records across experiments this way.

**What blocks query-time graphs in the existing pipelines [measured, code].**
- Graph uses **global PageRank** and prebuilt entity collections. Mix uses prebuilt relationship and
  entity collections.
- A per-question graph would need per-question embeddings, or incremental upserts into a cached
  collection, and a local ranking.
- That makes it a **new pipeline variant**. It cannot be done by re-pointing the existing four
  without changing their algorithms.

## 2. Offline simulation (20-question pilot, Mistral full graph) [measured]

**Method.** Because extraction is context-free, a query-focused graph over paragraph set S is exactly
the full graph's records restricted to S, so it can be simulated without new LLM calls.

Data used: the pilot's existing Mistral extraction (200 paragraphs) and the stored Dense top-10 lists
from the pilot run.

Script: `eval/hotpotqa/simulate_query_focused_graph.py`.
Output: `data/benchmarks/hotpotqa/query_focused_simulation_pilot20_mistral.json`.

**Selections:**
- **A**: all paragraphs (current design).
- **B_k**: Dense top-k paragraphs only.
- **C_k**: B_k plus one cheap expansion hop, i.e. paragraphs whose title appears in the question, or
  matches an entity extracted from the B_k paragraphs. This is the "related document" step and needs
  no extra LLM call to decide.

| Selection | Paragraphs / question | Unique paragraphs extracted (20 q, cached) | Share of corpus | Entities / relationships per question | Gold-paragraph coverage | Both gold paragraphs selected | Gold pair connected in the graph (≤ 3 hops) |
|---|---|---|---|---|---|---|---|
| **A. Full corpus** | 200 | 200 | 100% | 1,598 / 1,117 | **1.000** | 20/20 | **10/20** (bridge 9/10, comparison 1/10) |
| B_5 (Dense top 5) | 5 | 93 | 47% | 38 / 27 | 0.875 | 15/20 | 6/20 (bridge 6) |
| B_10 (Dense top 10) | 10 | 162 | 81% | 80 / 56 | 0.975 | 19/20 | 9/20 (bridge 8) |
| **C_5 (top 5 + title/entity expansion)** | **6.05** | **110** | **55%** | 47 / 32 | **0.975** | **19/20** | **8/20** (bridge 7) |
| C_10 | 11.55 | 165 | 83% | 93 / 65 | 0.975 | 19/20 | 9/20 (bridge 8) |

**What this shows:**
- **The expansion hop matters.**
  - B_5 misses a gold paragraph on 5 of 20 questions.
  - One title/entity-matching hop (C_5) recovers 4 of them: coverage rises from 0.875 to **0.975**,
    the same as Dense top 10.
  - It does this for only ~1 extra paragraph per question.
- **Cross-document paths are mostly kept.**
  - The full graph connects the two gold paragraphs for 9 of 10 bridge questions.
  - C_5 keeps 7 of those 9; B_5 keeps 6.
  - Comparison questions are rarely connected in any setting (their two entities are usually not
    linked by text), so graph paths can't be expected to help them.
- **The risk is real but bounded.** If the first retrieval misses a required paragraph and no
  extracted entity's name matches its title, the query-focused graph cannot contain the path. On
  this sample, 1 of 20 questions still lacks a gold paragraph under C_5, the same as Dense top 10.
- **Savings are limited in this benchmark** (about 45% for C_5). In the HotpotQA *distractor*
  setting each question brings only its own 10 paragraphs, so a query's neighbourhood is already
  most of its share of the corpus. In a FullWiki-style corpus the saving would be far larger. That
  is not measured here.

**n = 20.** These numbers are indicative, not conclusive.

## 3. Cost and latency

| | A. Full corpus (offline) | B. Query-focused (B_5) | C. Selective expansion (C_5) |
|---|---|---|---|
| Paragraphs extracted for 100q | 985 [measured] | ~460 [estimate, 47%] | **~540 [estimate, 55%]** |
| Mistral extraction time (~30 s/paragraph [measured]) | **~8.2 h, all up front** [estimate from measured rate] | ~3.8 h, spread over queries | **~4.5 h, spread over queries** |
| qwen2.5:3b extraction time (B+C prompt measured ~5.3 s; original prompt not yet measured) | ~1.5–2.5 h [estimate] | ~0.7–1.2 h | ~0.8–1.4 h |
| First (cold-cache) query latency, Mistral | ~1–2.5 s (graph already built) [measured, pilot] | ~2.5 min (5 × 30 s) [estimate] | **~3 min** (6 × 30 s) [estimate] |
| First (cold-cache) query latency, qwen2.5:3b | ~1–2.5 s | ~25–40 s [estimate] | ~30–50 s [estimate] |
| Repeated or overlapping queries (warm cache) | same | ≈ A plus a ~1–2 s graph assembly [estimate] | ≈ A plus ~1–2 s [estimate] |

**The trade-off:** query-focused construction moves extraction cost from indexing time to the first
query. With these models that means minutes of cold-start latency per question on this laptop, unless
the cache is warm.

## 4. Expected differences across the requested metrics [estimates unless marked]

| Metric | A. Full corpus | B. Query-focused | C. Selective expansion |
|---|---|---|---|
| Extraction volume (entities / relationships) | 100% | ~3–5% per question; ~47% of the corpus overall [measured, pilot] | ~3–6% per question; ~55% overall [measured, pilot] |
| Graph coverage of gold evidence | 1.000 [measured] | 0.875 [measured] | 0.975 [measured] |
| Cross-document gold paths | 10/20 [measured] | 6/20 [measured] | 8/20 [measured] |
| Gold-document Recall@5 / @10 | Graph 0.775 / 0.825, Mix 0.775 / 0.925, Dense 0.875 / 0.975 [measured, pilot] | Capped at 0.875 by construction; likely ≤ Dense | Capped at 0.975; likely ≈ Dense to Mix |
| Supporting-fact F1 | 0.12–0.30 [measured, pilot] | Likely ≈ Dense (the generator sees similar top-5) | Likely ≈ Dense |
| Answer F1 / EM | 0.57 / 0.45 for all four [measured, pilot] | Likely unchanged: in the pilot, answers were driven by the generator, not graph retrieval | Likely unchanged |
| Extraction cost | Highest, up front | ~53% lower | ~45% lower |
| Query latency | Lowest | Highest when cold | Highest when cold |

These retrieval and answer estimates assume the query-focused graph is ranked with the same evidence
rules as Graph or Mix. Recall cannot exceed the selection's gold coverage.

## 5. Recommendation: smallest scientifically defensible prototype

A **simulation-first evaluation of option C**, with no new LLM extraction, using the 100q
experiment's own full extraction once the Mistral run finishes (and the Qwen run's extraction if
approved):

1. **Re-run this simulation on the 100 questions** (CPU only, minutes). This measures coverage,
   cross-document paths and extraction share with n = 100 instead of 20.
2. **Add one new, separately named pipeline**, `query_focused_c5`, leaving the existing four
   untouched. Per question it would:
   - take Dense top-5 (the existing Pipeline A query);
   - add the title/entity expansion hop;
   - build the induced graph from the cached per-paragraph records;
   - BFS ≤ 2 hops from question-matched entities;
   - rank chunks by path support, falling back to Dense order.

   It would run through the existing benchmark runner on **the same 100 questions, gold evidence,
   generator and metrics** as the other four.
3. **Report extraction cost two ways:**
   - **measured**, as the number of distinct paragraphs it touches, times each paragraph's
     *recorded* extraction latency (cold cache);
   - **observed**, as query latency with the cache warm.

**Why this is the smallest defensible step:**
- **No new model inference for extraction.** Because extraction is context-free, induced-graph
  results are exactly what on-demand extraction would produce.
- **Identical controls** to the existing pipelines.
- **It answers the key question directly:** whether a ~55% extraction budget keeps the evidence
  quality of the full graph.

A live on-demand prototype (extracting at query time) would only add latency measurements on top.
It can follow if the simulated quality holds.

**Waiting for approval before implementing anything.**
