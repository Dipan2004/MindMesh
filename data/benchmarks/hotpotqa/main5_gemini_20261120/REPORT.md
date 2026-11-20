# Main five-pipeline benchmark (Experiment A) + local extraction comparison (Experiment B)

Run on 2026-11-20 and 2026-11-21. Each pipeline was run once on 100 HotpotQA dev-distractor questions
(the existing 100q sample, seed 20261119) over the same 985-paragraph corpus. Machine-readable numbers are in
`analysis.json`; the gold-evidence check is in `extraction_failures_gold_check.json`.

## Setup
- **Graph extraction:** OpenRouter `google/gemini-2.5-flash-lite` (served by Google AI Studio), with no local fallback.
  - Original MindMesh extraction prompt (sha256 in `cache_meta.json`), sent with a strict JSON schema
    (schema sha256 `5e15bcc2de5b…`); every response was also validated locally.
  - Temperature 0, seed 42, max_tokens 4096, reasoning off, `require_parameters=true`.
  - Frequency penalty was not sent, because the endpoints do not support it.
- **Retry policy:**
  - Network and HTTP errors, or invalid JSON: up to 2 retries.
  - Output cut off at max_tokens (`finish_reason=length`): rejected and recorded, never retried. The request is
    deterministic; a seed-change test on 5 failures (`retry_policy_test/`) reproduced identical truncated outputs.
- **Full-graph pipelines** (Dense, Graph, Mix, Mix Optimized) all used one shared index:
  `expanded_20261119/dev_distractor_100q_seed20261119/index/openrouter__google_gemini-2.5-flash-lite`
  (cache version `c801e9eaa4fa`).
- **C5** used its own on-demand Gemini cache, `c5/extraction_cache` (same model, prompt, schema and config; separate
  directory). Its retrieval runs on a Dense-only index, `c5/dense_index`.
- **Answer generator:** `qwen2.5:3b` (digest 357c53fb…) for all five pipelines.
  - Prompt `hotpotqa-json-v2-schema`; temperature 0, seed 42, num_ctx 8192, num_predict 256.
  - Retrieval depth 10; the generator sees the top 5.

## Verification
- **Completeness:** all five pipelines have exactly 100 unique rows; none missing or extra, and no row errors.
- **Same generator and settings everywhere:** confirmed from both run configs.
- **Dense consistency check:** Dense run inside the C5 driver gives the same top-10 and the same answer as full-index
  Dense on 100/100 questions.
- **Generation JSON parse failures:** Dense 5, Graph 2, Mix 2, Mix Optimized 4, C5 4. These are scored as produced.
- **Machine incidents:** no thermal sleeps or reboots during the run (no Kernel-Power 41/42/88/107 events after
  22:40). C5-run GPU temperature: mean 53.9 °C, max 86 °C.
- **API incidents:** 4 paragraphs got HTTP 429 (rate limit) on all 3 attempts during the parallel full-graph build.
  They are recorded as provider failures.

## Results (n = 100; paired bootstrap 95% CIs vs Dense, 10,000 resamples)

| Pipeline | R@5 | R@10 | MRR | Gold docs in top 10 | EM | F1 | SP P | SP R | SP F1 | Joint EM | Joint F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dense | 0.925 | 0.990 | 0.907 | 0.98 | 0.51 | 0.569 | 0.381 | 0.260 | 0.286 | 0.03 | 0.185 |
| Graph | 0.800 | 0.825 | 0.872 | 0.68 | 0.50 | 0.541 | 0.317 | 0.221 | 0.240 | 0.01 | 0.142 |
| Mix | 0.885 | 0.980 | 0.905 | 0.96 | 0.53 | 0.569 | 0.402 | 0.280 | 0.308 | 0.02 | 0.185 |
| Mix Optimized | 0.925 | 0.985 | 0.907 | 0.97 | 0.52 | 0.579 | 0.408 | 0.292 | 0.316 | 0.04 | 0.198 |
| query_focused_c5 | 0.945 | 1.000 | 0.786 | 1.00 | 0.54 | 0.604 | 0.281 | 0.221 | 0.228 | 0.04 | 0.160 |

**Differences vs Dense whose CI excludes 0:**
- Graph R@5: −0.125 [−0.185, −0.065]
- Graph R@10: −0.165 [−0.215, −0.110]
- C5 MRR: −0.121 [−0.189, −0.051]; C5 worse on 31 questions, better on 8
- Mix Optimized SP-F1: +0.030 [+0.004, +0.058]

**Everything else includes 0.** That covers all answer EM/F1 differences. For example, C5 EM is +0.03 with CI
[−0.05, +0.11], and C5 R@5 is +0.02 with CI [−0.02, +0.06].

**By question type (R@5 / answer F1):**

| Pipeline | Bridge | Comparison |
|---|---|---|
| Dense | 0.91 / 0.51 | 0.94 / 0.63 |
| Graph | 0.82 / 0.52 | 0.78 / 0.56 |
| Mix | 0.85 / 0.54 | 0.92 / 0.60 |
| Mix Optimized | 0.91 / 0.53 | 0.94 / 0.63 |
| C5 | 0.94 / 0.60 | 0.95 / 0.61 |

## Cost and time (reported separately)

| Item | Value |
|---|---|
| **Offline full-graph build: extraction** | 985 paragraphs. About 10 min wall clock (22:55:46 to 23:05:38, 8 parallel requests). Sum of per-request times 4,577 s; mean 4.65 s per request. |
| Build tokens / cost | 393,895 input / 1,658,728 output tokens, **$0.703** |
| Graph construction / embeddings + collections | 0.09 s / 359 s |
| Graph | 6,234 nodes, 6,661 edges; 706 cross-document nodes; 137 cross-document edges; 801 documents linked through shared entities; 575 weak components (largest 4,341) |
| **C5 on-demand extraction** | 516 paragraphs extracted (52% of the corpus), sequential, **2,195 s total**; mean 22.0 s per question |
| C5 tokens / cost | 201,714 input / 777,738 output tokens, **$0.331** |
| C5 cache | 567 lookups: 50 hits on valid records, 1 hit on a failed record, 516 misses (hit rate 9.0%) |
| **Online retrieval latency** (mean / median) | Dense 0.32 / 0.28 s · Graph 1.92 / 1.73 s · Mix 0.80 / 0.80 s · Mix Optimized 0.86 / 0.83 s · C5 excluding extraction 0.35 / 0.35 s |
| C5 retrieval including extraction | mean 22.3 s. Not comparable with the online latency of the other pipelines. |
| **Generation latency** (mean) | Dense 0.78 s · Graph 0.83 s · Mix 0.68 s · Mix Optimized 0.69 s · C5 0.70 s |

**Total API spend for this work: about $1.06.**
- Full build $0.703
- C5 $0.331
- Retry test $0.0085
- Smoke test $0.0063
- Provider pilots $0.0087

This is well under the $5 limit.

## Extraction failures (kept, counted, never substituted)
- **Full-graph build: 55 of 985 paragraphs (5.6%).**
  - 51 were truncated by repetition loops: 4,000+ output tokens on paragraphs of 700–2,000 characters.
  - 4 got HTTP 429 after retries.
- **Gold evidence in the full graph:** 7 of the 199 gold paragraphs have no extraction record. This affects 7
  questions (3 bridge, 4 comparison); each is missing 1 of its 2 gold paragraphs:
  `5ade51b4…` Children's Mercy Park · `5a901b98…` Food and Drug Administration · `5a76d15e…` The X-Files (season 6) ·
  `5a8b3979…` John Corabi · `5ac179bf…` Castor et Pollux · `5a7458ff…` Honey, I Blew Up the Kid ·
  `5a7f2200…` Kele Okereke (HTTP 429).
- **Effect on retrieval:** these paragraphs are still embedded and can be retrieved by the Dense component. The graph
  has no nodes from them.
- **C5 cache: 22 of 516 truncated (4.3%).** Of these, 6 gold paragraphs failed (listed in `analysis.json`).

## Interpretation (n = 100, one run; no claims beyond the CIs)
- **Graph (entity-graph retrieval alone)** retrieves significantly less gold evidence than Dense: R@5 −0.125, R@10 −0.165.
- **Mix and Mix Optimized match Dense on retrieval.** Mix Optimized's supporting-fact F1 is slightly higher
  (+0.03, CI excludes 0); its answer metrics do not differ.
- **C5 again reaches the highest recall** (R@10 = 1.00, all gold documents in the top 10). Its MRR is again
  significantly lower than Dense: −0.12 here, compared with −0.16 in the earlier qwen2.5:3b-extraction C5 run. This
  repeats the ranking-order finding.
- **C5's answer F1 is the highest (0.604), but the difference from Dense is not significant.**
- **C5 cost:** on-demand extraction touched 52% of the corpus, at 47% of the full-build API cost. It ran
  sequentially, at 22 s per question.
- **Cache reuse is low** (9%) in this distractor setting.

---

# Experiment B: local extraction-model comparison (small pilot, n = 2 paragraphs)

Output: `../local_extraction_comparison_20261120/experiment_b_report.json`. Same two paragraphs as the provider pilot
('New Mosque (Istanbul)' 523 characters; 'Rango (2011 film)' 800 characters). Original prompt and production local
settings: temperature 0, seed 42, num_predict 4096, num_ctx 8192, frequency penalty 0.25 with the [0.5, 1.0] retry
ladder. One cache per model; not used in Experiment A. Laptop on AC power.

| Model | Availability | Paragraph | Status | Wall s | Decode tok/s | In / out tokens | Attempts | Entities / relationships | Provenance (has sentence / verbatim) | Agreement vs Mistral (ent F1 / rel F1) |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:1.5b | installed | New Mosque | ok after 1 truncation | 29.7 | 160 | 822 / 4,394 | 2 | 1 / 0 | 1.00 / 1.00 | 0.00 / 0.00 |
| qwen2.5:1.5b | | Rango | **failed: truncated ×3** | 80.9 | 154 | 1,500 / 12,288 | 3 | 0 / 0 | n/a | n/a |
| qwen2.5:3b | installed | New Mosque | ok | 28.0 | 96 | 411 / 2,421 | 1 | 9 / 5 | 1.00 / 1.00 | 0.82 / 0.67 |
| qwen2.5:3b | | Rango | ok | 35.0 | 95 | 500 / 3,301 | 1 | 16 / 15 | 1.00 / 0.07 | 0.91 / 0.00 |
| qwen2.5:4b | **unavailable**: not installed, and not a published Qwen 2.5 size | | | | | | | | | |
| qwen2.5:7b | **unavailable**: not installed (not downloaded) | | | | | | | | | |

**Valid JSON rate:** 1.5b 1/2, 3b 2/2.

**Agreement:**
- 1.5b vs 3b on New Mosque: entities 0.00, relationships 0.00. The 1.5b model returned a single entity.
- vs Gemini, New Mosque: 3b entities 0.67 / relationships 0.00, a naming difference ('Yeni Cami').

**Interpretation:**
- qwen2.5:1.5b is not usable for MindMesh extraction: it truncates or loops and returns near-empty output.
- qwen2.5:3b is valid on both paragraphs, but on Rango its evidence sentences are not verbatim (0.07).
- Agreement is not correctness, and n = 2 supports no general claim.
