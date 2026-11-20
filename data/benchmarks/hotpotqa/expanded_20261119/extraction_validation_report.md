# Extraction optimisation and extraction-model validation (2026-11-19)

**Scope:** 30 fixed paragraphs (the first 30 pilot paragraphs by `chunk_id`), each run through:
- the current MindMesh extraction prompt ("baseline");
- **C**: the same prompt with JSON-schema constrained output;
- **B+C**: numbered sentences with a `sentence_id` field instead of retyped sentences, plus the schema;

all with `mistral:latest` (7.2B Q4_K_M, digest `6577803aa9a0…`), interleaved per paragraph.

Separately, **qwen2.5:3b** (3.1B Q4_K_M, digest `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`,
pulled with approval) was run on the same 30 paragraphs with B+C. Generation options are identical to production
(temperature 0, seed 42, `num_predict` 4096, `num_ctx` 8192, frequency penalty 0.25 with the retry ladder).

**Nothing was adopted.** The main 100q experiment stays paused at its 33-record checkpoint, and its auto-resume
launcher is disabled (`MindMesh_HotpotQA_resume.cmd.paused`).

## Important limitation

**There are no gold entity or relationship annotations for these paragraphs.** "Quality" below is
*agreement with the Mistral baseline extraction*, plus reference-free grounding checks. Agreement is
not correctness: a variant can disagree with the baseline and still be right, or agree and both be
wrong. Matching counts are not treated as matching quality.

## 1. Speed and tokens (Mistral arms, paired, n = 30)

| | Baseline | C | B+C |
|---|---|---|---|
| Time per paragraph: mean / median | 24.4 s / 23.4 s | 29.6 s / 24.2 s | **13.5 s / 11.5 s** |
| Total time | 732 s | 889 s (**−21% slower**) | **405 s (44.6% faster)** |
| Output tokens: mean / total | 1,191 / 35,721 | 1,173 / 35,202 (1.5% fewer) | **661 / 19,841 (44.5% fewer)** |
| Decode rate | 50.3 tok/s | 38.5 tok/s | 51.1 tok/s |
| Parse failures / retries | 0 / 0 | 0 / 0 | 0 / 0 |
| Paragraphs faster than baseline | — | 19 (11 slower) | **27 (3 slower)** |
| Mean per-paragraph time reduction, 95% bootstrap CI | — | −22% [−57%, +3%] | **37% [27%, 45%]** |
| Time reduction without the 3 slowest baseline paragraphs | — | −21% | 41% |
| Share of B+C's saving from the 3 longest paragraphs | — | — | 30% |

**Consistency:**
- **B+C's speed-up is consistent, not outlier-driven.** It was faster on 27 of 30 paragraphs, and
  still 41% faster with the three longest paragraphs removed.
- **C gave no speed benefit.** Its median per-paragraph change is about 0, it produced the same tokens,
  and 3 paragraphs decoded abnormally slowly: 11–23 tok/s versus ~50 ('Dariusz Wolski' 124 s,
  'Hortência Marcari' 84 s, 'Tenacious D Live' 77 s). Whether that is schema-constrained decoding
  or a thermal event at that moment cannot be separated. Telemetry missed the first ~5 minutes, and
  B+C, also schema-constrained, did not show it.
- **C's intended benefit cannot be tested on this sample.** C was meant to cut runaway outputs and
  retries, and the baseline had none here.

## 2. Extraction agreement with the Mistral baseline

| | C | B+C | qwen2.5:3b B+C |
|---|---|---|---|
| Entities extracted (baseline 226) | 228 | **192 (−15%)** | **154 (−32%)** |
| Relationships extracted (baseline 162) | 154 | **135 (−17%)** | **97 (−40%)** |
| Entity P / R / F1 (exact normalised name) | 0.956 / 0.960 / **0.958** | 0.854 / 0.726 / **0.785** | 0.784 / 0.531 / **0.633** |
| Relationship P / R / F1 (source, target pair) | 0.844 / 0.828 / **0.836** | 0.625 / 0.510 / **0.561** | 0.352 / 0.204 / **0.258** |
| Relationship F1 including type | 0.740 | 0.428 | 0.105 |
| Entity recall, counting renames as matches | 0.969 | 0.788 | 0.602 |
| Relationship recall, counting renames/reversals as matches | 0.873 | 0.650 | 0.293 |
| Paragraphs with truly missing relationships | 6 | **16** | **22** |
| Paragraphs missing over half of baseline relationships (after renames) | 2 | **8** | **19** |
| Duplicate entities / relationships within a paragraph | 1 / 0 | 0 / 7 | — |
| Same entity, different type / same pair, different relation type | 5 / 15 | 6 / 19 | — |

**Renamed vs missing:**
- Some of B+C's disagreements are renames, which is harmless or arguably better: 'Rudebox (song)' →
  'Rudebox', 'Pirates of the Caribbean film series' → 'Pirates of the Caribbean'.
- Many are genuine omissions:
  - Bagdad Mine lost its mineral relationships (azurite, chalcopyrite, chrysocolla);
  - Zakk Wylde lost 'Jeffrey Phillip Wielandt → Zakk Wylde' and '1994 → Pride & Glory';
  - Hasan Fehmi lost 'Istanbul'.
- Even after crediting renames, B+C misses about **35% of the baseline's relationships**, and in 8 of
  30 paragraphs over half. For reference, C, which uses the identical prompt, differs on about 13%:
  that's roughly the run-to-run noise floor for this measurement.

## 3. Reference-free grounding (all arms)

| | Baseline | C | B+C | qwen2.5:3b B+C |
|---|---|---|---|---|
| Entity names found in the paragraph | 0.982 | 0.983 | 0.984 | 0.981 |
| Relationship endpoints found in the paragraph | 0.975 | 0.974 | 0.978 | 0.969 |
| Evidence sentence verbatim in paragraph | 0.590 | 0.571 | **1.000** | **1.000** |
| Paragraph's title entity extracted | 29/30 | 29/30 | 29/30 | 27/30 |

- **Nobody hallucinates names.** All variants stay grounded in the text; the differences are in coverage.
- **B+C's sentence numbers fix the evidence field.** The baseline's "verbatim" evidence sentence is
  actually verbatim only 59% of the time. With sentence numbers it is exact by construction.

## 4. qwen2.5:3b (B+C) vs Mistral

| | Mistral baseline | Mistral B+C | qwen2.5:3b B+C |
|---|---|---|---|
| Time per paragraph: mean / median | 24.4 s / 23.4 s | 13.5 s / 11.5 s | **5.3 s / 4.4 s** |
| Output tokens, mean | 1,191 | 661 | 475 |
| Decode rate | 50 tok/s | 51 tok/s | **98 tok/s** |
| Parse failures | 0 | 0 | 0 |
| Relationship recall vs Mistral baseline (with renames) | — | 0.650 | **0.293** |

- **Qwen is about 4.6× faster** than the Mistral baseline (78% less time, 60% fewer tokens).
- **But its graph is much sparser.** It found about 30% of Mistral's relationships, and lost over
  half of them in 19 of 30 paragraphs.
- **Its timing comes from a later thermal window** (not interleaved). Its decode rate and output
  tokens are the more robust comparison.

### 4a. qwen2.5:3b detail (addendum)

- **Output tokens per paragraph:** mean 474.5, median 387, range 134–1,383. Mistral baseline: mean
  1,191, median 1,126. Mistral B+C: mean 661, median 567.
- **VRAM (`ollama ps`, at 8k context, 100% on GPU):**
  - qwen2.5:3b: **2.41 GB**.
  - mistral:latest: **5.50 GB**.
  - The GPU has 8 GB in total.
- **Relationships missing vs the Mistral baseline (after crediting renames):** 111 of 157.
  - 22 paragraphs have at least one missing relationship.
  - In 3 paragraphs Qwen extracted **no relationships at all** where Mistral had some: 'Ross Geller',
    'New Mosque (Istanbul)', 'Rango (2011 film)'.
  - Examples:
    - The Duke Steps Out → Joan Crawford
    - Dariusz Wolski → Aerosmith (0 of 16 baseline relationships found)
    - Hasan Fehmi → Committee of Union and Progress
    - Bagdad Mine → Arizona / azurite
    - Zakk Wylde: 1994 → Pride & Glory
- **Qwen relationships with no Mistral counterpart: 61.** They can't be verified without gold labels.
  - Some are renames: Rudebox → Robbie Williams, where the baseline used 'Rudebox (song)'.
  - Some look questionable on reading: 'The Duke Steps Out USED_FOR Vitaphone';
    'Vitaphone LOCATED_IN UCLA Film and TV Archives' (the film's materials are archived there, not
    the company).
  - At least one is an exact duplicate within a paragraph: 'Rudebox RELEASED_IN 2006' twice.
- These are listed as **questionable, not verified incorrect.**

## 5. GPU temperature, throttling and interruptions

- **Mistral arms:**
  - 315 samples at 5 s intervals: mean **88.2 °C**, max **92 °C**, 94% of samples ≥ 87 °C.
  - Median SM clock 1,935 MHz (min 1,740; max is 3,105).
  - The software thermal-slowdown flag (`0x20`) was set in **94%** of samples.
  - **One thermal hibernation/reboot at 20:40:29** stopped the run at 87/90 arm-runs; the last
    paragraph was rerun at 20:43.
- **qwen2.5:3b:** 32 samples, mean 87.6 °C, max 90 °C, median clock 2,145 MHz. No interruptions.
- Arms were interleaved per paragraph, so the Mistral comparisons share thermal conditions.

## 6. Against the adoption criteria

| Criterion | C | B+C | qwen2.5:3b B+C |
|---|---|---|---|
| Measurable reduction in time and tokens | **No** (−21% time, −1.5% tokens) | **Yes** (−45% / −45%) | **Yes** (−78% / −60%) |
| Reproducible across the sample | No | **Yes** (27/30 faster) | Yes |
| No material degradation in entity/relationship quality | Agreement near the noise floor | **Not shown**: relationship recall 0.65 with renames, 16 paragraphs with missing relationships | **No**: relationship recall 0.29 |
| No systematic loss of important relationships | Mostly | **Fails**: 8/30 paragraphs lose over half | **Fails**: 19/30 |

## 7. Recommendation

**Keep the baseline** (current Mistral extraction configuration) for the main experiment.

- **C:** no benefit on this sample. Not recommended.
- **B+C:** a large, consistent speed-up and exact evidence sentences, but it extracts noticeably
  fewer relationships. There is **insufficient evidence** that its graph quality is preserved, and
  the agreement data point to a real loss. Not recommended for the main experiment without an
  independent check. The most relevant check is downstream: rebuild the 20q pilot index with B+C
  and compare retrieval and answer metrics against the existing pilot run.
- **qwen2.5:3b:** much faster, but its graph is far sparser than Mistral's. Not recommended as the
  extraction model for this experiment. It is a legitimate separate condition only if reported as a
  different (weaker-extraction) configuration.

## Files

- `extraction_validation/`:
  - `results.jsonl`: 90 Mistral arm-runs, raw entities and relationships;
  - `summary.json`, `analysis.json`, `diff_classification.json`, `config.json`;
  - `gpu_telemetry.csv`, `run.log`, `run_chain.py`, `gpu_logger.py`.
- `extraction_validation_qwen2.5_3b/`: `results.jsonl`, `config.json`, `gpu_telemetry.csv`,
  `run.log`, `ollama_ps_at_end.json`.
- **Scripts:**
  - `eval/hotpotqa/validate_extraction_variants.py`
  - `eval/hotpotqa/analyze_extraction_validation.py`
  - `eval/hotpotqa/classify_extraction_diffs.py`
- **Main experiment checkpoint (untouched):** 33 records in
  `dev_distractor_100q_seed20261119/index/mistral_latest/extractions.jsonl`.
