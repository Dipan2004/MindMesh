# Local 7B extraction timing experiment (n = 2 paragraphs), 2026-11-21

A small isolated pilot, not a benchmark. Agreement is not correctness. Projections are estimates.
Numbers are in `report.json`; per-second hardware samples are in `hardware_samples.jsonl`.

## Why a 7B model and not 9B–12B
- No 9B–12B model is installed. Downloading gemma2:9b failed because drive C: had less than 2 GB free.
- At the user's direction the model store stays on C:, and a 7B model was used instead.
- `mistral:latest` (7.2B, Q4_K_M) was already installed, so nothing was downloaded.

## Setup
- **Paragraphs:** the same two as Experiment B, 'New Mosque (Istanbul)' (523 characters) and 'Rango (2011 film)'
  (800 characters). They were chosen as the median-length and 90th-percentile-length paragraphs of the 30-paragraph
  study, so they are longer than the corpus mean of 530 characters.
- **Extraction:** original MindMesh prompt, run through the production local extraction path (ExtractionCache) with
  temperature 0, seed 42, num_predict 4096, num_ctx 8192, and frequency penalty 0.25 with the [0.5, 1.0] retry ladder.
- **Caches:** one fresh cache per model.
- **Gemini reference:** the saved re-test results for the same paragraphs (OpenRouter, strict JSON schema,
  temperature 0, seed 42, max_tokens 4096).
- **Hardware:**
  - RTX 4060 Laptop GPU with 8 GB, i7-13700HX, 15.6 GB RAM.
  - On AC power.
  - Both models ran 100% on the GPU (`ollama ps`).

## Timing

| | qwen2.5:3b | mistral:latest (7B) | Gemini 2.5 Flash-Lite (API) |
|---|---|---|---|
| Model load (cold) | 3.4 s | 4.2 s | n/a |
| New Mosque | 24.9 s · 411 in / 2,419 out tokens · 98 tok/s | 32.2 s · 481 / 1,619 · 51 tok/s | 5.8 s · 398 / 2,376 · $0.00099 |
| Rango | 18.4 s · 500 / 1,782 · 98 tok/s | 56.4 s · 562 / 2,806 · 50 tok/s | **failed**: truncated at 4,096 tokens · 9.5 s · $0.00167 |
| Valid JSON | 2/2 | 2/2 | 1/2 |
| GPU memory (model loaded) | 2.4 GB | 5.4 GB | n/a |
| GPU temperature during extraction (mean / max) | 76 / 82 °C | 86 / 89 °C | n/a |
| GPU utilisation / SM clock (mean) | 87% / 2,478 MHz | 96% / 2,353 MHz | n/a |
| CPU (mean) / RAM used | 7% / 11.4 GB | 8% / 11.2 GB | n/a |
| GPU board power (mean, indicative only) | 84 W | 90 W | n/a |

## Extraction quality

| | qwen2.5:3b | mistral:latest | Gemini |
|---|---|---|---|
| Entities / relationships: New Mosque | 9 / 5 | 8 / 4 | 9 / 9 |
| Entities / relationships: Rango | 11 / **0** | 17 / 16 | failed |
| Evidence sentence present | 100% | 100% | 100% |
| Evidence sentence copied word for word: New Mosque / Rango | 1.00 / **0.00** | 0.33 / 1.00 | 1.00 / n/a |
| Agreement with mistral:latest, entity F1 / relationship F1: New Mosque | 0.82 / 0.67 | — | 0.82 / 0.00* |
| Agreement with mistral:latest, entity F1 / relationship F1: Rango | 0.71 / 0.00 | — | n/a |

\* Gemini names the mosque 'Yeni Cami', so exact-name relationship matching scores 0.

- **qwen2.5:3b is not stable across sessions.** On Rango it returned 11 entities and **0 relationships** this time,
  versus 16 / 15 in Experiment B, with identical settings. This matches the "bistable" behaviour documented in the
  project config.
- **mistral:latest reproduces its earlier output:** it matches its own earlier extraction (E3) exactly on both
  paragraphs (entity and relationship F1 = 1.0).

## Projections (estimates, not measurements)

| | Full corpus (985 paragraphs, 522,462 chars) | C5 workload observed in the 100q run (516 paragraphs, 249,163 chars) |
|---|---|---|
| qwen2.5:3b, from these 2 paragraphs (0.0327 s/char) | ~4.8 h | ~2.3 h |
| qwen2.5:3b, from the earlier measured C5 run (9.68 s/paragraph) | ~2.6 h | ~1.4 h (measured: 507 paragraphs in 1.36 h) |
| mistral:latest, from these 2 paragraphs (0.0670 s/char) | ~9.7 h | ~4.6 h |
| mistral:latest, from the 30-paragraph study (24.4 s/paragraph mean) | ~6.7 h | ~3.5 h |
| **Gemini (measured)** | **~10 min with 8 parallel requests, $0.70** | **0.61 h sequential, $0.33** |

- **The two-paragraph projections overstate time.** Both paragraphs are longer than average, and two points are a
  weak basis. The projections from larger earlier samples are more reliable.
- **None of the figures include:**
  - thermal throttling: this GPU sat at 86–89 °C under the 7B model, and earlier long runs caused thermal shutdowns
  - restarts
  - failure retries beyond those observed
- **Local runs are sequential.** One GPU runs one model at a time.

## Electricity
**Not estimated.**
- Only GPU board power from nvidia-smi is available (about 84–90 W during extraction). There is no CPU or
  whole-system meter.
- nvidia-smi gave an implausible 501 W reading earlier in this session.
- No energy or carbon comparison with the API is made.

## Feasibility
- **A 7B model fits.** mistral:latest fits fully on the 8 GB GPU (5.4 GB) and produced valid, richer, reproducible
  extractions on both paragraphs.
- **It is slow.** About 50 tokens/s, roughly half qwen2.5:3b's decode speed. A full-corpus build is projected at
  about 7–10 h.
- **It runs hot.** Sustained runs are at risk of thermal shutdowns on this laptop.
- **9B–12B models are blocked.** They need free space on C: (8 GB or more) and are likely to approach the 8 GB GPU
  limit at an 8k context.
