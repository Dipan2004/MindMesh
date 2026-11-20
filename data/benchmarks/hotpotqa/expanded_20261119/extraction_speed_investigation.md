# Extraction speed investigation (2026-11-19)

Read-only investigation. Nothing in the running experiment was changed. Measurements come from
the pilot's 200 saved extractions and LLM responses, and from a *copy* of the running 100q cache
(first 18 new paragraphs). No GPU was used.

## 1. Where the time goes

| Per paragraph | Pilot (n = 200) | Current 100q run (n = 18) |
|---|---|---|
| Wall time | mean 36.3 s, median 26.6 s, p90 75.6 s, max 259 s | mean 46.0 s, median 21.5 s, p90 90.8 s, max 306 s |
| Prompt to the LLM | ~392 tokens (254 template + 138 paragraph) | ~393 tokens |
| LLM output | **mean 1,110 tokens**, median 1,029, max 3,461 | **mean 1,229**, median 946, max 3,574 |
| Output ÷ input | 2.8× | 3.1× |
| Pure decode time at the measured 52 tok/s | **63% of wall time** | **60%** |
| Prompt tokenization (tiktoken, Rust) | 0.098 ms | 0.095 ms |
| JSON parse + validation (Python) | 0.07 ms | 0.06 ms |

Token counts use the cl100k tokenizer as an approximation of mistral's tokenizer (typically within ~10–15%).

**The remaining ~37–40% of wall time** is not Python. It is a mix of:
- (a) thermal throttling: the GPU sits at 89 °C and drops its clock, so the real decode rate falls
  below the 52 tok/s measured when cool;
- (b) Windows thermal hibernation pauses;
- (c) long degenerate outputs and retries (4/200 pilot paragraphs failed after the retry ladder, at
  up to 259 s each);
- (d) prompt evaluation and model load (well under 1 s).

**Output composition** (40 responses): verbatim `source_sentence` copies 33%, `description` text
20%, names / types / JSON syntax 47%.

## 2. Answers to the questions asked

1. **Where is the time spent?** In LLM decoding of about 1,100 output tokens per paragraph, plus
   thermal slowdowns and pauses. The prompt is short (~390 tokens).
2. **Rust tokenizers installed?** Yes: `tiktoken` 0.14 (Rust core), Hugging Face `tokenizers` 0.22
   (Rust), `orjson` 3.11 (Rust JSON) and `regex`. There is no Rust toolchain (`cargo` / `rustc`).
3. **Is tokenization the bottleneck?** **No.** The extraction loop does no tokenization at all;
   Ollama tokenizes inside its own C++ runtime. Even if it did, tokenizing a prompt takes
   **0.1 ms out of ~36,000 ms (0.0003%)**, and Python post-processing takes 0.07 ms.
4. **Can Rust preprocessing reduce LLM input or overhead?** **Not meaningfully.** Input is already
   small (~390 tokens; prompt evaluation is sub-second on this GPU), and the Python overhead is
   ~0.2 ms. Moving tokenization, parsing or JSON handling to Rust (`orjson` is already installed)
   would save **< 1 ms per paragraph, i.e. ~0% speed-up**. Shrinking the *input* cannot help much
   either, because the cost is in the *output*.

## 3. Options that would actually help

The only lever that matters is the number of tokens decoded per paragraph and the speed of
decoding them. None of the options below touches the four retrieval algorithms.

| Option | What changes | Estimated extraction speed-up | Expected quality impact | Needs |
|---|---|---|---|---|
| **A. Thermal fix** (GPU clock cap `nvidia-smi -lgc 0,1500`, or max-fan mode) | Hardware setting only; model, prompt and outputs unchanged | Removes most hibernation pauses and reboot gaps; decode ~5–15% slower when cool, but steadier. **Wall-time saving ~20–35%** with you away overnight | None (identical outputs) | Admin terminal (you) |
| **B. Sentence-index output** (preprocessing: number the paragraph's sentences, which HotpotQA already provides; the model returns `sentence_id` instead of re-typing the sentence; code restores the exact sentence) | **Extraction prompt and output format change** | Removes ~33% of output tokens → **~25–30% less decode time** | Likely neutral to positive: `source_sentence` becomes exactly verbatim by construction. Risk: a wrong index → wrong evidence sentence. Must be measured | Your approval (prompt change); re-extract under one prompt version |
| **C. JSON-schema constrained output** (as already done for answer generation) | Ollama gets the extraction JSON schema; prompt text unchanged | Removes runaway / unparseable outputs and retry passes. Cuts the **long tail** (p90 76 s, max 259–306 s): **~10–15%** overall | Should reduce parse failures (4/200 in the pilot); outputs can differ slightly from unconstrained decoding | Your approval (generation-setting change) |
| **D. Ollama request batching** (`OLLAMA_NUM_PARALLEL=2–3`, 2–3 paragraphs in flight) | Runtime setting; prompt and model unchanged | Decode on this GPU is memory-bandwidth bound, so batching typically yields **~1.4–1.8× throughput** | Batched temperature-0 decoding is not guaranteed bit-identical to single-stream; small differences possible. **More heat** on a machine that already overheats | Your approval; conflicts with the "never run GPU inference concurrently" rule |
| E. Rust tokenization / preprocessing | Move tokenization or parsing to Rust | **~0%** (< 1 ms of ~36 s) | None | Not worth doing |

Not proposed (out of bounds): changing the extraction model; dropping descriptions or relationships;
lowering `num_predict` (would truncate legitimate long outputs, max 3,574 tokens seen).

**Most defensible combination:** A (no effect on results), plus C and B as a single new versioned
extraction configuration if you accept a prompt and format change. Expected total: about **35–45%
less extraction time**, roughly 36 s → 20–23 s per paragraph, with throttling removed by A.

## 4. Validation experiment (before any adoption)

- **Paragraphs:** 30 from the pilot corpus. They already have mistral baseline extractions,
  latencies and responses, so no new baseline compute is needed. Use the first 30 by `chunk_id`
  (fixed, not cherry-picked).
- **Arms:**
  - baseline (existing records);
  - C only;
  - B + C;
  - optionally D (2 in flight) on top of B + C.
- **Measured per arm:**
  - time per paragraph and decode tokens/s (from Ollama's `eval_count` / `eval_duration`);
  - output tokens;
  - parse failures and retries;
  - entity count and relationship count;
  - **entity agreement with baseline:** normalised-name recall and precision;
  - **relationship agreement:** (source, target) pair overlap;
  - fraction of `source_sentence` values found verbatim in the paragraph.
- **Adoption gate:**
  - ≥ 25% less time per paragraph;
  - entity-name recall vs baseline ≥ 0.90 and relationship-pair recall ≥ 0.85;
  - parse failures not higher than baseline;
  - verbatim `source_sentence` rate not lower.
- **Cost:** about 30 × 3 arms × ~25 s ≈ **40 minutes of GPU**. It needs the GPU to itself: either
  pause the running experiment (safe; it resumes from the last saved paragraph), or run the
  validation after the 100q stage.

## 5. What adoption would mean for the running experiment

- Option **A** can be applied **now** without touching the experiment.
- Options **B–D** change how the graph is built. One index must not mix two extraction
  configurations, so adopting them means:
  - pause the run;
  - preserve the current 100q extractions as they are, in a separate folder;
  - restart the 100q extraction under the new versioned configuration;
  - record it in the methodology as a deviation;
  - make the pilot comparison caveat that its graph used the original prompt.

  Currently ~30 of 985 paragraphs (~15 min of work) would be redone.
