# Stage 1: qwen3:4b extraction check with JSON schema (30 paragraphs), 2026-11-20

Setup: qwen3:4b (Q4_K_M, digest 359d7dd4...), original MindMesh extraction prompt, think=false,
Ollama `format` = JSON schema (arm C), temperature 0, seed 42, num_predict 4096. Same 30 paragraphs as E3.
Machine-readable numbers: `stage1_comparison.json`.

| | qwen3:4b + schema | Mistral baseline | qwen2.5:3b (B+C, ref.) |
|---|---|---|---|
| Mean / median s per paragraph | 30.6 / 24.0 | 24.4 / 23.4 | 5.3 / 4.4 |
| Parse failures | **1 / 30** | 0 | 0 |
| Entities / relationships | 254 / 197 | 226 / 162 | 154 / 97 |
| Entity F1 vs Mistral | 0.68 | (Mistral C noise floor 0.96) | 0.63 |
| Relationship recall vs Mistral (renames counted) | 0.48 | (noise floor 0.87) | 0.29 |
| Evidence sentence verbatim | 0.90 | 0.59 | — |

- Failure: paragraph 30 'Rango (2011 film)'. All 3 attempts ran to the 4,096-token limit inside a valid-looking
  JSON object (truncated at line 276); 230.6 s lost. Recorded as a model failure.
- Infra: one CUDA "unknown error" crash during a thermal sleep (02:40:56-02:41:01, no reboot); resumed, no data lost.
- GPU: mean 85.4 °C, max 91 °C, median SM clock 1980 MHz; 3.9 GB VRAM, 100% GPU.

## Full 985-paragraph build estimate (estimate, not measured)
- 985 x 30.6 s (failure rate included) = **about 8.4 h**; 985 x 23.7 s (successful paragraphs only) = about 6.5 h.
- Thermal sleeps / CUDA crashes would add restarts (the build is resumable).

## Blocker before a full build
`build_index.py` uses the production extractor, which does **not** send the JSON schema. Without it qwen3:4b
writes prose (Stage 1 attempt 1, STOPPED.md). A code change is needed: pass the schema as `format` for qwen3
models, and preferably fail fast on runaway output instead of three 4,096-token attempts.
Not changed yet; awaiting approval.
