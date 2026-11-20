# Stage 1 (qwen3:4b extraction check) stopped at paragraph 1, 2026-11-20 ~02:27

- **Model:** `qwen3:4b`, 4.0B, Q4_K_M, digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`;
  capabilities completion, tools, thinking.
- **Settings sent:** the original MindMesh extraction prompt with temperature 0, seed 42, `num_predict` 4096,
  `num_ctx` 8192 and frequency penalty 0.25. **`think: False` was sent**, as recorded in `config.json`.
- **Observed on paragraph 1 ('The Duke Steps Out'):**
  - The model wrote plain-text reasoning ("We are given a text: …") instead of JSON.
  - Every attempt ran to the 4,096-token limit (Ollama log: 4,096 tokens in about 59 s, 69 tok/s).
  - All 3 retry-ladder attempts failed: 12,288 tokens, 231.7 s, "No JSON object found".
- **Why it was stopped:**
  - Projected Stage 1 time: about 2 h, mostly failures.
  - Projected 985-paragraph build time: about 60 h.
  - This is the outcome the Stage 1 gate exists to catch.
- **Preserved:** `results.jsonl` (1 row), `config.json`, `run.log`, `gpu_telemetry.csv`.
  Nothing else was changed.
