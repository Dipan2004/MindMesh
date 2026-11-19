# Freeze amendment 3 (2026-11-19)

Base: `freeze_20261119_amend2/` (kept unmodified, as are amend1 and the original freeze).

**File changed:** `src/mindmesh/benchmarks/hotpotqa/generation.py`

**Failure:** the first full run (`runs/gen-mistral_latest__idx-mistral_latest__72597386/`, kept as is)
lost 44 of 80 generations to `ResponseError: prediction aborted, token repeat limit reached`.
Reproduced: with the generic `format="json"` grammar, mistral:latest wrote
`"supporting_facts": [4, 1], ` and then tried to open a second pair. The grammar only allowed
whitespace at that point, so the model emitted tabs until Ollama's repetition guard aborted the
request, discarding an answer that was often already written (for example "Talib Kweli", correct).

**Change:** Ollama now receives an explicit JSON schema as `format`: `answer` is a string and
`supporting_facts` is a list of two-integer `[passage, sentence]` arrays. The prompt text, the
retrieval code, retrieval/fusion settings and the other generation options are unchanged.
`PROMPT_VERSION` is now `hotpotqa-json-v2-schema`, so the re-run writes to a new run directory.

**Check before the re-run:** all 44 previously failing generations completed through the real
`generate()` (counted for success only; answers were not scored, nothing was tuned).
Approved by the user.
