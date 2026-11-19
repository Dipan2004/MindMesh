# Freeze amendment 2 (2026-11-19)

Base: `freeze_20261119_amend1/` (kept unmodified; original `freeze_20261119/` also kept).

**File changed:** `src/mindmesh/benchmarks/hotpotqa/generation.py` (supporting-fact parsing in `generate`)

**Failure found by the 2-question diagnostic smoke test:** mistral:latest returns
`"supporting_facts": [3, 0]` (a flat pair) instead of `[[3, 0]]`. The parser only accepted a list
of pairs and silently dropped the flat form, so supporting-fact P/R/F1 and joint EM/F1 were 0 on
every question regardless of what the model cited.

**Change:** a flat list of integers is now read as consecutive [passage, sentence] pairs. The
requested `[[p, s], ...]` form is handled as before. Verified on 6 synthetic cases (all pass).
Scoring parser only. The prompt, retrieval code, retrieval/fusion settings and generation settings
are unchanged. Approved by the user before the full benchmark began.

The smoke-test results in `runs/smoke_2q_diagnostic/` were produced before this fix and are
kept unchanged as a diagnostic record.
