# DOCUMENT.md — Build Log (Function-Level Tracking)

This file is the single record of what was actually built and how. Every
task in `TASKS.md` gets one entry here, added **immediately** after that
task passes its test — not batched, not summarized later from memory.

An entry that doesn't include real pasted command output is invalid.
"Implemented and tested successfully" with no output is not a log entry —
delete it and redo it properly.

---

## Entry Template (copy this for every task)

```
## Task <ID> — <title>

**Status:** [x] done / [!] blocked / [~] in progress
**Date:** <date completed>
**Files touched:** <exact paths>

### Function(s) implemented

- `function_name(arg: type, ...) -> return_type`
  - **Purpose:** one sentence, what it actually does
  - **How:** 2-4 sentences — the real approach (which library call, which
    algorithm, which config value it reads). If it wraps a cached/external
    call, say which.
  - **Edge cases handled:** what happens on empty input, on failure, on
    an unexpected type — and how you verified it (don't just assert you
    handled it)

### Config values used

- `CONFIG_VAR_NAME` = <value used this run> (from CONFIG.md default / .env
  override — say which)

### Test run

**Command:**
```
<exact command>
```

**Output (pasted verbatim):**
```
<real stdout/stderr, not paraphrased>
```

### Notes / deviations from TASKS.md

<anything that didn't go as planned, any function signature that changed
from what TASKS.md specified, and why>

---
```

---

## Build Log

(Entries go below, in task order. Nothing pre-filled — this section starts
empty and fills in as `TASKS.md` is worked through.)

### Phase 0 — Scaffolding
<!-- Task 0.1, 0.2 entries go here -->

### Phase 1 — Ingestion
<!-- Task 1.1–1.5 entries go here -->

### Phase 2 — Graph
<!-- Task 2.1–2.3 entries go here -->

### Phase 3 — Embeddings + Vector Store
<!-- Task 3.1–3.3 entries go here -->

### Phase 4 — Retrieval Pipelines
<!-- Task 4.1–4.4 entries go here -->

### Phase 5 — LLM Layer
<!-- Task 5.1–5.2 entries go here -->

### Phase 6 — Mode Toggle + Observability
<!-- Task 6.1–6.2 entries go here -->

### Phase 7 — API + Production Hardening
<!-- Task 7.1–7.3 entries go here -->

### Phase 8 — Evaluation
<!-- Task 8.1–8.6 entries go here -->

### Phase 9 — Stress Test
<!-- Task 9.1 entry goes here -->

### Phase 10 — Paper Artifacts
<!-- Task 10.1–10.2 entries go here -->

---

## Running Summary (update the counts as you go)

| Phase | Tasks total | Done | Blocked |
|---|---|---|---|
| 0 — Scaffolding | 2 | 0 | 0 |
| 1 — Ingestion | 5 | 0 | 0 |
| 2 — Graph | 3 | 0 | 0 |
| 3 — Embeddings/Vector Store | 3 | 0 | 0 |
| 4 — Retrieval Pipelines | 4 | 0 | 0 |
| 5 — LLM Layer | 2 | 0 | 0 |
| 6 — Mode Toggle | 2 | 0 | 0 |
| 7 — API/Hardening | 3 | 0 | 0 |
| 8 — Evaluation | 6 | 0 | 0 |
| 9 — Stress Test | 1 | 0 | 0 |
| 10 — Paper Artifacts | 2 | 0 | 0 |
| **Total** | **33** | **0** | **0** |
