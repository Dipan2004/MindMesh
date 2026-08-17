# graphrag-lite

Lightweight graph-RAG: structural graph filtering + Zvec hybrid retrieval +
small local LLM. Full context, gap analysis, and design rationale in
`PRD.md`.

## Read these in order

1. **`PRD.md`** — the research problem, the gap, why this design (chunking,
   extraction, embedding, retrieval, ablations). Read this first for *why*.
2. **`ARCHITECTURE.md`** — component diagram, file structure, data flow.
   Read this for *what the system looks like*.
3. **`CONFIG.md`** — every tunable variable in one place, with defaults and
   the range to test each across. Read this before touching any number in
   code.
4. **`TASKS.md`** — the build plan. 33 tasks across 10 phases, each with an
   exact test command and a required pasted-output proof. This is the
   actual work order — build in this sequence, not out of order, since
   later tasks depend on earlier ones (Phase 4 needs Phase 2 and 3 done).
5. **`DOCUMENT.md`** — the build log. Empty until you start Task 0.1, then
   fills in one entry per task, in order, with real command output. This is
   what you show your mentor to prove the system isn't a black box.

## Quick start (once Phase 0-6 are done)

```bash
pip install -e .
cp .env.example .env   # fill in local model names / Grok key if using API mode
python scripts/build_index.py --stage all       # run full indexing
python scripts/run_query.py --pipeline b --mode local --query "your question"
```

## Non-negotiable rule while building

No task gets marked done without:
1. A real, runnable test (the exact command is in `TASKS.md`)
2. Actual pasted output in `DOCUMENT.md` (not "it worked")

If you're using Claude Code to execute `TASKS.md`, tell it exactly that —
see the note at the bottom of `TASKS.md`.
