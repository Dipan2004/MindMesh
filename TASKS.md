# TASKS.md — Build Plan (No Black Box)

## Rules for every task (read this before Task 0.1)

1. **One task = one file (or a tight cluster of 2-3 files that can't be
   tested independently).** Never bundle unrelated functions into one task
   just to move faster.
2. **A task is not "done" until it has a passing test AND a DOCUMENT.md
   entry.** Code that runs but has no test, or a test that passes but isn't
   logged in DOCUMENT.md, is an incomplete task — mark it `BLOCKED`, not
   `DONE`.
3. **Definition of Done = paste actual command output.** "It works" /
   "tests pass" is not acceptable evidence. Every task below has an exact
   command to run; the DOCUMENT.md entry must include the real stdout, not
   a paraphrase or a claim that it succeeded.
4. **No function may silently swallow an exception.** If something can
   fail (LLM call, Zvec query, file read), the failure must be visible in
   the test — either the test causes it or the test proves the error path
   returns/raises what it's supposed to.
5. **Every constant is read from `config.py` (which reads `CONFIG.md`'s
   defaults / `.env` overrides).** A task that hardcodes a chunk size,
   model name, or hop count instead of importing it from config fails
   review — go fix it before marking the task done.
6. **After finishing a task, update DOCUMENT.md immediately** — not at the
   end of the day, not after three more tasks. Use the template in
   `DOCUMENT.md`'s header. This is mandatory, not optional, per task.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done (has test +
DOCUMENT.md entry) · `[!]` blocked (say why in DOCUMENT.md)

---

## Phase 0 — Scaffolding

### Task 0.1 — Repo skeleton + config loader
**Files:** `pyproject.toml`, `.env.example`, `src/graphrag_lite/config.py`
**Build:**
- `config.py` exposes a single `Config` object (pydantic `BaseSettings` or
  plain dataclass) that reads every variable in `CONFIG.md` §1-11 from env
  vars, falling back to the defaults listed there.
- Function: `get_config() -> Config` (cached singleton).
**Test:** `python -c "from graphrag_lite.config import get_config; c = get_config(); print(c.CHUNK_SIZE_EXTRACTION, c.EMBEDDING_MODEL, c.SLM_MODEL)"`
**DoD:** paste the printed line — must show the CONFIG.md defaults exactly.

### Task 0.2 — LLM call cache
**Files:** `src/graphrag_lite/cache/llm_cache.py`
**Build:**
- `cached_call(prompt: str, model: str, fn: Callable[[str], str]) -> str`
  — hashes `(prompt, model)`, checks `diskcache` at `CACHE_DIR`, calls `fn`
  on miss, stores result, returns it either way.
- `cache_stats() -> dict` — returns hit/miss counters for the session.
**Test:** write a small script that calls `cached_call` twice with the same
prompt against a dummy `fn` that increments a counter and returns a fixed
string. Assert the counter is 1 (not 2) after the second call.
**DoD:** paste the script's printed assertion result + the counter value.

---

## Phase 1 — Ingestion

### Task 1.1 — Document loader
**Files:** `src/graphrag_lite/ingestion/loader.py`
**Build:** `load_documents(dir_path: str) -> list[Document]` where
`Document = {doc_id: str, text: str, source_path: str}`. Must handle
`.txt` and `.md`. Must raise (not silently skip) on an unreadable file.
**Test:** put 3 sample `.md` files in `data/raw/`, run
`python -c "from graphrag_lite.ingestion.loader import load_documents; docs = load_documents('data/raw'); print(len(docs)); print(docs[0])"`
**DoD:** paste output — must show `3` and a real `Document` dict with
non-empty `text`.

### Task 1.2 — Token-based chunker
**Files:** `src/graphrag_lite/ingestion/chunker.py`
**Build:** `chunk_document(doc: Document, size: int, overlap_pct: float, tokenizer) -> list[Chunk]`
where `Chunk = {chunk_id, doc_id, text, token_count, start_offset, end_offset}`.
Must use the real tokenizer from config, not `len(text)//4`.
**Test:** chunk one doc at `CHUNK_SIZE_EXTRACTION`, assert every chunk's
`token_count <= CHUNK_SIZE_EXTRACTION`, assert consecutive chunks share
`overlap_pct` worth of tokens (print the actual overlap token count per
boundary, don't just assert True silently).
**DoD:** paste the printed per-chunk token counts and overlap counts for
one document.

### Task 1.3 — Entity/relationship extraction (single pass, no gleaning yet)
**Files:** `src/graphrag_lite/ingestion/extractor.py`
**Build:** `extract_entities_relationships(chunk: Chunk, llm_client) -> ExtractionResult`
where `ExtractionResult = {entities: list[Entity], relationships: list[Relationship]}`,
`Entity = {name, type, description, source_sentence}`,
`Relationship = {source, target, type, description, source_sentence}`.
Prompt must force exact `source_sentence` output (PRD §4). Must go through
`cached_call` from Task 0.2.
**Test:** run extraction on one real chunk from Task 1.1's sample docs
against the local SLM. Manually verify at least 2 entities and 1
relationship are correct against the source text (eyeball check, but the
raw output must be pasted, not summarized).
**DoD:** paste the full `ExtractionResult` JSON for one chunk, plus the
source chunk text next to it for verification.

### Task 1.4 — Gleaning loop
**Files:** `src/graphrag_lite/ingestion/gleaning.py`
**Build:** `glean(chunk: Chunk, first_pass: ExtractionResult, llm_client, max_gleanings: int) -> ExtractionResult`
— re-prompts up to `max_gleanings` times with "did you miss any
entities/relationships?", merges new findings into the result, dedupes.
**Test:** run on the same chunk as Task 1.3 with `MAX_GLEANINGS=1`. Print
entity/relationship count before and after gleaning.
**DoD:** paste before/after counts. If gleaning adds 0 new items on your
sample chunk, that's a valid result — paste it anyway, don't cherry-pick a
chunk that "shows improvement."

### Task 1.5 — Full ingestion CLI
**Files:** `scripts/build_index.py` (stage 1 only, up through extraction)
**Build:** wires 1.1 → 1.2 → 1.3 → 1.4 across all documents in `data/raw/`,
writes extraction results to `data/cache/extractions.jsonl`.
**Test:** `python scripts/build_index.py --stage extract` on the full
sample corpus.
**DoD:** paste console output (doc count, chunk count, entity count,
relationship count, cache hit/miss stats from Task 0.2) + `wc -l data/cache/extractions.jsonl`.

---

## Phase 2 — Graph

### Task 2.1 — Graph builder
**Files:** `src/graphrag_lite/graph/builder.py`
**Build:** `build_graph(extractions: list[ExtractionResult]) -> nx.DiGraph`
— dedupes entities by normalized name, adds typed nodes/edges with
provenance (`source_doc`, `source_sentence`) as attributes.
**Test:** build the graph from Task 1.5's output. Print
`graph.number_of_nodes()`, `graph.number_of_edges()`, and one full node's
attribute dict, one full edge's attribute dict.
**DoD:** paste all four printed values.

### Task 2.2 — Ranking (PageRank + degree)
**Files:** `src/graphrag_lite/graph/ranking.py`
**Build:** `rank_pagerank(graph) -> dict[node, float]`,
`rank_degree(graph) -> dict[node, int]`. Both must be pure functions over
the graph — no side effects, no config coupling beyond damping factor if
tunable.
**Test:** run both on Task 2.1's graph, print the top-5 nodes by each
method side by side.
**DoD:** paste the two top-5 lists. If they're identical, note that
explicitly — it's a legitimate (if uninteresting) result on a small graph.

### Task 2.3 — Bounded BFS traversal
**Files:** `src/graphrag_lite/graph/traversal.py`
**Build:** `bfs_expand(graph, seed_nodes: list[str], hops: int) -> list[str]`
— returns all nodes within `hops` of any seed, **unranked** (ranking is a
separate step per PRD §9, don't conflate them in this function).
**Test:** pick 2 known seed nodes from Task 2.1's printed node list, run
`bfs_expand` with `hops=2`, print the returned node count and list.
**DoD:** paste the seed nodes used, the hop count, and the full returned
list.

---

## Phase 3 — Embeddings + Vector Store

### Task 3.1 — Embedder
**Files:** `src/graphrag_lite/embeddings/embedder.py`
**Build:** `embed_batch(texts: list[str], model: str) -> list[list[float]]`
— calls the configured embedding model (Ollama for `nomic-embed-text` /
`Qwen3-Embedding-0.6B`, or `sentence-transformers` for `all-MiniLM-L6-v2`).
Must return vectors whose length matches the model's actual known
dimension — assert this inside the function, don't just trust the API.
**Test:** embed 3 sample sentences, print each vector's length and the
first 5 floats of each.
**DoD:** paste the three vector-length numbers (must match each other)
and the sample float slices.

### Task 3.2 — Zvec client — entities collection
**Files:** `src/graphrag_lite/vectorstore/zvec_client.py`
**Build:** `insert_entities(entities: list[Entity], embeddings) -> None`,
`hybrid_query(collection: str, query_text: str, query_vector, top_k: int, scalar_filters: dict = None) -> list[ScoredResult]`.
**Test:** insert Task 2.1's extracted entities (embedded via Task 3.1),
then run one hybrid query with a real query string related to your sample
corpus, `top_k=5`. Print the 5 returned results with their scores.
**DoD:** paste the 5 scored results — must show non-zero, differentiated
scores (all-identical scores means the fusion isn't working, flag as
`[!]` blocked, don't mark done).

### Task 3.3 — Zvec client — chunks collection
**Files:** same file, additive
**Build:** `insert_chunks(chunks: list[Chunk], embeddings) -> None`,
reuses `hybrid_query` with `collection="chunks_v1"`.
**Test:** same shape as 3.2, against chunks instead of entities.
**DoD:** paste the 5 scored chunk results.

---

## Phase 4 — Retrieval Pipelines

### Task 4.1 — Pipeline A (flat baseline)
**Files:** `src/graphrag_lite/retrieval/pipeline_a.py`
**Build:** `retrieve_flat(query: str) -> list[Evidence]` — embed query →
`hybrid_query` on chunks collection only, `top_k=ZVEC_TOP_N_FLAT`, no graph
involved at all.
**Test:** run on 3 sample queries from your eventual gold set (draft them
now if not written yet). Print the retrieved evidence for each.
**DoD:** paste all 3 queries with their retrieved evidence.

### Task 4.2 — Pipeline B step 1+2 — entry points + ranked expansion
**Files:** `src/graphrag_lite/retrieval/pipeline_b.py`
**Build:** `get_candidates(query: str) -> list[RankedNode]` — embed query →
`hybrid_query` on entities (top-k = `ZVEC_TOP_K_ENTRY`) → `bfs_expand` from
those → `rank_pagerank` (default per CONFIG.md) → truncate to
`TOP_N_CANDIDATES_AFTER_RANKING`.
**Test:** same 3 queries as Task 4.1. Print entry points, raw BFS count
before ranking, and final ranked candidate list.
**DoD:** paste entry points → BFS count → final ranked list for all 3
queries.

### Task 4.3 — Pipeline B step 3 — scoped re-query
**Files:** same file, additive
**Build:** `retrieve_structural(query: str) -> list[Evidence]` — calls
`get_candidates`, verbalizes each candidate node, runs a second
`hybrid_query` scoped to just those candidates (via scalar filter or
pre-filtered subset), returns top `TOP_N_FINAL_EVIDENCE`.
**Test:** same 3 queries. Print final evidence list, and print it next to
Pipeline A's output for the same query — this is the comparison the whole
paper hinges on, so the two outputs must be visibly logged side by side.
**DoD:** paste the Pipeline A vs Pipeline B evidence side-by-side for all
3 queries.

### Task 4.4 — Serialization (TOON + JSON)
**Files:** `src/graphrag_lite/retrieval/serialize.py`
**Build:** `to_toon(evidence: list[Evidence]) -> str`,
`to_json(evidence: list[Evidence]) -> str`. Both must round-trip losslessly
(a `from_toon`/`from_json` parse-back check counts as part of the test).
**Test:** serialize Task 4.3's evidence both ways, print both strings and
their token counts (via the real tokenizer from config, not char count).
**DoD:** paste both serialized strings + both token counts — this is the
first real data point for the TOON-vs-JSON ablation, log it as such.

---

## Phase 5 — LLM Layer

### Task 5.1 — LLM client factory
**Files:** `src/graphrag_lite/llm/client_factory.py`
**Build:** `get_llm_client(mode: Literal["local","api"], api_key: str = None) -> LLMClient`
— `local` returns an Ollama-backed client using `config.SLM_MODEL`, `api`
returns a Grok-backed client using the passed key (session-only, never
written to disk or logged — verify this explicitly in the test).
**Test:** instantiate both modes, run one trivial completion
("say the word 'ready'") through each, print the two responses. Then grep
the process's env/log output to confirm the API key string never appears
in it.
**DoD:** paste both completions + the grep result showing no key leakage
(empty grep output = pass, paste that empty result explicitly, don't just
claim it).

### Task 5.2 — Answer generation with citations
**Files:** `src/graphrag_lite/llm/generate.py`
**Build:** `generate_answer(query: str, evidence_str: str, client: LLMClient) -> Answer`
where `Answer = {text, cited_evidence_ids: list[str], confidence: float|None}`.
Prompt must require the model to cite which evidence IDs it used.
**Test:** run end-to-end: Task 4.3's Pipeline B evidence → Task 4.4's TOON
serialization → this function, for one query. Print the full `Answer`.
**DoD:** paste the full `Answer` object, and manually verify the cited
evidence IDs actually exist in the evidence that was passed in (paste that
check's result too — a hallucinated citation ID is a real bug to catch
here, not later).

---

## Phase 6 — Mode Toggle + Observability

### Task 6.1 — Tracing toggle
**Files:** `src/graphrag_lite/observability/tracing.py`
**Build:** `set_tracing(mode: Literal["local","api"]) -> None` — sets
`LANGCHAIN_TRACING_V2` env var per CONFIG.md §8 rule (on for local, off for
api). Must be called once at startup, not per-request.
**Test:** call with `mode="local"`, print `os.environ["LANGCHAIN_TRACING_V2"]`;
call with `mode="api"`, print it again.
**DoD:** paste both printed values (`true` then `false`).

### Task 6.2 — End-to-end query CLI (both modes)
**Files:** `scripts/run_query.py`
**Build:** CLI wiring `client_factory` + `tracing` + `pipeline_a`/`pipeline_b`
+ `serialize` + `generate` — `python scripts/run_query.py --pipeline b --mode local --query "..."`.
**Test:** run one query in local mode through Pipeline B end to end.
**DoD:** paste full console output including the final answer + citations.

---

## Phase 7 — API + Production Hardening

### Task 7.1 — FastAPI skeleton
**Files:** `src/graphrag_lite/api/main.py`, `routes.py`
**Build:** `POST /query {pipeline, mode, api_key?, query} -> Answer`,
`GET /health`.
**Test:** `uvicorn` up, `curl localhost:8000/health`, then one `curl` to
`/query`.
**DoD:** paste both curl outputs.

### Task 7.2 — Async hardening (CPU-bound offload)
**Files:** `api/main.py` (modify), `api/middleware.py`
**Build:** wrap `bfs_expand`/`rank_pagerank` calls in `asyncio.to_thread`
inside the route handler. Add `httpx.AsyncClient` connection pooling +
semaphore for outbound Grok calls (limit from CONFIG.md, verified against
Grok's actual current rate limit — check docs, don't assume a number).
**Test:** fire 20 concurrent requests at `/query` (a small local script,
not the full stress test yet) with `mode=local`, confirm the server
doesn't block/serialize them — print per-request start/end timestamps to
show overlap.
**DoD:** paste the timestamp log showing overlapping request windows.

### Task 7.3 — Circuit breaker + retry
**Files:** `api/middleware.py` (additive)
**Build:** exponential backoff retry (`RETRY_MAX_ATTEMPTS`,
`RETRY_BACKOFF_BASE_S`) wrapping outbound Grok calls; circuit breaker opens
after `CIRCUIT_BREAKER_FAILURE_THRESHOLD` consecutive failures, returns 503
instead of retrying further.
**Test:** point the client at a deliberately-wrong Grok endpoint/key to
force failures, confirm the breaker opens after the configured threshold
and subsequent calls fail fast (measure latency of the fast-fail vs the
retried calls).
**DoD:** paste the failure log showing the threshold being hit and the
fast-fail latency after.

---

## Phase 8 — Evaluation

### Task 8.1 — Gold Q&A set
**Files:** `eval/gold_qa_set.json`
**Build:** 30-50 hand-written `{query, ground_truth_answer, ground_truth_evidence_node_ids}`
entries over your actual corpus. This is manual work, not code — no
function to test, but it's still a task with a DoD.
**DoD:** paste the full JSON file (or first 5 + last 5 entries + total
count if long) — must be real questions against your real ingested corpus,
not placeholder text.

### Task 8.2 — Accuracy/efficiency eval harness
**Files:** `eval/run_accuracy_eval.py`
**Build:** runs every gold question through Pipeline A and Pipeline B
(local mode, tracing on), scores correctness (LLM-as-judge or fuzzy match),
groundedness (cited IDs ⊆ retrieved evidence IDs), pulls latency/token
count from LangSmith trace metadata via their SDK.
**Test:** run on the full gold set.
**DoD:** paste the full results table (per-question scores + aggregate
accuracy/latency/tokens for A vs B) — this is a real paper result, log it
completely, not truncated.

### Task 8.3 — Embedding ablation
**Files:** `eval/ablations/embedding_ablation.py`
**Build:** re-run Task 8.2's harness three times, swapping
`EMBEDDING_MODEL` between `nomic-embed-text`, `Qwen3-Embedding-0.6B`,
`all-MiniLM-L6-v2` each time (re-embed + re-index between runs — don't
reuse stale vectors from a different model).
**DoD:** paste the 3-way comparison table.

### Task 8.4 — SLM ablation
**Files:** `eval/ablations/slm_ablation.py`
**Build:** re-run Task 8.2's harness with `SLM_MODEL` = `qwen3:4b`,
`qwen3:1.7b`, `phi4-mini` (extraction stays on the primary model unless
you're also testing extraction-model sensitivity separately — state which
in the DOCUMENT.md entry).
**DoD:** paste the 3-way comparison table.

### Task 8.5 — TOON vs JSON ablation
**Files:** `eval/ablations/toon_vs_json_ablation.py`
**Build:** re-run Task 8.2's harness with `SERIALIZATION_FORMAT=toon` vs
`json`, same evidence, same model — isolate the serialization variable
only.
**DoD:** paste the 2-way comparison table (tokens, latency, accuracy) —
this is the PRD §11 ablation, don't skip logging whether TOON's savings
actually held or got offset by prompt tax on your corpus.

### Task 8.6 — Ranking method ablation (PageRank vs degree)
**Files:** add a flag to `eval/run_accuracy_eval.py` or a small ablation
script.
**Build:** re-run with `RANKING_METHOD=pagerank` vs `degree`.
**DoD:** paste the 2-way comparison table.

---

## Phase 9 — Stress Test

### Task 9.1 — Async stress-test script
**Files:** `stress/stress_test.py`
**Build:** `asyncio` + `httpx` ramp across `STRESS_CONCURRENCY_LEVELS`
against the running FastAPI server in `mode=api`. Logs per-stage latency
(retrieval-only vs LLM-call — instrument both separately, not just
end-to-end), success/error rate by failure type, p50/p95/p99.
**Pre-check:** confirm Grok's actual current rate limit before setting
`STRESS_SEMAPHORE_LIMIT` — check the API docs at build time, don't reuse a
number from an old conversation.
**Test:** run the full ramp (10 → 50 → 200 → 1000).
**DoD:** paste the full results table for all 4 concurrency levels —
latency percentiles, success rate, requests/sec, and any errors
encountered (with their actual error type, not "some failed").

---

## Phase 10 — Paper Artifacts

### Task 10.1 — Results compilation
**Files:** `eval/results_summary.md` (generated, not hand-written)
**Build:** a script that reads every ablation's output JSON and produces
one combined markdown table for the paper.
**DoD:** paste the final combined table.

### Task 10.2 — Non-goals verification pass
**Files:** none — this is a review task
**Build:** re-read PRD §19 (Non-Goals). Confirm nothing built during the
project accidentally implemented something listed as out of scope (e.g.
don't discover halfway through that Pipeline B quietly grew a trained GNN).
**DoD:** paste a short checklist confirming each non-goal still holds,
with a one-line note per item.

---

## How to run this with Claude Code

Point Claude Code at this file with an instruction like: *"Work through
TASKS.md in order. For each task: implement it, run its exact test command,
paste the real output into DOCUMENT.md using the template at the top of
that file, then move to the next task. Do not mark a task `[x]` without a
DOCUMENT.md entry containing real command output. Stop and flag `[!]` if a
test fails — do not silently work around it or skip to the next task."*
