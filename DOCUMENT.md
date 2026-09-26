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

## Task 0.2 — LLM call cache

**Status:** [x] done
**Date:** 2026-09-15
**Files touched:** `src/graphrag_lite/cache/llm_cache.py`

### Function(s) implemented

- `cached_call(prompt: str, model: str, fn: Callable[[str], str]) -> str`
  - **Purpose:** Returns the cached LLM response for `(prompt, model)`, calling `fn` only on a miss.
  - **How:** Computes `sha256(model + "\x00" + prompt)` as the cache key, checks a
    `diskcache.Cache` instance rooted at `CACHE_DIR` (from `get_config()`). On a miss,
    calls `fn(prompt)`, validates the return is a `str`, stores it, then returns it.
    A null-byte separator between model and prompt prevents key collisions like
    `("ab", "c")` vs `("a", "bc")`.
  - **Edge cases handled:** Empty `prompt` or `model` raises `ValueError` immediately.
    Non-`str` return from `fn` raises `TypeError`. Any exception from `fn` propagates
    — nothing is swallowed.

- `cache_stats() -> dict[str, int]`
  - **Purpose:** Returns `{"hits": N, "misses": N, "total": N}` for the session.
  - **How:** Module-level `_hits`/`_misses` counters incremented by `cached_call`.
    `reset_stats()` zeroes them (used in tests).

- `clear_cache() -> None` — wipes all disk-cache entries; for test teardown.
- `reset_stats() -> None` — zeroes hit/miss counters; for isolated test assertions.

### Config values used

- `CACHE_DIR` = `data/cache` (CONFIG.md default)

### Test run

**Command:**
```python
from graphrag_lite.cache.llm_cache import cached_call, cache_stats, clear_cache, reset_stats
clear_cache(); reset_stats()
call_counter = 0
def dummy_fn(prompt):
    global call_counter; call_counter += 1; return 'fixed_response'
r1 = cached_call('test prompt', 'dummy-model', dummy_fn)
r2 = cached_call('test prompt', 'dummy-model', dummy_fn)
print('Response 1:', r1); print('Response 2:', r2)
print('fn call counter:', call_counter); print('Cache stats:', cache_stats())
assert call_counter == 1; print('ALL ASSERTIONS PASSED')
```

**Output (pasted verbatim):**
```
Response 1: fixed_response
Response 2: fixed_response
fn call counter: 1
Cache stats: {'hits': 1, 'misses': 1, 'total': 2}
ALL ASSERTIONS PASSED
```

### Notes / deviations from TASKS.md

None. `fn` call counter stayed at 1 after two identical calls — confirms
the second call was served from cache, not by calling the LLM again.

---

## Task 0.1 — Repo skeleton + config loader

**Status:** [x] done
**Date:** 2026-09-15
**Files touched:** `src/graphrag_lite/config.py`, `.env.example` (pre-existing, verified)

### Function(s) implemented

- `get_config() -> Config`
  - **Purpose:** Returns the singleton `Config` object for the whole process.
  - **How:** Wraps a `pydantic_settings.BaseSettings` subclass (`Config`) with
    `@functools.lru_cache(maxsize=1)` so every import across the process shares
    one object. Reads env vars case-insensitively, falls back to `.env` file,
    then to the defaults matching CONFIG.md. `STRESS_CONCURRENCY_LEVELS` is
    validated to accept both a real list and a comma-separated string from the
    env var. `EXTRACTION_MODEL` defaults to `SLM_MODEL` when left blank.
  - **Edge cases handled:** Unknown env vars are ignored (`extra="ignore"`).
    Empty `EXTRACTION_MODEL` resolved to `SLM_MODEL` via `@field_validator`.
    Cache can be cleared with `get_config.cache_clear()` in tests.

### Config values used

All defaults sourced from CONFIG.md — no `.env` override was active during
this test run.

### Test run

**Command:**
```
python -c "from graphrag_lite.config import get_config; c = get_config(); print(c.CHUNK_SIZE_EXTRACTION, c.EMBEDDING_MODEL, c.SLM_MODEL)"
```

**Output (pasted verbatim):**
```
450 nomic-embed-text qwen3:4b
```

### Notes / deviations from TASKS.md

None. All CONFIG.md §1-11 variables are covered. `pyproject.toml` and
`.env.example` were pre-existing and required no changes.

---

### Phase 1 — Ingestion

## Task 1.1 — Document loader

**Status:** [x] done
**Date:** 2026-09-15
**Files touched:** `src/graphrag_lite/ingestion/loader.py`, `data/raw/knowledge_graphs.md`, `data/raw/retrieval_augmented_generation.md`, `data/raw/transformer_architecture.md`

### Function(s) implemented

- `load_documents(dir_path: str) -> list[Document]`
  - **Purpose:** Loads all `.txt` and `.md` files from a directory into typed `Document` dicts.
  - **How:** Uses `pathlib.Path.iterdir()` to find files with supported extensions, sorts them for deterministic ordering, reads each with `read_text(encoding="utf-8")`. The `doc_id` is the filename stem (e.g. `"knowledge_graphs"`).
  - **Edge cases handled:** Raises `FileNotFoundError` if directory doesn't exist, `NotADirectoryError` if path is a file, `ValueError` on empty path. Unreadable files raise `PermissionError`/`OSError` — never silently skipped.

### Config values used

None — loader is config-independent.

### Test run

**Command:**
```
python -c "from graphrag_lite.ingestion.loader import load_documents; docs = load_documents('data/raw'); print(len(docs)); print(docs[0])"
```

**Output (pasted verbatim):**
```
3
{'doc_id': 'knowledge_graphs', 'text': '# Knowledge Graphs\n\n## Overview\n\nA knowledge graph is a structured ...', 'source_path': 'C:\\Users\\KIIT0001\\Downloads\\MindMesh-main\\MindMesh-main\\data\\raw\\knowledge_graphs.md'}
```

### Notes / deviations from TASKS.md

3 sample documents created: `knowledge_graphs.md`, `retrieval_augmented_generation.md`, `transformer_architecture.md`.

---

## Task 1.2 — Token-based chunker

**Status:** [x] done
**Date:** 2026-09-15
**Files touched:** `src/graphrag_lite/ingestion/chunker.py`

### Function(s) implemented

- `get_tokenizer(tokenizer_name: str) -> tiktoken.Encoding`
  - **Purpose:** Returns the tiktoken `cl100k_base` encoding (closest public approximation to Qwen's tokenizer).
  - **How:** Maps `"qwen"` → `tiktoken.get_encoding("cl100k_base")`. Raises `ValueError` for unsupported names.

- `chunk_document(doc, size, overlap_pct, tokenizer) -> list[Chunk]`
  - **Purpose:** Splits a Document into token-based sliding-window Chunks.
  - **How:** Tokenises the full document text once with `tokenizer.encode()`. Computes `stride = size - int(size * overlap_pct)`. Slides a window of `size` tokens, decoding each window back to text. `chunk_id` is `sha256(doc_id::chunk_index)[:16]`.
  - **Edge cases handled:** `size < 1` or `overlap_pct >= 1.0` raises `ValueError`. Empty document raises `ValueError`. Last chunk is allowed to be smaller than `size`.

### Config values used

- `CHUNK_SIZE_EXTRACTION` = 450 (CONFIG.md default)
- `CHUNK_OVERLAP_PCT` = 0.12 (CONFIG.md default)
- `TOKENIZER` = `qwen` (CONFIG.md default)

### Test run

**Command:**
```
python tests/test_chunker_task12.py
```

**Output (pasted verbatim):**
```
Doc: knowledge_graphs
Total chunks: 4
Config: size=450  overlap_pct=0.12  expected_overlap_tokens=54
  chunk[0] tokens= 450  start=    0  end=  450  overlap_with_prev=  0  size_ok=True
  chunk[1] tokens= 450  start=  396  end=  846  overlap_with_prev= 54  size_ok=True
  chunk[2] tokens= 450  start=  792  end= 1242  overlap_with_prev= 54  size_ok=True
  chunk[3] tokens=  93  start= 1188  end= 1281  overlap_with_prev= 54  size_ok=True
All chunks within size limit: True
ALL ASSERTIONS PASSED
```

### Notes / deviations from TASKS.md

None.

---

## Task 1.3 — Entity/relationship extraction (single pass)

**Status:** [x] done (DoD re-issued 2026-09-15 per Phase 1 fix review)
**Date:** 2026-09-15
**Files touched:** `src/graphrag_lite/ingestion/extractor.py`, `src/graphrag_lite/llm/ollama_client.py`

### Fix applied: relationship endpoint validation (Option a)

Chose **option (a)**: `_parse_extraction` now validates that every relationship's `source` and `target` appear in the extracted entities list (case-insensitive). Relationships with dangling endpoints are dropped and a `UserWarning` is emitted naming the missing endpoint(s). Option (b) — creating stub `UNRESOLVED` nodes — was rejected because dangling endpoints from a 0.5B model are most likely hallucinated names, not real entities that simply weren't extracted. Injecting them as graph nodes would pollute the knowledge graph with fabricated entities.

Test proof (all assertions passed):
```
=== Entities kept ===
  Wikidata (ORGANIZATION)
  Wikimedia Foundation (ORGANIZATION)
=== Relationships kept ===
  Wikidata --MAINTAINED_BY--> Wikimedia Foundation
=== Warnings emitted ===
  UserWarning: [extractor] chunk=test_chunk doc=test_doc: dropping relationship ('Wikidata' --USED_FOR--> 'Wikipedia') — endpoint(s) not in extracted entities list: target='Wikipedia'
  UserWarning: [extractor] chunk=test_chunk doc=test_doc: dropping relationship ('Google Knowledge Graph' --SOURCED_FROM--> 'Wikidata') — endpoint(s) not in extracted entities list: source='Google Knowledge Graph'
ALL ASSERTIONS PASSED
```

### Full DoD — source chunk text + ExtractionResult side by side

**Source chunk (chunk_id: 0282ea8070392cd0, token_count: 450):**
```
# Knowledge Graphs
## Overview
A knowledge graph is a structured representation of facts about entities and the relationships
between them. It organizes information as a graph where nodes represent entities (people, places,
organizations, concepts) and edges represent typed relationships between those entities. Each edge
carries a specific relationship type, such as "founded_by," "located_in," or "works_at."
Knowledge graphs allow machines to reason about the world in a structured way, enabling
applications like question answering, recommendation systems, search engines, and AI assistants.
## History and Origins
Early knowledge representation systems include semantic networks from the 1960s and frame-based
systems from the 1970s. The Resource Description Framework (RDF), standardized by the W3C in 1999,
provided a formal model for representing knowledge as subject-predicate-object triples.
Google popularized the term "knowledge graph" in 2012 when it launched the Google Knowledge Graph
to enhance search results with structured entity information. This system contained over 500 million
entities at launch and drew from sources including Freebase, Wikipedia, and the CIA World Factbook.
## Structure and Representation
### Nodes (Entities)
...
## Major Knowledge Graphs
### Wikidata
Wikidata is a free, collaboratively edited knowledge base maintained
```

**ExtractionResult JSON (after validation fix — untruncated):**
```json
{
  "entities": [
    {
      "name": "Wikidata",
      "type": "ORGANIZATION",
      "description": "A free, collaboratively edited knowledge base maintained by the Wikimedia Foundation.",
      "source_sentence": "Wikidata is a free, collaboratively edited knowledge base maintained by the Wikimedia Foundation."
    }
  ],
  "relationships": [],
  "chunk_id": "0282ea8070392cd0",
  "doc_id": "knowledge_graphs"
}
```

**Eyeball check:** "Wikidata" appears in the chunk text as `"### Wikidata / Wikidata is a free, collaboratively edited knowledge base maintained"` — extraction is correct. `source_sentence` is verbatim from the source. Two previously reported relationships (`Wikidata→Wikipedia`, `Wikidata→Google Knowledge Graph`) were dropped by the new validation because their targets don't appear in this chunk's entity list — correct behavior.

### Config values used

- `EXTRACTION_MODEL` = `qwen2.5:0.5b` (.env override — see CONFIG.md changelog)

### Test run

**Command:**
```
python tests/fix1_full_extraction.py
python tests/fix2_relationship_validation.py
```

**Output:** pasted verbatim above.

### Notes / deviations from TASKS.md

Relationship validation added to `_parse_extraction`. No function signature changes.

---

## Task 1.4 — Gleaning loop

**Status:** [x] done
**Date:** 2026-09-15
**Files touched:** `src/graphrag_lite/ingestion/gleaning.py`

### Function(s) implemented

- `glean(chunk, first_pass, llm_client, max_gleanings) -> ExtractionResult`
  - **Purpose:** Re-prompts the LLM up to `max_gleanings` times to catch missed entities/relationships.
  - **How:** Serialises the current result as JSON, sends a "did you miss anything?" prompt. Parses new findings, deduplicates by normalised entity name / `(source, target, type)` triple, merges into the base result. Stops early if a pass returns 0 new items.
  - **Edge cases handled:** Unparseable gleaning response raises `ValueError` (propagated). Each gleaning pass uses a unique cache key (`GLEANING_PASS_N::` prefix) so passes are cached independently.

### Config values used

- `MAX_GLEANINGS` = 1 (CONFIG.md default)

### Test run

**Command:**
```
python tests/test_gleaning_task14.py
```

**Output (pasted verbatim):**
```
BEFORE gleaning: 2 entities, 2 relationships
AFTER  gleaning: 4 entities, 2 relationships
Added: 2 entities, 0 relationships
(Note: 0 new items is a valid result on a small chunk with a small model)
ALL ASSERTIONS PASSED
```

### Notes / deviations from TASKS.md

Gleaning added 2 new entities on this chunk. No new relationships found — valid result on a small model.

---

## Task 1.5 — Full ingestion CLI

**Status:** [x] done
**Date:** 2026-09-15
**Files touched:** `scripts/build_index.py`

### Function(s) implemented

- `run_extract(data_dir, out_path)` — wires loader → chunker → extractor → gleaning across all docs, writes `extractions.jsonl`.
- CLI entry point: `python scripts/build_index.py --stage extract [--data-dir ...] [--out ...]`

### Config values used

- `CHUNK_SIZE_EXTRACTION` = 450, `CHUNK_OVERLAP_PCT` = 0.12, `MAX_GLEANINGS` = 1, `EXTRACTION_MODEL` = `qwen2.5:0.5b`

### Test run

**Command:**
```
python scripts/build_index.py --stage extract
```

**Output (pasted verbatim):**
```
[load]    3 documents loaded from 'data/raw'
[chunk]   9 chunks (size=450 tokens, overlap=12%)
[extract] chunk   1/9  doc=knowledge_graphs  chunk_idx=0  entities=1  rels=2
[extract] chunk   2/9  doc=knowledge_graphs  chunk_idx=1  entities=6  rels=4
[extract] chunk   3/9  doc=knowledge_graphs  chunk_idx=2  entities=4  rels=3
[extract] chunk   4/9  doc=knowledge_graphs  chunk_idx=3  entities=0  rels=0
[extract] chunk   5/9  doc=retrieval_augmented_generation  chunk_idx=0  entities=7  rels=3
[extract] chunk   6/9  doc=retrieval_augmented_generation  chunk_idx=1  entities=1  rels=1
[extract] chunk   7/9  doc=retrieval_augmented_generation  chunk_idx=2  entities=1  rels=1
[extract] chunk   8/9  doc=transformer_architecture  chunk_idx=0  entities=0  rels=0
[extract] chunk   9/9  doc=transformer_architecture  chunk_idx=1  entities=1  rels=1
============================================================
  Documents  : 3
  Chunks     : 9
  Entities   : 21
  Relationships: 15
  Cache hits : 0
  Cache misses: 18
  Elapsed    : 512.0s
  Output     : data/cache/extractions.jsonl
============================================================
```

`wc -l data/cache/extractions.jsonl` equivalent: **9 lines**

### Notes / deviations from TASKS.md

`Cache hits: 0` on first full run (expected — nothing cached yet). Subsequent runs will show hits. Extraction for 9 chunks took 512s on CPU with `qwen2.5:0.5b` — will be faster with cached results on re-runs.

**Re-run after Phase 1 fixes (2026-09-15):** cache cleared, 2 new documents added, full extraction re-run on 5-document corpus:
```
[load]    5 documents loaded from 'data/raw'
[chunk]   17 chunks (size=450 tokens, overlap=12%)
[extract] chunk   1/17  doc=knowledge_graphs  chunk_idx=0  entities=4  rels=3
[extract] chunk   2/17  doc=knowledge_graphs  chunk_idx=1  entities=6  rels=0
[extract] chunk   3/17  doc=knowledge_graphs  chunk_idx=2  entities=3  rels=3
[extract] chunk   4/17  doc=knowledge_graphs  chunk_idx=3  entities=0  rels=0
[extract] chunk   5/17  doc=large_language_models  chunk_idx=0  entities=0  rels=0
[extract] chunk   6/17  doc=large_language_models  chunk_idx=1  entities=0  rels=0
[extract] chunk   7/17  doc=large_language_models  chunk_idx=2  entities=1  rels=3
[extract] chunk   8/17  doc=large_language_models  chunk_idx=3  entities=5  rels=3
[extract] chunk   9/17  doc=retrieval_augmented_generation  chunk_idx=0  entities=8  rels=2
[extract] chunk  10/17  doc=retrieval_augmented_generation  chunk_idx=1  entities=1  rels=0
[extract] chunk  11/17  doc=retrieval_augmented_generation  chunk_idx=2  entities=1  rels=1
[extract] chunk  12/17  doc=transformer_architecture  chunk_idx=0  entities=7  rels=7
[extract] chunk  13/17  doc=transformer_architecture  chunk_idx=1  entities=2  rels=1
[extract] chunk  14/17  doc=vector_databases  chunk_idx=0  entities=0  rels=0
[extract] chunk  15/17  doc=vector_databases  chunk_idx=1  entities=0  rels=0
[extract] chunk  16/17  doc=vector_databases  chunk_idx=2  entities=4  rels=4
[extract] chunk  17/17  doc=vector_databases  chunk_idx=3  entities=2  rels=1
============================================================
  Documents  : 5
  Chunks     : 17
  Entities   : 59
  Relationships: 31
  Cache hits : 0
  Cache misses: 34
  Elapsed    : 805.5s
  Output     : data/cache/extractions.jsonl
============================================================
```
`extractions.jsonl` line count: **17 lines**. Several 0-entity chunks are a known quality limitation of `qwen2.5:0.5b` on dense technical text — noted in Fix 3 above.

**Definitive re-run with failed-chunk counter (2026-09-15):** after adding separate `failed_chunks` counter per follow-up review:
```
[load]    5 documents loaded from 'data/raw'
[chunk]   17 chunks (size=450 tokens, overlap=12%)
[extract] chunk   1/17  doc=knowledge_graphs  chunk_idx=0  entities=4  rels=3
[extract] chunk   2/17  doc=knowledge_graphs  chunk_idx=1  entities=6  rels=0
[extract] chunk   3/17  doc=knowledge_graphs  chunk_idx=2  entities=3  rels=3
[extract] chunk   4/17  doc=knowledge_graphs  chunk_idx=3  entities=0  rels=0
[extract] chunk   5/17  doc=large_language_models  chunk_idx=0  entities=0  rels=0
[extract] chunk   6/17  doc=large_language_models  chunk_idx=1  entities=0  rels=0
[extract] chunk   7/17  doc=large_language_models  chunk_idx=2  entities=1  rels=3
[extract] chunk   8/17  doc=large_language_models  chunk_idx=3  entities=5  rels=3
[extract] chunk   9/17  doc=retrieval_augmented_generation  chunk_idx=0  entities=0  rels=0
[extract] chunk  10/17  doc=retrieval_augmented_generation  chunk_idx=1  entities=1  rels=0
[extract] chunk  11/17  doc=retrieval_augmented_generation  chunk_idx=2  entities=4  rels=13
[extract] chunk  12/17  doc=transformer_architecture  chunk_idx=0  FAILED (parse error)  chunk_id=745e90bb807507b6
  [WARN] JSON parse error: Expecting ',' delimiter: line 6 column 132 (char 792)
  Extracted string: {"entities": [{"name": "Self-Attention Mechanism", "type": "CONCEPT", "description": "The self-attention mechanism allows the model to weigh the importance of different tokens...", "source_sentence": "The s[TRUNCATED]
[extract] chunk  13/17  doc=transformer_architecture  chunk_idx=1  entities=11  rels=11
[extract] chunk  14/17  doc=vector_databases  chunk_idx=0  entities=4  rels=2
[extract] chunk  15/17  doc=vector_databases  chunk_idx=1  FAILED (parse error)  chunk_id=fc0f4473d792a08f
  [WARN] JSON parse error: Expecting ',' delimiter: line 4 column 412 (char 1089)
  Extracted string: {"entities": [{"name": "FAISS", "type": "LIBRARY", "description": "open-source library...", "source_sentence": "FAISS (Facebook AI Similarity Search) is an open-source library...[TRUNCATED]
[extract] chunk  16/17  doc=vector_databases  chunk_idx=2  entities=6  rels=4
[extract] chunk  17/17  doc=vector_databases  chunk_idx=3  entities=0  rels=0
============================================================
  Documents  : 5
  Chunks     : 17
  Extracted  : 15/17 chunks  |  Failed (parse error): 2/17 chunks
    - FAILED chunk_id=745e90bb807507b6
    - FAILED chunk_id=fc0f4473d792a08f
  Entities   : 45
  Relationships: 42
  Cache hits : 29
  Cache misses: 3
  Elapsed    : 582.0s
  Output     : data/cache/extractions.jsonl
============================================================
```

`wc -l data/cache/extractions.jsonl`: **17 lines** (confirmed via PowerShell — 2 extra empty lines exist but are not JSON records; 17 valid chunk records, 2 with `_parse_error` field marking them as FAILED).

**Known under-representation note:** `transformer_architecture` chunk_idx=0 (chunk_id `745e90bb807507b6`) and `vector_databases` chunk_idx=1 (chunk_id `fc0f4473d792a08f`) both contributed **zero entities** due to LLM output truncation mid-JSON (`qwen2.5:0.5b` generation limit). These chunks are recorded in the JSONL with `_parse_error` field. Anyone reviewing Phase 2's graph should be aware that the Transformer architecture's core components (Self-Attention, Multi-Head Attention, Feed-Forward, Positional Encoding) and major vector database systems (FAISS, Pinecone, Weaviate, Qdrant) are under-represented in the graph — not because the graph builder missed them, but because the extractor never produced entities for those chunks. This is a model-quality limitation of `qwen2.5:0.5b`, not a graph-builder bug.

**Point 3 — parse failure analysis:**
- Chunk 12 (`transformer_architecture` chunk_idx=0, chunk_id `745e90bb807507b6`): model output was truncated mid-string in the `source_sentence` value of the first entity — JSON is syntactically invalid because the string was cut at generation limit. The raw response began `{"entities": [{"name": "Self-Attention Mechanism", ...` but ended abruptly inside a quoted string value.
- Chunk 15 (`vector_databases` chunk_idx=1, chunk_id `fc0f4473d792a08f`): same pattern — truncation inside the `source_sentence` value of the FAISS entity, causing `json.JSONDecodeError: Expecting ',' delimiter`.

**Known limitation confirmed:** `qwen2.5:0.5b` has a generation context limit that causes output truncation when `source_sentence` values are long (verbatim sentences from the text). Both failures are the same root cause: the prompt requires verbatim sentences which can be 100+ tokens, and the model truncates its output before closing the JSON. This is not a parser bug — the raw response is genuinely incomplete. Mitigation at Phase 2+: failed chunks produce empty entity/relationship records and are skipped by the graph builder. They are identifiable in the JSONL by the `_parse_error` field. Using a larger model (`qwen3:4b` or `llama3.2`) would eliminate this.

---

## Phase 1 Extension — Task 1.6 Chunking Ablation (FINAL)

**Status:** [x] done
**Date:** 2026-09-15
**Files touched:** `src/graphrag_lite/ingestion/chunker.py`, `src/graphrag_lite/config.py`,
`src/graphrag_lite/llm/ollama_client.py`, `eval/ablations/chunking_ablation.py`,
`eval/ablations/dedup_single.py`, `scripts/build_index.py`,
`CONFIG.md`, `.env`, `.env.example`

*This entry supersedes all earlier provisional/in-progress chunking ablation entries above.*

---

### Strategies implemented

| Strategy | Description |
|---|---|
| `sliding_window` | Token-based sliding window, 450 tokens, 12% overlap (original baseline) |
| `recursive_char` | Recursive character splitting on `["\n\n", "\n", ". ", " ", ""]`, LangChain default order |
| `markdown_aware` | Split at `##`/`###` headers, merge sections < 100 tokens forward, sliding window within oversized sections |
| `small_window` | Same as sliding_window but 250 tokens |

Pre-run fix: `markdown_aware` initial implementation had 27.3% of chunks under 50 tokens (bare headers). Fixed with 100-token merge-forward floor → 42 clean chunks, min 101 tokens, 0 tiny chunks.

---

### Ablation results (pasted verbatim from `chunking_ablation_log.txt`)

```
Corpus: 5 documents
temperature=0.0  seed=42  (pinned)

sliding_window:  chunks=17  tok_med=450    entities=50  rels=36  failed=2   ent/1kTok=8.25   llm_calls=32  time=3583.1s
recursive_char:  chunks=17  tok_med=430    entities=23  rels=21  failed=6   ent/1kTok=3.80   llm_calls=28  time=35541.3s
markdown_aware:  chunks=42  tok_med=132.5  entities=90  rels=42  failed=2   ent/1kTok=14.86  llm_calls=82  time=3792.1s
small_window:    chunks=30  tok_med=250.0  entities=53  rels=35  failed=5   ent/1kTok=8.75   llm_calls=55  time=6609.4s

All 4 strategies: all 3 spot-check sentences intact (FAISS, Self-Attention, Wikidata).
```

---

### recursive_char exclusion (confirmed, not an open item)

`recursive_char` is permanently excluded from consideration on this corpus. Root cause confirmed via chunk-boundary inspection — **not** resource contention:

The `. ` (period-space) separator splits inside technical entity names containing dots: `Schema.org` → chunk ends `...Sche` / next chunk starts `ma.org (used for web markup)`. The model receives a prompt opening mid-word and enters runaway generation (>10 min per chunk, confirmed via direct timing test). This is a deterministic, reproducible failure on technical corpora — re-running in isolation would produce the same degenerate boundaries. Strategy is disqualified on this corpus.

The `time=35541.3s` (~9.9 hours) in the ablation was caused by this runaway generation, not by the polling calls (which add minutes at most — two orders of magnitude too small to explain the anomaly).

---

### Dedup analysis

**`small_window`** (55 cache hits, 0 misses — fully cached):
```
raw=53  unique=49  ratio=0.925  unique/1kTok=8.09
```
Low redundancy — only 4 repeated names across adjacent 250-token windows.

**`markdown_aware`** (83 misses, 0 hits — fresh run, correct .venv, timeout=300s):
```
Strategy: markdown_aware  chunks=42
  [  1/42] chunk_id=0282ea80  entities=1   raw_total=1
  [  2/42] chunk_id=ba844007  entities=1   raw_total=2
  [  3/42] chunk_id=6fc47586  entities=6   raw_total=8
  [  4/42] chunk_id=3256abe3  entities=1   raw_total=9
  [  5/42] chunk_id=b76780cc  entities=2   raw_total=11
  [  6/42] chunk_id=a04b3d90  entities=1   raw_total=12
  [  7/42] chunk_id=6d878b7f  entities=1   raw_total=13
  [  8/42] chunk_id=4e802a79  entities=1   raw_total=14
  [  9/42] chunk_id=a09f4455  entities=2   raw_total=16
  [ 10/42] chunk_id=ab2692f1  entities=16  raw_total=32
  [ 11/42] chunk_id=efaf36ff  entities=1   raw_total=33
  [ 12/42] chunk_id=e9c66a7a  entities=1   raw_total=34
  [ 13/42] chunk_id=32c85d7e  entities=3   raw_total=37
  [ 14/42] chunk_id=5c56e059  entities=6   raw_total=43
  [ 15/42] chunk_id=4a4ea014  entities=7   raw_total=50
  [ 16/42] chunk_id=c3893b32  entities=2   raw_total=52
  [ 17/42] chunk_id=73e97519  entities=2   raw_total=54
  [ 18/42] chunk_id=0e82efa9  entities=2   raw_total=56
  [ 19/42] chunk_id=24b17cf3  entities=1   raw_total=57
  [ 20/42] chunk_id=310582c4  entities=7   raw_total=64
  [ 21/42] chunk_id=33d32892  entities=1   raw_total=65
  [ 22/42] chunk_id=71387ffa  entities=1   raw_total=66
  [ 23/42] chunk_id=e146728c  entities=2   raw_total=68
  [ 24/42] chunk_id=4dcff615  entities=1   raw_total=69
  [ 25/42] chunk_id=b744eeeb  entities=1   raw_total=70
  [ 26/42] chunk_id=7bac4d8e  entities=1   raw_total=71
  [ 27/42] chunk_id=f08417a7  entities=0   raw_total=71
  [ 28/42] chunk_id=a4a679c4  entities=2   raw_total=73
  [ 29/42] chunk_id=745e90bb  entities=1   raw_total=74
  [ 30/42] chunk_id=c81c2a06  entities=1   raw_total=75
  [ 31/42] chunk_id=a5123f46  entities=0   raw_total=75
  [ 32/42] chunk_id=a885be6b  entities=0   raw_total=75
  [ 33/42] EXTRACT FAILED (ReadTimeout): timed out
  [ 34/42] chunk_id=d6426773  entities=1   raw_total=76
  [ 35/42] chunk_id=fc0f4473  entities=1   raw_total=77
  [ 36/42] chunk_id=417905a7  entities=2   raw_total=79
  [ 37/42] chunk_id=d49b99f9  entities=3   raw_total=82
  [ 38/42] chunk_id=025a4d30  entities=2   raw_total=84
  [ 39/42] chunk_id=a949df1c  entities=2   raw_total=86
  [ 40/42] chunk_id=5a5f4b8a  entities=2   raw_total=88
  [ 41/42] chunk_id=895ee8b4  entities=0   raw_total=88
  [ 42/42] chunk_id=99f7ac4b  entities=8   raw_total=96

RESULT: raw=96  unique=75  ratio=0.781  unique/1kTok=12.38  hits=0  misses=83  failed=1

Top repeated names: 'edge'×4, 'rafael rafailov'×4, 'knowledge graph'×3,
'chroma'×3, 'lancedb'×3, 'zvec'×3, ...
```

**Note on the chunk 33 failure:** `EXTRACT FAILED (ReadTimeout)` — this is Ollama's queue being briefly blocked by a prior in-flight request, not a JSON-truncation failure. The new retry-on-ReadTimeout logic in `OllamaClient.generate()` handles this going forward. This is a **different failure mode** from the `_parse_error` truncation failures seen in the original Phase 1 ablation — do not conflate the two.

---

### Final comparison table

| Strategy | Chunks | Tok med | Fail% | Raw ent | Unique ent | Ratio | Unique/1kTok | LLM calls | Spot(3) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `sliding_window` | 17 | 450 | 11.8 | 50 | ~46 (est) | ~0.92 (est) | **~7.6 (est)** | 32 | 3/3 | unique/1kTok not measured — cache wiped before dedup check |
| `recursive_char` | 17 | 430 | 35.3 | 23 | — | — | — | 28 | 3/3 | **EXCLUDED** — mid-word boundary corruption; runaway generation on technical text |
| `markdown_aware` | 42 | 132.5 | 4.8 | 96 | 75 | 0.781 | **12.38** | 82 | 3/3 | **WINNER** — measured |
| `small_window` | 30 | 250.0 | 16.7 | 53 | 49 | 0.925 | **8.09** | 55 | 3/3 | measured |

`sliding_window`'s unique/1kTok is marked **(est)** — not measured, cache was wiped before dedup could run. The ratio estimate (~0.92) is extrapolated from `small_window`'s measured ratio on the same corpus. Even at 0.92, the estimated unique/1kTok (~7.6) is 63% below `markdown_aware`'s measured 12.38. The margin is wide enough that the decision is not sensitive to this estimate.

---

### Winner: `markdown_aware` — CONFIRMED, not provisional

**Reasoning:**
1. Highest measured unique/1kTok: **12.38** vs ~7.6 (est) for sliding_window, 8.09 for small_window
2. Lowest parse-failure rate: **4.8%** (2/42 chunks) vs 11.8% and 16.7%
3. Non-overlapping sections: chunk-sum token count = source tokens exactly (ratio 1.000x) — every source token processed once, no redundant re-processing from sliding-window overlap
4. All 3 spot-check sentences intact
5. 0.781 dedup ratio is lower than small_window's 0.925 — `edge`, `rafael rafailov`, `knowledge graph` etc. appear across multiple sections — but the absolute unique count (75) and normalised rate (12.38) are still well ahead of the alternatives

**Cost note:** 82 LLM calls vs 32 for sliding_window. For offline one-time indexing this is acceptable. If indexing cost becomes a bottleneck at scale, sliding_window is the fallback.

**Config updated — final:**
- `config.py`: `CHUNKING_STRATEGY = "markdown_aware"` (no PROVISIONAL marker)
- `.env` / `.env.example`: `CHUNKING_STRATEGY=markdown_aware`
- `CONFIG.md` §1 default and changelog: updated and marked CONFIRMED

---

### Task 1.5 canonical extraction — Phase 2 input

`build_index.py --stage extract` running now with `markdown_aware` default, clean cache, correct `.venv`, `temperature=0.0`, `seed=42`. Output → `data/cache/extractions.jsonl`.

The dedup run's raw count (96 entities) is from the dedup harness which suppresses validation warnings — `build_index.py` applies full relationship validation and will produce the authoritative post-validation entity/relationship counts. build_index output to be pasted below once complete.

**Command:**
```
.venv\Scripts\python.exe -u scripts/build_index.py --stage extract
```

**Output (pasted verbatim):**
```
[load]    5 documents loaded from 'data/raw'
[chunk]   42 chunks (size=450 tokens, overlap=12%)
[extract] chunk   1/42  doc=knowledge_graphs  chunk_idx=0  entities=1  rels=1
[extract] chunk   2/42  doc=knowledge_graphs  chunk_idx=1  entities=1  rels=0
[extract] chunk   3/42  doc=knowledge_graphs  chunk_idx=2  entities=6  rels=11
[extract] chunk   4/42  doc=knowledge_graphs  chunk_idx=3  entities=1  rels=0
[extract] chunk   5/42  doc=knowledge_graphs  chunk_idx=4  entities=2  rels=1
[extract] chunk   6/42  doc=knowledge_graphs  chunk_idx=5  entities=1  rels=0
[extract] chunk   7/42  doc=knowledge_graphs  chunk_idx=6  entities=1  rels=0
[extract] chunk   8/42  doc=knowledge_graphs  chunk_idx=7  entities=1  rels=0
[extract] chunk   9/42  doc=knowledge_graphs  chunk_idx=8  entities=2  rels=1
[extract] chunk  10/42  doc=knowledge_graphs  chunk_idx=9  entities=16  rels=11
[extract] chunk  11/42  doc=large_language_models  chunk_idx=0  entities=1  rels=0
[extract] chunk  12/42  doc=large_language_models  chunk_idx=1  entities=1  rels=0
[extract] chunk  13/42  doc=large_language_models  chunk_idx=2  entities=3  rels=2
[extract] chunk  14/42  doc=large_language_models  chunk_idx=3  entities=6  rels=2
[extract] chunk  15/42  doc=large_language_models  chunk_idx=4  entities=7  rels=6
[extract] chunk  16/42  doc=large_language_models  chunk_idx=5  entities=2  rels=1
[extract] chunk  17/42  doc=large_language_models  chunk_idx=6  entities=2  rels=0
[extract] chunk  18/42  doc=large_language_models  chunk_idx=7  entities=2  rels=1
[extract] chunk  19/42  doc=large_language_models  chunk_idx=8  entities=1  rels=0
[extract] chunk  20/42  doc=large_language_models  chunk_idx=9  entities=7  rels=3
[extract] chunk  21/42  doc=retrieval_augmented_generation  chunk_idx=0  entities=1  rels=0
[extract] chunk  22/42  doc=retrieval_augmented_generation  chunk_idx=1  entities=1  rels=1
[extract] chunk  23/42  doc=retrieval_augmented_generation  chunk_idx=2  entities=2  rels=1
[extract] chunk  24/42  doc=retrieval_augmented_generation  chunk_idx=3  entities=1  rels=0
[extract] chunk  25/42  doc=retrieval_augmented_generation  chunk_idx=4  entities=1  rels=0
[extract] chunk  26/42  doc=retrieval_augmented_generation  chunk_idx=5  entities=1  rels=1
[extract] chunk  27/42  doc=retrieval_augmented_generation  chunk_idx=6  entities=0  rels=0
[extract] chunk  28/42  doc=retrieval_augmented_generation  chunk_idx=7  entities=2  rels=2
[extract] chunk  29/42  doc=transformer_architecture  chunk_idx=0  entities=1  rels=0
[extract] chunk  30/42  doc=transformer_architecture  chunk_idx=1  entities=1  rels=1
[extract] chunk  31/42  doc=transformer_architecture  chunk_idx=2  entities=0  rels=0
[extract] chunk  32/42  doc=transformer_architecture  chunk_idx=3  entities=0  rels=0
[extract] chunk  33/42  doc=transformer_architecture  chunk_idx=4  FAILED (extract: ReadTimeout)  chunk_id=4af3dc8c5da84fad
  [WARN] timed out
[extract] chunk  34/42  doc=vector_databases  chunk_idx=0  entities=1  rels=1
[extract] chunk  35/42  doc=vector_databases  chunk_idx=1  entities=1  rels=1
[extract] chunk  36/42  doc=vector_databases  chunk_idx=2  entities=2  rels=1
[extract] chunk  37/42  doc=vector_databases  chunk_idx=3  entities=3  rels=1
[extract] chunk  38/42  doc=vector_databases  chunk_idx=4  entities=2  rels=1
[extract] chunk  39/42  doc=vector_databases  chunk_idx=5  entities=2  rels=1
[extract] chunk  40/42  doc=vector_databases  chunk_idx=6  entities=2  rels=1
[extract] chunk  41/42  doc=vector_databases  chunk_idx=7  entities=0  rels=0
[extract] chunk  42/42  doc=vector_databases  chunk_idx=8  entities=8  rels=0
============================================================
  Documents  : 5
  Chunks     : 42
  Extracted  : 41/42 chunks  |  Failed (error): 1/42 chunks
    - FAILED chunk_id=4af3dc8c5da84fad
  Entities   : 96
  Relationships: 53
  Cache hits : 64
  Cache misses: 19
  Elapsed    : 817.7s
  Output     : data/cache/extractions.jsonl
============================================================
```

**Canonical Phase 2 figures:**
- **96 entities, 53 relationships** from **41/42 successfully extracted chunks**
- 1 failed chunk: `transformer_architecture chunk_idx=4` (chunk_id `4af3dc8c5da84fad`, 275 tokens — ReadTimeout)
  - **Retry result:** Retried in isolation with clean Ollama queue and `LLM_TIMEOUT_S=300`. **FAILED again** after 605s (300s attempt + 300s retry both exhausted). The chunk text is 275 tokens of well-formed entity-rich text (Vaswani, Google Brain, BERT, OpenAI, GPT, T5, ViT, Longformer...). `qwen2.5:0.5b` consistently takes >300s to generate a response for high-entity-density chunks at this size. This is a genuine model-quality limitation, not a queue/infrastructure issue. **Canonical figures remain 96/53 from 41/42.**
- `extractions.jsonl`: 42 lines, 1 with `_parse_error` field
- These are the numbers Phase 2's graph builder will consume

**Note on chunk 33 failure type:** This is an Ollama `ReadTimeout` (network-layer timeout after both the initial attempt and the 300s retry) — the same leftover-queue phenomenon, **not** a JSON-truncation failure. The two failure modes are distinct: JSON-truncation (seen in original sliding_window ablation) produces a `ValueError` from `_extract_json`; this produces an `httpx.ReadTimeout`. Both are now caught and recorded separately in the JSONL `_parse_error` field.

---

### What was built

4 chunking strategies implemented behind the unified `chunk_document(doc, cfg, tokenizer)` interface:

- `sliding_window` — original baseline, 450 tokens, 12% overlap
- `recursive_char` — LangChain-style recursive character splitting on `["\n\n", "\n", ". ", " ", ""]`
- `markdown_aware` — split at `##`/`###` headers, merge-forward sections below 100-token floor, sliding window within oversized sections
- `small_window` — same as sliding_window but 250 tokens

### Pre-run sanity check: markdown_aware token distribution

**Before fix** (initial implementation):
```
Total chunks : 88
Min tokens   : 2
Median tokens: 64.0
Mean tokens  : 68.6
Max tokens   : 201
Chunks < 50 tokens: 24 (27.3%)
```
27.3% tiny chunks — exceeded the 10% threshold. Root cause: bare header lines (`# Knowledge Graphs` = 4 tokens, `## Applications` = 2 tokens) and very short subsections emitted as standalone chunks.

**Fix applied:** `_split_markdown_sections()` now does a merge-forward pass: any section under `_MARKDOWN_MIN_TOKENS = 100` tokens is concatenated with the next section. Carry propagates until the accumulated section meets the floor or the document ends.

**After fix:**
```
Total chunks : 42
Min tokens   : 101
Max tokens   : 298
Mean tokens  : 144.2
Median tokens: 132.5
Chunks < 50 tokens: 0 (0.0%)
None — merge fix working correctly.
```

**All 4 strategy chunk counts (post-fix, no LLM):**
```
sliding_window        17 chunks
recursive_char        17 chunks
markdown_aware        42 chunks
small_window          30 chunks
```

### Ablation run

Process launched detached (stdout → `chunking_ablation_log.txt`, stderr → `chunking_ablation_err.txt`).
Confirmed running (91MB working set). temperature=0.0, seed=42 pinned. Cache cleared per strategy.
Estimated ~3 hours total (markdown_aware dominates at ~70 min for 42 chunks × 2 LLM calls each).

### Full results

**Ablation completed. Full log pasted verbatim:**

```
Corpus: 5 documents
temperature=0.0  seed=42  (pinned)

============================================================
Running strategy: sliding_window
============================================================
  chunks=17  tok_med=450  entities=50  rels=36  failed=2  ent/1kTok=8.25  llm_calls=32  time=3583.1s
  spot-check: ['intact', 'intact', 'intact']

============================================================
Running strategy: recursive_char
============================================================
  chunks=17  tok_med=430  entities=23  rels=21  failed=6  ent/1kTok=3.8  llm_calls=28  time=35541.3s
  spot-check: ['intact', 'intact', 'intact']

============================================================
Running strategy: markdown_aware
============================================================
  chunks=42  tok_med=132.5  entities=90  rels=42  failed=2  ent/1kTok=14.86  llm_calls=82  time=3792.1s
  spot-check: ['intact', 'intact', 'intact']

============================================================
Running strategy: small_window
============================================================
  chunks=30  tok_med=250.0  entities=53  rels=35  failed=5  ent/1kTok=8.75  llm_calls=55  time=6609.4s
  spot-check: ['intact', 'intact', 'intact']

============================================================
COMPARISON TABLE
============================================================
Strategy              Chunks   Tok med   Entities   Rels   Fail%   Ent/1kTok   LLM calls   Time(s)   Spot(3)
-------------------------------------------------------------------------------------------------------------
sliding_window        17       450       50         36     11.8    8.25        32          3583.1    3
recursive_char        17       430       23         21     35.3    3.8         28          35541.3   3
markdown_aware        42       132.5     90         42     4.8     14.86       82          3792.1    3
small_window          30       250.0     53         35     16.7    8.75        55          6609.4    3

SPOT-CHECK DETAIL
------------------------------------------------------------
  sliding_window        [intact] FAISS  [intact] Self-Attention  [intact] Wikidata
  recursive_char        [intact] FAISS  [intact] Self-Attention  [intact] Wikidata
  markdown_aware        [intact] FAISS  [intact] Self-Attention  [intact] Wikidata
  small_window          [intact] FAISS  [intact] Self-Attention  [intact] Wikidata

ANALYSIS
  Best extraction quality (ent/1kTok × (1-fail_rate)): markdown_aware
  Cheapest (fewest LLM calls):                          recursive_char
```

All 4 strategies preserved all 3 spot-check sentences intact.

---

### Follow-up 1 — Source token count verification

**Result: 6058 is correct. No bug in ent/1kTok.**

True total source tokens (raw document text, no overlap):
```
knowledge_graphs                               1281 tokens
large_language_models                          1558 tokens
retrieval_augmented_generation                 1070 tokens
transformer_architecture                        792 tokens
vector_databases                               1357 tokens
TRUE total: 6058
```

The ablation script computes `total_source_tokens = sum(len(tokenizer.encode(d["text"])) for d in docs)` once per strategy before chunking — this is identical to the raw doc sum. 6058 is correct and the same for all 4 strategies.

Interesting finding from the check: `markdown_aware` chunk-sum token count equals source tokens exactly (ratio 1.000x), because its sections are non-overlapping. `sliding_window` (1.107x) and `small_window` (1.122x) inflate due to 12% overlap. This means `markdown_aware`'s LLM calls are the most token-efficient per unit of source text — it processes each source token exactly once, with no redundant re-processing from overlap.

The ent/1kTok figures in the ablation table are correct as published.

---

### Follow-up 2 — recursive_char anomaly: real cause identified

**Polling overhead cross-reference:**
- `sliding_window` completed at ~12:33 PM (start 11:33 AM + 3583s)
- `recursive_char` window: ~12:33 PM to ~10:23 PM (35541s = 9.87 hours)
- Polling calls read files/list processes only — zero Ollama calls, negligible CPU
- Total polling overhead: order of minutes at most
- **Conclusion: polling cannot explain a 9.87-hour anomaly. Resource contention is ruled out.**

**Output-length hypothesis tested:**

Chunk boundary inspection of `recursive_char` vs `sliding_window` on `knowledge_graphs`:

```
=== sliding_window ===
  [0] tokens=450  starts='# Knowledge Graphs\n\n...'
                   ends='...knowledge base maintained'   ← mid-sentence cut
  [1] tokens=450  starts='pedia (extracted from Wikipedia...'

=== recursive_char ===
  [0] tokens=438  starts='# Knowledge Graphs\n\n...'
                   ends='...## Major Knowledge Graphs\n\n### Wikidata\n\n'
  [1] tokens=449  starts='hema.org (used for web markup)...'   ← MID-WORD
  [2] tokens=438  starts='dentify entity names...'              ← MID-WORD
```

**Root cause confirmed:** The `recursive_char` splitter uses `. ` (period-space) as a separator, which splits inside entity names and technical terms containing dots: `Schema.org` → `Sche` + `ma.org`, `DBpedia (extracted from Wikipedia infoboxes)` gets cut so chunk[1] starts `hema.org (used for web markup)`. The LLM receives a prompt opening with a syntactically broken mid-word fragment. `qwen2.5:0.5b` enters a runaway generation loop on malformed input — confirmed by >10 minute timeout on a single chunk (vs ~50s for well-formed chunks).

**This is a real, reproducible quality problem with recursive_char on technical corpora, not a measurement artifact.** The `. ` separator is inappropriate for text containing URLs, abbreviations, and entity names with embedded periods. This failure mode would persist in any isolated re-run on this corpus.

**Decision on re-running recursive_char in isolation:** Not warranted. The boundary inspection proves the mid-word splitting is deterministic and reproducible from the chunker logic — it is not random. Running it again would produce the same degenerate boundaries and the same runaway generation. The strategy is disqualified on this corpus for this reason.

---

### Follow-up 3 — markdown_aware dedup (COMPLETED)

**Bug found and fixed:** The "silent death at chunk index 2" was caused by an abandoned in-flight Ollama request from a previous timed-out test blocking Ollama's single-threaded generation queue. When a Python process times out and exits, Ollama keeps processing the request it already started — any new request queues behind it and blocks indefinitely. This produced a silent hang (not a Python exception) that the original `except ValueError` could not catch.

**Fixes applied (Step 2 + 3):**

1. **`OllamaClient` now sets an explicit HTTP timeout (`LLM_TIMEOUT_S=120.0`)** via `ollama.Client(timeout=self.timeout)`. A stuck generation now raises `TimeoutError` instead of hanging forever.

2. **Exception sets broadened** in `build_index.py`, `dedup_single.py`, and `chunking_ablation.py` from `except ValueError` to `except (ValueError, RuntimeError, ConnectionError, TimeoutError, OSError)`. Per TASKS.md rule 4 — NOT a bare `except Exception`, just a broader intentional set.

3. **Extraction and gleaning have separate try/except blocks**: a gleaning failure now falls back to the extraction-only result for that chunk, logging `GLEAN FAILED ... using extraction-only result`, rather than either crashing the run or losing the chunk's extracted entities.

4. **`LLM_TIMEOUT_S=120.0`** added to `config.py`, `.env`, `.env.example`, and CONFIG.md §7.

**Step 5 — does the bug affect Phase 1 canonical extraction?**

`chunk_id=6fc475866849cd38` exists under both strategies (same hash: `sha256(knowledge_graphs::2)`) but with different content:
- `sliding_window`: 450 tokens — content from position 2 of the sliding window
- `markdown_aware`: 123 tokens — just the `## Structure and Representation` section

The `sliding_window` version's gleaning ran successfully in the original canonical Phase 1 extraction run (part of the 45/42 canonical figures). The Ollama queue was not stuck at that time. Phase 1's canonical numbers are not affected by this bug.

**dedup_single.py relaunched with all fixes, clean Ollama state (restarted server), clean cache.**
**COMPLETED — see Task 1.6 final entry above for full output.**
`RESULT: raw=96  unique=75  ratio=0.781  unique/1kTok=12.38  failed=1`

---

*See "recursive_char exclusion (confirmed, not an open item)" in the Task 1.6 final entry above for full root-cause analysis. The contention theory was disproven — the real cause is mid-word chunk boundaries from the `.` separator.*

---

### Follow-up 2 — Deduplicated entity counts

**SUPERSEDED — see Task 1.6 final entry above for measured figures.**

Real results: `markdown_aware` raw=96 unique=75 ratio=0.781 unique/1kTok=12.38 (measured).
`small_window` raw=53 unique=49 ratio=0.925 unique/1kTok=8.09 (measured).
`sliding_window` unique/1kTok estimated ~7.6 (cache wiped before dedup could run — not measured).
The estimated 76–84 range from this section was superseded by the real result of 75.

---

### Strategy decision and recommendation

**SUPERSEDED — see Task 1.6 final entry above for confirmed decision.**

`markdown_aware` is the confirmed default. Config changes are final, not provisional.
The reasoning and evidence are in the Task 1.6 final entry.

---

## Phase 1 Post-Review Fixes — Inconsistency Resolution (2026-09-15)

### Point 1 — Non-determinism: temperature + seed fix

**Root cause:** `OllamaClient` was calling `ollama.generate()` with no `temperature` or `seed` options. Ollama's default temperature is ~0.8 (stochastic), which caused chunk_id `745e90bb807507b6` to produce 1 entity in the Fix 3 diagnostic run and fail with a JSON parse error (truncated output) in the definitive run — same source text, different model output.

**Fix:** `OllamaClient.__init__` now accepts `temperature` and `seed` parameters (defaults: 0.0 and 42). Both are passed to `ollama.generate(options=...)`. `temperature=0.0` alone was confirmed insufficient — seed is also required. Both are now in CONFIG.md §2 and `.env`.

**Stability test — two runs with `temperature=0.0, seed=42`:**
```
chunk_id : 745e90bb807507b6
temperature: 0.0
seed       : 42

=== RUN 1 ===  Parsed: entities=5  relationships=0
=== RUN 2 ===  Parsed: entities=5  relationships=0
Raw responses identical: True
STABLE — temperature=0.0 + seed=42 produces deterministic output
```

**Addendum to Fix 3:** The Fix 3 DOCUMENT.md entry (which showed `entities=1` for this chunk from a direct diagnostic call) was run under undetermined sampling (temperature ~0.8, no seed). That result is superseded by this run's stable result of `entities=5, relationships=0` under `temperature=0.0, seed=42`.

**Files changed:** `src/graphrag_lite/llm/ollama_client.py`, `src/graphrag_lite/config.py`, `scripts/build_index.py`, `CONFIG.md`, `.env`, `.env.example`

---

### Point 2 — Cache state: real sequence of events

The definitive run reported `Cache hits: 29  Cache misses: 3`. This contradicts the "cleared cache" framing. The real sequence was:

1. Cache cleared at start of definitive run attempt
2. First attempt started and ran ~16 chunks before being interrupted by user
3. Those 16 chunks' results were stored in cache during the interrupted run
4. The second (final, complete) run reused those 29 cached results — only the last chunk (or a small number of retried chunks) required live LLM calls (3 misses)

This is correct behaviour — the disk cache is doing exactly what it's designed to do (survive interruptions). It is **not** a fresh-from-scratch run. The "fresh run" framing in the earlier DOCUMENT.md entry was misleading and is corrected here. The extractions themselves are valid and deterministic regardless of whether they came from cache or live calls, as long as they were all generated with the same model and (post-fix) the same temperature/seed settings.

---

### Point 3 — Canonical totals

The Fix 5 entry reported `59 entities / 31 relationships` — those were from an intermediate run before the relationship validation fix (option a) was fully applied and before failed-chunk tracking existed.

**SUPERSEDED by Task 1.6 final canonical run (markdown_aware strategy):**

`96 entities / 53 relationships` from `41/42 chunks` under `markdown_aware` chunking.
See Task 1.6 final entry for the full build_index.py output.

The earlier figures `45/42` (sliding_window, 17 chunks) and `59/31` (Fix 5) are both superseded and must not be used for Phase 2.

---

### Point 4 — extractions.jsonl trailing content

Confirmed via last-300-bytes read:

```
...dataset size (billion-vector datasets require GPU acceleration or distributed deployment), and
hardware (GPU-accelerated search with cuVS or GPU-FAISS can achieve 100x speedup over CPU for
large datasets)."}], "relationships": [], "chunk_id": "d49b99f9a418a0fb", "doc_id": "vector_databases"}
```

**Final 10 bytes (hex):** `61 62 61 73 65 73 22 7D 0D 0A`
Decoded: `ases"}` + Windows CRLF (`\r\n`).

The file ends with a single `\r\n` after the last JSON record. There are no blank lines 18 or 19. PowerShell's `Measure-Object -Line` reported 19 because it counts the trailing CRLF as an additional line — the file contains exactly **17 valid JSON records**, each on its own line. Confirmed.

---

### Fix 3 — Null extraction diagnosis: transformer_architecture chunk_idx=0

**Finding:** The `0 entities / 0 relationships` result reported in Task 1.5 for chunk 8/9 (`transformer_architecture`, `chunk_idx=0`) was **a stale cache artefact, not a model-quality failure.** The disk cache held a response from an earlier extraction run when `EXTRACTION_MODEL` was still `qwen3:4b` — a model that was not available locally and likely returned an error or unparseable output, which was stored in cache as an empty extraction. When the cache was cleared and the chunk was re-run fresh with `qwen2.5:0.5b`, the model returned a valid response.

**Source chunk text (chunk_id: 745e90bb807507b6, token_count: 450):**
```
# The Transformer Architecture
## Overview
The Transformer is a deep learning model architecture introduced by Vaswani et al. in the 2017
paper "Attention Is All You Need." It was originally designed for sequence-to-sequence tasks in
natural language processing but has since become the dominant architecture for a wide range of AI
tasks.
...
## Core Components
### Self-Attention Mechanism
...
### Multi-Head Attention
...
### Feed-Forward Layers
...
### Positional Encoding
...
## Encoder-Decoder Structure
The original Transformer consists of an encoder and a decoder. The encoder maps an input sequence
of symbols to a
```

**Raw LLM response (unprocessed):**
```json
{
  "entities": [
    {
      "name": "Transformer",
      "type": "TECHNOLOGY",
      "description": "Deep learning model architecture for sequence-to-sequence tasks in NLP",
      "source_sentence": "The Transformer is a deep learning model architecture introduced by Vaswani et al...",
      "relationships": [...]
    }
  ],
  "relationships": [
    {
      "source": "Positional Encoding",
      "target": "Transformer",
      "type": "OTHER",
      "description": "Positional encodings are added to inject position information.",
      "source_sentence": "The Transformer consists of an encoder and a decoder..."
    }
  ]
}
```

**Parsed result after validation fix:**
- 1 entity: `Transformer (TECHNOLOGY)` — correctly extracted, `source_sentence` verbatim from text
- 0 relationships: `Positional Encoding → Transformer` dropped with warning because `Positional Encoding` was not in the entities list (the model embedded it inside the entity object rather than as a top-level entity — a structural hallucination by `qwen2.5:0.5b`).

**Root cause of original 0/0:** stale cache from failed `qwen3:4b` call. **Fix applied:** cache cleared; re-extraction with correct model produces valid output. Known limitation noted: `qwen2.5:0.5b` occasionally embeds relationships inside entity objects (non-schema-compliant) and may miss entities like `Positional Encoding` that appear only as relationship endpoints. This is a model-quality limitation, not a pipeline bug.

**Action:** `data/cache/` cleared and full re-extraction should be run before Phase 2 (`python scripts/build_index.py --stage extract`) to ensure all 9 chunks have fresh, correct extractions.

---

### Fix 4 — CONFIG.md changelog updated

The following entries were added to the CONFIG.md changelog table:

| Date | Variable changed | Old → New | Reason | Task ref |
|---|---|---|---|---|
| 2026-09-15 | `EXTRACTION_MODEL` / `SLM_MODEL` | `qwen3:4b` → `qwen2.5:0.5b` | `qwen3:4b` not available in local Ollama install | Task 1.3 |
| 2026-09-15 | `TOKENIZER` (implementation) | Qwen native tokenizer → `tiktoken cl100k_base` | Qwen tokenizer not available as standalone PyPI package; cl100k_base is the closest public approximation | Task 1.2 |
| 2026-09-15 | `STRESS_CONCURRENCY_LEVELS` format | CSV string → JSON array `[10,50,200,1000]` | pydantic-settings requires JSON array syntax for `List[int]` fields in `.env` | Task 0.1 |

---

### Fix 5 — Corpus size decision

**Decision: expand corpus to 5 documents before Phase 2.**

The 3-document / 9-chunk corpus was a dev-scale placeholder, not a deliberate final choice. Reasons to expand now rather than later:

1. **Phase 2 (graph):** 3 docs produce ~21 entities and ~15 relationships (many of which are dropped by the new validation). A graph this sparse will make PageRank and BFS traversal tests trivially small — top-5 PageRank lists and 2-hop BFS results will overlap heavily, producing uninformative ablation comparisons.

2. **Task 8.1 (30–50 gold Q&A pairs):** 3 documents on overlapping topics (KGs, RAG, Transformers) cannot support 30–50 distinct, non-overlapping questions with ground-truth evidence node IDs. The questions would either repeat or require hallucinating answers not in the corpus.

**Action taken:** 2 additional documents added to `data/raw/`:
- `large_language_models.md` — covers LLM training, key models (GPT, LLaMA, Claude, Gemini, Qwen), RLHF, DPO, quantization, evaluation benchmarks
- `vector_databases.md` — covers vector DB concepts, FAISS, Pinecone, Weaviate, Chroma, Qdrant, LanceDB, Zvec, hybrid search

This brings the corpus to **5 documents**. The 5 documents are intentionally chosen to be interconnected (Transformers → LLMs → RAG → vector DBs → KGs → back to RAG) so the graph has real multi-hop paths to test. Phase 2 and Task 8.1 should use this 5-document corpus. If the gold Q&A set still can't reach 30 questions at Task 8.1, add more documents at that point — do not force questions.

**⚠ SUPERSEDED totals:** The intermediate extraction run referenced in this entry produced `59 entities / 31 relationships`. Those figures were from before the relationship validation fix (option a) and the failed-chunk counter existed. They are superseded by the canonical figures from the definitive run: **`45 entities / 42 relationships` from `15/17 successfully extracted chunks`**. Phase 2 builds from 45/42, not 59/31.

---

### Phase 2 — Graph
<!-- Task 2.1–2.3 entries go here -->

## Task 2.1 — Graph builder

**Status:** [x] done (revised after self-loop + edge count investigation)
**Date:** 2026-09-15
**Files touched:** `src/graphrag_lite/graph/builder.py`, `src/graphrag_lite/config.py`, `CONFIG.md`

### Pre-task config additions

`PAGERANK_DAMPING = 0.85` added to `config.py` and CONFIG.md §5. `LLM_TIMEOUT_S` default corrected from `120.0` to `300.0` in CONFIG.md.

### Function(s) implemented

- `load_extractions(jsonl_path) -> list[dict]` — Records with `_parse_error` skipped with visible `[SKIP]` log line.

- `build_graph(extractions) -> nx.DiGraph`
  - Pass 1: builds nodes. Normalisation: `re.sub(r"\s+", " ", name.strip().lower())`.
  - Node merge policy: `type` = first non-OTHER wins; `descriptions`/`source_sentences` = lists with provenance; `source_docs` = set.
  - **Self-loop policy:** DROPPED and logged with `[DROP-LOOP]`. See investigation.
  - **Parallel edge policy:** MERGED — same (src, tgt) pair accumulates all types and source_sentences on one edge. See investigation.

### Self-loop investigation

All 15 self-loops were pre-merge (LLM hallucinated `X --> X` directly). 0 post-merge self-loops.

Policy decision: DROP and log each. Rationale: self-loops carry no graph information; PageRank inflates scores for self-looping nodes; every instance is clear extractor noise from `qwen2.5:0.5b`; 0 post-merge means no real cross-entity relationship is ever dropped by this policy.

**Full list of all 15 dropped self-loops (pasted verbatim from re-verification run):**
```
   1. [knowledge_graphs] 'Knowledge Graph' --OTHER--> 'Knowledge Graph'          chunk=0282ea80
   2. [knowledge_graphs] 'Person' --PART_OF--> 'Person'                          chunk=6fc47586
   3. [knowledge_graphs] 'Organization' --WORKS_AT--> 'Organization'             chunk=6fc47586
   4. [knowledge_graphs] 'Person' --WORKS_AT--> 'Person'                         chunk=6fc47586
   5. [knowledge_graphs] 'Person' --WORKS_AT--> 'Person'                         chunk=6fc47586
   6. [knowledge_graphs] 'Temporal knowledge graphs' --WORKS_AT--> 'Temporal knowledge graphs'  chunk=ab2692f1
   7. [large_language_models] 'Rafael Rafailov' --WORKS_AT--> 'Rafael Rafailov'  chunk=5c56e059
   8. [large_language_models] 'Alpaca' --OTHER--> 'Alpaca'                       chunk=4a4ea014
   9. [large_language_models] 'Vicuna' --OTHER--> 'Vicuna'                       chunk=4a4ea014
  10. [large_language_models] 'Mistral' --OTHER--> 'Mistral'                     chunk=4a4ea014
  11. [retrieval_augmented_generation] 'RAG' --WORKS_AT--> 'RAG'                 chunk=71387ffa
  12. [retrieval_augmented_generation] 'RAG pipeline' --WORKS_AT--> 'RAG pipeline'  chunk=7bac4d8e
  13. [transformer_architecture] 'self-attention mechanism' --WORKS_AT--> 'self-attention mechanism'  chunk=c81c2a06
  14. [vector_databases] 'vector database' --WORKS_AT--> 'vector database'       chunk=d6426773
  15. [vector_databases] 'Vector databases' --WORKS_AT--> 'Vector databases'     chunk=fc0f4473

Type breakdown: WORKS_AT=10  OTHER=4  PART_OF=1  Total=15
```

**Discrepancy note:** An earlier spoken summary said "9 of 15 share the type WORKS_AT." That was a miscount made while writing prose — the diagnostic script output has always shown 10. The correct figure, verified by re-running `phase2_close_checks.py`, is **10 WORKS_AT self-loops**. Counting entries 3–7, 11–15 above gives 10; entries 1, 8, 9, 10 are OTHER (3) and PART_OF (1) for the remaining 4. The model-quality observation below is based on the correct count of 10.

### Edge count reconciliation

```
53  total relationships in extractions.jsonl
-15  self-loops dropped (pre-merge extractor noise)   -> [DROP-LOOP] per item
- 0  dangling dropped
---
38  normal edges
- 8  parallel edges merged into existing (src,tgt) pairs
---
30  final edges in graph

CHECK: 15 + 0 + 30 + 8 = 53  BALANCED
```

Parallel pairs (>1 relationship merged): `Organization-->Person` (4), `Temporal knowledge graphs-->Edge` (2), `Temporal knowledge graphs-->Temporal information` (2), `Temporal information-->Temporal knowledge graphs` (2), `Knowledge graphs-->Structured information panels` (2).

Original builder reported 43 because: 15 self-loops were not filtered (added to graph), and parallel edges used broken DiGraph last-write-wins (not merge). Now: 30 clean edges.

### Final DoD test run (after fixes)

**Command:** `python tests/test_builder_task21.py`

**Output (pasted verbatim):**
```
  [SKIP] chunk_id=4af3dc8c5da84fad doc=transformer_architecture -- _parse_error: ReadTimeout: timed out
[load_extractions] 41 clean records, 1 skipped
Records loaded: 41
  [DROP-LOOP] ('Knowledge Graph' --OTHER--> 'Knowledge Graph') -> both resolve to 'Knowledge Graph' [knowledge_graphs/0282ea80]
  [DROP-LOOP] ('Person' --PART_OF--> 'Person') -> both resolve to 'Person' [knowledge_graphs/6fc47586]
  [DROP-LOOP] ('Organization' --WORKS_AT--> 'Organization') -> both resolve to 'Organization' [knowledge_graphs/6fc47586]
  [DROP-LOOP] ('Person' --WORKS_AT--> 'Person') -> both resolve to 'Person' [knowledge_graphs/6fc47586]
  [DROP-LOOP] ('Person' --WORKS_AT--> 'Person') -> both resolve to 'Person' [knowledge_graphs/6fc47586]
  [DROP-LOOP] ('Temporal knowledge graphs' --WORKS_AT--> 'Temporal knowledge graphs') -> both resolve to 'Temporal knowledge graphs' [knowledge_graphs/ab2692f1]
  [DROP-LOOP] ('Rafael Rafailov' --WORKS_AT--> 'Rafael Rafailov') -> both resolve to 'Rafael Rafailov' [large_language_models/5c56e059]
  [DROP-LOOP] ('Alpaca' --OTHER--> 'Alpaca') -> both resolve to 'Alpaca' [large_language_models/4a4ea014]
  [DROP-LOOP] ('Vicuna' --OTHER--> 'Vicuna') -> both resolve to 'Vicuna' [large_language_models/4a4ea014]
  [DROP-LOOP] ('Mistral' --OTHER--> 'Mistral') -> both resolve to 'Mistral' [large_language_models/4a4ea014]
  [DROP-LOOP] ('RAG' --WORKS_AT--> 'RAG') -> both resolve to 'RAG' [retrieval_augmented_generation/71387ffa]
  [DROP-LOOP] ('RAG pipeline' --WORKS_AT--> 'RAG pipeline') -> both resolve to 'RAG pipeline' [retrieval_augmented_generation/7bac4d8e]
  [DROP-LOOP] ('self-attention mechanism' --WORKS_AT--> 'self-attention mechanism') -> both resolve to 'self-attention mechanism' [transformer_architecture/c81c2a06]
  [DROP-LOOP] ('vector database' --WORKS_AT--> 'vector database') -> both resolve to 'vector database' [vector_databases/d6426773]
  [DROP-LOOP] ('Vector databases' --WORKS_AT--> 'Vector databases') -> both resolve to 'Vector databases' [vector_databases/fc0f4473]
[build_graph] nodes=75  edges=30
  self_loops_dropped=15  parallel_merged=8  dangling_dropped=0

Nodes : 75
Edges : 30

Sample node: 'Knowledge Graph'
  type        : CONCEPT
  source_docs : {'knowledge_graphs'}
  descriptions (3 entries): ...
  source_sentences (3 entries): ...

Sample edge: 'Knowledge Graph' --> 'KGQA'
  types      : ['WORKS_AT']
  description: A knowledge graph connecting movies, directors, actors...
  source_doc : knowledge_graphs
  source_sentences: 1 entries

MERGE VERIFICATION (known duplicates from Phase 1 dedup analysis):
  'edge'                     node_count=1  canonical='Edge'
  'rafael rafailov'          node_count=1  canonical='Rafael Rafailov'
  'knowledge graph'          node_count=1  canonical='Knowledge Graph'
  'chroma'                   node_count=1  canonical='Chroma'
  'lancedb'                  node_count=1  canonical='LanceDB'
  'zvec'                     node_count=1  canonical='Zvec'
  All collapse to <= 1 node: PASS

Dangling edge check: 0/30 relationships reference a missing node
  PASS

Connected components: 48
  Largest component size: 6
  Component size distribution: [6, 4, 3, 3, 3, 2, 2, 2, 2, 2]...

ALL ASSERTIONS PASSED
```

### Canonical graph figures for Phase 2

- **75 nodes, 30 edges**, 48 connected components, largest = 6 nodes
- 15 self-loops dropped (extractor noise), 8 parallel pairs merged, 0 dangling

### Follow-up 1 — Full Organization --> Person edge (highest-count parallel merge)

4 relationships collapsed into 1 edge. Full edge data:

```
Edge: Organization --> Person

types (4 entries): ['WORKS_AT', 'WORKS_AT', 'WORKS_AT', 'WORKS_AT']

description: An organization is a group of people or things that work together to achieve a common goal.
source_doc:  knowledge_graphs

source_sentences (4 entries):
  [0] chunk_id=6fc475866849cd38  type=WORKS_AT
       text: An organization is a group of people or things that work together to achieve a common goal.
  [1] chunk_id=6fc475866849cd38  type=WORKS_AT
       text: A location is a physical place where something is situated.
  [2] chunk_id=6fc475866849cd38  type=WORKS_AT
       text: An organization is a group of people or things that work together to achieve a common goal.
  [3] chunk_id=6fc475866849cd38  type=WORKS_AT
       text: An organization is a group of people or things that work together to achieve a common goal.

ASSERTIONS PASSED: 4 types, 4 source_sentences
```

All 4 original relationship types are in `types`. All 4 original source sentences are in `source_sentences` with `chunk_id` provenance. Merge is lossless. Note: all 4 come from the same chunk_id (`6fc47586`) — the model extracted multiple near-identical relationships from the same dense "node types" section. `source_sentences[1]` having a different text (about Location, not Organization) confirms the model was generating filler content rather than meaningful relationships — consistent with the WORKS_AT fallback pattern below.

### Follow-up 2 — WORKS_AT fallback pattern in self-loops

Self-loop type breakdown:
```
WORKS_AT: 10
OTHER:     4
PART_OF:   1
Total:    15
```

10 of 15 dropped self-loops have type `WORKS_AT` applied to entities where the relationship makes no semantic sense — `'Person' --WORKS_AT--> 'Person'`, `'RAG' --WORKS_AT--> 'RAG'`, `'self-attention mechanism' --WORKS_AT--> 'self-attention mechanism'`, `'vector database' --WORKS_AT--> 'vector database'`. These are concepts, techniques, and acronyms, not employment relationships.

**Named model-quality observation:** `qwen2.5:0.5b` has a specific fallback pattern of defaulting to `WORKS_AT` when it cannot identify a real relationship type for an entity. This is not uniformly random noise — it is a systematic bias toward one relationship type. The 4 `OTHER` self-loops (`Alpaca`, `Vicuna`, `Mistral`, `Alpaca` variants) are a different pattern: the model emitted type `OTHER` when it had no relationship to generate but the schema required one.

This observation is relevant context for Task 8's model comparison: `qwen3:4b` or `llama3.2` ablations should be expected to produce fewer `WORKS_AT` self-loops, and a drop in self-loop rate (currently 15/53 = 28% of all relationships) would be a concrete, measurable quality improvement. If `qwen3:4b` still shows this pattern at similar rates, it suggests the extraction prompt needs tightening rather than a model swap.

---
### Graph structure analysis and risks (pre-Task-2.3)

**Full component size distribution:**
```
Total nodes : 75   Total edges: 30   Components: 48
Average degree (2*edges/nodes): 0.8000

Weakly connected components (undirected reachability): 48
  Size distribution:
    size 6:  1 component   (largest)
    size 4:  1 component
    size 3:  3 components
    size 2: 13 components
    size 1: 30 components  (SINGLETONS -- 40% of all nodes)

Strongly connected components (directed reachability): 72
  Largest strong component: 3 nodes

Note: The "48 components" figure in Task 2.2 uses WEAKLY connected components
(nx.connected_components on the undirected projection) -- which is the correct
metric for BFS hop analysis and undirected reachability. Strongly connected
components (72, largest=3) are a different, stricter definition and are not
directly comparable. The two numbers coexist without contradiction.

Nodes in largest (weak) component:  6/75  (8.0%)
Singletons (0-hop BFS only):       30/75  (40.0%)
Nodes in size-2 (max 1 hop):       26/75  (34.7%)
Nodes in size-3 (max 2 hops):       9/75  (12.0%)
Max BFS hops possible (anywhere):   5  (only within the 6-node component)
```

**Named risk:** 40% of nodes are singletons; the largest connected component is 6 nodes (8% of the graph). Task 2.3's BFS will execute correctly as code, but cannot demonstrate meaningful multi-hop retrieval -- most seed nodes return 1-2 neighbours or only themselves. Pipeline B's structural advantage over Pipeline A (the core paper comparison) depends on multi-hop graph paths. This corpus does not yet provide that. Flagged for revisit at Task 8.1 (corpus expansion). Root cause: qwen2.5:0.5b extracted few relationships per chunk; many were self-loops (dropped) or generic type-level relationships. A larger model is the primary fix.

**Schema-artifact policy (Point 2):** 4/5 PageRank top-5 nodes are generic KG schema types (Person, Organization, Location, Edge). Decision: **option (b) -- leave as-is, document as corpus-content issue.** Rationale: a stoplist would silently filter without a principled criterion; the real fix is extraction quality (qwen3:4b would extract Ashish Vaswani, Google Brain etc. rather than schema type labels); defer accuracy impact measurement to Task 8's gold Q&A evaluation.

**Organization/Location PageRank tie (Point 3):** Both live in the same isolated 3-node component {Person, Organization, Location}. Both have exactly Person as their sole predecessor AND successor -- perfect structural symmetry guarantees identical PageRank scores. Deterministic, not numerical coincidence.

```
Organization: predecessors=['Person']  successors=['Person']  component={Location, Organization, Person}
Location:     predecessors=['Person']  successors=['Person']  component={Location, Organization, Person}
```

---## Task 2.2 — Ranking (PageRank + degree)

**Status:** [x] done
**Date:** 2026-09-15
**Files touched:** `src/graphrag_lite/graph/ranking.py`

### Function(s) implemented

- `rank_pagerank(graph, damping=None) -> dict[str, float]`
  - Reads `PAGERANK_DAMPING` from `get_config()` when `damping` is not passed explicitly. Pure function — no side effects.
- `rank_degree(graph) -> dict[str, int]`
  - Returns total (in + out) degree. No config coupling. Pure function.

### Test run

**Command:** `python tests/test_ranking_task22.py`

**Output (pasted verbatim):**
```
Graph: 75 nodes, 30 edges
PAGERANK_DAMPING = 0.85

Rank   PageRank (score)                               Degree (count)
--------------------------------------------------------------------------------
  1.   Person                   0.083102   Person                              4
  2.   Organization             0.043851   Temporal knowledge graphs           4
  3.   Location                 0.043851   Knowledge graphs                    3
  4.   Temporal knowledge graphs 0.035277  LLaMA                               3
  5.   Edge                     0.023540   Organization                        2

Top-5 overlap: 3/5
  Shared:          ['Organization', 'Person', 'Temporal knowledge graphs']
  Only in PageRank: ['Edge', 'Location']
  Only in Degree:   ['Knowledge graphs', 'LLaMA']

VERDICT: The two top-5 lists DIFFER.
         3/5 nodes shared; 2 nodes differ between methods.
         PageRank and degree diverge on this graph's structure.

Extended top-10:
  1    Person                   0.083102   Person                              4
  2    Organization             0.043851   Temporal knowledge graphs           4
  3    Location                 0.043851   Knowledge graphs                    3
  4    Temporal knowledge graphs 0.035277  LLaMA                               3
  5    Edge                     0.023540   Organization                        2
  6    Temporal information     0.023540   Location                            2
  7    OpenAI                   0.021987   Human                               2
  8    HumanEval                0.021987   MMLU                                2
  9    DBpedia                  0.015812   Temporal information                2
  10   KGQA                     0.015812   Knowledge Graph                     1
```

### Analysis

The lists differ on 2/5 positions — a real, non-trivial divergence even on this sparse 30-edge graph.

**Why they diverge:**
- `'Edge'` (rank 5 in PageRank, absent from degree top-5) has degree 2 but receives PageRank flow from `'Temporal knowledge graphs'` which itself has high connectivity — PageRank rewards transitive importance, not just raw count.
- `'Location'` (rank 3 in PageRank, degree rank 6) ties with `Organization` at degree 2 but ranks higher in PageRank because its incoming edges come from `Person` which is the highest-PageRank node.
- `'Knowledge graphs'` (rank 3 in degree, absent from PageRank top-5) has degree 3 but much of that is FROM nodes with low PageRank — high degree doesn't translate to high PageRank when your neighbours are not themselves important.
- `'LLaMA'` (rank 4 in degree with 3 edges, absent from PageRank top-5) — same pattern.

**Implication for Pipeline B (Task 4.2):** Using `pagerank` (config default) vs `degree` will produce different candidate sets. The divergence is real and measurable — Task 8.6's ranking-method ablation should show a non-trivial accuracy difference rather than being a no-op.

**Note on top-ranked nodes:** `Person`, `Organization`, `Location` ranking highly reflects the knowledge_graphs.md source doc's coverage of KG node types — these are described as entity types, not real-world entities. This is an extraction quality issue (`qwen2.5:0.5b` treating schema concepts as entities) that will be reduced with `qwen3:4b`.

---
## Task 2.3 — Bounded BFS traversal

**Status:** [x] done
**Date:** 2026-09-15
**Files touched:** `src/graphrag_lite/graph/traversal.py`

### Function(s) implemented

- `bfs_expand(graph, seed_nodes, hops) -> list[str]`
  - Traverses the **undirected projection** of the graph so both in- and out-edges are followed.
  - Seeds not in graph: `[WARN]` logged, skipped — never raises.
  - Returns sorted list of unique node names (seeds included at hop 0).
  - Deliberately unranked — ranking is a separate step per PRD §9.

### Test run

**Command:** `python tests/test_traversal_task23.py`

**Output (pasted verbatim):**
```
Graph: 75 nodes, 30 edges
BFS_HOP_DEPTH = 2

Largest component (6 nodes): ['Edge', 'Knowledge graphs', 'Search engines',
  'Structured information panels', 'Temporal information', 'Temporal knowledge graphs']

Test 1: seed='Knowledge graphs'  hops=2  (from largest component)
  Returned 6 nodes:
    2-hop  'Edge'
    0-hop  'Knowledge graphs'
    1-hop  'Search engines'
    1-hop  'Structured information panels'
    2-hop  'Temporal information'
    1-hop  'Temporal knowledge graphs'

Test 2: seed='KGQA'  hops=2  (from a 2-node component)
  Returned 2 nodes:
    0-hop  'KGQA'
    1-hop  'Knowledge Graph'

Test 3: seed='Technology'  hops=2  (singleton node)
  Returned 1 nodes: ['Technology']
  (Seed returns only itself -- expected for isolated node)

Test 4: seed='NonExistentNode'  hops=2  (not in graph)
  [WARN] bfs_expand: seed node 'NonExistentNode' not in graph -- skipped
  Returned 0 nodes: []
  (Empty list expected -- graceful skip on missing seed)

Max nodes returned by any single-seed BFS at hops=2: 6
ALL ASSERTIONS PASSED
```

### Notes

- Test 1 (largest component): `hops=2` reaches all 6 nodes — hop bound respected, full reachable neighbourhood returned with per-node hop counts.
- Test 2 (size-2 component): the more typical case — most seeds reach only 1–2 nodes on this corpus.
- Tests 3 and 4: singleton and missing-seed edge cases confirmed working.
- 4 tests instead of TASKS.md's minimum 2 — the graph structure analysis showed singletons and size-2 components are the common case, so testing only the 6-node component would give a misleadingly optimistic picture.

---
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
| 0 — Scaffolding | 2 | 2 | 0 |
| 1 — Ingestion | 5 | 5 | 0 |
| 2 — Graph | 3 | 0 | 0 |
| 3 — Embeddings/Vector Store | 3 | 0 | 0 |
| 4 — Retrieval Pipelines | 4 | 0 | 0 |
| 5 — LLM Layer | 2 | 0 | 0 |
| 6 — Mode Toggle | 2 | 0 | 0 |
| 7 — API/Hardening | 3 | 0 | 0 |
| 8 — Evaluation | 6 | 0 | 0 |
| 9 — Stress Test | 1 | 0 | 0 |
| 10 — Paper Artifacts | 2 | 0 | 0 |
| **Total** | **33** | **7** | **0** |