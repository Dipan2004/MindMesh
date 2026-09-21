# ARCHITECTURE.md — Lightweight Graph-RAG (Zvec + Local SLM)

Companion to `PRD.md`. This is the system's shape — components, data flow,
and file layout — that `TASKS.md` builds piece by piece. Nothing here is a
placeholder; every box below maps to a real file in `src/graphrag_lite/`.

---

## 1. Component Diagram

```
                              ┌─────────────────────────────┐
                              │        INDEXING (offline)    │
                              │                               │
  data/raw/*.txt/.md ───────▶│  loader.py                    │
                              │      │                        │
                              │      ▼                        │
                              │  chunker.py (token-based)     │
                              │      │                        │
                              │      ▼                        │
                              │  extractor.py ◀── llm_cache.py│
                              │  (LLM extract + gleaning loop)│
                              │      │                        │
                              │      ▼                        │
                              │  builder.py (NetworkX graph)  │
                              │      │                        │
                              │      ▼                        │
                              │  embedder.py (batch embed)    │
                              │      │                        │
                              │      ▼                        │
                              │  zvec_client.py (insert)       │
                              └──────────────┬────────────────┘
                                             ▼
                                    ┌──────────────────┐
                                    │   Zvec store      │
                                    │  (entities coll.  │
                                    │   + chunks coll.) │
                                    └─────────┬─────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    ▼                                                 ▼
        ┌───────────────────────┐                       ┌───────────────────────┐
        │   PIPELINE A (base)    │                       │   PIPELINE B (proposed)│
        │                        │                       │                        │
        │ query embed            │                       │ query embed            │
        │  → zvec flat search    │                       │  → zvec entry search   │
        │  → top-N chunks        │                       │  → traversal.py BFS    │
        │                        │                       │  → ranking.py PPR/deg  │
        │                        │                       │  → zvec re-query       │
        │                        │                       │  → top-N evidence      │
        └───────────┬────────────┘                       └───────────┬────────────┘
                    │                                                 │
                    └────────────────────┬────────────────────────────┘
                                         ▼
                              serialize.py (TOON / JSON)
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │  client_factory.py             │
                         │  mode = "local" → Ollama SLM   │
                         │  mode = "api"   → Grok          │
                         └───────────────┬─────────────────┘
                                         ▼
                              generate.py (answer + citations)
                                         │
                          ┌──────────────┴───────────────┐
                          ▼                               ▼
              tracing.py (LangSmith,             stress_test.py logger
              local mode only)                    (api mode only)
```

## 2. Layers and Responsibilities

| Layer | Directory | Responsibility | Depends on |
|---|---|---|---|
| Ingestion | `src/graphrag_lite/ingestion/` | Load docs, chunk, extract entities/relationships | `cache/` |
| Cache | `src/graphrag_lite/cache/` | Memoize every LLM call by prompt hash | none |
| Graph | `src/graphrag_lite/graph/` | Build NetworkX graph, rank/traverse it | `ingestion/` output |
| Embeddings | `src/graphrag_lite/embeddings/` | Turn text into vectors (entities, chunks, queries) | none |
| Vector store | `src/graphrag_lite/vectorstore/` | Zvec insert/query wrapper (hybrid search) | `embeddings/` |
| Retrieval | `src/graphrag_lite/retrieval/` | Pipeline A, Pipeline B, serialization | `graph/`, `vectorstore/` |
| LLM | `src/graphrag_lite/llm/` | Local/API client factory, generation | `retrieval/` output |
| API | `src/graphrag_lite/api/` | FastAPI endpoints, async hardening | all of the above |
| Observability | `src/graphrag_lite/observability/` | LangSmith tracing toggle | `api/` |

## 3. Full File Structure

```
graphrag-lite/
├── PRD.md
├── ARCHITECTURE.md
├── CONFIG.md                      # every tunable variable, one place
├── TASKS.md                       # task breakdown, source of truth for build order
├── DOCUMENT.md                    # running build log — updated after EVERY task
├── README.md
├── pyproject.toml
├── .env.example
│
├── src/graphrag_lite/
│   ├── __init__.py
│   ├── config.py                  # loads CONFIG.md values from .env / defaults
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   └── llm_cache.py           # disk-cache wrapper: cached_call(prompt, model) -> str
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py              # load_documents(dir) -> list[Document]
│   │   ├── chunker.py             # chunk_document(doc, size, overlap) -> list[Chunk]
│   │   ├── extractor.py           # extract_entities_relationships(chunk) -> ExtractionResult
│   │   └── gleaning.py            # glean(chunk, first_pass, n) -> ExtractionResult
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py             # build_graph(extractions) -> nx.DiGraph
│   │   ├── ranking.py             # rank_pagerank(graph) / rank_degree(graph) -> dict[node, score]
│   │   └── traversal.py           # bfs_expand(graph, seeds, hops) -> list[node]
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py            # embed_batch(texts, model) -> list[vector]
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── zvec_client.py         # insert_entities(), insert_chunks(), hybrid_query()
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── pipeline_a.py          # retrieve_flat(query) -> Evidence[]
│   │   ├── pipeline_b.py          # retrieve_structural(query) -> Evidence[]
│   │   └── serialize.py           # to_toon(evidence) / to_json(evidence) -> str
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client_factory.py      # get_llm_client(mode, api_key=None) -> LLMClient
│   │   └── generate.py            # generate_answer(evidence_str, query, client) -> Answer
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app entrypoint
│   │   ├── routes.py              # /index, /query, /health
│   │   └── middleware.py          # semaphore, circuit breaker, retry/backoff
│   │
│   └── observability/
│       ├── __init__.py
│       └── tracing.py             # set_tracing(enabled: bool)
│
├── eval/
│   ├── gold_qa_set.json           # 30-50 Q&A pairs w/ ground-truth answer + evidence nodes
│   ├── run_accuracy_eval.py       # runs gold set through A and B, scores both
│   └── ablations/
│       ├── embedding_ablation.py  # nomic vs Qwen3-Embedding-0.6B vs MiniLM
│       ├── slm_ablation.py        # qwen3:4b vs qwen3:1.7b vs phi4-mini
│       └── toon_vs_json_ablation.py
│
├── stress/
│   └── stress_test.py             # asyncio+httpx ramp test, 10→50→200→1000
│
├── scripts/
│   ├── build_index.py             # CLI: run full indexing pipeline
│   └── run_query.py               # CLI: run one query through A or B
│
├── tests/
│   ├── unit/                      # one test file per src module, see TASKS.md
│   └── integration/
│       ├── test_indexing_e2e.py
│       └── test_query_e2e.py
│
└── data/
    ├── raw/                       # source documents go here
    └── cache/                     # llm_cache.py disk cache lives here
```

## 4. Why This Shape (tied back to constraints)

- **No framework dependency (no Cognee, no GraphRAG monorepo)** — confirmed
  in PRD §20: both are too heavy/coupled to fork cleanly. This structure is
  from-scratch, one responsibility per file, each independently testable.
- **Cache sits beside ingestion, not inside it** — extraction and generation
  both call through it; keeping it a separate module means Task tests can
  verify caching behavior (cache hit/miss) in isolation from extraction logic.
- **Pipeline A and B are separate files, not a shared class with an if/else**
  — the whole point of the paper is comparing them; keeping them physically
  separate makes it impossible to accidentally leak logic between the
  "baseline" and "proposed" arms.
- **`config.py` + `CONFIG.md` is the single source of truth for every tunable
  number** (chunk size, hop depth, top-k, model name, ...) — no magic
  numbers buried in function bodies. Every constant referenced in `TASKS.md`
  traces back to an entry in `CONFIG.md`.
- **`api/` only exists to support the production-hardening side quest and the
  stress test** — the core research pipeline (indexing + eval) never needs
  FastAPI at all; `scripts/build_index.py` and `eval/run_accuracy_eval.py`
  call the library functions directly. This keeps the research path testable
  without spinning up a server.

## 5. Data Flow — Two Critical Paths

**Path 1: Indexing (offline, once)**
`loader → chunker → extractor(+gleaning, +cache) → builder → embedder → zvec_client.insert`

**Path 2: Query (Pipeline B, the one under test)**
`query → embedder.embed_batch([query]) → zvec_client.hybrid_query(entities) → traversal.bfs_expand → ranking.rank_pagerank → zvec_client.hybrid_query(scoped) → serialize.to_toon → client_factory.get_llm_client → generate.generate_answer`

Every arrow above is a real function call implemented in a numbered task in
`TASKS.md` — nothing in this path is precomputed or mocked once the tasks
are complete.
