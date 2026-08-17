# CONFIG.md — Every Tunable Variable in One Place

Rule: if a number or model name is used more than once, it lives here and
in `src/graphrag_lite/config.py` — never hardcoded inline in a task's code.
When a task changes a default, update this table AND the changelog at the
bottom, so DOCUMENT.md entries can reference "CONFIG.md v(N)" instead of
re-explaining the value.

## 1. Chunking

| Variable | Default | Range to test | Set in |
|---|---|---|---|
| `CHUNK_SIZE_EXTRACTION` | 450 tokens | 300–600 | `.env` / `config.py` |
| `CHUNK_OVERLAP_PCT` | 12% | 10–15% | `.env` / `config.py` |
| `CHUNK_SIZE_CITATION` | 1200 tokens | 1000–1500 | `.env` / `config.py` |
| `TOKENIZER` | `qwen` tokenizer (via `tiktoken`-compatible call) | — | `config.py` |

## 2. Extraction

| Variable | Default | Notes |
|---|---|---|
| `MAX_GLEANINGS` | 1 | Raise only after measuring recall at 1; each increment = 1 extra LLM call per chunk |
| `EXTRACTION_MODEL` | same as `SLM_MODEL` (local mode) | Ablation: try extraction on Grok too, compare noise |

## 3. Embedding

| Variable | Default | Ablation set |
|---|---|---|
| `EMBEDDING_MODEL` | `nomic-embed-text` (Ollama) | `Qwen3-Embedding-0.6B`, `all-MiniLM-L6-v2` (legacy baseline) |
| `EMBEDDING_DIM` | model-dependent, read at runtime, never hardcoded | — |
| `EMBEDDING_BATCH_SIZE` | 64 | tune for CPU RAM budget |

## 4. Vector Store (Zvec)

| Variable | Default | Notes |
|---|---|---|
| `ZVEC_COLLECTION_ENTITIES` | `entities_v1` | scalar fields: `entity_type`, `source_doc`, `pagerank_score` |
| `ZVEC_COLLECTION_CHUNKS` | `chunks_v1` | scalar fields: `source_doc`, `chunk_index` |
| `ZVEC_TOP_K_ENTRY` | 8 | entry-point search (Pipeline B step 1) |
| `ZVEC_TOP_N_FLAT` | 8 | Pipeline A flat search count |
| `ZVEC_FUSION_METHOD` | RRF (built-in) | alt: weighted fusion |

## 5. Graph Retrieval (Pipeline B)

| Variable | Default | Range to test |
|---|---|---|
| `BFS_HOP_DEPTH` | 2 | 2–3 |
| `RANKING_METHOD` | `pagerank` | alt: `degree` — must A/B both, PRD §9 |
| `TOP_N_CANDIDATES_AFTER_RANKING` | 20 | before second Zvec pass |
| `TOP_N_FINAL_EVIDENCE` | 6 | passed to the LLM |

## 6. Serialization

| Variable | Default | Notes |
|---|---|---|
| `SERIALIZATION_FORMAT` | `toon` | ablation vs `json`, PRD §11 — measure tokens/latency/accuracy for both |

## 7. LLM / Mode Toggle

| Variable | Default | Notes |
|---|---|---|
| `LLM_MODE` | `local` | `local` or `api` — this single flag also drives tracing (see §8) |
| `SLM_MODEL` (local mode) | `qwen3:4b` | ablation: `qwen3:1.7b`, `phi4-mini` |
| `API_MODEL` (api mode) | Grok (model name TBD at build time — check current Grok API docs, don't hardcode a version that may be deprecated) | user-pasted key, session-only, never persisted to disk/log |

## 8. Observability

| Variable | Default | Notes |
|---|---|---|
| `LANGCHAIN_TRACING_V2` | `true` when `LLM_MODE=local`, `false` when `LLM_MODE=api` | set programmatically by `observability/tracing.py`, not manually |
| `LANGCHAIN_PROJECT` | `graphrag-lite-eval` | keep eval runs grouped separately from ad-hoc dev runs |

## 9. Caching

| Variable | Default | Notes |
|---|---|---|
| `CACHE_BACKEND` | `diskcache` (SQLite-backed) | key = `hash(prompt + model_name)` |
| `CACHE_DIR` | `data/cache/` | gitignored |

## 10. Stress Test

| Variable | Default | Notes |
|---|---|---|
| `STRESS_CONCURRENCY_LEVELS` | `[10, 50, 200, 1000]` | ramp order, don't skip steps |
| `STRESS_TIMEOUT_S` | 30 | per-request timeout before counted as failure |
| `STRESS_SEMAPHORE_LIMIT` | tied to Grok's actual rate limit — check before first run | see TASKS.md Task 6.1 |

## 11. Production Hardening (API layer)

| Variable | Default | Notes |
|---|---|---|
| `UVICORN_WORKERS` | `os.cpu_count()` | multi-process for CPU-bound graph filtering |
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | 5 consecutive failures | then 503 + cooldown |
| `RETRY_BACKOFF_BASE_S` | 0.5 | exponential |
| `RETRY_MAX_ATTEMPTS` | 3 | |

---

## Changelog

| Date | Variable changed | Old → New | Reason | Task ref |
|---|---|---|---|---|
| — | — | — | — | (fill in as tasks change defaults) |
