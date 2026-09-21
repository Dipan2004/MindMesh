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
| `CHUNKING_STRATEGY` | `markdown_aware` (confirmed default) | `sliding_window`, `recursive_char`, `markdown_aware`, `small_window` | `.env` / `config.py` |

## 2. Extraction

| Variable | Default | Notes |
|---|---|---|
| `MAX_GLEANINGS` | 1 | Raise only after measuring recall at 1; each increment = 1 extra LLM call per chunk |
| `EXTRACTION_MODEL` | same as `SLM_MODEL` (local mode) | Ablation: try extraction on Grok too, compare noise |
| `EXTRACTION_TEMPERATURE` | 0.0 | Greedy/deterministic. Keep at 0.0 for extraction and gleaning so cache hits are meaningful and results are stable across re-runs. Raise only if you want diversity in generation (not applicable for structured JSON extraction). |
| `EXTRACTION_SEED` | 42 | Fixed RNG seed. temperature=0.0 alone is not sufficient for determinism on all models/platforms — seed must also be set. Confirmed via stability test. |

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
| `LLM_TIMEOUT_S` | 120.0 | Seconds before an Ollama generate() call raises TimeoutError. Without this, an abandoned in-flight request from a killed process blocks Ollama's queue indefinitely (root cause of "silent hang" bug confirmed during Phase 1 chunking ablation). |

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
| 2026-09-15 | `EXTRACTION_MODEL` / `SLM_MODEL` | `qwen3:4b` → `qwen2.5:0.5b` | `qwen3:4b` not available in local Ollama install; `qwen2.5:0.5b` (397 MB) was fastest available model. Paper ablations should be re-run with `qwen3:4b` once pulled. | Task 1.3 |
| 2026-09-15 | `TOKENIZER` (implementation) | Qwen native tokenizer → `tiktoken cl100k_base` | Qwen's exact tokenizer is not available as a standalone PyPI package. `cl100k_base` (GPT-4 BPE) is the closest public approximation — vocabulary overlap is high but not identical. Token counts may differ by ±5% from true Qwen counts on the same text. Noted as a known approximation, not a silent substitution. | Task 1.2 |
| 2026-09-15 | `STRESS_CONCURRENCY_LEVELS` format | `10,50,200,1000` (CSV string) → `[10,50,200,1000]` (JSON array) | pydantic-settings tries to JSON-parse `List[int]` fields before custom validators fire; CSV string caused a `JSONDecodeError`. JSON array format is the correct syntax for pydantic-settings list fields in `.env` files. | Task 0.1 / Task 1.3 |
| 2026-09-15 | `EXTRACTION_TEMPERATURE` | unset (Ollama default ~0.8) → `0.0` | Temperature was never pinned; non-deterministic sampling caused the same chunk to succeed in one run and fail with truncated JSON in another. Fixed to 0.0 (greedy) so extraction results are stable across re-runs and cache hits are meaningful. | Phase 1 fix-up |
| 2026-09-15 | `CHUNKING_STRATEGY` | `sliding_window` → `markdown_aware` | **CONFIRMED** after chunking ablation (Task 1.6): lowest fail rate (4.8%), highest measured unique/1kTok (12.38 vs ~7.4 est. for sliding_window). recursive_char excluded — mid-word boundary corruption from character-level separator on technical text (confirmed, not contention). small_window eliminated (higher fail rate, no quality gain). | Task 1.6 |


