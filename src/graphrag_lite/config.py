"""
config.py — Single source of truth for every tunable variable.

Reads from environment variables (or a .env file), falling back to the
defaults documented in CONFIG.md. Every value used anywhere in the
pipeline must be imported from here — no inline magic numbers.

Usage:
    from graphrag_lite.config import get_config
    cfg = get_config()
    print(cfg.CHUNK_SIZE_EXTRACTION)
"""

from __future__ import annotations

import functools
from typing import List, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """
    All configuration variables for graphrag_lite.

    Pydantic-settings reads from environment variables (case-insensitive)
    and from a .env file in the working directory, then falls back to the
    defaults below.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",        # ignore unrecognised env vars silently
    )

    # ------------------------------------------------------------------ #
    # §1  Chunking (CONFIG.md §1)
    # ------------------------------------------------------------------ #
    CHUNK_SIZE_EXTRACTION: int = Field(
        default=450,
        description="Token budget for extraction chunks (300-600).",
    )
    CHUNK_OVERLAP_PCT: float = Field(
        default=0.12,
        description="Fraction of chunk size used as overlap (0.10-0.15).",
    )
    CHUNK_SIZE_CITATION: int = Field(
        default=1200,
        description="Token budget for citation-display chunks (1000-1500).",
    )
    TOKENIZER: str = Field(
        default="qwen",
        description=(
            "Tokenizer family: 'qwen' uses cl100k_base via tiktoken "
            "(closest public approximation to Qwen's tokenizer)."
        ),
    )
    CHUNKING_STRATEGY: str = Field(
        default="markdown_aware",
        description=(
            "Active chunking strategy: sliding_window | recursive_char | "
            "markdown_aware | small_window. "
            "CONFIRMED default after chunking ablation (Phase 1 extension Task 1.6): "
            "lowest fail rate (4.8%), highest unique/1kTok (12.38 measured), "
            "non-overlapping sections = zero redundant token processing."
        ),
    )

    # ------------------------------------------------------------------ #
    # §2  Extraction
    # ------------------------------------------------------------------ #
    MAX_GLEANINGS: int = Field(
        default=1,
        description="Re-prompts after first extraction pass. Raise only after measuring recall.",
    )
    EXTRACTION_MODEL: str = Field(
        default="",
        description=(
            "Model used for entity/relationship extraction. "
            "Defaults to SLM_MODEL when empty."
        ),
    )
    EXTRACTION_TEMPERATURE: float = Field(
        default=0.0,
        description=(
            "Sampling temperature for extraction and gleaning LLM calls. "
            "0.0 = greedy / deterministic. Keep low so results are stable "
            "across re-runs and cache hits are meaningful."
        ),
    )
    EXTRACTION_SEED: int = Field(
        default=42,
        description=(
            "Fixed RNG seed for Ollama extraction calls. "
            "Combined with temperature=0.0 this produces deterministic output. "
            "temperature=0.0 alone is insufficient on some models/platforms."
        ),
    )
    LLM_NUM_PREDICT: int = Field(
        default=4096,
        description=(
            "Maximum tokens the model may generate per call (Ollama num_predict). "
            "Caps runaway generation on entity-rich chunks. "
            "4096, raised from 2048 on 2026-10-22: the extraction failures were "
            "NOT timeouts but TRUNCATIONS. On qwen2.5:1.5b, chunk a885be6b "
            "returned done_reason='length' with eval_count=2048 (exactly the "
            "cap) and unparseable JSON; at 4096 it returned done_reason='stop' "
            "with eval_count=842 in 101.8s and parsed cleanly. A chunk that "
            "stops early costs no extra time, so a higher cap only costs wall "
            "time on chunks that were previously broken. "
            "NOTE: a higher cap does NOT fix a genuine runaway -- chunk "
            "4af3dc8c hit the cap at BOTH 2048 and 4096 and still failed to "
            "parse. Runaway chunks need a different fix, not a bigger cap. "
            "It is also not true that hitting the cap means the response was "
            "worth keeping: chunk a4a679c4 at 4096 produced a COMPLETE "
            "20-entity / 20-relationship extraction whose only defect was a "
            "trailing comma, and _extract_json now repairs that."
        ),
    )
    LLM_THINK: Literal["auto", "on", "off"] = Field(
        default="auto",
        description=(
            "Whether to enable the model's reasoning/thinking mode. "
            "'auto' (default) does not send the parameter, leaving the "
            "model's own default. 'off' sends think=False. 'on' sends "
            "think=True. "
            "qwen3-family models are reasoning models: they return their "
            "chain-of-thought in a SEPARATE 'thinking' field, which consumes "
            "the whole num_predict budget and leaves the visible 'response' "
            "empty -- so extraction returns nothing that parses. Use 'off' "
            "for any qwen3-family extraction model."
        ),
    )
    LLM_NUM_CTX: int = Field(
        default=8192,
        description=(
            "Ollama runtime context window (num_ctx) sent per request. "
            "8192, raised from the Ollama default of 4096 on 2026-10-22. "
            "The default 4096 is NOT a hard wall on generation: Ollama grows "
            "the runtime context to fit num_predict (measured: 8750 tokens "
            "in one request, prompt_eval=558 + eval=8192, on a runner that "
            "'ollama ps' showed at CONTEXT=4096). But the bound IS real for "
            "a request served by a runner still at 4096: the prompt occupies "
            "the first tokens, so a 403-token extraction prompt leaves only "
            "3693 to generate into, and the response comes back with "
            "done_reason='length' and an eval_count BELOW num_predict -- "
            "which looks identical to a num_predict truncation but needs a "
            "different fix. Raising num_ctx to 8192 moved that same request "
            "to the full eval_count=4096. Set LLM_NUM_PREDICT below "
            "LLM_NUM_CTX minus the prompt size, or raise LLM_NUM_CTX."
        ),
    )
    LLM_FREQUENCY_PENALTY: float = Field(
        default=0.25,
        description=(
            "Ollama frequency_penalty sent per request. 0.0 = do not send "
            "the parameter (llama.cpp's own default, a no-op). "
            "Unlike repeat_penalty -- which applies the SAME flat penalty "
            "to every token that has ever appeared -- a frequency penalty "
            "scales with HOW MANY TIMES a token has appeared, so the 84th "
            "copy of a line is penalised far harder than the first. "
            "That is the shape a degenerate repetition loop needs. "
            "Measured 2026-10-22 on qwen2.5:1.5b, chunk 4af3dc8c, "
            "num_predict=4096, 3 runs per setting: "
            "no penalty -> eval=4096, done=length, parse=FAIL (0/3); "
            "0.25 -> eval=2583, done=stop, parse=ok 3/3, 21 entities / "
            "15 relationships, zero repetition; "
            "0.5 -> parse=ok 3/3 but only 10 entities / 8 relationships; "
            "0.75 -> parse=ok 3/3 but only 8 entities / 3 relationships; "
            "1.0 and 1.5 -> the model abandons a record mid-field and the "
            "JSON is left unterminated (parse=FAIL). "
            "0.25 is the default because it is the only setting that both "
            "breaks the loop AND recovers the full entity set: the chunk "
            "names ~22 real entities, and 0.25 recovers 21 of them, while "
            "0.5 and 0.75 have traded the repetition for UNDER-extraction. "
            "NOTE the scope of that claim: it is a WITHIN-CHUNK comparison "
            "against one dense chunk. Against the corpus, 10 and 8 "
            "entities are ABOVE the median -- the 39 chunks that parsed "
            "in the corpus run have a median of 5 entities and 21 of 39 "
            "have fewer than 6 -- so do not read it as a corpus-level "
            "statement, and do not add a minimum-entity-count guard on "
            "the strength of it (it would reject more than half the "
            "legitimate extractions). "
            "A flat repeat_penalty does not work at all (83x at no penalty, "
            "37x at 1.05, 48x at 1.15 -- still truncated and unparseable, "
            "and non-monotonic in the penalty). "
            "CAVEAT, measured on the full corpus 2026-10-22: the penalty "
            "REDUCES the failure rate but does not eliminate it. A "
            "full-corpus re-extraction at 0.25 parsed 39 of 42 chunks; the "
            "3 failures were all repetition loops. One of them, 4af3dc8c, "
            "had passed 3/3 in isolation with the SAME settings -- the "
            "model is BISTABLE, settling into one of several stable "
            "attractors according to the server's numeric state rather "
            "than the prompt, the seed, or the settings. An 18-run probe "
            "(tests/exp_reextract_failures.py, no cache in the path) "
            "confirmed this directly: two identical requests back-to-back "
            "in one process gave a loop and a success respectively, and it "
            "REFUTED the num_ctx hypothesis (the corpus run's ctx=8192 "
            "parses 4af3dc8c 1/2, ctx unset 0/2). It also showed the "
            "failure RATE is a sample: the corpus run parsed 0/3 of those "
            "chunks, the probe re-running the identical settings parsed "
            "2/6. "
            "CONTROL ARM, same corpus, same day (tests/exp_reextract_corpus.py "
            "at fp=0, i.e. the parameter not sent at all): 35 of 42 parsed, "
            "7 failures -- 6 repetition loops and 1 malformed JSON "
            "(done=stop, 689 tokens, a MISSING COMMA, 35 s instead of 300 s; "
            "this quiet mode is easy to miss next to the loud loops). So the "
            "penalty is worth +4 chunks NET, and the two arms disagree on 8 "
            "of 42 chunks IN BOTH DIRECTIONS: 6 chunks parse only at 0.25, "
            "and 2 (4af3dc8c, f08417a7) parse only at 0. An earlier "
            "documented claim that 'the penalty turned 3 failures into 0' "
            "was an inference from the 0.25 arm alone and was WRONG. "
            "The correct reading: the penalty shifts the dice, it does not "
            "make chunks safe, and NO single setting makes extraction "
            "reliable. Treat this knob as a mitigation and RETRY a failed "
            "chunk with escalated settings (LLM_RETRY_FREQUENCY_PENALTIES); "
            "do not treat one successful extraction as evidence the chunk "
            "is safe. 0.0 restores the old behaviour exactly."
        ),
    )
    LLM_RETRY_FREQUENCY_PENALTIES: List[float] = Field(
        default=[0.5, 1.0],
        description=(
            "Escalation ladder for a chunk whose extraction failed to "
            "parse. After the first attempt fails, the extractor retries "
            "with each of these frequency_penalty values in order. "
            "Rationale (2026-10-22): the model is BISTABLE -- it settles "
            "into one of several stable attractors depending on the "
            "server's numeric state, so the same chunk and settings can "
            "parse in one run and fall into a repetition loop in the next. "
            "Changing the settings is a different decode and therefore a "
            "different attractor, so a retry has a good chance of "
            "succeeding. Measured on the pathological chunk: 0.5 parses "
            "3/3 but recovers only 10 of ~22 entities, 1.0 stops the loop "
            "but truncates the record mid-field, so neither is a good "
            "STANDALONE value -- they are escalation steps, and 0.25 "
            "remains the first attempt. An empty list disables the "
            "ladder (one attempt only, as before). "
            "MEASURED END TO END (2026-10-22, later that day). The corpus "
            "was re-run through the production path with this ladder: "
            "42 of 42 chunks parsed, 0 failures, 245 raw entities and "
            "171 raw relationships -- better than both no-retry control "
            "arms (fp=0.25: 39/42; fp=0: 35/42) on every axis. The "
            "RESCUE rate was measured separately on the four known-hard "
            "chunks, 3 repetitions, each with a fresh cache so it is a "
            "fresh decode: P(attempt 0 fails) = 4/12, P(rescued given "
            "attempt 0 failed) = 4/4 = 100%, P(chunk ultimately fails) = "
            "0/12. EVERY rescue came from the first rung (0.5); the 1.0 "
            "rung was never needed. "
            "TWO CAVEATS, both measured. (i) A PARSE success is not a "
            "CONTENT success: one of the four rescues (4af3dc8c at "
            "fp=0.5) came back with 8 entities and 8 relationship "
            "objects whose source, target, type and description were all "
            "EMPTY STRINGS -- valid JSON, zero usable edges. The higher "
            "penalty buys parseability by pushing the model towards "
            "placeholders. (ii) The ladder does not address a separate "
            "content defect: the extractor's endpoint-consistency filter "
            "drops 53 of the 171 relationships the model emits (31%), "
            "45 of them because the model named an entity in a "
            "relationship that it did not also list as an entity. See "
            "DOCUMENT.md open item 8."
        ),
    )

    EXTRACTION_STUB_ENDPOINTS: bool = Field(
        default=False,
        description=(
            "If True, a relationship endpoint that is NOT in the chunk's "
            "entity list is RECOVERED instead of dropped, provided the "
            "endpoint's name occurs verbatim in the chunk text (casefold, "
            "word boundaries). The recovered endpoint is added to the entity "
            "list with type OTHER and the marker attribute stub=True, and "
            "the relationship is kept. Endpoints whose names do NOT occur "
            "in the text are still dropped with a warning. "
            "DEFAULT OFF: the endpoint-consistency filter is a safety "
            "property (a dangling edge is a hallucination), and turning it "
            "off silently is not an option. "
            "RATIONALE (2026-10-22): the filter drops 53 of the 171 "
            "relationships the model emits (31%), 45 of them because the "
            "model named an entity in a relationship that it did not also "
            "list as an entity. Those are mostly NOT hallucinations -- the "
            "name is in the chunk text. The verbatim check is the guard "
            "that keeps genuinely fabricated endpoints out. "
            "See DOCUMENT.md open item 8 and CONFIG.md changelog."
        ),
    )

    # ------------------------------------------------------------------ #
    # §3  Embedding
    # ------------------------------------------------------------------ #
    EMBEDDING_MODEL: str = Field(
        default="nomic-embed-text",
        description="Ollama model name or sentence-transformers model id.",
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=64,
        description="Texts per embed_batch call. Tune for CPU RAM budget.",
    )
    EMBEDDING_DIM: int = Field(
        default=0,
        description=(
            "Actual embedding dimension measured from the model's runtime output. "
            "Value 0 means not yet measured. Never set this manually — it is "
            "populated by embed_batch() on the first successful call. "
            "This is NOT the source of truth for vector length; the model output is."
        ),
    )

    # ------------------------------------------------------------------ #
    # §4  Vector Store (Zvec)
    # ------------------------------------------------------------------ #
    ZVEC_ROOT: str = Field(
        default="",
        description=(
            "Root directory holding the Zvec collections. Defaults to CACHE_DIR "
            "when empty. Set this explicitly so an experiment can point "
            "CACHE_DIR at its own private LLM-response cache while still "
            "reading the canonical collections built at data/cache."
        ),
    )
    ZVEC_COLLECTION_ENTITIES: str = Field(default="entities_v1")
    ZVEC_COLLECTION_CHUNKS: str = Field(default="chunks_v1")
    ZVEC_TOP_K_ENTRY: int = Field(
        default=8,
        description="Top-k entities returned by Pipeline B entry-point search.",
    )
    ZVEC_TOP_N_FLAT: int = Field(
        default=8,
        description="Top-N chunks returned by Pipeline A flat search.",
    )
    ZVEC_FUSION_METHOD: str = Field(
        default="rrf",
        description="Zvec fusion strategy: 'rrf' or 'weighted'.",
    )

    # ------------------------------------------------------------------ #
    # §5  Graph Retrieval (Pipeline B)
    # ------------------------------------------------------------------ #
    BFS_HOP_DEPTH: int = Field(
        default=2,
        description="BFS expansion depth from entry-point entities (2-3).",
    )
    RANKING_METHOD: Literal["pagerank", "degree"] = Field(
        default="pagerank",
        description="Node-ranking strategy used in Pipeline B candidate selection.",
    )
    PAGERANK_DAMPING: float = Field(
        default=0.85,
        description=(
            "NetworkX PageRank damping factor. Standard default is 0.85. "
            "Read from config so ablations can vary it without code changes."
        ),
    )
    TOP_N_CANDIDATES_AFTER_RANKING: int = Field(
        default=20,
        description="Candidates kept after structural ranking, before second Zvec pass.",
    )
    TOP_N_FINAL_EVIDENCE: int = Field(
        default=6,
        description="Final evidence pieces passed to the LLM.",
    )
    PIPELINE_B_SIM_WEIGHT: float = Field(
        default=0.85,
        description=(
            "Weight of the QUERY-DEPENDENT similarity term in Pipeline B's "
            "candidate score. The similarity is normalised to [0,1] against "
            "the best entry-point score before weighting, because raw Zvec "
            "RRF scores are ~0.02-0.03 while normalised PageRank is ~0-1; "
            "without normalisation PageRank dominates and the same node "
            "wins every query. Set to 1.0 to make PageRank a pure tiebreak."
        ),
    )
    PIPELINE_B_PR_WEIGHT: float = Field(
        default=0.15,
        description=(
            "Weight of the query-INDEPENDENT PageRank term in Pipeline B's "
            "candidate score. Kept small so it only breaks ties between "
            "equally query-relevant candidates. Must sum to 1.0 with "
            "PIPELINE_B_SIM_WEIGHT."
        ),
    )
    PIPELINE_B_DROP_TYPE_LABELS: bool = Field(
        default=True,
        description=(
            "If True, generic taxonomy nodes (Person, Organization, Edge, "
            "...) are removed from the ranked candidate list, not just from "
            "the entry points. They are the highest-PageRank nodes in the "
            "graph and are never meaningful retrieval targets."
        ),
    )

    # ------------------------------------------------------------------ #
    # §6  Serialization
    # ------------------------------------------------------------------ #
    SERIALIZATION_FORMAT: Literal["toon", "json"] = Field(
        default="toon",
        description="Evidence serialization format sent to the LLM.",
    )

    # ------------------------------------------------------------------ #
    # §7  LLM / Mode Toggle
    # ------------------------------------------------------------------ #
    LLM_MODE: Literal["local", "api"] = Field(
        default="local",
        description=(
            "'local' → Ollama/SLM_MODEL + LangSmith tracing ON. "
            "'api' → Grok API + tracing OFF."
        ),
    )
    SLM_MODEL: str = Field(
        default="qwen3:4b",
        description="Ollama model tag used in local mode.",
    )
    API_MODEL: str = Field(
        default="",
        description=(
            "Grok model name for api mode. Check current Grok docs at build time; "
            "do not hardcode a version that may be deprecated."
        ),
    )
    LLM_TIMEOUT_S: float = Field(
        default=300.0,
        description=(
            "Seconds before an Ollama generate() call raises TimeoutError. "
            "CPU inference on qwen2.5:0.5b takes 30-120s+ per call; gleaning "
            "prompts with 6+ entities reach 1500+ tokens and can exceed 120s. "
            "Raised to 300s after httpcore.ReadTimeout at chunk 32/42. "
            "Also surfaces stuck Ollama queue states from abandoned requests."
        ),
    )
    ZVEC_ADOPT_UNTAGGED: bool = Field(
        default=False,
        description=(
            "If True, a Zvec collection without a chunking_strategy.txt tag "
            "is adopted and tagged with the current CHUNKING_STRATEGY rather "
            "than raising RuntimeError. Set to True (or env var =1) only for "
            "legacy collections built before the strategy tag was introduced. "
            "Logs a warning when used."
        ),
    )

    # ------------------------------------------------------------------ #
    # §8  Observability
    # ------------------------------------------------------------------ #
    LANGCHAIN_PROJECT: str = Field(
        default="graphrag-lite-eval",
        description="LangSmith project name. Keep eval runs grouped separately.",
    )
    LANGCHAIN_API_KEY: str = Field(
        default="",
        description="LangSmith API key. Only needed in local mode with tracing enabled.",
    )

    # ------------------------------------------------------------------ #
    # §9  Caching
    # ------------------------------------------------------------------ #
    CACHE_BACKEND: str = Field(
        default="diskcache",
        description="Cache backend. Only 'diskcache' is implemented.",
    )
    CACHE_DIR: str = Field(
        default="data/cache",
        description="Directory for the disk-cache SQLite store.",
    )

    # ------------------------------------------------------------------ #
    # §10  Stress Test
    # ------------------------------------------------------------------ #
    STRESS_CONCURRENCY_LEVELS: List[int] = Field(
        default=[10, 50, 200, 1000],
        description="Ramp levels for the async stress test (Task 9.1).",
    )
    STRESS_TIMEOUT_S: int = Field(
        default=30,
        description="Per-request timeout in seconds during the stress test.",
    )
    STRESS_SEMAPHORE_LIMIT: int = Field(
        default=50,
        description=(
            "Max in-flight Grok requests. Must not exceed Grok's actual "
            "rate limit — verify against current API docs before first run."
        ),
    )

    # ------------------------------------------------------------------ #
    # §11  Production Hardening
    # ------------------------------------------------------------------ #
    UVICORN_WORKERS: int = Field(
        default=0,
        description=(
            "Uvicorn worker count. 0 = auto (os.cpu_count()). "
            "Set explicitly to override."
        ),
    )
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(
        default=5,
        description="Consecutive failures before the circuit breaker opens.",
    )
    RETRY_BACKOFF_BASE_S: float = Field(
        default=0.5,
        description="Base sleep time (seconds) for exponential backoff retry.",
    )
    RETRY_MAX_ATTEMPTS: int = Field(
        default=3,
        description="Max retry attempts for outbound Grok calls.",
    )

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #

    def effective_extraction_model(self) -> str:
        """
        Return the model to use for extraction.
        Falls back to SLM_MODEL when EXTRACTION_MODEL is not explicitly set.
        Use this instead of reading EXTRACTION_MODEL directly.
        """
        return self.EXTRACTION_MODEL if self.EXTRACTION_MODEL else self.SLM_MODEL

    def effective_think(self) -> bool | None:
        """
        Translate LLM_THINK into the value Ollama's `think` parameter wants.

        Returns
        -------
        bool | None
            None  -> do not send the parameter; the model decides (default)
            False -> explicitly disable reasoning
            True  -> explicitly request reasoning

        Use this instead of reading LLM_THINK directly.
        """
        if self.LLM_THINK == "off":
            return False
        if self.LLM_THINK == "on":
            return True
        return None


@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Return the singleton Config instance.

    Cached after the first call so every import across the process shares
    the same object and reads the same values. Call ``get_config.cache_clear()``
    in tests that need a fresh read.
    """
    return Config()
