"""
pipeline3_lightrag/adapter.py — Pipeline 3: Genuine upstream LightRAG v1.5.8.

SOURCE
------
LightRAG v1.5.8 by HKUDS (Hong Kong University of Science and Technology)
  Repository:  https://github.com/HKUDS/LightRAG
  Package:     lightrag-hku
  License:     MIT
  Located at:  LightRAG/ (vendored in this repository with its own .git)

This adapter is a genuine integration of the upstream LightRAG library.
It is NOT the MindMesh-native experimental implementations in
src/graphrag_lite/retrieval/lightrag/  (those are adapted from the paper).

This adapter creates a fully isolated working directory for LightRAG:
  data/pipeline3_workdir/

It does NOT:
  - Modify the canonical MindMesh graph (data/cache/)
  - Modify any existing experiment results
  - Use the same Zvec collections as Pipeline A/B

LightRAG modes supported: local, global, hybrid, naive, mix, bypass.

SETUP REQUIRED (GPU task)
-------------------------
Before running queries, the LightRAG working directory must be populated:
  python run_pipeline.py --pipeline pipeline_3_lightrag --setup-only
This ingests data/raw/ documents into LightRAG's own storage using the
configured LLM (requires Ollama). Flagged as GPU_DEFERRED_TASKS.md G7.

For dry-run and import tests, no GPU inference is needed.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
# Insert LightRAG source into path so it can be imported without pip install
_LIGHTRAG_SRC = str(ROOT / "LightRAG")
if _LIGHTRAG_SRC not in sys.path:
    sys.path.insert(0, _LIGHTRAG_SRC)

from mindmesh.pipelines.base import PipelineAdapter, PipelineResult


import re


_PLACEHOLDER_PATTERNS = (
    re.compile(r'^\s*<[^>]+>\s*$'),
    re.compile(r'^\s*<\|[^|]+\|>\s*$'),
)


def _is_placeholder_text(value):
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    for pattern in _PLACEHOLDER_PATTERNS:
        if pattern.match(text):
            return True
    lowered = text.lower()
    if lowered in {
        'entity_name',
        'entity_type',
        'entity_description',
        'source_entity',
        'target_entity',
        'relationship_keywords',
        'relationship_description',
    }:
        return True
    if '<' in text and '>' in text and (
        '<entity_name>' in text
        or '<source_entity>' in text
        or '<target_entity>' in text
        or '<relationship_description>' in text
        or '<relationship_keywords>' in text
    ):
        return True
    return False


def _has_placeholder_in_record(record):
    for key in ('entity_name', 'src_id', 'tgt_id', 'source_entity', 'target_entity'):
        if key in record and _is_placeholder_text(record.get(key)):
            return True
    for key in ('description', 'entity_description', 'relationship_description'):
        if key in record and _is_placeholder_text(record.get(key)):
            return True
    for key in ('keywords', 'relationship_keywords'):
        if key in record and _is_placeholder_text(record.get(key)):
            return True
    for key in ('entity_type', 'type'):
        if key in record and _is_placeholder_text(record.get(key)):
            return True
    # reject empty descriptions
    for key in ('description', 'entity_description', 'relationship_description'):
        if key in record:
            v = record.get(key)
            if v is None or (isinstance(v, str) and not v.strip()):
                return True
    return False
def _make_pipeline3_extraction_validator():
    def _validate(chunk_key, chunk_text, maybe_nodes, maybe_edges):
        filtered_nodes = {}
        for name, records in (maybe_nodes or {}).items():
            if _is_placeholder_text(name):
                continue
            name_str = str(name).strip()
            if not name_str or name_str == '<entity_name>':
                continue
            cleaned_records = []
            for rec in records or []:
                if not isinstance(rec, dict):
                    continue
                if _has_placeholder_in_record(rec):
                    continue
                # ensure endpoint not placeholder
                n = rec.get('entity_name') or rec.get('name') or name_str
                if _is_placeholder_text(n):
                    continue
                cleaned_records.append(rec)
            if cleaned_records:
                filtered_nodes[name_str] = cleaned_records

        filtered_edges = {}
        for pair, records in (maybe_edges or {}).items():
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            src, tgt = pair[0], pair[1]
            if _is_placeholder_text(src) or _is_placeholder_text(tgt):
                continue
            src_s = str(src).strip()
            tgt_s = str(tgt).strip()
            if not src_s or not tgt_s or src_s == tgt_s:
                continue
            if src_s == '<source_entity>' or tgt_s == '<target_entity>':
                continue
            cleaned_records = []
            for rec in records or []:
                if not isinstance(rec, dict):
                    continue
                if _has_placeholder_in_record(rec):
                    continue
                rsrc = rec.get('src_id') or rec.get('source_entity') or src_s
                rtgt = rec.get('tgt_id') or rec.get('target_entity') or tgt_s
                if _is_placeholder_text(rsrc) or _is_placeholder_text(rtgt):
                    continue
                if str(rsrc).strip() == str(rtgt).strip():
                    continue
                cleaned_records.append(rec)
            if cleaned_records:
                filtered_edges[(src_s, tgt_s)] = cleaned_records

        had_candidates = bool(maybe_nodes) or bool(maybe_edges)
        if had_candidates and not filtered_nodes and not filtered_edges:
            raise ValueError(
                f'Pipeline3 extraction produced only placeholder/malformed records for chunk {chunk_key}; refusing to create fake graph data'
            )
        if not filtered_nodes and not filtered_edges and chunk_text and str(chunk_text).strip():
            raise ValueError(
                f'Pipeline3 extraction produced no entities/relationships for non-empty chunk {chunk_key}'
            )
        return filtered_nodes, filtered_edges
    return _validate



# LightRAG working directory — isolated from MindMesh data/cache
_DEFAULT_WORKDIR = ROOT / "data" / "pipeline3_workdir"

# Valid LightRAG query modes
LIGHTRAG_MODES = ("local", "global", "hybrid", "naive", "mix", "bypass")


def _check_lightrag_importable() -> tuple[bool, str]:
    """Return (ok, error_message) for the LightRAG import check."""
    try:
        import lightrag
        from lightrag import LightRAG, QueryParam
        return True, ""
    except ImportError as exc:
        return False, str(exc)


def _make_ollama_llm_func(model: str, host: str = "http://localhost:11434", timeout: float | None = None):
    """
    Create an async LLM function compatible with LightRAG using Ollama.

    LightRAG's llm_model_func must be an async callable:
        async def func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str
    """
    import httpx

    if timeout is None:
        timeout = float(os.environ.get("PIPELINE3_OLLAMA_LLM_TIMEOUT", "600"))

    async def ollama_llm(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list = [],
        **kwargs,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history_messages:
            messages.append(msg)
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            response = await client.post(
                f"{host}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "seed": 42,
                        "num_predict": 4096,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]

    return ollama_llm

def _extract_ollama_embeddings(payload: Any) -> list:
    """Return the ``embeddings`` list from an Ollama ``/api/embed`` response.

    Ollama responds with ``{"embeddings": [[float, ...], ...], ...}``.  A
    malformed payload is rejected rather than partially consumed.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            "Ollama embedding response must be a JSON object, "
            f"got {type(payload).__name__}"
        )
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, (list, tuple)):
        raise ValueError(
            "Ollama embedding response is missing a list 'embeddings' field"
        )
    return list(embeddings)


def _normalize_embeddings(
    raw_embeddings: Any,
    texts: str | list[str] | tuple[str, ...],
    embedding_dim: int,
) -> "np.ndarray":
    """Convert provider embeddings into LightRAG's strict 2D float32 array.

    Contract: shape ``(len(texts), embedding_dim)``, one row per input text in
    input order.  Malformed vectors are rejected, never reshaped or padded.
    """
    import numpy as np

    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

    if isinstance(texts, str):
        input_texts = [texts]
    else:
        input_texts = list(texts)

    expected_count = len(input_texts)
    if expected_count == 0:
        if isinstance(raw_embeddings, (list, tuple)) and raw_embeddings:
            raise ValueError(
                "Embedding count mismatch: empty input batch returned "
                f"{len(raw_embeddings)} vector(s)"
            )
        return np.empty((0, embedding_dim), dtype=np.float32)

    if not isinstance(raw_embeddings, (list, tuple)):
        raise ValueError(
            "Embedding provider must return a list of vectors, "
            f"got {type(raw_embeddings).__name__}"
        )
    if len(raw_embeddings) != expected_count:
        raise ValueError(
            f"Embedding count mismatch: expected {expected_count} vector(s) "
            f"for {expected_count} input text(s) but got {len(raw_embeddings)}"
        )

    vectors = []
    for index, vector in enumerate(raw_embeddings):
        if not isinstance(vector, (list, tuple)):
            raise ValueError(
                f"Embedding {index} must be a sequence of numbers, "
                f"got {type(vector).__name__}"
            )
        if len(vector) != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch at row {index}: expected "
                f"{embedding_dim} but got {len(vector)}"
            )
        try:
            array = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Embedding {index} contains non-numeric values"
            ) from exc
        if array.ndim != 1:
            raise ValueError(
                f"Embedding {index} must be one-dimensional, "
                f"got shape {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(
                f"Embedding {index} contains NaN or infinite values"
            )
        vectors.append(array)

    result = np.stack(vectors, axis=0)
    if result.shape != (expected_count, embedding_dim):
        raise ValueError(
            f"Embedding shape mismatch: expected "
            f"({expected_count}, {embedding_dim}) but got {result.shape}"
        )
    if result.dtype != np.float32:
        result = result.astype(np.float32, copy=False)
    return result


async def _post_ollama_embed(host: str, model: str, texts: list[str]) -> Any:
    """POST to Ollama's batch embedding endpoint and return the JSON payload."""
    import httpx

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{host}/api/embed",
            json={"model": model, "input": texts},
        )
        response.raise_for_status()
        return response.json()


def _make_ollama_embed_func(
    model: str = "nomic-embed-text",
    host: str = "http://localhost:11434",
    embedding_dim: int = 768,
):
    """
    Create an EmbeddingFunc instance compatible with LightRAG v1.5.8 using Ollama.

    LightRAG v1.5.8 requires the wrapped function to return a 2D NumPy array of
    shape ``(len(texts), embedding_dim)``.  Ollama returns a Python list, so the
    adapter converts and validates it before LightRAG sees it.
    """
    sys.path.insert(0, _LIGHTRAG_SRC)
    from lightrag.utils import EmbeddingFunc

    async def _embed(texts: str | list[str]) -> "np.ndarray":
        input_texts = [texts] if isinstance(texts, str) else list(texts)
        if not input_texts:
            return _normalize_embeddings([], input_texts, embedding_dim)

        payload = await _post_ollama_embed(host, model, input_texts)
        raw_embeddings = _extract_ollama_embeddings(payload)
        return _normalize_embeddings(raw_embeddings, input_texts, embedding_dim)

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        func=_embed,
        model_name=model,
    )


class Pipeline3LightRAGAdapter(PipelineAdapter):
    """
    Genuine LightRAG Pipeline 3 adapter.

    Uses the upstream lightrag-hku package at LightRAG/ to provide:
      - Local retrieval (entity-focused)
      - Global retrieval (relationship-focused)
      - Hybrid retrieval
      - Naive retrieval (pure vector, like Pipeline A)
      - Mix retrieval (hybrid + naive)

    Requires a one-time ingestion step (GPU-intensive, see GPU_DEFERRED_TASKS.md).
    """

    name = "pipeline_3_lightrag"
    description = (
        "Genuine upstream LightRAG v1.5.8 by HKUDS (MIT license). "
        "Modes: local, global, hybrid, naive, mix. "
        "Requires separate ingestion step (GPU task G7)."
    )
    supports_modes = LIGHTRAG_MODES

    def __init__(
        self,
        model: str,
        exp_dir: Path | None = None,
        top_k: int = 6,
        workdir: Path | None = None,
        llm_host: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        embedding_dim: int = 768,
        llm_timeout: float | None = None,
        llm_max_async: int | None = None,
        embedding_func_max_async: int | None = None,
    ) -> None:
        super().__init__(model=model, exp_dir=exp_dir, top_k=top_k)
        self.workdir = workdir or _DEFAULT_WORKDIR
        self.llm_host = llm_host
        self.embed_model = embed_model
        self.embedding_dim = embedding_dim
        self.llm_timeout = llm_timeout
        if llm_max_async is None:
            llm_max_async = int(os.environ.get("PIPELINE3_LLM_MAX_ASYNC", "1"))
        if embedding_func_max_async is None:
            embedding_func_max_async = int(os.environ.get("PIPELINE3_EMBED_MAX_ASYNC", "1"))
        self.llm_max_async = llm_max_async
        self.embedding_func_max_async = embedding_func_max_async
        self._rag = None  # LightRAG instance, created lazily
    def _check_importable(self) -> None:
        """Raise ImportError with a helpful message if LightRAG is not importable."""
        ok, err = _check_lightrag_importable()
        if not ok:
            raise ImportError(
                f"LightRAG is not importable: {err}\n"
                f"The LightRAG source is vendored at LightRAG/ in this repository.\n"
                f"It should be importable via sys.path. Check that "
                f"{ROOT / 'LightRAG'} exists and contains lightrag/."
            )

    def _get_rag(self):
        """Return the LightRAG instance, creating it if needed (lazy init)."""
        if self._rag is not None:
            return self._rag

        self._check_importable()
        from lightrag import LightRAG

        self.workdir.mkdir(parents=True, exist_ok=True)

        llm_func = _make_ollama_llm_func(self.model, self.llm_host, timeout=self.llm_timeout)
        embed_func = _make_ollama_embed_func(
            self.embed_model,
            self.llm_host,
            embedding_dim=self.embedding_dim,
        )

        addon_params = {
            "entity_type_prompt_file": "mindmesh_concrete.yml",
            "entity_extraction_use_json": False,
        }
        kwargs = dict(
            working_dir=str(self.workdir),
            llm_model_func=llm_func,
            embedding_func=embed_func,
            addon_params=addon_params,
            kg_extraction_validator=_make_pipeline3_extraction_validator(),
        )
        if self.llm_max_async is not None:
            kwargs["llm_model_max_async"] = int(self.llm_max_async)
        if self.embedding_func_max_async is not None:
            kwargs["embedding_func_max_async"] = int(self.embedding_func_max_async)
        self._rag = LightRAG(**kwargs)
        return self._rag

    def is_indexed(self) -> bool:
        """Return True if the LightRAG working directory has been indexed."""
        # LightRAG creates kv_store_*.json files in working_dir after indexing
        if not self.workdir.exists():
            return False
        kv_files = list(self.workdir.glob("kv_store_*.json"))
        return len(kv_files) > 0
    def index_documents(self, docs_dir: Path | None = None) -> dict:
        """
        Ingest documents from docs_dir (default: data/raw/) into LightRAG.

        GPU-INTENSIVE: This calls the LLM for entity/relationship extraction.
        It is tagged as GPU task G7 in GPU_DEFERRED_TASKS.md.
        Only call this explicitly — never called automatically.
        """
        import asyncio
        from pathlib import Path as _Path

        source_dir = docs_dir or (ROOT / "data" / "raw")
        if not source_dir.exists():
            return {"error": f"Source directory not found: {source_dir}"}

        rag = self._get_rag()
        docs = []
        for f in sorted(source_dir.iterdir()):
            if f.suffix in (".txt", ".md"):
                docs.append(f.read_text(encoding="utf-8"))

        if not docs:
            return {"error": "No .txt or .md documents found", "dir": str(source_dir)}

        async def _run():
            await rag.ainsert(docs)

        asyncio.run(_run())
        return {
            "indexed": len(docs),
            "workdir": str(self.workdir),
            "source_dir": str(source_dir),
        }

    def query(
        self,
        question: str,
        mode: str = "hybrid",
        generate_answer: bool = True,
    ) -> PipelineResult:
        """
        Execute a LightRAG query.

        Parameters
        ----------
        question : str
            Natural language question.
        mode : str
            One of: local, global, hybrid, naive, mix, bypass.
        generate_answer : bool
            LightRAG always generates an answer; set False to retrieve
            context only (uses only_need_context=True).

        Returns
        -------
        PipelineResult
            Retrieved chunks, entities, relationships, and answer.
        """
        import asyncio

        if mode not in LIGHTRAG_MODES:
            return PipelineResult(
                query=question,
                pipeline=self.name,
                model=self.model,
                mode=mode,
                errors=[
                    f"Unknown mode {mode!r}. Supported: {', '.join(LIGHTRAG_MODES)}"
                ],
            )

        if not self.is_indexed():
            return PipelineResult(
                query=question,
                pipeline=self.name,
                model=self.model,
                mode=mode,
                errors=[
                    "LightRAG working directory is not indexed. "
                    "Run: python run_pipeline.py --pipeline pipeline_3_lightrag --setup-only "
                    "(GPU task G7 — requires LLM inference)."
                ],
            )

        try:
            self._check_importable()
            from lightrag import QueryParam

            rag = self._get_rag()

            async def _query():
                param = QueryParam(
                    mode=mode,
                    top_k=self.top_k,
                    only_need_context=not generate_answer,
                )
                return await rag.aquery(question, param=param)

            t0 = time.monotonic()
            raw = asyncio.run(_query())
            total_lat = time.monotonic() - t0

            # LightRAG returns a string (answer or context)
            answer = raw if generate_answer else None
            context = raw if not generate_answer else ""

            return PipelineResult(
                query=question,
                pipeline=self.name,
                model=self.model,
                mode=mode,
                # LightRAG does not return structured chunk dicts natively;
                # we return the raw string in context and note the limitation.
                retrieved_chunks=[],
                context=context,
                answer=answer,
                total_latency_s=total_lat,
                raw_output=raw,
                errors=[],
            )

        except Exception as exc:
            return PipelineResult(
                query=question,
                pipeline=self.name,
                model=self.model,
                mode=mode,
                errors=[f"{type(exc).__name__}: {exc}"],
            )

    def dry_run(self) -> dict:
        base = super().dry_run()
        ok, err = _check_lightrag_importable()
        base.update({
            "lightrag_importable":  ok,
            "lightrag_import_error": err if not ok else None,
            "workdir":              str(self.workdir),
            "is_indexed":           self.is_indexed(),
            "embed_model":          self.embed_model,
            "embedding_dim":        self.embedding_dim,
            "llm_host":             self.llm_host,
            "note": (
                "Pipeline 3 requires index_documents() before queries. "
                "This is GPU task G7 — see GPU_DEFERRED_TASKS.md."
            ),
        })
        return base
