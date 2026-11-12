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


def _make_ollama_llm_func(model: str, host: str = "http://localhost:11434"):
    """
    Create an async LLM function compatible with LightRAG using Ollama.

    LightRAG's llm_model_func must be an async callable:
        async def func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str
    """
    import httpx

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

        async with httpx.AsyncClient(timeout=300) as client:
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


def _make_ollama_embed_func(model: str = "nomic-embed-text", host: str = "http://localhost:11434"):
    """
    Create an EmbeddingFunc instance compatible with LightRAG v1.5.8 using Ollama.

    LightRAG v1.5.8 requires embedding_func to be an EmbeddingFunc dataclass
    with .func and .embedding_dim attributes.
    """
    import httpx
    sys.path.insert(0, _LIGHTRAG_SRC)
    from lightrag.utils import EmbeddingFunc

    async def _embed(texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{host}/api/embed",
                json={"model": model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]

    return EmbeddingFunc(embedding_dim=768, func=_embed)


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
    ) -> None:
        super().__init__(model=model, exp_dir=exp_dir, top_k=top_k)
        self.workdir = workdir or _DEFAULT_WORKDIR
        self.llm_host = llm_host
        self.embed_model = embed_model
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

        llm_func = _make_ollama_llm_func(self.model, self.llm_host)
        embed_func = _make_ollama_embed_func(self.embed_model, self.llm_host)

        self._rag = LightRAG(
            working_dir=str(self.workdir),
            llm_model_func=llm_func,
            embedding_func=embed_func,
        )
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
            "llm_host":             self.llm_host,
            "note": (
                "Pipeline 3 requires index_documents() before queries. "
                "This is GPU task G7 — see GPU_DEFERRED_TASKS.md."
            ),
        })
        return base
