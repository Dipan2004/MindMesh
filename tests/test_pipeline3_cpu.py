"""
test_pipeline3_cpu.py — CPU-safe tests for Pipeline 3 (upstream LightRAG).

No model inference, no document indexing, and no calls to Ollama.  Every
LightRAG storage lives in a temporary directory which is removed afterwards.

Covers:
  1.  LightRAG import
  2.  EmbeddingFunc wrapper construction
  3.  Adapter embedding-function integration
  4.  LightRAG initialization with a deterministic dummy embedding function
  5.  Storage backend configuration
  6.  Working-directory isolation
  7.  QueryParam configuration for supported modes
  8.  Correct behavior when querying before indexing
  9.  Dry-run behavior without inference

Run:
    python -m pytest -vv -s tests/test_pipeline3_cpu.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT / "src"), str(ROOT / "LightRAG"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lightrag import LightRAG, QueryParam  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

from mindmesh.pipelines.base import PipelineResult  # noqa: E402
from mindmesh.pipelines.pipeline3_lightrag.adapter import (  # noqa: E402
    LIGHTRAG_MODES,
    Pipeline3LightRAGAdapter,
    _check_lightrag_importable,
    _make_ollama_embed_func,
)

EMBED_DIM = 8


async def _dummy_embed(texts, *args, **kwargs):
    return np.zeros((len(texts), EMBED_DIM), dtype=float)


def _dummy_embedding_func() -> EmbeddingFunc:
    return EmbeddingFunc(embedding_dim=EMBED_DIM, func=_dummy_embed)


async def _forbidden_llm(*args, **kwargs):
    raise AssertionError("LLM must not be called in a CPU-only test")


async def _forbidden_embed(*args, **kwargs):
    raise AssertionError("Embedding model must not be called in a CPU-only test")


def _make_rag(workdir: Path, workspace: str) -> LightRAG:
    return LightRAG(
        working_dir=str(workdir),
        llm_model_func=_forbidden_llm,
        embedding_func=_dummy_embedding_func(),
        workspace=workspace,
    )


def test_lightrag_import():
    ok, err = _check_lightrag_importable()
    assert ok, f"LightRAG is not importable: {err}"

    import lightrag as lightrag_pkg

    assert lightrag_pkg.__file__
    assert "LightRAG" in lightrag_pkg.__file__ or "lightrag" in lightrag_pkg.__file__.lower()
    assert LightRAG is not None
    assert QueryParam is not None


def test_embedding_func_wrapper():
    wrapper = _dummy_embedding_func()
    assert isinstance(wrapper, EmbeddingFunc)
    assert wrapper.embedding_dim == EMBED_DIM

    vectors = asyncio.run(wrapper(["alpha", "beta"]))
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, EMBED_DIM)
    assert not vectors.any()


def test_adapter_embedding_function_integration():
    embed = _make_ollama_embed_func()
    assert isinstance(embed, EmbeddingFunc)
    assert embed.embedding_dim == 768
    assert callable(embed.func)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp) / "adapter_workdir"
        adapter = Pipeline3LightRAGAdapter(model="qwen2.5:1.5b", workdir=workdir)
        captured: dict = {}

        class FakeRAG:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch("lightrag.LightRAG", FakeRAG):
            rag = adapter._get_rag()

        assert isinstance(rag, FakeRAG)
        assert isinstance(captured["embedding_func"], EmbeddingFunc)
        assert captured["embedding_func"].embedding_dim == 768
        assert captured["working_dir"] == str(workdir)
        assert callable(captured["llm_model_func"])


def test_lightrag_init_with_dummy_embedding():
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp) / "init_workdir"

        async def _run():
            rag = _make_rag(workdir, "test_pipeline3_cpu_init")
            try:
                assert workdir.exists()
                assert rag.working_dir == str(workdir)
                assert isinstance(rag.embedding_func, EmbeddingFunc)
                assert rag.embedding_func.embedding_dim == EMBED_DIM
                for attr in (
                    "full_docs",
                    "text_chunks",
                    "entities_vdb",
                    "relationships_vdb",
                    "chunks_vdb",
                    "chunk_entity_relation_graph",
                    "llm_response_cache",
                    "doc_status",
                ):
                    assert getattr(rag, attr) is not None, f"{attr} not created"
            finally:
                await rag.finalize_storages()

        asyncio.run(_run())


def test_storage_backend_configuration():
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp) / "storage_workdir"

        async def _run():
            rag = _make_rag(workdir, "test_pipeline3_cpu_storage")
            assert rag.kv_storage == "JsonKVStorage"
            assert rag.vector_storage == "NanoVectorDBStorage"
            assert rag.graph_storage == "NetworkXStorage"
            assert rag.doc_status_storage == "JsonDocStatusStorage"

            await rag.initialize_storages()
            try:
                assert type(rag.full_docs).__name__ == "JsonKVStorage"
                assert type(rag.chunk_entity_relation_graph).__name__ == "NetworkXStorage"
                assert type(rag.entities_vdb).__name__ == "NanoVectorDBStorage"
                assert type(rag.doc_status).__name__ == "JsonDocStatusStorage"
            finally:
                await rag.finalize_storages()

        asyncio.run(_run())


def test_initialize_storages_sets_pipeline_status():
    from lightrag.kg.shared_storage import get_namespace_data

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp) / "pipeline_status_workdir"

        async def _run():
            rag = _make_rag(workdir, "test_pipeline3_cpu_status")
            await rag.initialize_storages()
            try:
                status = await get_namespace_data(
                    "pipeline_status", workspace=rag.workspace
                )
                assert "busy" in status
                assert status["busy"] is False
            finally:
                await rag.finalize_storages()

        asyncio.run(_run())


def test_working_directory_isolation():
    with tempfile.TemporaryDirectory() as tmp:
        custom_workdir = Path(tmp) / "isolated_workdir"
        adapter = Pipeline3LightRAGAdapter(
            model="qwen2.5:1.5b", workdir=custom_workdir
        )

        canonical = ROOT / "data" / "cache"
        assert adapter.workdir == custom_workdir
        assert adapter.workdir != canonical
        assert canonical not in adapter.workdir.parents

        captured: dict = {}

        class FakeRAG:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch("lightrag.LightRAG", FakeRAG):
            adapter._get_rag()

        assert captured["working_dir"] == str(custom_workdir)
        assert custom_workdir.exists()
        assert custom_workdir.is_dir()


def test_query_param_supported_modes():
    adapter = Pipeline3LightRAGAdapter(model="qwen2.5:1.5b")
    assert tuple(adapter.supports_modes) == LIGHTRAG_MODES

    for mode in LIGHTRAG_MODES:
        param = QueryParam(mode=mode, top_k=6, only_need_context=True)
        assert param.mode == mode
        assert param.top_k == 6
        assert param.only_need_context is True


def test_query_before_indexing_returns_error():
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp) / "unindexed_workdir"
        adapter = Pipeline3LightRAGAdapter(model="qwen2.5:1.5b", workdir=workdir)

        with patch.object(
            adapter,
            "_get_rag",
            side_effect=AssertionError("query must not initialize LightRAG when unindexed"),
        ):
            result = adapter.query("What is HNSW?", mode="hybrid")

        assert isinstance(result, PipelineResult)
        assert result.errors, "An unindexed query must return an error result"
        assert "not indexed" in result.errors[0].lower()
        assert result.answer is None
        assert adapter.is_indexed() is False


def test_dry_run_without_inference():
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp) / "dry_run_workdir"
        adapter = Pipeline3LightRAGAdapter(model="qwen2.5:1.5b", workdir=workdir)

        with patch.object(
            adapter,
            "_get_rag",
            side_effect=AssertionError("dry_run must not initialize LightRAG"),
        ):
            dry = adapter.dry_run()

        assert dry["dry_run"] is True
        assert dry["pipeline"] == "pipeline_3_lightrag"
        assert dry["lightrag_importable"] is True
        assert dry["is_indexed"] is False
        assert dry["workdir"] == str(workdir)
        assert not workdir.exists(), "dry_run must not create the storage directory"
