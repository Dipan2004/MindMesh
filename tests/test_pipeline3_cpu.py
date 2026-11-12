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
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (
    str(ROOT / "src"),
    str(ROOT / "LightRAG"),
    str(ROOT),
    str(ROOT / "eval"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lightrag import LightRAG, QueryParam  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

from manual_pipeline3_integration import (  # noqa: E402
    _indexing_success_criteria,
    _insertion_success_criteria,
    _is_placeholder_label,
    _status_buckets,
    _storage_flush_evidence,
)

from mindmesh.pipelines.base import PipelineResult  # noqa: E402
from mindmesh.pipelines.pipeline3_lightrag.adapter import (  # noqa: E402
    LIGHTRAG_MODES,
    Pipeline3LightRAGAdapter,
    _check_lightrag_importable,
    _extract_ollama_embeddings,
    _make_ollama_embed_func,
    _normalize_embeddings,
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


def test_adapter_passes_configured_embedding_dim_to_factory():
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp) / "configured_dim_workdir"
        adapter = Pipeline3LightRAGAdapter(
            model="qwen2.5:1.5b",
            workdir=workdir,
            embedding_dim=EMBED_DIM,
        )
        captured: dict = {}

        class FakeRAG:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch("lightrag.LightRAG", FakeRAG):
            adapter._get_rag()

        assert adapter.embedding_dim == EMBED_DIM
        assert captured["embedding_func"].embedding_dim == EMBED_DIM


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


# ---------------------------------------------------------------------------
# Embedding contract: Ollama list payload -> LightRAG 2D float32 ndarray
# ---------------------------------------------------------------------------

def _vector(start: float = 0.1) -> list[float]:
    return [start + (i * 0.01) for i in range(EMBED_DIM)]


def test_single_text_embedding_conversion():
    result = _normalize_embeddings([_vector()], "alpha", EMBED_DIM)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, EMBED_DIM)
    np.testing.assert_allclose(result[0], np.asarray(_vector(), dtype=np.float32))


def test_multi_text_embedding_conversion_preserves_order():
    first = _vector(1.0)
    second = _vector(2.0)
    result = _normalize_embeddings([first, second], ["alpha", "beta"], EMBED_DIM)
    assert result.shape == (2, EMBED_DIM)
    np.testing.assert_allclose(result[0], np.asarray(first, dtype=np.float32))
    np.testing.assert_allclose(result[1], np.asarray(second, dtype=np.float32))


def test_embedding_conversion_returns_numpy_array():
    result = _normalize_embeddings([_vector()], ["alpha"], EMBED_DIM)
    assert type(result) is np.ndarray


def test_embedding_conversion_is_two_dimensional():
    result = _normalize_embeddings([_vector()], ["alpha"], EMBED_DIM)
    assert result.ndim == 2
    assert result.shape == (1, EMBED_DIM)


def test_embedding_conversion_uses_float32():
    result = _normalize_embeddings([_vector()], ["alpha"], EMBED_DIM)
    assert result.dtype == np.float32


def test_embedding_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError, match="dimension mismatch"):
        _normalize_embeddings([_vector()[:-1]], ["alpha"], EMBED_DIM)


def test_embedding_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="count mismatch"):
        _normalize_embeddings([_vector()], ["alpha", "beta"], EMBED_DIM)


def test_empty_batch_returns_empty_2d_array_without_provider_call():
    result = _normalize_embeddings([], [], EMBED_DIM)
    assert result.shape == (0, EMBED_DIM)
    assert result.dtype == np.float32

    with pytest.raises(ValueError, match="empty input batch"):
        _normalize_embeddings([_vector()], [], EMBED_DIM)

    embed = _make_ollama_embed_func(embedding_dim=EMBED_DIM)
    with patch(
        "mindmesh.pipelines.pipeline3_lightrag.adapter._post_ollama_embed",
        new=AsyncMock(side_effect=AssertionError("provider must not be called")),
    ):
        result = asyncio.run(embed([]))
    assert result.shape == (0, EMBED_DIM)
    assert result.dtype == np.float32


def test_invalid_embedding_payloads_are_rejected():
    with pytest.raises(ValueError, match="missing a list"):
        _extract_ollama_embeddings({"not_embeddings": []})

    with pytest.raises(ValueError, match="list of vectors"):
        _normalize_embeddings("not-a-list", ["alpha"], EMBED_DIM)

    with pytest.raises(ValueError, match="non-numeric"):
        _normalize_embeddings([["x"] * EMBED_DIM], ["alpha"], EMBED_DIM)

    with pytest.raises(ValueError, match="must be a sequence"):
        _normalize_embeddings([42], ["alpha"], EMBED_DIM)


def test_nan_and_infinity_are_rejected():
    nan_vector = _vector()
    nan_vector[0] = float("nan")
    with pytest.raises(ValueError, match="NaN or infinite"):
        _normalize_embeddings([nan_vector], ["alpha"], EMBED_DIM)

    inf_vector = _vector()
    inf_vector[0] = float("inf")
    with pytest.raises(ValueError, match="NaN or infinite"):
        _normalize_embeddings([inf_vector], ["alpha"], EMBED_DIM)


def test_adapter_embedding_func_integrates_with_upstream_wrapper():
    embed = _make_ollama_embed_func(embedding_dim=EMBED_DIM)
    assert isinstance(embed, EmbeddingFunc)
    assert embed.embedding_dim == EMBED_DIM

    async def _run():
        with patch(
            "mindmesh.pipelines.pipeline3_lightrag.adapter._post_ollama_embed",
            new=AsyncMock(
                return_value={"embeddings": [_vector(1.0), _vector(2.0)]}
            ),
        ):
            multi = await embed(["alpha", "beta"])
        assert isinstance(multi, np.ndarray)
        assert multi.shape == (2, EMBED_DIM)
        assert multi.dtype == np.float32
        np.testing.assert_allclose(
            multi[0], np.asarray(_vector(1.0), dtype=np.float32)
        )

        with patch(
            "mindmesh.pipelines.pipeline3_lightrag.adapter._post_ollama_embed",
            new=AsyncMock(return_value={"embeddings": [_vector(3.0)]}),
        ):
            single = await embed("alpha")
        assert single.shape == (1, EMBED_DIM)
        assert single.dtype == np.float32

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Extraction and indexing success-criteria diagnostics (CPU-only)
# ---------------------------------------------------------------------------

def test_placeholder_entity_record_is_rejected():
    from lightrag.operate import _handle_single_entity_extraction

    record = [
        "entity",
        "<entity_name>",
        "<entity_type>",
        "<entity_description>",
    ]
    assert _handle_single_entity_extraction(record, "chunk-1", 0) is None


def test_malformed_entity_record_is_rejected():
    from lightrag.operate import _handle_single_entity_extraction

    assert _handle_single_entity_extraction(
        ["entity", "Hybrid Search", "concept"], "chunk-1", 0
    ) is None


def test_valid_entity_record_is_accepted():
    from lightrag.operate import _handle_single_entity_extraction

    record = _handle_single_entity_extraction(
        ["entity", "Hybrid Search", "concept", "Combines dense and sparse retrieval."],
        "chunk-1",
        0,
    )
    assert record is not None
    assert record["entity_name"] == "Hybrid Search"
    assert record["entity_type"] == "concept"


def test_placeholder_graph_labels_are_detected():
    assert _is_placeholder_label("<entity_name>")
    assert _is_placeholder_label("<source_entity>")
    assert not _is_placeholder_label("Hybrid Search")


def test_status_buckets_classify_failed_and_pending_documents():
    buckets = _status_buckets({
        "doc-1": {"present": True, "status": "processed", "error_msg": None},
        "doc-2": {"present": True, "status": "failed", "error_msg": "flush failed"},
        "doc-3": {"present": True, "status": "pending", "error_msg": None},
        "doc-4": {"present": False, "status": None, "error_msg": None},
    })
    assert buckets["processed"] == ["doc-1"]
    assert buckets["failed"] == ["doc-2"]
    assert buckets["pending_or_active"] == ["doc-3"]
    assert buckets["missing"] == ["doc-4"]


def test_insertion_success_requires_processed_status_not_just_no_exception():
    assert _insertion_success_criteria([{"ok": True}], True) is True
    assert _insertion_success_criteria([{"ok": True}], False) is False
    assert _insertion_success_criteria(
        [{"ok": True}, {"ok": False}], True
    ) is False
    assert _insertion_success_criteria([], True) is False


def test_indexing_success_criteria_rejects_failed_status_and_flush():
    result = _indexing_success_criteria(
        docs_present=True,
        statuses_all_processed=False,
        flush_ok=False,
        graph_nonempty=True,
        non_placeholder_count=2,
    )
    assert result["ok"] is False
    assert "document_status_incomplete" in result["failures"]
    assert "storage_flush_missing" in result["failures"]

    success = _indexing_success_criteria(
        docs_present=True,
        statuses_all_processed=True,
        flush_ok=True,
        graph_nonempty=True,
        non_placeholder_count=2,
    )
    assert success == {"ok": True, "failures": []}


def test_storage_flush_evidence_requires_all_vector_files():
    complete = _storage_flush_evidence([
        {"path": "pipeline3/vdb_entities.json"},
        {"path": "pipeline3/vdb_relationships.json"},
        {"path": "pipeline3/vdb_chunks.json"},
    ])
    assert complete["ok"] is True

    incomplete = _storage_flush_evidence([
        {"path": "pipeline3/vdb_entities.json"},
    ])
    assert incomplete["ok"] is False
    assert set(incomplete["missing"]) == {
        "vdb_relationships.json",
        "vdb_chunks.json",
    }

    entities_and_chunks_only = _storage_flush_evidence(
        [
            {"path": "pipeline3/vdb_entities.json"},
            {"path": "pipeline3/vdb_chunks.json"},
        ],
        expect_entities=True,
        expect_relationships=False,
        expect_chunks=True,
    )
    assert entities_and_chunks_only["ok"] is True


def test_placeholder_entity_output_rejected_by_validator():
    from mindmesh.pipelines.pipeline3_lightrag.adapter import _make_pipeline3_extraction_validator

    validator = _make_pipeline3_extraction_validator()
    maybe_nodes = {"<entity_name>": [{"entity_name": "<entity_name>", "entity_type": "<entity_type>", "description": "<entity_description>"}]}
    maybe_edges = {}
    try:
        validator("chunk-1", "some text", maybe_nodes, maybe_edges)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_placeholder_relationship_output_rejected_by_validator():
    from mindmesh.pipelines.pipeline3_lightrag.adapter import _make_pipeline3_extraction_validator

    validator = _make_pipeline3_extraction_validator()
    maybe_nodes = {}
    maybe_edges = {
        ("<source_entity>", "<target_entity>"): [
            {
                "src_id": "<source_entity>",
                "tgt_id": "<target_entity>",
                "description": "<relationship_description>",
                "keywords": "<relationship_keywords>",
            }
        ]
    }
    try:
        validator("chunk-1", "some text", maybe_nodes, maybe_edges)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_malformed_extraction_output_rejected_when_only_placeholders():
    from mindmesh.pipelines.pipeline3_lightrag.adapter import _make_pipeline3_extraction_validator

    validator = _make_pipeline3_extraction_validator()
    maybe_nodes = {"X": [{"entity_name": "X", "entity_type": "<entity_type>", "description": "bad"}]}
    maybe_edges = {}
    try:
        validator("chunk-1", "some text", maybe_nodes, maybe_edges)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_valid_entity_and_relationship_output_passes_validator():
    from mindmesh.pipelines.pipeline3_lightrag.adapter import _make_pipeline3_extraction_validator

    validator = _make_pipeline3_extraction_validator()
    maybe_nodes = {
        "Hybrid Search": [
            {"entity_name": "Hybrid Search", "entity_type": "method", "description": "Combines vector and keyword search."}
        ],
        "Vector Search": [
            {"entity_name": "Vector Search", "entity_type": "method", "description": "Semantic retrieval using embeddings."}
        ],
    }
    maybe_edges = {
        ("Hybrid Search", "Vector Search"): [
            {"src_id": "Hybrid Search", "tgt_id": "Vector Search", "description": "Hybrid search combines vector search.", "keywords": "combines, leverages"}
        ]
    }
    nodes, edges = validator("chunk-1", "some text", maybe_nodes, maybe_edges)
    assert "Hybrid Search" in nodes
    assert ("Hybrid Search", "Vector Search") in edges


def test_empty_extraction_output_reported_as_extraction_failure():
    from mindmesh.pipelines.pipeline3_lightrag.adapter import _make_pipeline3_extraction_validator

    validator = _make_pipeline3_extraction_validator()
    try:
        validator("chunk-1", "hybrid search combines vector and keyword search", {}, {})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "no entities/relationships" in str(exc)


def test_no_placeholder_labels_in_final_graph():
    from mindmesh.pipelines.pipeline3_lightrag.adapter import _is_placeholder_text

    assert _is_placeholder_text("<source_entity>") is True
    assert _is_placeholder_text("<target_entity>") is True
    assert _is_placeholder_text("Hybrid Search") is False
    assert _is_placeholder_text("") is True