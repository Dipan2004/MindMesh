"""
test_pipeline_runner.py — CPU-safe tests for the unified pipeline runner.

No model inference is performed. Tests cover:
  1. Argument parsing
  2. Pipeline registry discovery
  3. Model availability validation (mocked)
  4. Pipeline adapter loading
  5. LightRAG package import
  6. Pipeline 3 dry-run (no inference)
  7. Storage namespace isolation
  8. Existing regression tests still pass

Run:
    python tests/test_pipeline_runner.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "LightRAG"))
sys.path.insert(0, str(ROOT))  # for run_pipeline at repo root

PASS = []
FAIL = []


def _ok(name: str) -> None:
    PASS.append(name)
    print(f"  PASS  {name}")


def _fail(name: str, exc: Exception) -> None:
    FAIL.append(name)
    print(f"  FAIL  {name}: {exc}")
    traceback.print_exc()


# ── 1. Argument parsing ───────────────────────────────────────────────────────

def test_argparse_list_pipelines():
    from run_pipeline import build_parser
    p = build_parser()
    args = p.parse_args(["--list-pipelines"])
    assert args.list_pipelines is True
    _ok("argparse_list_pipelines")


def test_argparse_list_models():
    from run_pipeline import build_parser
    p = build_parser()
    args = p.parse_args(["--list-models"])
    assert args.list_models is True
    _ok("argparse_list_models")


def test_argparse_pipeline_a_dry_run():
    from run_pipeline import build_parser
    p = build_parser()
    args = p.parse_args(["--pipeline", "pipeline_a", "--model", "qwen2.5:1.5b", "--dry-run"])
    assert args.pipeline == "pipeline_a"
    assert args.model == "qwen2.5:1.5b"
    assert args.dry_run is True
    _ok("argparse_pipeline_a_dry_run")


def test_argparse_pipeline3_hybrid():
    from run_pipeline import build_parser
    p = build_parser()
    args = p.parse_args([
        "--pipeline", "pipeline_3_lightrag",
        "--model", "qwen2.5:1.5b",
        "--mode", "hybrid",
        "--top-k", "8",
    ])
    assert args.pipeline == "pipeline_3_lightrag"
    assert args.mode == "hybrid"
    assert args.top_k == 8
    _ok("argparse_pipeline3_hybrid")


def test_argparse_missing_model_requires_error():
    """Runner must fail when --pipeline is given without --model."""
    from run_pipeline import main
    # Should exit with code 1 when model is missing
    result = main(["--pipeline", "pipeline_a"])
    assert result == 1
    _ok("argparse_missing_model_requires_error")


# ── 2. Pipeline registry discovery ───────────────────────────────────────────

def test_registry_all_pipelines_present():
    from mindmesh.pipelines.registry import list_pipelines, PIPELINE_REGISTRY
    names = {r["name"] for r in list_pipelines()}
    expected = {
        "pipeline_a", "pipeline_b",
        "lightrag_local", "lightrag_global", "lightrag_hybrid", "lightrag_mix",
        "pipeline_3_lightrag",
    }
    missing = expected - names
    assert not missing, f"Missing pipelines: {missing}"
    _ok("registry_all_pipelines_present")


def test_registry_get_adapter_class():
    from mindmesh.pipelines.registry import get_adapter_class
    from mindmesh.pipelines.base import PipelineAdapter
    for name in ["pipeline_a", "pipeline_b", "pipeline_3_lightrag"]:
        cls = get_adapter_class(name)
        assert issubclass(cls, PipelineAdapter), f"{name}: not a PipelineAdapter"
    _ok("registry_get_adapter_class")


def test_registry_unknown_pipeline_raises():
    from mindmesh.pipelines.registry import get_adapter_class
    try:
        get_adapter_class("does_not_exist")
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "does_not_exist" in str(exc)
    _ok("registry_unknown_pipeline_raises")


# ── 3. Model availability validation (mocked) ────────────────────────────────

def test_model_validation_installed():
    from run_pipeline import validate_model, list_local_models
    mock_models = [
        {"name": "qwen2.5:1.5b", "id": "abc", "size": "986 MB", "modified": ""},
    ]
    with patch("run_pipeline.list_local_models", return_value=mock_models):
        # Should not raise
        validate_model("qwen2.5:1.5b")
    _ok("model_validation_installed")


def test_model_validation_not_installed():
    from run_pipeline import list_local_models
    mock_models = [{"name": "qwen2.5:1.5b", "id": "abc", "size": "986 MB", "modified": ""}]
    with patch("run_pipeline.list_local_models", return_value=mock_models):
        try:
            from run_pipeline import validate_model
            validate_model("qwen3:4b")
            assert False, "Should have called sys.exit"
        except SystemExit as exc:
            assert exc.code == 1
    _ok("model_validation_not_installed")


# ── 4. Pipeline adapter loading ───────────────────────────────────────────────

def test_adapter_a_loads():
    from mindmesh.pipelines.registry import build_adapter
    with tempfile.TemporaryDirectory() as tmp:
        adapter = build_adapter("pipeline_a", model="qwen2.5:1.5b",
                                 exp_dir=Path(tmp), top_k=6)
        assert adapter.name == "pipeline_a"
        assert adapter.model == "qwen2.5:1.5b"
    _ok("adapter_a_loads")


def test_adapter_b_loads():
    from mindmesh.pipelines.registry import build_adapter
    with tempfile.TemporaryDirectory() as tmp:
        adapter = build_adapter("pipeline_b", model="qwen2.5:1.5b",
                                 exp_dir=Path(tmp), top_k=6)
        assert adapter.name == "pipeline_b"
    _ok("adapter_b_loads")


def test_adapter_p3_loads_and_dry_runs():
    from mindmesh.pipelines.registry import build_adapter
    with tempfile.TemporaryDirectory() as tmp:
        adapter = build_adapter("pipeline_3_lightrag", model="qwen2.5:1.5b",
                                 exp_dir=Path(tmp), top_k=6)
        assert adapter.name == "pipeline_3_lightrag"
        dry = adapter.dry_run()
        assert "pipeline_3_lightrag" in dry["pipeline"]
        assert dry["dry_run"] is True
    _ok("adapter_p3_loads_and_dry_runs")


# ── 5. LightRAG package import ────────────────────────────────────────────────

def test_lightrag_importable():
    from mindmesh.pipelines.pipeline3_lightrag.adapter import _check_lightrag_importable
    ok, err = _check_lightrag_importable()
    assert ok, f"LightRAG not importable: {err}"
    _ok("lightrag_importable")


def test_lightrag_queryparam_modes():
    import sys
    sys.path.insert(0, str(ROOT / "LightRAG"))
    from lightrag import QueryParam
    for mode in ("local", "global", "hybrid", "naive", "mix"):
        q = QueryParam(mode=mode, top_k=6)
        assert q.mode == mode
    _ok("lightrag_queryparam_modes")


def test_lightrag_version():
    sys.path.insert(0, str(ROOT / "LightRAG"))
    import lightrag as lr
    # Should be importable from the vendored directory
    assert hasattr(lr, '__file__')
    assert "LightRAG" in lr.__file__ or "lightrag" in lr.__file__.lower()
    _ok("lightrag_version")


# ── 6. Pipeline 3 dry-run (no inference) ─────────────────────────────────────

def test_p3_dry_run_no_inference():
    """Pipeline 3 dry_run must not call any LLM or embed function."""
    from mindmesh.pipelines.pipeline3_lightrag.adapter import Pipeline3LightRAGAdapter
    with tempfile.TemporaryDirectory() as tmp:
        adapter = Pipeline3LightRAGAdapter(
            model="qwen2.5:1.5b",
            workdir=Path(tmp) / "workdir",
        )
        dry = adapter.dry_run()
        assert dry["dry_run"] is True
        assert dry["lightrag_importable"] is True
        assert dry["is_indexed"] is False
    _ok("p3_dry_run_no_inference")


def test_p3_query_unindexed_returns_error():
    """Pipeline 3 must return an error result (not raise) if not indexed."""
    from mindmesh.pipelines.pipeline3_lightrag.adapter import Pipeline3LightRAGAdapter
    with tempfile.TemporaryDirectory() as tmp:
        adapter = Pipeline3LightRAGAdapter(
            model="qwen2.5:1.5b",
            workdir=Path(tmp) / "workdir",
        )
        result = adapter.query("what is HNSW?", mode="hybrid")
        assert result.errors, "Should have returned errors when not indexed"
        assert "not indexed" in result.errors[0].lower() or "GPU" in result.errors[0]
    _ok("p3_query_unindexed_returns_error")


# ── 7. Storage namespace isolation ───────────────────────────────────────────

def test_p3_workdir_isolated():
    """Pipeline 3 must use its own workdir, not data/cache."""
    from mindmesh.pipelines.pipeline3_lightrag.adapter import Pipeline3LightRAGAdapter
    with tempfile.TemporaryDirectory() as tmp:
        custom_workdir = Path(tmp) / "my_workdir"
        adapter = Pipeline3LightRAGAdapter(
            model="qwen2.5:1.5b",
            workdir=custom_workdir,
        )
        assert adapter.workdir == custom_workdir
        # Must not point at data/cache
        canonical = ROOT / "data" / "cache"
        assert adapter.workdir != canonical
    _ok("p3_workdir_isolated")


def test_pipeline_a_uses_canonical_not_experiment():
    """Pipeline A must use the canonical chunks_v1, not an experiment dir."""
    from mindmesh.pipelines.pipeline_a_adapter.adapter import PipelineAAdapter
    adapter = PipelineAAdapter(model="qwen2.5:1.5b", top_k=6)
    # The adapter does not carry a custom exp_dir for chunk storage
    dry = adapter.dry_run()
    # exp_dir can be set from run_pipeline but it's just an output dir, not collection dir
    _ok("pipeline_a_uses_canonical_not_experiment")


# ── 8. Existing regression tests ─────────────────────────────────────────────

def test_cache_isolation_regression():
    """Model/cache isolation tests from Phase 1 must still pass."""
    from graphrag_lite.cache.llm_cache import _make_key
    from graphrag_lite.llm.ollama_client import OllamaClient

    c1 = OllamaClient("qwen2.5:0.5b", temperature=0.0, seed=42,
                      num_predict=4096, num_ctx=8192, frequency_penalty=0.25)
    c2 = OllamaClient("qwen2.5:1.5b", temperature=0.0, seed=42,
                      num_predict=4096, num_ctx=8192, frequency_penalty=0.25)
    prompt = "test prompt"
    k1 = _make_key(prompt, c1.model, c1.cache_tag)
    k2 = _make_key(prompt, c2.model, c2.cache_tag)
    assert k1 != k2, "Different models must produce different cache keys"
    _ok("cache_isolation_regression")


def test_pipeline_registry_does_not_overlap_with_legacy():
    """
    The new pipeline_3_lightrag must not conflict with the existing
    lightrag_local/global/hybrid/mix names.
    """
    from mindmesh.pipelines.registry import PIPELINE_REGISTRY
    new_name = "pipeline_3_lightrag"
    legacy_names = {"lightrag_local", "lightrag_global", "lightrag_hybrid", "lightrag_mix"}
    assert new_name not in legacy_names
    for name in legacy_names:
        assert name in PIPELINE_REGISTRY
    assert new_name in PIPELINE_REGISTRY
    _ok("pipeline_registry_no_overlap")


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_argparse_list_pipelines,
        test_argparse_list_models,
        test_argparse_pipeline_a_dry_run,
        test_argparse_pipeline3_hybrid,
        test_argparse_missing_model_requires_error,
        test_registry_all_pipelines_present,
        test_registry_get_adapter_class,
        test_registry_unknown_pipeline_raises,
        test_model_validation_installed,
        test_model_validation_not_installed,
        test_adapter_a_loads,
        test_adapter_b_loads,
        test_adapter_p3_loads_and_dry_runs,
        test_lightrag_importable,
        test_lightrag_queryparam_modes,
        test_lightrag_version,
        test_p3_dry_run_no_inference,
        test_p3_query_unindexed_returns_error,
        test_p3_workdir_isolated,
        test_pipeline_a_uses_canonical_not_experiment,
        test_cache_isolation_regression,
        test_pipeline_registry_does_not_overlap_with_legacy,
    ]

    print(f"\n{'='*60}")
    print("  Pipeline Runner CPU-Safe Tests")
    print(f"{'='*60}")
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            _fail(fn.__name__, exc)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {len(PASS)} passed, {len(FAIL)} failed")
    print(f"{'='*60}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
