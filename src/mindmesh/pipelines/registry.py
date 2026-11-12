"""
registry.py — Pipeline registry for MindMesh.

Maps pipeline names to their adapter classes.  The runner imports
this module to discover available pipelines without hardcoded conditionals.

Adding a new pipeline:
  1. Create a subclass of PipelineAdapter in a dedicated module.
  2. Register it in PIPELINE_REGISTRY below.
  3. That is all — run_pipeline.py picks it up automatically.

Pipeline names follow the convention:
  pipeline_a              — flat vector baseline (Pipeline A)
  pipeline_b              — structural graph filter (Pipeline B)
  lightrag_local          — MindMesh-adapted LightRAG Local (experimental)
  lightrag_global         — MindMesh-adapted LightRAG Global (experimental)
  lightrag_hybrid         — MindMesh-adapted LightRAG Hybrid (experimental)
  lightrag_mix            — MindMesh-adapted LightRAG Mix (experimental)
  pipeline_3_lightrag     — Genuine upstream LightRAG v1.5.8 (Pipeline 3)
"""

from __future__ import annotations

from pathlib import Path
from typing import Type

from mindmesh.pipelines.base import PipelineAdapter


# ── Pipeline adapter imports ──────────────────────────────────────────────────
# Each import is lazy-ish: the module-level class definition is cheap;
# the actual LLM/graph setup happens in the adapter's __init__.

from mindmesh.pipelines.pipeline_a_adapter.adapter import PipelineAAdapter
from mindmesh.pipelines.pipeline_b_adapter.adapter import PipelineBAdapter
from mindmesh.pipelines.lightrag_experimental.adapter import (
    LightRAGLocalAdapter,
    LightRAGGlobalAdapter,
    LightRAGHybridAdapter,
    LightRAGMixAdapter,
)
from mindmesh.pipelines.pipeline3_lightrag.adapter import Pipeline3LightRAGAdapter


# ── Registry ──────────────────────────────────────────────────────────────────

#: Maps pipeline name → adapter class.
PIPELINE_REGISTRY: dict[str, Type[PipelineAdapter]] = {
    "pipeline_a":          PipelineAAdapter,
    "pipeline_b":          PipelineBAdapter,
    "lightrag_local":      LightRAGLocalAdapter,
    "lightrag_global":     LightRAGGlobalAdapter,
    "lightrag_hybrid":     LightRAGHybridAdapter,
    "lightrag_mix":        LightRAGMixAdapter,
    "pipeline_3_lightrag": Pipeline3LightRAGAdapter,
}


def list_pipelines() -> list[dict]:
    """Return info about every registered pipeline."""
    rows = []
    for name, cls in PIPELINE_REGISTRY.items():
        rows.append({
            "name":        name,
            "class":       cls.__name__,
            "description": getattr(cls, "description", ""),
            "modes":       list(getattr(cls, "supports_modes", ("default",))),
        })
    return rows


def get_adapter_class(pipeline_name: str) -> Type[PipelineAdapter]:
    """
    Return the adapter class for the given pipeline name.

    Raises
    ------
    ValueError
        If the pipeline name is not registered.
    """
    if pipeline_name not in PIPELINE_REGISTRY:
        available = ", ".join(sorted(PIPELINE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown pipeline {pipeline_name!r}. "
            f"Available: {available}. "
            f"Use --list-pipelines to see all options."
        )
    return PIPELINE_REGISTRY[pipeline_name]


def build_adapter(
    pipeline_name: str,
    model: str,
    exp_dir: Path | None = None,
    top_k: int = 6,
    **kwargs,
) -> PipelineAdapter:
    """
    Instantiate and return the pipeline adapter for the given name.

    Parameters
    ----------
    pipeline_name : str
        One of the registered pipeline names.
    model : str
        The model tag to use for LLM calls (must be installed locally).
    exp_dir : Path | None
        Experiment directory for pipelines that need private Zvec collections.
    top_k : int
        Number of evidence items to retrieve.
    **kwargs
        Additional arguments forwarded to the adapter constructor.

    Returns
    -------
    PipelineAdapter
        A ready-to-use pipeline adapter (no inference performed yet).
    """
    cls = get_adapter_class(pipeline_name)
    return cls(model=model, exp_dir=exp_dir, top_k=top_k, **kwargs)
