"""
base.py — Pipeline adapter base class and standardized result type.

Every pipeline in MindMesh exposes the same interface so run_pipeline.py
does not need a large collection of conditionals.

Standardized result schema
--------------------------
Every pipeline returns a PipelineResult.  The fields are:

  query              — the original query string
  pipeline           — the pipeline name (e.g. "pipeline_a", "pipeline_3_lightrag")
  model              — the LLM/extraction model used
  mode               — the retrieval mode within the pipeline (e.g. "hybrid")
  retrieved_chunks   — list of retrieved evidence dicts
  entities           — list of relevant entities (graph-based pipelines)
  relationships      — list of relevant relationships (graph-based pipelines)
  context            — the assembled context string passed to the LLM
  answer             — the generated answer (None if eval_only=True)
  source_docs        — set of source document IDs in the retrieved set
  retrieval_latency_s — retrieval wall time
  generation_latency_s — generation wall time (0.0 if eval_only)
  total_latency_s    — total wall time
  errors             — list of error strings (empty on success)
  raw_output         — pipeline-specific raw output for debugging
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineResult:
    """Standardized result returned by every pipeline adapter."""
    query:               str
    pipeline:            str
    model:               str
    mode:                str                = "default"
    retrieved_chunks:    list[dict]         = field(default_factory=list)
    entities:            list[dict]         = field(default_factory=list)
    relationships:       list[dict]         = field(default_factory=list)
    context:             str                = ""
    answer:              str | None         = None
    source_docs:         list[str]          = field(default_factory=list)
    retrieval_latency_s: float              = 0.0
    generation_latency_s: float            = 0.0
    total_latency_s:     float              = 0.0
    errors:              list[str]          = field(default_factory=list)
    raw_output:          Any                = None

    @property
    def chunk_ids(self) -> list[str]:
        return [c.get("chunk_id", c.get("id", "")) for c in self.retrieved_chunks]

    @property
    def retrieved_count(self) -> int:
        return len(self.retrieved_chunks)

    def to_dict(self) -> dict:
        return {
            "query":               self.query,
            "pipeline":            self.pipeline,
            "model":               self.model,
            "mode":                self.mode,
            "retrieved_count":     self.retrieved_count,
            "chunk_ids":           self.chunk_ids,
            "source_docs":         self.source_docs,
            "entities_count":      len(self.entities),
            "relationships_count": len(self.relationships),
            "answer":              self.answer,
            "retrieval_latency_s": round(self.retrieval_latency_s, 4),
            "generation_latency_s": round(self.generation_latency_s, 4),
            "total_latency_s":     round(self.total_latency_s, 4),
            "errors":              self.errors,
            "retrieved_chunks":    self.retrieved_chunks,
        }


class PipelineAdapter(ABC):
    """
    Abstract base class for all pipeline adapters.

    Subclasses implement ``query()`` to execute retrieval (and optionally
    answer generation).  The adapter is responsible for:
      - Validating its own dependencies at construction time.
      - Returning a PipelineResult with canonical chunk_ids preserved.
      - NOT performing model inference in ``__init__``.
    """

    #: Override in subclasses
    name: str = "base"
    description: str = "Base pipeline adapter"
    supports_modes: tuple[str, ...] = ("default",)

    def __init__(
        self,
        model: str,
        exp_dir: Path | None = None,
        top_k: int = 6,
    ) -> None:
        self.model = model
        self.exp_dir = exp_dir
        self.top_k = top_k

    @abstractmethod
    def query(
        self,
        question: str,
        mode: str = "default",
        generate_answer: bool = False,
    ) -> PipelineResult:
        """
        Execute retrieval for the given question.

        Parameters
        ----------
        question : str
            Natural language question.
        mode : str
            Pipeline-specific retrieval mode (e.g. "hybrid" for Pipeline 3).
        generate_answer : bool
            If True, generate a natural-language answer. If False, return
            retrieved evidence only (no LLM generation call).

        Returns
        -------
        PipelineResult
        """

    def dry_run(self) -> dict:
        """
        Return configuration information without executing any inference.
        Used by --dry-run in run_pipeline.py.
        """
        return {
            "pipeline": self.name,
            "description": self.description,
            "model": self.model,
            "top_k": self.top_k,
            "exp_dir": str(self.exp_dir) if self.exp_dir else None,
            "supports_modes": list(self.supports_modes),
            "dry_run": True,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model!r})"
