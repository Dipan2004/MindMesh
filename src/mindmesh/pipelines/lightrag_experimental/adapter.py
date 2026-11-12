"""
lightrag_experimental/adapter.py — Adapters for MindMesh-native LightRAG retrievers.

These wrap the four experimental LightRAG-style retrievers already in
src/graphrag_lite/retrieval/lightrag/:
  - retrieve_local
  - retrieve_global
  - retrieve_hybrid
  - retrieve_mix

IMPORTANT DISTINCTION
---------------------
These are MindMesh-NATIVE implementations adapted from the LightRAG paper
(arXiv:2410.05779). They are NOT the upstream lightrag-hku package.

The upstream package-based Pipeline 3 is in:
  src/mindmesh/pipelines/pipeline3_lightrag/adapter.py

These experimental adapters are preserved as separate research baselines.
They require an experiment directory with pre-built Zvec collections
(entities/, relationships/, chunks/).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from mindmesh.pipelines.base import PipelineAdapter, PipelineResult


def _default_exp_dir() -> Path:
    """Return the default experiment directory (Phase 1 validation results)."""
    return ROOT / "data" / "experiments" / "qwen2_5_1_5b_validate"


def _load_graph(exp_dir: Path):
    """Load the graph and PageRank from an experiment directory's extractions."""
    import io, contextlib
    from graphrag_lite.graph.builder import build_graph, load_extractions
    from graphrag_lite.graph.ranking import rank_pagerank

    # Try experiment extractions first, fall back to canonical
    ext_paths = [
        exp_dir / "extractions.jsonl",
        ROOT / "data" / "cache_v2" / "extractions_v2_off.jsonl",
        ROOT / "data" / "cache" / "extractions.jsonl",
    ]
    records = None
    for p in ext_paths:
        if p.exists():
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                records = load_extractions(str(p))
            break
    if records is None:
        raise FileNotFoundError(
            "No extractions.jsonl found. Run an extraction experiment first."
        )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        G = build_graph(records)
    pr = rank_pagerank(G)
    return G, pr


class _ExperimentalBase(PipelineAdapter):
    """Shared base for all experimental LightRAG adapters."""

    def __init__(
        self,
        model: str,
        exp_dir: Path | None = None,
        top_k: int = 6,
    ) -> None:
        effective_dir = exp_dir or _default_exp_dir()
        super().__init__(model=model, exp_dir=effective_dir, top_k=top_k)
        self._G = None
        self._pr = None

    def _get_graph(self):
        if self._G is None:
            self._G, self._pr = _load_graph(self.exp_dir)
        return self._G, self._pr

    def _chunks_to_result(
        self, question: str, chunks: list[dict], retrieval_lat: float
    ) -> PipelineResult:
        source_docs = sorted({c.get("source_doc", "") for c in chunks if c.get("source_doc")})
        return PipelineResult(
            query=question,
            pipeline=self.name,
            model=self.model,
            mode="default",
            retrieved_chunks=chunks,
            source_docs=source_docs,
            retrieval_latency_s=retrieval_lat,
            total_latency_s=retrieval_lat,
        )


class LightRAGLocalAdapter(_ExperimentalBase):
    name = "lightrag_local"
    description = (
        "MindMesh-native LightRAG Local: entity search -> one-hop edge "
        "expansion -> provenance lookup. Adapted from arXiv:2410.05779. "
        "NOT the upstream lightrag-hku package."
    )
    supports_modes = ("default",)

    def query(self, question: str, mode: str = "default", generate_answer: bool = False) -> PipelineResult:
        from graphrag_lite.retrieval.lightrag import retrieve_local
        G, pr = self._get_graph()
        t0 = time.monotonic()
        chunks = retrieve_local(question, G, pr, self.exp_dir, top_k=self.top_k)
        return self._chunks_to_result(question, chunks, time.monotonic() - t0)


class LightRAGGlobalAdapter(_ExperimentalBase):
    name = "lightrag_global"
    description = (
        "MindMesh-native LightRAG Global: relationship search -> endpoint "
        "expansion -> provenance lookup. Adapted from arXiv:2410.05779."
    )
    supports_modes = ("default",)

    def query(self, question: str, mode: str = "default", generate_answer: bool = False) -> PipelineResult:
        from graphrag_lite.retrieval.lightrag import retrieve_global
        G, pr = self._get_graph()
        t0 = time.monotonic()
        chunks = retrieve_global(question, G, self.exp_dir, top_k=self.top_k)
        return self._chunks_to_result(question, chunks, time.monotonic() - t0)


class LightRAGHybridAdapter(_ExperimentalBase):
    name = "lightrag_hybrid"
    description = (
        "MindMesh-native LightRAG Hybrid: merge of Local + Global evidence. "
        "Adapted from arXiv:2410.05779."
    )
    supports_modes = ("default",)

    def query(self, question: str, mode: str = "default", generate_answer: bool = False) -> PipelineResult:
        from graphrag_lite.retrieval.lightrag import retrieve_hybrid
        G, pr = self._get_graph()
        t0 = time.monotonic()
        chunks = retrieve_hybrid(question, G, pr, self.exp_dir, top_k=self.top_k)
        return self._chunks_to_result(question, chunks, time.monotonic() - t0)


class LightRAGMixAdapter(_ExperimentalBase):
    name = "lightrag_mix"
    description = (
        "MindMesh-native LightRAG Mix: Hybrid graph + direct chunk vector search. "
        "Adapted from arXiv:2410.05779."
    )
    supports_modes = ("default",)

    def query(self, question: str, mode: str = "default", generate_answer: bool = False) -> PipelineResult:
        from graphrag_lite.retrieval.lightrag import retrieve_mix
        G, pr = self._get_graph()
        t0 = time.monotonic()
        chunks = retrieve_mix(question, G, pr, self.exp_dir, top_k=self.top_k)
        return self._chunks_to_result(question, chunks, time.monotonic() - t0)
