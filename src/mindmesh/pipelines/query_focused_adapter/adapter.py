"""
query_focused_adapter/adapter.py -- prototype pipeline "query_focused_c5".

Dense top-5 + one title/entity expansion hop -> temporary per-question graph
from cached per-paragraph extractions -> <= 2-hop BFS from question entities ->
rank by path support, dense order as fallback.  See
graphrag_lite/retrieval/query_focused.py.  The four existing pipelines are untouched.
"""
from __future__ import annotations

import time
from pathlib import Path

from mindmesh.pipelines.base import PipelineAdapter, PipelineResult


class QueryFocusedC5Adapter(PipelineAdapter):
    name = "query_focused_c5"
    description = ("Query-focused graph (prototype): Dense top-5 + title/entity expansion, "
                   "temporary per-question graph from cached extractions, 2-hop BFS ranking.")
    supports_modes = ("default",)

    def __init__(self, model: str, exp_dir: Path | None = None, top_k: int = 10) -> None:
        if exp_dir is None:
            raise ValueError("query_focused_c5 needs an index directory with extractions.jsonl")
        super().__init__(model=model, exp_dir=exp_dir, top_k=top_k)

    def query(self, question: str, mode: str = "default", generate_answer: bool = False) -> PipelineResult:
        from graphrag_lite.retrieval.query_focused import retrieve_query_focused
        t0 = time.monotonic()
        ev, info = retrieve_query_focused(question, self.exp_dir, top_k=self.top_k)
        lat = time.monotonic() - t0
        return PipelineResult(
            query=question, pipeline=self.name, model=self.model, mode=mode,
            retrieved_chunks=ev, source_docs=sorted({e["source_doc"] for e in ev if e["source_doc"]}),
            retrieval_latency_s=lat, total_latency_s=lat, raw_output=info,
        )
