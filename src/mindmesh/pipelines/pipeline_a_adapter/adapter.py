"""
Pipeline A adapter — flat vector-only baseline.

Wraps graphrag_lite.retrieval.pipeline_a.retrieve_flat().
No changes to the underlying implementation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from mindmesh.pipelines.base import PipelineAdapter, PipelineResult


class PipelineAAdapter(PipelineAdapter):
    name = "pipeline_a"
    description = (
        "Flat vector-only baseline. Single hybrid (HNSW+BM25) query "
        "against the canonical chunks collection. No graph."
    )
    supports_modes = ("default",)

    def query(
        self,
        question: str,
        mode: str = "default",
        generate_answer: bool = False,
    ) -> PipelineResult:
        from graphrag_lite.retrieval.pipeline_a import retrieve_flat

        t0 = time.monotonic()
        evidence = retrieve_flat(question)
        if self.top_k:
            evidence = evidence[:self.top_k]
        retrieval_lat = time.monotonic() - t0

        chunks = [
            {
                "chunk_id":    e.chunk_id,
                "source_doc":  e.source_doc,
                "chunk_index": e.chunk_index,
                "text":        e.text,
                "score":       e.score,
                "source":      e.source,
            }
            for e in evidence
        ]
        source_docs = sorted({c["source_doc"] for c in chunks if c["source_doc"]})

        # Optional answer generation
        answer = None
        gen_lat = 0.0
        if generate_answer and evidence:
            from graphrag_lite.llm.client_factory import get_llm_client
            from graphrag_lite.llm.generator import AnswerGenerator
            client = get_llm_client("local")
            gen = AnswerGenerator(client=client)
            ev_dicts = [{"text": e.text, "chunk_id": e.chunk_id} for e in evidence]
            t_gen = time.monotonic()
            ans = gen.generate_answer(question, ev_dicts)
            gen_lat = time.monotonic() - t_gen
            answer = ans.text

        total_lat = time.monotonic() - t0
        return PipelineResult(
            query=question,
            pipeline=self.name,
            model=self.model,
            mode=mode,
            retrieved_chunks=chunks,
            source_docs=source_docs,
            answer=answer,
            retrieval_latency_s=retrieval_lat,
            generation_latency_s=gen_lat,
            total_latency_s=total_lat,
        )
