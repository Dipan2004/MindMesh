"""
query_focused_ondemand.py -- query-focused C5 with genuine on-demand extraction.

Same retrieval design as the prototype (graphrag_lite.retrieval.query_focused, unchanged
and imported here): Dense top-5 seeds -> title/entity expansion hop -> temporary graph ->
<= 2-hop BFS from question entities -> path-support ranking with dense fallback.

The only difference: paragraph records come from a LazyRecords store backed by an
ExtractionCache. Asking whether a paragraph has a record looks it up in the cache and,
on a miss, extracts it right then (measured). Only `ok` entries are records; failed
entries are excluded from graph construction and traversal.
"""

from __future__ import annotations

import contextlib
import io
import time

from graphrag_lite.graph.builder import _norm, build_graph
from graphrag_lite.retrieval import query_focused as QF
from graphrag_lite.retrieval.extraction_cache import ExtractionCache


class LazyRecords:
    """Mapping-like view: chunk_id -> ok extraction record, extracting cache misses on demand."""

    def __init__(self, cache: ExtractionCache, paras: dict[str, dict]):
        self.cache, self.paras = cache, paras
        self.start_question("")

    def start_question(self, qid: str) -> None:
        self.qid = qid
        self.memo: dict[str, dict | None] = {}
        self.q = {"lookups": 0, "hits_ok": 0, "hits_failed": 0, "misses": 0, "extracted_ok": 0,
                  "extracted_failed_model": 0, "extracted_failed_infra": 0,
                  "extraction_wall_s": 0.0, "extraction_eval_tokens": 0, "extraction_eval_s": 0.0,
                  "extraction_cost_usd": 0.0, "extraction_prompt_tokens": 0,
                  "extracted_chunk_ids": []}

    def _ensure(self, cid: str) -> dict | None:
        if cid in self.memo:
            return self.memo[cid]
        p = self.paras.get(cid)
        if p is None:
            self.memo[cid] = None
            return None
        self.q["lookups"] += 1
        e = self.cache.get(p["text"])
        if e is not None:
            self.q["hits_ok" if e["status"] == "ok" else "hits_failed"] += 1
        else:
            self.q["misses"] += 1
            e = self.cache.extract(p, self.qid)
            self.q[f"extracted_{e['status']}"] += 1
            self.q["extraction_wall_s"] += e["extraction_wall_s"]
            self.q["extraction_eval_tokens"] += e["eval_tokens"]
            self.q["extraction_eval_s"] += e["eval_s"]
            self.q["extraction_cost_usd"] += e.get("cost_usd") or 0.0
            self.q["extraction_prompt_tokens"] += e.get("prompt_tokens") or 0
            self.q["extracted_chunk_ids"].append(cid)
        self.memo[cid] = e
        return e

    def __contains__(self, cid: str) -> bool:
        e = self._ensure(cid)
        return e is not None and e["status"] == "ok"

    def __getitem__(self, cid: str) -> dict:
        e = self._ensure(cid)
        if e is None or e["status"] != "ok":
            raise KeyError(cid)
        return {"chunk_id": e["chunk_id"], "doc_id": e["doc_id"], "chunk_index": 0, "text": e["text"],
                "title": e["title"], "entities": e["entities"], "relationships": e["relationships"]}


class OnDemandIndex(QF.QueryFocusedIndex):
    """Same title indexes and expansion as the prototype; records are lazy and cache-backed."""

    def __init__(self, paras: dict[str, dict], cache: ExtractionCache):  # noqa: D401 -- no super().__init__
        self.exp_dir = None
        self.titles = {cid: p["title"] for cid, p in paras.items()}
        self.by_full_title, self.by_base_title = {}, {}
        for cid, t in self.titles.items():
            self.by_full_title.setdefault(_norm(t), []).append(cid)
            self.by_base_title.setdefault(QF._base_title(t), []).append(cid)
        self.records = LazyRecords(cache, paras)
        self.touched: set[str] = set()


def retrieve_c5_ondemand(query: str, qid: str, idx: OnDemandIndex, top_k: int = 10) -> tuple[list[str], dict]:
    """Return (ranked chunk ids, diagnostics incl. timing and cache accounting)."""
    t0 = time.perf_counter()
    dense = QF.dense_ranking(query, top_k)                       # exactly the Pipeline A query
    t_dense = time.perf_counter() - t0
    idx.records.start_question(qid)
    t1 = time.perf_counter()
    ranked, info = QF.rank_query_focused(query, idx, dense, top_k)   # unchanged prototype logic
    t_rank_total = time.perf_counter() - t1
    q = dict(idx.records.q)
    # graph construction timed on its own (rebuild of the same ok records; traversal = remainder)
    sel = info.get("selected_paragraphs", [])
    recs = [idx.records[c] for c in sel if c in idx.records]          # all memoised: no extraction here
    t2 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        build_graph(recs)
    t_graph = time.perf_counter() - t2
    t_graph_traverse = max(t_rank_total - q["extraction_wall_s"], 0.0)
    info.update({
        "dense_ids": dense, "valid_records_used": len(recs),
        "selected_failed_or_missing": len(sel) - len(recs),
        "timing_s": {"dense_retrieval": round(t_dense, 4), "extraction": round(q["extraction_wall_s"], 3),
                     "graph_construction": round(t_graph, 4),
                     "traversal_and_ranking": round(max(t_graph_traverse - t_graph, 0.0), 4),
                     "graph_construction_plus_traversal": round(t_graph_traverse, 4),
                     "retrieval_total": round(time.perf_counter() - t0 - t_graph, 3)},
        "cache": q,
    })
    return ranked, info
