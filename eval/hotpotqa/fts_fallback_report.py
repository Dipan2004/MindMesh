"""
eval/hotpotqa/fts_fallback_report.py -- account for Zvec keyword-parser (BM25/FTS) fallbacks in a run.

    python eval/hotpotqa/fts_fallback_report.py --run-dir <exp>/runs/<run_id> --log <run_full.log>

Read-only. Nothing is re-queried.

How fallbacks are identified: graphrag_lite.vectorstore.zvec_client.hybrid_query sends a
dense (HNSW) + BM25 query; if Zvec's FTS parser rejects the query text it logs
"FTS query parse failed" and the code re-runs the search vector-only. Both pipelines'
first-stage search goes through that same function (Dense: retrieve_flat; C5:
query_focused.dense_ranking) with the same query text and topk. Each warning line in the
raw log is attributed to the next "[R i/100] <pipeline> <qid>" row (stdout and stderr are
written to the same file in order).

Also verifies identical first-stage behaviour directly: for every question, the C5 row's
stored dense_ids must equal the Dense row's retrieved ids.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROW = re.compile(r"^\[R (\d+)/(\d+)\] (\S+)\s+(\S+)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    args = ap.parse_args(argv)
    pending: list[str] = []
    events = []
    total_rows = 0
    for line in args.log.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "FTS query parse failed" in line:
            pending.append(line.split("FTS query parse failed:", 1)[1].strip()[:90])
            continue
        m = ROW.match(line)
        if m:
            total_rows += 1
            if pending:
                events.append({"question_no": int(m.group(1)), "pipeline": m.group(3), "qid": m.group(4),
                               "warnings": pending})
                pending = []
    rows = [json.loads(l) for l in (args.run_dir / "retrieval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {(r["pipeline"], r["qid"]): r for r in rows}
    qids = sorted({r["qid"] for r in rows})
    identical, differ = 0, []
    for x in qids:
        a, c = by.get(("pipeline_a", x)), by.get(("c5_ondemand", x))
        if a and c and "diagnostics" in c:
            if a["retrieved_chunk_ids"] == c["diagnostics"]["dense_ids"][:len(a["retrieved_chunk_ids"])]:
                identical += 1
            else:
                differ.append(x)
    calls = {"pipeline_a": sum(1 for r in rows if r["pipeline"] == "pipeline_a"),
             "c5_ondemand": sum(1 for r in rows if r["pipeline"] == "c5_ondemand")}
    fb = {p: sorted({e["qid"] for e in events if e["pipeline"] == p}) for p in calls}
    out = {
        "first_stage_queries": calls,
        "fallback_queries_vector_only": {p: len(v) for p, v in fb.items()},
        "hybrid_hnsw_bm25_queries": {p: calls[p] - len(fb[p]) for p in calls},
        "fallback_rate": {p: round(len(fb[p]) / calls[p], 4) if calls[p] else None for p in calls},
        "affected_question_ids": sorted(set(fb["pipeline_a"]) | set(fb["c5_ondemand"])),
        "fallback_in": ("both" if fb["pipeline_a"] and fb["c5_ondemand"] else
                        "Dense only" if fb["pipeline_a"] else "C5 only" if fb["c5_ondemand"] else "none"),
        "same_questions_fell_back_in_both": set(fb["pipeline_a"]) == set(fb["c5_ondemand"]),
        "warning_lines_total": sum(len(e["warnings"]) for e in events),
        "warning_messages": sorted({w for e in events for w in e["warnings"]}),
        "dense_first_stage_identical_questions": identical, "dense_first_stage_differs": differ,
        "method": "warnings attributed to the next retrieval row in the raw log; nothing re-queried",
        "events": events,
    }
    (args.run_dir / "fts_fallback_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "events"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
