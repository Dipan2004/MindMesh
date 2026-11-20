"""
eval/hotpotqa/query_focused_cost.py -- extraction-cost accounting for query_focused_c5.

    python eval/hotpotqa/query_focused_cost.py --bench-dir <dir> --run-id <run with query_focused_c5>

Cold-cost model (measured inputs, no estimation of extraction time):
  * every paragraph a question selects must be extracted once; later questions reuse the cache;
  * the cost of extracting a paragraph = the latency RECORDED for that paragraph when the full
    index was built (same model, same prompt, same settings; extraction is context-free).
Reports total cold extraction time vs full-corpus extraction, the per-question cold cost
(paragraphs new to the cache at that question, in run order), and the share of the corpus touched.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--index", default="mistral_latest")
    args = ap.parse_args(argv)
    ext = {json.loads(l)["chunk_id"]: json.loads(l) for l in
           (args.bench_dir / "index" / args.index / "extractions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    rows = [json.loads(l) for l in (args.bench_dir / "runs" / args.run_id / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip() and json.loads(l)["pipeline"] == "query_focused_c5"]
    lat = {c: r.get("latency_s", 0.0) for c, r in ext.items()}
    seen: set[str] = set()
    per_q = []
    for r in rows:
        sel = r.get("retrieval_diagnostics", {}).get("selected_paragraphs", [])
        new = [c for c in sel if c not in seen]
        seen.update(sel)
        per_q.append({"qid": r["qid"], "selected": len(sel), "new_to_cache": len(new),
                      "cold_extraction_s": round(sum(lat.get(c, 0.0) for c in new), 1),
                      "warm_retrieval_s": r["retrieval_latency_s"]})
    full_s = sum(lat.values())
    touched_s = sum(lat.get(c, 0.0) for c in seen)
    cold = [p["cold_extraction_s"] for p in per_q]
    out = {
        "questions": len(per_q), "corpus_paragraphs": len(ext), "paragraphs_touched": len(seen),
        "share_of_corpus_touched": round(len(seen) / len(ext), 4) if ext else None,
        "full_corpus_extraction_h": round(full_s / 3600, 2),
        "query_focused_extraction_h": round(touched_s / 3600, 2),
        "extraction_time_saving_pct": round(100 * (1 - touched_s / full_s), 1) if full_s else None,
        "selected_paragraphs_per_question_mean": round(st.fmean(p["selected"] for p in per_q), 2) if per_q else None,
        "cold_extraction_per_question_s": {"mean": round(st.fmean(cold), 1), "median": round(st.median(cold), 1),
                                           "max": max(cold)} if cold else None,
        "warm_retrieval_per_question_s_mean": round(st.fmean(p["warm_retrieval_s"] for p in per_q), 3) if per_q else None,
        "note": "extraction latencies were recorded under this laptop's thermal throttling",
    }
    dst = args.bench_dir / "runs" / args.run_id / "query_focused_cost.json"
    dst.write_text(json.dumps({**out, "per_question": per_q}, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
