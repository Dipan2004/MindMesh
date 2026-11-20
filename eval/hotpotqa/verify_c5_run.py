"""
eval/hotpotqa/verify_c5_run.py -- completeness check for a Dense vs on-demand C5 run (read-only).

    python eval/hotpotqa/verify_c5_run.py --run-dir <exp>/runs/<run_id>

Checks: question IDs match questions.json; every question has exactly one retrieval row and
one generation row per pipeline; no duplicates or extras; errors, empty retrievals, missing C5
diagnostics and generation parse failures are counted; one generator for all rows.
Writes verification.json into the run dir; exit 0 only if complete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PIPES = ("pipeline_a", "c5_ondemand")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    rd = ap.parse_args(argv).run_dir
    cfg = json.loads((rd / "run_config.json").read_text(encoding="utf-8"))
    qids = cfg["question_ids"]
    qs = json.loads((Path(cfg["bench_dir"]) / "questions.json").read_text(encoding="utf-8"))
    out = {"question_ids_match_questions_json": qids == [q["_id"] for q in qs], "n_questions": len(qids)}
    for name in ("retrieval", "results"):
        rows = [json.loads(l) for l in (rd / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        keys = [(r["pipeline"], r["qid"]) for r in rows]
        sec = {"rows": len(rows), "duplicates": len(keys) - len(set(keys)),
               "rows_with_errors": sum(1 for r in rows if r["errors"])}
        for p in PIPES:
            got = [k[1] for k in keys if k[0] == p]
            sec[p] = {"rows": len(got), "missing": sorted(set(qids) - set(got)), "extra": sorted(set(got) - set(qids))}
        if name == "retrieval":
            sec["empty_retrieval"] = sum(1 for r in rows if not r["retrieved_chunk_ids"])
            sec["c5_rows_missing_diagnostics"] = sum(1 for r in rows if r["pipeline"] == "c5_ondemand" and "diagnostics" not in r)
        else:
            sec["generation_parse_failures"] = sum(1 for r in rows if not r["generation_parse_ok"])
            sec["generators"] = sorted({r["generation_model"] for r in rows})
        out[name] = sec
    out["complete"] = (out["question_ids_match_questions_json"]
                       and all(out[n]["duplicates"] == 0 and all(not out[n][p]["missing"] and not out[n][p]["extra"]
                                                                 and out[n][p]["rows"] == len(qids) for p in PIPES)
                               for n in ("retrieval", "results"))
                       and out["retrieval"]["c5_rows_missing_diagnostics"] == 0
                       and len(out["results"]["generators"]) == 1)
    out["finished_at"] = cfg.get("finished_at")
    (rd / "verification.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0 if out["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
