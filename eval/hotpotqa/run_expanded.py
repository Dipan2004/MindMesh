"""
eval/hotpotqa/run_expanded.py -- idempotent driver for the expanded HotpotQA experiments.

    python eval/hotpotqa/run_expanded.py            # run / resume all stages
    python eval/hotpotqa/run_expanded.py --status   # print status only

Stages (each skipped when its completion check passes, so it is safe to re-run
after a crash, reboot or thermal shutdown):

  prepare   nested samples: 100q = first 50 bridge + 50 comparison of a seeded
            100-per-type draw; 200q = the whole draw (100q is a strict subset)
  index100  MindMesh extraction + graph + Zvec index for the 100q corpus
  smoke     2 questions x 4 pipelines (diagnostic only)
  run100    4 pipelines x 100 questions
  qc100     completeness / duplicates / failures / metric recomputation
  index200  index for the 200q corpus (100q extractions reused by chunk_id)
  run200    4 pipelines x 200 questions          (only if qc100 is valid)
  qc200

Retrieval code is not touched.  Extraction records or result rows that failed
only because Ollama was unreachable (machine shutting down) are moved to a
quarantine file and retried; real model failures are kept as failures.

Stops on the first failed check and records it in status.json; a stage that
failed for a non-transient reason is not retried automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

EXP = ROOT / "data" / "benchmarks" / "hotpotqa" / "expanded_20261119"
SEED = 20261119
DRAW = 100
MODEL = "mistral:latest"
B = {100: EXP / f"dev_distractor_100q_seed{SEED}", 200: EXP / f"dev_distractor_200q_seed{SEED}"}
PILOT_EXT = ROOT / "data/benchmarks/hotpotqa/dev_distractor_20q_seed42/index/mistral_latest/extractions.jsonl"
PIPELINES = ["pipeline_a", "pipeline_b", "lightrag_mix", "lightrag_mix_opt"]
STARTUP_LAUNCHER = "MindMesh_HotpotQA_resume.cmd"   # per-user Startup folder, removed on completion
TRANSIENT = ("Failed to connect", "ConnectError", "ConnectionError", "ConnectTimeout",
             "RemoteDisconnected", "Connection refused", "actively refused", "ReadTimeout",
             "Server disconnected", "WinError 10061", "WinError 10054")
STATUS = EXP / "status.json"
LOG = EXP / "driver.log"
PY = sys.executable


def log(msg: str) -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(**kw) -> dict:
    st = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    if kw:
        st.update(kw, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        STATUS.write_text(json.dumps(st, indent=2), encoding="utf-8")
    return st


def run(step: str, args: list[str]) -> int:
    (EXP / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    log(f"[{step}] $ {' '.join(args)}")
    with open(EXP / "logs" / f"{step}.log", "a", encoding="utf-8") as fh:
        fh.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(args)}\n")
        fh.flush()
        rc = subprocess.call([PY, "-u", *args], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    log(f"[{step}] exit {rc}")
    return rc


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def quarantine(path: Path, is_transient) -> int:
    """Move transient-failure lines out of an append-only jsonl into <name>.quarantine.jsonl."""
    rows = jsonl(path)
    bad = [r for r in rows if is_transient(r)]
    if not bad:
        return 0
    keep = [r for r in rows if not is_transient(r)]
    with open(path.with_suffix(".quarantine.jsonl"), "a", encoding="utf-8") as q:
        for r in bad:
            q.write(json.dumps(dict(r, _quarantined_at=time.strftime("%Y-%m-%dT%H:%M:%S")), ensure_ascii=False) + "\n")
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    os.replace(tmp, path)
    return len(bad)


def transient_ext(r: dict) -> bool:
    return any(t in str(r.get("_parse_error", "")) for t in TRANSIENT)


def transient_row(r: dict) -> bool:
    return any(t in e for e in r.get("errors", []) for t in TRANSIENT)


# ── environment guards ───────────────────────────────────────────────────────

def wait_for_ollama(timeout_s: int = 600) -> bool:
    import ollama
    t0 = time.monotonic()
    started = False
    while time.monotonic() - t0 < timeout_s:
        try:
            ollama.Client(timeout=10).list()
            return True
        except Exception:
            if not started:
                app = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama app.exe"
                if app.exists():
                    subprocess.Popen([str(app)], close_fds=True)
                    log("started Ollama app")
                started = True
            time.sleep(5)
    return False


def clear_stale_gpu_lock() -> None:
    lock = ROOT / "data/benchmarks/.gpu_inference.lock"
    if not lock.exists():
        return
    try:
        pid = int(json.loads(lock.read_text()).get("pid", -1))
    except Exception:
        pid = -1
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
                         capture_output=True, text=True).stdout
    if "hotpotqa" not in out:
        lock.unlink(missing_ok=True)
        log(f"removed stale GPU lock (pid {pid})")


def single_instance() -> bool:
    lock = EXP / "driver.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
                                 capture_output=True, text=True).stdout
            if "run_expanded" in out:
                return False
        except Exception:
            pass
    lock.write_text(str(os.getpid()))
    return True


# ── stage checks ─────────────────────────────────────────────────────────────

def sample_ids(n: int) -> list[str]:
    p = B[n] / "sample_ids.json"
    return json.loads(p.read_text(encoding="utf-8"))["ids"] if p.exists() else []


def index_dir(n: int) -> Path:
    return B[n] / "index" / "mistral_latest"


def index_complete(n: int) -> bool:
    m = index_dir(n) / "index_meta.json"
    if not m.exists():
        return False
    meta = json.loads(m.read_text(encoding="utf-8"))
    total = len(jsonl(B[n] / "corpus.jsonl"))
    ext_ids = {r["chunk_id"] for r in jsonl(index_dir(n) / "extractions.jsonl")}
    return (meta.get("smoke_test_limit") is None and meta.get("chunks_indexed") == total
            and len(ext_ids) == total and m.stat().st_mtime >= (index_dir(n) / "extractions.jsonl").stat().st_mtime)


def seed_extractions(n: int, sources: list[Path]) -> None:
    """Reuse identical-chunk extractions (same model, same settings) from earlier indexes."""
    dst = index_dir(n) / "extractions.jsonl"
    if dst.exists():
        return
    wanted = {json.loads(l)["chunk_id"] for l in (B[n] / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    got: dict[str, dict] = {}
    prov = {}
    for src in sources:
        for r in jsonl(src):
            if r["chunk_id"] in wanted and r["chunk_id"] not in got and not transient_ext(r):
                got[r["chunk_id"]] = r
                prov[r["chunk_id"]] = str(src.relative_to(ROOT))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in got.values()), encoding="utf-8")
    (index_dir(n) / "seeded_extractions.json").write_text(json.dumps(
        {"note": "extractions reused by identical chunk_id from earlier indexes built with the same "
                 "model (mistral:latest), same extraction code and settings, gleanings 0",
         "count": len(got), "sources": prov}, indent=2), encoding="utf-8")
    log(f"[index{n}] seeded {len(got)} extractions from earlier indexes")


def run_dir(n: int) -> Path | None:
    runs = B[n] / "runs"
    if not runs.exists():
        return None
    cands = [d for d in runs.iterdir() if d.is_dir() and d.name.startswith("gen-mistral_latest")]
    return max(cands, key=lambda d: d.stat().st_mtime) if cands else None


def qc(n: int, rd: Path) -> dict:
    """Stage C checks. Recomputes every aggregate from the raw rows."""
    from mindmesh.benchmarks.hotpotqa.metrics import answer_metrics, joint_metrics, retrieval_metrics, sp_metrics
    qs = json.loads((B[n] / "questions.json").read_text(encoding="utf-8"))
    qids = [q["_id"] for q in qs]
    rows = jsonl(rd / "results.jsonl")
    summ = json.loads((rd / "summary.json").read_text(encoding="utf-8")) if (rd / "summary.json").exists() else {}
    out = {"n_questions": n, "run_dir": str(rd.relative_to(ROOT)), "checks": {}, "per_pipeline": {}}
    keys = [(r["pipeline"], r["qid"]) for r in rows]
    out["checks"]["rows_expected"] = len(PIPELINES) * n
    out["checks"]["rows_found"] = len(rows)
    out["checks"]["duplicate_rows"] = len(keys) - len(set(keys))
    out["checks"]["question_ids_unique"] = len(set(qids)) == len(qids)
    gold = {q["_id"]: q for q in qs}
    mismatch = []
    for p in PIPELINES:
        pr = [r for r in rows if r["pipeline"] == p]
        missing = sorted(set(qids) - {r["qid"] for r in pr})
        extra = sorted({r["qid"] for r in pr} - set(qids))
        failed_gen = [r["qid"] for r in pr if any(e.startswith("generation") for e in r["errors"])]
        failed_ret = [r["qid"] for r in pr if any(e.startswith("retrieval") for e in r["errors"])]
        bad_out = [r["qid"] for r in pr if not r.get("generation_parse_ok")]
        for r in pr:   # recompute per-row metrics from the stored raw prediction
            q = gold[r["qid"]]
            a = answer_metrics(r["pred_answer"], q["answer"]); s = sp_metrics(r["pred_sp"], q["supporting_facts"])
            j = joint_metrics(a, s); rm = retrieval_metrics(r["retrieved_chunk_ids"], q["gold_chunk_ids"])
            for k, v in [("ans_em", a["em"]), ("ans_f1", a["f1"]), ("sp_f1", s["f1"]), ("joint_f1", j["f1"]),
                         ("recall@10", rm["recall@10"]), ("mrr", rm["mrr"])]:
                if abs(r[k] - v) > 1e-9:
                    mismatch.append((p, r["qid"], k))
        agg_ok = True
        if p in summ:
            for k in ("recall@5", "recall@10", "mrr", "ans_em", "ans_f1", "sp_f1"):
                v = sum(r[k] for r in pr) / len(pr) if pr else 0
                agg_ok &= abs(round(v, 4) - summ[p][k]) < 1e-4
        out["per_pipeline"][p] = {"rows": len(pr), "missing": missing, "extra": extra,
                                  "failed_generation": failed_gen, "failed_retrieval": failed_ret,
                                  "invalid_or_truncated_output": bad_out, "summary_matches_raw": agg_ok and p in summ}
    out["checks"]["metric_recompute_mismatches"] = mismatch[:20]
    worst_fail = max(len(v["failed_generation"]) + len(v["failed_retrieval"]) for v in out["per_pipeline"].values())
    out["valid"] = (out["checks"]["rows_found"] == out["checks"]["rows_expected"]
                    and out["checks"]["duplicate_rows"] == 0 and out["checks"]["question_ids_unique"]
                    and not mismatch
                    and all(not v["missing"] and not v["extra"] and v["summary_matches_raw"] for v in out["per_pipeline"].values())
                    and worst_fail <= 0.05 * n)
    out["validity_rule"] = ("all rows present once, IDs match the sample, every per-row metric recomputes from raw "
                            "predictions, summary.json matches raw rows, and at most 5% failed queries per pipeline")
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)
    EXP.mkdir(parents=True, exist_ok=True)
    if args.status:
        print(json.dumps(status(), indent=2))
        return 0
    if not single_instance():
        print("driver already running")
        return 0
    st = status()
    if st.get("state") in ("failed", "complete"):
        log(f"state is {st['state']} ({st.get('detail')}); nothing to do")
        return 0
    log("driver start")
    if not wait_for_ollama():
        status(state="waiting", detail="Ollama not reachable")
        log("Ollama not reachable; will retry at next start")
        return 1
    clear_stale_gpu_lock()

    # prepare
    for n in (100, 200):
        if not (B[n] / "sample_ids.json").exists():
            if run("prepare", ["eval/hotpotqa/prepare_sample.py", "--draw-per-type", str(DRAW), "--seed", str(SEED),
                               "--n-bridge", str(n // 2), "--n-comparison", str(n // 2), "--out", str(B[n])]) != 0:
                status(state="failed", stage="prepare", detail=f"prepare_sample {n}q failed")
                return 1
    if not set(sample_ids(100)) <= set(sample_ids(200)):
        status(state="failed", stage="prepare", detail="100q sample is not a subset of 200q")
        return 1

    def build(n: int, sources: list[Path]) -> bool:
        if index_complete(n):
            return True
        status(state="running", stage=f"index{n}")
        seed_extractions(n, sources)
        q = quarantine(index_dir(n) / "extractions.jsonl", transient_ext)
        if q:
            log(f"[index{n}] quarantined {q} extraction records that failed only because Ollama was unreachable")
        for attempt in range(3):
            rc = run(f"index{n}", ["eval/hotpotqa/build_index.py", "--model", MODEL, "--bench-dir", str(B[n])])
            q = quarantine(index_dir(n) / "extractions.jsonl", transient_ext)
            if q:
                log(f"[index{n}] quarantined {q} transient extraction failures; retrying")
                if not wait_for_ollama():
                    return False
                continue
            if rc == 0 and index_complete(n):
                return True
        status(state="failed", stage=f"index{n}", detail=f"build_index exit {rc}")
        return False

    def bench(n: int, step: str, extra: list[str]) -> bool:
        status(state="running", stage=step)
        rd_name = extra[extra.index("--run-id") + 1] if "--run-id" in extra else None
        for attempt in range(3):
            rd = (B[n] / "runs" / rd_name) if rd_name else run_dir(n)
            if rd is not None:
                q = quarantine(rd / "results.jsonl", transient_row)
                if q:
                    log(f"[{step}] quarantined {q} rows that failed only because Ollama was unreachable")
            rc = run(step, ["eval/hotpotqa/run_benchmark.py", "--model", MODEL, "--bench-dir", str(B[n]), *extra])
            rd = (B[n] / "runs" / rd_name) if rd_name else run_dir(n)
            if rd is not None and any(transient_row(r) for r in jsonl(rd / "results.jsonl")):
                if not wait_for_ollama():
                    return False
                continue
            if rc == 0:
                return True
        status(state="failed", stage=step, detail=f"run_benchmark exit {rc}")
        return False

    # 100q
    if not build(100, [PILOT_EXT]):
        return 1
    smoke_dir = B[100] / "runs" / "smoke_2q"
    if not (smoke_dir / "summary.json").exists():
        if not bench(100, "smoke", ["--limit", "2", "--run-id", "smoke_2q"]):
            return 1
        rows = jsonl(smoke_dir / "results.jsonl")
        if len(rows) != 8 or any(r["errors"] for r in rows) or not all(r["generation_parse_ok"] for r in rows) \
                or not all(r["retrieved_chunk_ids"] for r in rows):
            status(state="failed", stage="smoke", detail="smoke test: missing rows, errors, empty retrieval or invalid output")
            log("[smoke] FAILED; stopping for diagnosis")
            return 1
        log("[smoke] passed: 8/8 rows, retrieval non-empty, 0 errors, outputs valid")
    if not (EXP / "qc_100q.json").exists() or not json.loads((EXP / "qc_100q.json").read_text())["valid"]:
        if not bench(100, "run100", []):
            return 1
        rep = qc(100, run_dir(100))
        (EXP / "qc_100q.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        log(f"[qc100] valid={rep['valid']}")
        if not rep["valid"]:
            status(state="failed", stage="qc100", detail="100q results failed validation; 200q not started")
            return 1

    # 200q (only after a valid 100q, and only when not held for approval)
    if (EXP / "HOLD_200Q").exists():
        status(state="held", stage="awaiting_approval_200q",
               detail="100q complete and validated; 200q held by HOLD_200Q until the user approves")
        log("[200q] held: HOLD_200Q present; stopping after the 100q stage")
        return 0
    if not build(200, [index_dir(100) / "extractions.jsonl", PILOT_EXT]):
        return 1
    if not (EXP / "qc_200q.json").exists() or not json.loads((EXP / "qc_200q.json").read_text())["valid"]:
        if not bench(200, "run200", []):
            return 1
        rep = qc(200, run_dir(200))
        (EXP / "qc_200q.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        log(f"[qc200] valid={rep['valid']}")
        if not rep["valid"]:
            status(state="failed", stage="qc200", detail="200q results failed validation")
            return 1

    status(state="complete", stage="done", detail="100q and 200q complete and validated")
    log("all stages complete")
    startup = Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs/Startup" / STARTUP_LAUNCHER
    if startup.exists():
        startup.unlink()
        log(f"removed auto-resume launcher {startup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
