"""Read-only health line for the expanded experiment: GPU, progress, checkpoint integrity, processes."""
import json, subprocess, sys
from pathlib import Path

E = Path(__file__).resolve().parent
ext = E / "dev_distractor_100q_seed20261119/index/mistral_latest/extractions.jsonl"
bad_json = 0; ids = []; perr = 0
for line in ext.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        r = json.loads(line); ids.append(r["chunk_id"]); perr += "_parse_error" in r
    except Exception:
        bad_json += 1
dups = len(ids) - len(set(ids))
gpu = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,utilization.gpu,memory.used",
                      "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
procs = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object CommandLine -match 'run_expanded|build_index|run_benchmark' | Measure-Object).Count"],
                       capture_output=True, text=True).stdout.strip()
boot = subprocess.run(["powershell", "-NoProfile", "-Command",
                       "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('s')"], capture_output=True, text=True).stdout.strip()
st = json.loads((E / "status.json").read_text(encoding="utf-8")) if (E / "status.json").exists() else {}
alert = []
if bad_json: alert.append(f"BAD_JSON={bad_json}")
if dups: alert.append(f"DUPLICATES={dups}")
if procs in ("", "0") and st.get("state") not in ("held", "complete"): alert.append("DRIVER_NOT_RUNNING")
print(f"records={len(ids)}/985 unique={len(set(ids))} parse_err={perr} gpu[T,MHz,util,MiB]={gpu} "
      f"procs={procs} stage={st.get('stage')} state={st.get('state')} boot={boot}"
      + (f" ALERT {' '.join(alert)}" if alert else ""))
