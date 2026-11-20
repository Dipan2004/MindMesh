"""
Passive telemetry for the running 100q experiment (read-only; does not touch the run).
Every 30 s: CPU %, RAM used, GPU temp / SM clock / util / VRAM / power / event reasons,
plus the extraction record count -> resources.csv.
Every 10 min: copies Ollama's server*.log files here (Ollama rotates them on restart).
"""
import csv, glob, os, shutil, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent / "dev_distractor_100q_seed20261119/index/mistral_latest/extractions.jsonl"
OUT = HERE / "resources.csv"
LOGDIR = Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama"
try:
    import psutil
except Exception:
    psutil = None

Q = "temperature.gpu,clocks.sm,utilization.gpu,memory.used,power.draw,clocks_event_reasons.active"
new = not OUT.exists()
last_copy = 0.0
while True:
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        g = subprocess.run(["nvidia-smi", f"--query-gpu={Q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip().split(", ")
        row.update(dict(zip(["gpu_temp_c", "sm_clock_mhz", "gpu_util_pct", "vram_used_mib", "power_w", "event_reasons"], g)))
    except Exception:
        pass
    if psutil:
        row["cpu_pct"] = psutil.cpu_percent(interval=1)
        vm = psutil.virtual_memory(); row["ram_used_gb"] = round(vm.used / 2**30, 2); row["ram_pct"] = vm.percent
    try:
        row["records"] = sum(1 for l in EXT.read_text(encoding="utf-8").splitlines() if l.strip())
    except Exception:
        row["records"] = ""
    cols = ["time", "gpu_temp_c", "sm_clock_mhz", "gpu_util_pct", "vram_used_mib", "power_w", "event_reasons",
            "cpu_pct", "ram_used_gb", "ram_pct", "records"]
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader(); new = False
        w.writerow({c: row.get(c, "") for c in cols})
    if time.time() - last_copy > 600:
        for src in glob.glob(str(LOGDIR / "server*.log")):
            try:
                p = Path(src)
                shutil.copy2(p, HERE / f"ollama_{p.stem}_created{int(p.stat().st_ctime)}.log")
            except Exception:
                pass
        last_copy = time.time()
    time.sleep(29)
