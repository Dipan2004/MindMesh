"""Append one nvidia-smi sample every 5 s to gpu_telemetry.csv (read-only monitoring)."""
import subprocess, time
from pathlib import Path
F = Path(__file__).with_name("gpu_telemetry.csv")
Q = "timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu,clocks_event_reasons.active"
while True:
    try:
        line = subprocess.run(["nvidia-smi", f"--query-gpu={Q}", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        if line:
            with open(F, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        pass
    time.sleep(5)
