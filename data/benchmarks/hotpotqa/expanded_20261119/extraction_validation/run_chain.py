"""Run (1) the rest of the mistral 3-arm validation, then (2) qwen2.5:3b B+C on the same 30 paragraphs.
Each phase gets its own GPU telemetry file. Both phases are resumable (done rows are skipped)."""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
VAL = Path(__file__).resolve().parent
QW = VAL.parent / "extraction_validation_qwen2.5_3b"
PY = sys.executable.replace("pythonw.exe", "python.exe")
Q = "timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu,clocks_event_reasons.active"


def phase(name, args, telemetry: Path, log: Path):
    stop = False
    logger = subprocess.Popen([PY, "-c", (
        "import subprocess,time\n"
        f"F=r'{telemetry}'\n"
        "while True:\n"
        f"    l=subprocess.run(['nvidia-smi','--query-gpu={Q}','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()\n"
        "    open(F,'a',encoding='utf-8').write(l+'\\n') if l else None\n"
        "    time.sleep(5)\n")])
    try:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} {name}\n"); fh.flush()
            return subprocess.call([PY, "-u", "eval/hotpotqa/validate_extraction_variants.py", *args],
                                   cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    finally:
        logger.kill()


QW.mkdir(parents=True, exist_ok=True)
rc1 = phase("mistral 3-arm (resume)", ["--n", "30"], VAL / "gpu_telemetry.csv", VAL / "run.log")
rc2 = phase("qwen2.5:3b BC", ["--n", "30", "--model", "qwen2.5:3b", "--arms", "BC", "--out", str(QW)],
            QW / "gpu_telemetry.csv", QW / "run.log") if rc1 == 0 else None
(VAL / "chain_done.txt").write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} mistral_rc={rc1} qwen_rc={rc2}\n")
