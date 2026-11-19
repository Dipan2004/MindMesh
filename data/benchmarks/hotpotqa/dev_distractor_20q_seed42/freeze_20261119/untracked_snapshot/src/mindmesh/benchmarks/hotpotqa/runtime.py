"""
runtime.py -- local-model validation, hardware capture and GPU discipline.

Rules enforced here:
  * exact installed Ollama tags only (``mistral`` == ``mistral:latest`` is the
    only normalisation, because that is Ollama's own naming rule)
  * no automatic pulls, no silent fallback, no cloud models (``*:cloud`` /
    remote stubs are refused)
  * a cross-process lock so two GPU inference experiments never overlap
  * explicit model unload (keep_alive=0) to free VRAM between runs
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LOCK_PATH = ROOT / "data" / "benchmarks" / ".gpu_inference.lock"


class ModelNotAvailable(SystemExit):
    pass


def installed_models() -> dict[str, dict]:
    import ollama
    out = {}
    for m in ollama.Client().list().models:
        out[m.model] = {
            "size_bytes": int(m.size or 0),
            "digest": m.digest,
            "parameter_size": getattr(m.details, "parameter_size", None),
            "quantization": getattr(m.details, "quantization_level", None),
            "family": getattr(m.details, "family", None),
        }
    return out


def is_cloud_model(tag: str, info: dict) -> bool:
    t = tag.lower()
    return t.endswith(":cloud") or t.endswith("-cloud") or info.get("size_bytes", 0) < 1_000_000


def resolve_local_model(tag: str) -> tuple[str, dict]:
    """Return (exact_installed_tag, info) or exit with a clear message."""
    models = installed_models()
    candidates = [tag] if ":" in tag else [tag, f"{tag}:latest"]
    for c in candidates:
        if c in models:
            if is_cloud_model(c, models[c]):
                raise ModelNotAvailable(
                    f"[REFUSED] {c!r} is an Ollama cloud model. MindMesh benchmarks run "
                    f"local inference only.")
            return c, models[c]
    local = [m for m, i in models.items() if not is_cloud_model(m, i)]
    raise ModelNotAvailable(
        f"[REFUSED] Ollama model {tag!r} is not installed locally.\n"
        f"Installed local models: {', '.join(local) or '(none)'}\n"
        f"Nothing was downloaded and no other model was substituted. Install it "
        f"yourself with `ollama pull {tag}` if you want to use it.")


def _run(cmd: list[str], timeout: int = 20) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def gpu_info() -> dict:
    q = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
              "--format=csv,noheader,nounits"])
    if not q or q.startswith("unavailable"):
        return {"nvidia_smi": q or "no output", "gpus": []}
    gpus = []
    for line in q.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append({"name": parts[0], "memory_total_mib": int(parts[1]),
                         "memory_used_mib": int(parts[2]), "driver": parts[3]})
    cuda = _run(["nvidia-smi"]).split("CUDA Version:")
    return {"gpus": gpus, "cuda_driver_version": cuda[1].split()[0] if len(cuda) > 1 else None}


def hardware_info() -> dict:
    info = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "ollama_version": _run(["ollama", "--version"]).replace("ollama version is ", ""),
    }
    try:
        import psutil  # optional
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        if platform.system() == "Windows":
            out = _run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
            try:
                info["ram_gb"] = round(int(out) / 2**30, 1)
            except ValueError:
                pass
            info["cpu"] = _run(["powershell", "-NoProfile", "-Command",
                                "(Get-CimInstance Win32_Processor).Name"]) or info["cpu"]
    info.update(gpu_info())
    return info


def ollama_ps() -> list[dict]:
    """Loaded models and how much of each sits in VRAM (from /api/ps)."""
    import ollama
    out = []
    try:
        for m in ollama.Client().ps().models:
            size = int(m.size or 0)
            vram = int(getattr(m, "size_vram", 0) or 0)
            out.append({
                "model": m.model, "size_bytes": size, "size_vram_bytes": vram,
                "gpu_fraction": round(vram / size, 4) if size else 0.0,
                "processor": "GPU" if size and vram >= size else ("CPU" if vram == 0 else "CPU/GPU split"),
                "context_length": getattr(m, "context_length", None),
            })
    except Exception as exc:  # noqa: BLE001
        out.append({"error": str(exc)})
    return out


def unload_model(tag: str) -> None:
    """Ask Ollama to drop the model from (V)RAM immediately."""
    import ollama
    try:
        ollama.Client().generate(model=tag, prompt="", keep_alive=0)
    except Exception:
        pass


class GpuLock:
    """Exclusive cross-process lock: one GPU inference experiment at a time."""

    def __init__(self, label: str):
        self.label = label
        self.fd = None

    def __enter__(self):
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LOCK_PATH.exists():
            try:
                held = json.loads(LOCK_PATH.read_text())
                pid = int(held.get("pid", -1))
            except Exception:
                held, pid = {}, -1
            if pid > 0 and _pid_alive(pid):
                raise SystemExit(
                    f"[REFUSED] Another GPU inference experiment is running "
                    f"(pid {pid}: {held.get('label')}). Runs are sequential by design.")
            LOCK_PATH.unlink(missing_ok=True)  # stale lock
        self.fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(self.fd, json.dumps({"pid": os.getpid(), "label": self.label,
                                      "started": time.strftime("%Y-%m-%dT%H:%M:%S")}).encode())
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        LOCK_PATH.unlink(missing_ok=True)
        return False


def _pid_alive(pid: int) -> bool:
    if platform.system() == "Windows":
        out = _run(["tasklist", "/FI", f"PID eq {pid}", "/NH"])
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
