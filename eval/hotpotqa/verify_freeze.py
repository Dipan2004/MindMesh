"""
eval/hotpotqa/verify_freeze.py -- check a HotpotQA freeze folder.

    python eval/hotpotqa/verify_freeze.py data/benchmarks/hotpotqa/dev_distractor_20q_seed42/freeze_20261119

Checks
  1. every file in untracked_snapshot/ is hashed in freeze_manifest.json, with the same hash
  2. git HEAD (from the manifest) + working_tree_vs_HEAD.patch + untracked_snapshot/
     reproduces every hashed file byte-for-byte (rebuilt in a temp dir, repo untouched)
  3. the live working tree still matches the manifest (no edits since the freeze)
Writes verify_report.json into the freeze folder; exit code 0 only if all pass.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    fz = Path((argv or sys.argv[1:])[0]).resolve()
    man = json.loads((fz / "freeze_manifest.json").read_text(encoding="utf-8"))
    hashes: dict[str, str] = man["file_sha256"]
    report: dict = {"freeze_dir": str(fz), "git_head": man["git_head"], "n_hashed": len(hashes)}

    # 1. snapshot coverage
    snap = fz / "untracked_snapshot"
    snap_files = sorted(p.relative_to(snap).as_posix() for p in snap.rglob("*") if p.is_file())
    missing = [f for f in snap_files if f not in hashes]
    mismatch = [f for f in snap_files if f in hashes and sha(snap / f) != hashes[f]]
    report["check1_snapshot_files"] = len(snap_files)
    report["check1_not_in_manifest"] = missing
    report["check1_hash_mismatch"] = mismatch
    ok1 = not missing and not mismatch and len(snap_files) > 0

    # 2. rebuild HEAD + patch + snapshot in a temp dir
    tmp = Path(tempfile.mkdtemp(prefix="mm_freeze_"))
    try:
        tar_bytes = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", man["git_head"],
                                    "src", "eval", "ui/lightrag_ui/src", ".gitignore"],
                                   capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
            tf.extractall(tmp)
        subprocess.run(["git", "init", "-q", str(tmp)], check=True)
        r = subprocess.run(["git", "-C", str(tmp), "apply", "--whitespace=nowarn",
                            str(fz / "working_tree_vs_HEAD.patch")], capture_output=True, text=True)
        report["check2_patch_apply"] = "ok" if r.returncode == 0 else r.stderr.strip()[:500]
        for f in snap_files:
            dst = tmp / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snap / f, dst)
        rebuilt_missing, rebuilt_mismatch = [], []
        for f, h in hashes.items():
            p = tmp / f
            if not p.exists():
                rebuilt_missing.append(f)
            elif sha(p) != h:
                rebuilt_mismatch.append(f)
        report["check2_missing"] = rebuilt_missing
        report["check2_mismatch"] = rebuilt_mismatch
        ok2 = r.returncode == 0 and not rebuilt_missing and not rebuilt_mismatch
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 3. live tree unchanged since the freeze
    live_mismatch = [f for f, h in hashes.items() if not (ROOT / f).exists() or sha(ROOT / f) != h]
    report["check3_live_tree_changed"] = live_mismatch
    ok3 = not live_mismatch

    report["pass"] = {"snapshot_covered": ok1, "reproducible": ok2, "live_tree_unchanged": ok3}
    (fz / "verify_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(main())
