"""Phase 2 preflight check."""
import sys, json, pathlib, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

print("=== PHASE 2 PREFLIGHT ===")

# 1. Ollama
r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30)
models = [l.split()[0] for l in r.stdout.strip().splitlines()[1:] if l.strip()]
print("Ollama available:", r.returncode == 0)
print("qwen2.5:1.5b installed:", "qwen2.5:1.5b" in models)

# 2. Canonical chunks
import zvec
ROOT = pathlib.Path(__file__).parent.parent
col = zvec.open(str(ROOT / "data/cache/chunks_v1"))
res = col.query(queries=[zvec.Query("text_vec", vector=[0.0]*768,
                param=zvec.HnswQueryParam(ef=300))], topk=100)
ids = [r.id for r in res]
col.close()
print(f"Canonical chunks: {len(ids)} (expected 42)")

# 3. Phase 1 baseline
p1 = ROOT / "data/experiments/qwen2_5_1_5b_validate/validation_20261109_223804.json"
print(f"Phase 1 validation intact: {p1.exists()}  ({p1.stat().st_size if p1.exists() else 0} bytes)")

# 4. Cache identity fix
src = (ROOT / "src/graphrag_lite/ingestion/extractor.py").read_text(encoding="utf-8")
fix = 'getattr(llm_client' in src
print(f"Cache identity fix active: {fix}")

# 5. Dev set
dev = json.loads((ROOT / "eval/gold_dev_set.json").read_text(encoding="utf-8"))
nq = len(dev["queries"])
print(f"Dev set: {nq} questions")

# 6. Check existing experiments
exp_dir = ROOT / "data/experiments"
for d in sorted(exp_dir.iterdir()) if exp_dir.exists() else []:
    print(f"  Existing experiment: {d.name}")

all_ok = ("qwen2.5:1.5b" in models and len(ids) == 42 and p1.exists() and fix and nq == 20)
print()
print("PREFLIGHT:", "PASS" if all_ok else "FAIL")
