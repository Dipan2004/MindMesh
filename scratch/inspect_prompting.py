from pathlib import Path
import json

adapter_path = Path("src/mindmesh/pipelines/lightrag_experimental/adapter.py")
generator_path = Path("src/graphrag_lite/llm/generator.py")

print("=== adapter.py ===")
with open(adapter_path, "r", encoding="utf-8") as f:
    text = f.read()
    for line in text.splitlines():
        if "num_predict" in line or "generate(" in line or "options" in line or "prompt" in line:
            print("  ", line)

print("\n=== generator.py ===")
with open(generator_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i, l in enumerate(lines[:60]):
        print(f"  {i+1}: {l.rstrip()}")
