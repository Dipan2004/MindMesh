import json, re
from pathlib import Path

runs = [
    ("pipeline_a", "lightrag_qwen2.5_1.5b_pipeline_a_20261118_024952"),
    ("lightrag_mix_opt", "lightrag_qwen2.5_1.5b_lightrag_mix_opt_20261118_203834")
]

HEX_RE = re.compile(r"\[([a-f0-9]{8,16})\]")
IDX_RE = re.compile(r"\[(\d+)\]")

for name, rdir in runs:
    with open(Path("data/experiments") / rdir / "retrieval_results.json", encoding="utf-8") as f:
        res = json.load(f)["results"]
    print(f"=== {name} ===")
    hex_total = 0
    idx_total = 0
    for r in res:
        ans = r.get("answer", "") or ""
        hex_cites = HEX_RE.findall(ans)
        idx_cites = IDX_RE.findall(ans)
        hex_total += len(hex_cites)
        idx_total += len(idx_cites)
        print(f"  {r['query_id']}: hex={hex_cites}, idx={idx_cites}")
    print(f"Total hex citations: {hex_total}, Total index citations: {idx_total}\n")
