"""Check what v2 extraction files contain."""
import sys, json, pathlib, collections
ROOT = pathlib.Path(__file__).parent.parent

for fname in ["extractions_v2_off.jsonl", "extractions_v2_on.jsonl"]:
    p = ROOT / "data" / "cache_v2" / fname
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    errs = sum(1 for r in recs if "_parse_error" in r)
    total_e = sum(len(r.get("entities",[])) for r in recs)
    total_r = sum(len(r.get("relationships",[])) for r in recs)
    # Check if gleaning field is present
    has_gleaning = sum(1 for r in recs if r.get("gleaning"))
    # Check first record keys
    r0 = recs[0]
    print(f"\n{fname}:")
    print(f"  Records: {len(recs)}, errors: {errs}")
    print(f"  Total entities: {total_e}, Total rels: {total_r}")
    print(f"  Has gleaning field: {has_gleaning}")
    print(f"  First record keys: {list(r0.keys())}")
    print(f"  First chunk entities: {len(r0.get('entities',[]))}")

# Also check the graph_v2_report
report = json.loads((ROOT / "data/cache_v2/graph_v2_report.json").read_text(encoding="utf-8"))
print("\ngraph_v2_report.json keys:", list(report.keys()))
print("off arm nodes/edges:", report.get("off",{}).get("nodes"), "/", report.get("off",{}).get("edges"))
print("on arm nodes/edges:", report.get("on",{}).get("nodes"), "/", report.get("on",{}).get("edges"))
