import urllib.request, json, sys

BASE = "http://localhost:8000"
results = []

def check(label, url, method="GET", data=None, expected_keys=None):
    try:
        req = urllib.request.Request(url, method=method,
              data=json.dumps(data).encode() if data else None,
              headers={"Content-Type":"application/json"} if data else {})
        with urllib.request.urlopen(req, timeout=8) as r:
            body = json.loads(r.read())
            status = r.status
        ok_keys = all(k in body for k in (expected_keys or []))
        result = {"label": label, "status": status, "ok": True, "sample": str(body)[:100], "keys_ok": ok_keys}
    except Exception as e:
        result = {"label": label, "status": None, "ok": False, "sample": str(e)[:100], "keys_ok": False}
    results.append(result)
    icon = "OK  " if result["ok"] else "FAIL"
    print(f"  {icon}  {label:38} HTTP {result['status']}  {result['sample'][:70]}")

def check_static(label, path, expect_text):
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=5) as r:
            ct = r.headers.get("content-type","")
            ok = expect_text in ct
            result = {"label": label, "status": r.status, "ok": ok, "sample": f"ct={ct[:50]}"}
    except Exception as e:
        result = {"label": label, "status": None, "ok": False, "sample": str(e)[:80]}
    results.append(result)
    icon = "OK  " if result["ok"] else "FAIL"
    print(f"  {icon}  {label:38} HTTP {result['status']}  {result['sample'][:70]}")

print("\n=== MindMesh UI + API Validation ===\n")
check("GET /health",              f"{BASE}/health",                  expected_keys=["status"])
check("GET /auth-status",         f"{BASE}/auth-status",             expected_keys=["authenticated"])
check("GET /auth/verify",         f"{BASE}/auth/verify",             expected_keys=["valid"])
check("GET /documents",           f"{BASE}/documents",               expected_keys=["documents","total"])
check("GET /graph/label/list",    f"{BASE}/graph/label/list?limit=5",expected_keys=["labels"])
check("GET /graphs",              f"{BASE}/graphs",                  expected_keys=["nodes","edges"])
check("GET /ui/customization",    f"{BASE}/ui/customization",        expected_keys=["title"])
check("POST /query/lightrag",     f"{BASE}/query/lightrag",
      method="POST",
      data={"query":"What is HNSW?","mode":"naive","only_need_context":True,"top_k":3},
      expected_keys=["response"])
check("POST /query (original A)", f"{BASE}/query",
      method="POST",
      data={"pipeline":"a","query":"What is HNSW?","top_k":3},
      expected_keys=["answer_text"])
check_static("GET /webui/index.html",  "/webui/index.html", "text/html")
check_static("GET /webui/ (SPA root)", "/webui/",           "text/html")

passed = sum(1 for r in results if r["ok"])
failed = len(results) - passed
print(f"\n{'='*55}")
print(f"  TOTAL: {passed}/{len(results)} passed  {failed} failed")
print(f"{'='*55}")
with open("validate_ui_results.json","w") as f:
    json.dump(results, f, indent=2)
sys.exit(0 if failed == 0 else 1)
