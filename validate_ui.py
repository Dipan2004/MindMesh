"""Validate the full UI + API integration. Run while API is up on port 8000."""
import urllib.request, json, sys, time

BASE = "http://localhost:8000"
results = []

def check(label, url, method="GET", data=None, expected_keys=None):
    try:
        req = urllib.request.Request(url, method=method,
              data=json.dumps(data).encode() if data else None,
              headers={"Content-Type":"application/json"} if data else {})
        with urllib.request.urlopen(req, timeout=5) as r:
            body = json.loads(r.read())
            status = r.status
        ok_keys = all(k in body for k in (expected_keys or []))
        result = {"label": label, "status": status, "ok": True,
                  "sample": str(body)[:120], "keys_ok": ok_keys}
    except Exception as e:
        result = {"label": label, "status": None, "ok": False,
                  "sample": str(e)[:120], "keys_ok": False}
    results.append(result)
    icon = "OK" if result["ok"] else "FAIL"
    print(f"  {icon:4}  {label:35}  HTTP {result['status']}  {result['sample'][:80]}")

# Static asset test
def check_static(label, path, expect_content_type):
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=5) as r:
            ct = r.headers.get("content-type","")
            ok = expect_content_type in ct or r.status == 200
            result = {"label": label, "status": r.status, "ok": ok,
                      "sample": f"content-type: {ct[:60]}", "keys_ok": ok}
    except Exception as e:
        result = {"label": label, "status": None, "ok": False,
                  "sample": str(e)[:80], "keys_ok": False}
    results.append(result)
    icon = "OK" if result["ok"] else "FAIL"
    print(f"  {icon:4}  {label:35}  HTTP {result['status']}  {result['sample'][:80]}")

print("\n=== UI + API Validation ===\n")

# API endpoints
check("GET /health",           f"{BASE}/health",        expected_keys=["status"])
check("GET /auth-status",      f"{BASE}/auth-status",   expected_keys=["authenticated"])
check("GET /auth/verify",      f"{BASE}/auth/verify",   expected_keys=["valid"])
check("GET /documents",        f"{BASE}/documents",     expected_keys=["documents","total"])
check("GET /graph/label/list", f"{BASE}/graph/label/list?limit=5", expected_keys=["labels"])
check("GET /graphs",           f"{BASE}/graphs",        expected_keys=["nodes","edges"])
check("GET /ui/customization", f"{BASE}/ui/customization", expected_keys=["title"])

# Query endpoint
check("POST /query/lightrag naive", f"{BASE}/query/lightrag", method="POST",
      data={"query":"What is HNSW?","mode":"naive","only_need_context":True,"top_k":3},
      expected_keys=["response"])

# Static UI assets
check_static("GET /webui/index.html",    "/webui/index.html",    "text/html")
check_static("GET /webui/ (spa root)",   "/webui/",              "text/html")

passed = sum(1 for r in results if r["ok"])
failed = len(results) - passed
print(f"\n{'='*60}")
print(f"Results: {passed}/{len(results)} passed, {failed} failed")
with open("validate_ui_results.json","w") as f:
    json.dump(results, f, indent=2)
print("Full results saved to validate_ui_results.json")
sys.exit(0 if failed == 0 else 1)
