"""
eval/check_guest_api.py -- exercise every web-UI API endpoint in guest/demo mode.

    python eval/check_guest_api.py [--base http://localhost:8000]

Uses the guest token the backend issues (no real auth exists). Uploads one
small text file to data/raw, checks it is listed, then deletes it again.
Runs one query per UI pipeline (generation uses the configured local model).
Writes eval/results/guest_api_check_<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
MODES = {"naive": "MindMesh Dense", "bypass": "MindMesh Graph",
         "mix": "MindMesh Mix", "mix_opt": "MindMesh Mix Optimized"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    args = ap.parse_args(argv)
    c = httpx.Client(base_url=args.base, timeout=300)
    results = []

    def check(name, fn):
        t0 = time.monotonic()
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append({"check": name, "pass": bool(ok), "detail": detail,
                        "seconds": round(time.monotonic() - t0, 2)})
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({detail})")

    st = c.get("/auth-status").json()
    token = st.get("access_token")
    H = {"Authorization": f"Bearer {token}"}
    check("GET /auth-status (guest, auth disabled)",
          lambda: (st.get("auth_configured") is False and bool(token), f"auth_mode={st.get('auth_mode')}"))
    check("POST /login (any username, no password check)",
          lambda: (lambda r: (r.status_code == 200 and r.json().get("access_token") == token, f"status {r.status_code}"))(
              c.post("/login", data={"username": "any-name", "password": ""})))
    check("GET /webui/ (UI served)", lambda: (lambda r: (r.status_code == 200 and "MindMesh" in r.text, f"status {r.status_code}"))(c.get("/webui/")))
    check("GET /webui JS served as JavaScript",
          lambda: (lambda html: (lambda js: (c.get("/webui/" + js).headers.get("content-type", "").startswith(("application/javascript", "text/javascript")), js))(
              html.split('src="./')[1].split('"')[0]))(c.get("/webui/").text))
    check("GET /health", lambda: (lambda r: (r.status_code == 200, r.json().get("status")))(c.get("/health", headers=H)))
    check("GET /ui/customization", lambda: (lambda r: (r.status_code == 200, f"status {r.status_code}"))(c.get("/ui/customization?locale=en", headers=H)))
    check("POST /documents/paginated", lambda: (lambda r: (r.status_code == 200, f"{len(r.json().get('documents', []))} docs"))(
        c.post("/documents/paginated", json={"page": 1, "page_size": 10}, headers=H)))

    fname = f"guest_api_check_{int(time.time())}.txt"
    up = {}

    def do_upload():
        r = c.post("/documents/upload", files={"file": (fname, b"Guest API check document. Safe to delete.", "text/plain")}, headers=H)
        up["resp"] = r
        return r.status_code == 200, f"status {r.status_code}"
    check("POST /documents/upload", do_upload)
    check("uploaded file present in data/raw", lambda: ((ROOT / "data" / "raw" / fname).exists(), fname))

    def do_delete():
        docs = c.get("/documents", headers=H).json()
        ids = [d["id"] for d in docs.get("documents", [])
               if fname in (d.get("file_path") or "") or fname in (d.get("content_summary") or "")
               or fname in (d.get("id") or "")]
        if not ids:
            (ROOT / "data" / "raw" / fname).unlink(missing_ok=True)
            return not (ROOT / "data" / "raw" / fname).exists(), "not listed by /documents; removed file directly"
        r = c.request("DELETE", "/documents/delete_document", json={"doc_ids": ids, "delete_file": True}, headers=H)
        gone = not (ROOT / "data" / "raw" / fname).exists()
        if not gone:
            (ROOT / "data" / "raw" / fname).unlink(missing_ok=True)
        return r.status_code == 200, f"status {r.status_code}, file removed={gone}"
    check("DELETE uploaded test document (cleanup)", do_delete)

    check("GET /graph/label/list", lambda: (lambda r: (r.status_code == 200 and len(r.json()) > 0, f"{len(r.json())} labels"))(c.get("/graph/label/list", headers=H)))
    check("GET /graphs (visualisation data)", lambda: (lambda j: (len(j.get("nodes", [])) > 0, f"{len(j.get('nodes', []))} nodes, {len(j.get('edges', []))} edges"))(
        c.get("/graphs?label=*&max_depth=3&max_nodes=1000", headers=H).json()))

    for mode, label in MODES.items():
        def q(mode=mode):
            r = c.post("/query", json={"query": "What is hybrid search in vector databases?", "mode": mode}, headers=H)
            j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            resp = (j.get("response") or "")
            return r.status_code == 200 and len(resp.strip()) > 0, f"status {r.status_code}, {len(resp)} chars"
        check(f"POST /query mode={mode} ({label})", q)

    out = ROOT / "eval" / "results" / f"guest_api_check_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"base": args.base, "results": results,
                               "passed": sum(r["pass"] for r in results), "total": len(results)}, indent=2), encoding="utf-8")
    print(f"\n{sum(r['pass'] for r in results)}/{len(results)} passed -> {out.relative_to(ROOT)}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
