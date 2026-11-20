"""
live_index.py -- the web app's searchable index, updated automatically when documents are uploaded.

    from graphrag_lite.api import live_index
    live_index.startup()                 # at server start (runs in a background thread)
    live_index.enqueue_file(path)        # after an upload -> full pipeline in the background
    live_index.active_dir()              # index folder every UI pipeline should query

Layout (data/live_index/):
    records.jsonl     every indexed chunk: chunk text + extracted entities/relationships (+ source info)
    state.json        active version, per-document status, job history
    v<N>/             one complete index version: extractions.jsonl, graph.graphml-free collections
                      (chunks/, entities/, relationships/, relationship_records.json)

Pipeline for an uploaded file (one background worker, one job at a time):
    load text -> chunk (CHUNKING_STRATEGY, default markdown_aware) -> extract entities and relationships
    with the local extraction model (production extractor, retry ladder, gleaning as configured)
    -> rebuild the graph -> embed and write a NEW index version -> switch the server to it.
A new version is written for every update, so queries never read a half-written index and Windows file
locks on the version in use do not matter. Older versions are removed when no longer active.

Seeding: the first time, the live index is built from the existing canonical index (data/cache:
extractions.jsonl + chunk text from chunks_v1). The canonical data itself is never modified.

Settings (environment variables, optional):
    LIVE_INDEX_EXTRACTION_MODEL   local Ollama model used for extraction (default qwen2.5:3b)
    LIVE_INDEX_DISABLED=1         do not use the live index (the app falls back to the canonical stores)
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
LIVE_ROOT = ROOT / "data" / "live_index"
RECORDS = LIVE_ROOT / "records.jsonl"
STATE = LIVE_ROOT / "state.json"
RAW_DIR = ROOT / "data" / "raw"
CANONICAL = ROOT / "data" / "cache"
INDEXABLE = {".txt", ".md", ".json"}
EXTRACTION_MODEL = os.environ.get("LIVE_INDEX_EXTRACTION_MODEL", "qwen2.5:3b")
CHUNK_STRATEGY = "markdown_aware"
MIN_FREE_MB = 200          # an index version for a small corpus is ~16 MB; keep a safety margin

_lock = threading.RLock()
_queue: "queue.Queue[dict]" = queue.Queue()
_worker: threading.Thread | None = None
_state: dict = {"active": None, "docs": {}, "current_job": None, "history": []}
_ready = threading.Event()


# ── helpers ──────────────────────────────────────────────────────────────────
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def doc_key(file_name: str) -> str:
    """Same document id the web UI already uses (md5 of the file name)."""
    return hashlib.md5(file_name.encode()).hexdigest()[:16]


def _save_state() -> None:
    LIVE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_state, indent=2), encoding="utf-8")
    os.replace(tmp, STATE)


def _load_records() -> list[dict]:
    if not RECORDS.exists():
        return []
    return [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_records(records: list[dict]) -> None:
    tmp = RECORDS.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, RECORDS)


def disabled() -> bool:
    return os.environ.get("LIVE_INDEX_DISABLED", "") == "1"


def active_dir() -> Path | None:
    """Index folder the UI pipelines should query, or None (use the canonical stores)."""
    if disabled():
        return None
    a = _state.get("active")
    return LIVE_ROOT / a if a else None


def _activate(version_dir: Path) -> None:
    """Point the process-wide MindMesh config at version_dir (same mechanism as the benchmark tooling)."""
    os.environ.update({
        "ZVEC_ROOT": str(version_dir),
        "CACHE_DIR": str(version_dir),
        "ZVEC_COLLECTION_CHUNKS": "chunks",
        "ZVEC_COLLECTION_ENTITIES": "entities",
        "CHUNKING_STRATEGY": CHUNK_STRATEGY,
        "ZVEC_ADOPT_UNTAGGED": "0",
    })
    from graphrag_lite.config import get_config
    get_config.cache_clear()
    try:
        import graphrag_lite.retrieval.pipeline_b as pb
        pb._G = pb._PR = pb._PR_MAX = None
    except Exception:  # noqa: BLE001
        pass
    try:
        import graphrag_lite.cache.llm_cache as lc
        lc._cache = None
    except Exception:  # noqa: BLE001
        pass


# ── index build ──────────────────────────────────────────────────────────────
def _build_version(records: list[dict]) -> Path:
    """Write a complete new index version from records; return its folder (not yet active)."""
    from graphrag_lite.embeddings.embedder import embed_batch
    from graphrag_lite.graph.builder import build_graph
    if str(ROOT / "eval") not in sys.path:
        sys.path.insert(0, str(ROOT / "eval"))
    from run_model_experiment import build_experiment_collections

    free_mb = shutil.disk_usage(LIVE_ROOT).free / 2**20
    if free_mb < MIN_FREE_MB:
        raise OSError(f"not enough disk space to build the index: {free_mb:.0f} MB free on the data drive, "
                      f"at least {MIN_FREE_MB} MB needed. Free some space and re-upload or restart the server.")
    existing = [int(p.name[1:]) for p in LIVE_ROOT.glob("v*") if p.name[1:].isdigit()]
    vdir = LIVE_ROOT / f"v{max(existing, default=0) + 1}"
    vdir.mkdir(parents=True)
    try:
        return _fill_version(vdir, records)
    except BaseException:
        shutil.rmtree(vdir, ignore_errors=True)      # never leave a half-written version behind
        raise


def _fill_version(vdir: Path, records: list[dict]) -> Path:
    from graphrag_lite.embeddings.embedder import embed_batch
    from graphrag_lite.graph.builder import build_graph
    from run_model_experiment import build_experiment_collections
    with open(vdir / "extractions.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            if "_parse_error" not in r:
                fh.write(json.dumps({k: r[k] for k in ("chunk_id", "doc_id", "chunk_index", "text", "entities",
                                                      "relationships")}, ensure_ascii=False) + "\n")
    clean = [r for r in records if "_parse_error" not in r]
    embed_batch(["dimension probe"])            # sets cfg.EMBEDDING_DIM before collections are created
    with contextlib.redirect_stdout(io.StringIO()):
        G = build_graph(clean)
        meta = build_experiment_collections(G, records, vdir, model=EXTRACTION_MODEL)
    for kind in ("entities", "chunks", "relationships"):
        if (vdir / kind).exists():
            (vdir / kind / "chunking_strategy.txt").write_text(CHUNK_STRATEGY)
    info = {"built_at": _now(), "chunks": meta.get("chunk_count"), "entities": meta.get("entity_count"),
            "relationships": meta.get("relationship_count"), "graph_nodes": G.number_of_nodes(),
            "graph_edges": G.number_of_edges(), "documents": len({r["doc_id"] for r in records})}
    (vdir / "version.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return vdir


def _publish(records: list[dict]) -> dict:
    """Build a new version from records, make it active, prune old versions."""
    vdir = _build_version(records)
    with _lock:
        _write_records(records)
        previous = _state.get("active")
        _state["active"] = vdir.name
        _activate(vdir)
        _save_state()
    for p in LIVE_ROOT.glob("v*"):
        if p.is_dir() and p.name not in (vdir.name, previous):
            shutil.rmtree(p, ignore_errors=True)   # still-open handles on Windows: retried next time
    return json.loads((vdir / "version.json").read_text(encoding="utf-8"))


def _seed_from_canonical() -> list[dict]:
    """Canonical extraction records + their chunk text (read-only use of data/cache)."""
    from graphrag_lite.retrieval.lightrag.chunk_lookup import fetch_chunks_by_ids
    ext = [json.loads(l) for l in (CANONICAL / "extractions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = fetch_chunks_by_ids([r["chunk_id"] for r in ext], exp_dir=None)
    out = []
    for r in ext:
        t = texts.get(r["chunk_id"])
        if not t or not t.get("text"):
            continue
        out.append({"chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "chunk_index": t.get("chunk_index", 0),
                    "text": t["text"], "entities": r.get("entities", []), "relationships": r.get("relationships", []),
                    "source": "canonical"})
    return out


# ── per-document pipeline ────────────────────────────────────────────────────
def _set_doc(key: str, **kw) -> None:
    with _lock:
        d = _state["docs"].setdefault(key, {})
        d.update(kw, updated_at=_now())
        _save_state()


def _extract_chunks(chunks: list[dict], key: str) -> list[dict]:
    from graphrag_lite.config import get_config
    from graphrag_lite.ingestion.extractor import extract_entities_relationships
    from graphrag_lite.ingestion.gleaning import glean
    from graphrag_lite.llm.ollama_client import OllamaClient
    cfg = get_config()
    client = OllamaClient(model=EXTRACTION_MODEL, temperature=cfg.EXTRACTION_TEMPERATURE, seed=cfg.EXTRACTION_SEED,
                          timeout=cfg.LLM_TIMEOUT_S, num_predict=cfg.LLM_NUM_PREDICT, num_ctx=cfg.LLM_NUM_CTX,
                          think=cfg.effective_think(), frequency_penalty=cfg.LLM_FREQUENCY_PENALTY)
    out = []
    for i, ch in enumerate(chunks, 1):
        _set_doc(key, stage=f"extracting entities and relationships ({i}/{len(chunks)} chunks)")
        rec = {"chunk_id": ch["chunk_id"], "doc_id": ch["doc_id"], "chunk_index": ch["chunk_index"],
               "text": ch["text"], "source": "upload", "extraction_model": EXTRACTION_MODEL}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ext = extract_entities_relationships(ch, client)
                if cfg.MAX_GLEANINGS > 0:
                    ext = glean(ch, ext, client, max_gleanings=cfg.MAX_GLEANINGS)
            rec["entities"], rec["relationships"] = ext.get("entities", []), ext.get("relationships", [])
        except Exception as exc:  # noqa: BLE001 -- recorded; the chunk is still searchable as text
            rec["entities"], rec["relationships"] = [], []
            rec["_parse_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        out.append(rec)
    return out


def _run_index_job(job: dict) -> None:
    from graphrag_lite.config import get_config
    from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
    path = Path(job["path"]); key = job["key"]; t0 = time.monotonic()
    _set_doc(key, status="processing", stage="loading", file_name=path.name, track_id=job["track_id"], error=None)
    if path.suffix.lower() not in INDEXABLE:
        raise ValueError(f"{path.suffix} files are stored but cannot be indexed yet (supported: .txt, .md, .json)")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("the file is empty")
    doc = {"doc_id": path.stem, "text": text, "source_path": str(path)}
    _set_doc(key, stage="chunking")
    chunks = chunk_document(doc, get_config(), get_tokenizer())
    if not chunks:
        raise ValueError("no chunks were produced")
    new = _extract_chunks(chunks, key)
    for r in new:
        r["file_name"] = path.name
    _set_doc(key, stage="building graph and embeddings")
    with _lock:
        records = [r for r in _load_records() if r.get("file_name") != path.name]
    info = _publish(records + new)
    failed = sum("_parse_error" in r for r in new)
    _set_doc(key, status="processed", stage="done", chunks=len(new), failed_chunks=failed,
             entities=sum(len(r["entities"]) for r in new), relationships=sum(len(r["relationships"]) for r in new),
             content_length=len(text), duration_s=round(time.monotonic() - t0, 1), index_version=_state["active"],
             index=info)
    log.info("Live index: %s indexed (%d chunks, %d failed) -> %s", path.name, len(new), failed, _state["active"])


def _run_remove_job(job: dict) -> None:
    name = job["file_name"]
    with _lock:
        records = _load_records()
    kept = [r for r in records if r.get("file_name") != name]
    if len(kept) != len(records):
        _publish(kept)
    with _lock:
        _state["docs"].pop(job["key"], None)
        _save_state()


def _worker_loop() -> None:
    _ready.wait()
    while True:
        job = _queue.get()
        try:
            with _lock:
                _state["current_job"] = {**job, "started_at": _now()}
            _safe_save()
            try:
                (_run_remove_job if job["kind"] == "remove" else _run_index_job)(job)
                outcome = "ok"
            except Exception as exc:  # noqa: BLE001
                log.exception("Live index job failed: %s", job)
                outcome = f"failed: {type(exc).__name__}: {exc}"
                if job["kind"] == "index":
                    with _lock:
                        _state["docs"].setdefault(job["key"], {}).update(
                            status="failed", stage="failed", error=f"{type(exc).__name__}: {exc}", updated_at=_now())
            with _lock:
                _state["current_job"] = None
                _state["history"] = (_state["history"] + [{**job, "finished_at": _now(), "outcome": outcome}])[-50:]
            _safe_save()
        except BaseException:  # noqa: BLE001 -- the worker must survive anything (e.g. a full disk)
            log.exception("Live index worker error")
            _state["current_job"] = None
        finally:
            _queue.task_done()


def _safe_save() -> None:
    try:
        with _lock:
            _save_state()
    except OSError as exc:
        log.error("Live index: could not save state (%s); status kept in memory", exc)


# ── public API ───────────────────────────────────────────────────────────────
def enqueue_file(path: Path) -> str:
    """Queue a stored upload for the full pipeline. Returns the track id."""
    path = Path(path)
    key = doc_key(path.name)
    track_id = f"index-{key}-{int(time.time())}"
    _set_doc(key, status="pending", stage="queued", file_name=path.name, track_id=track_id, error=None,
             created_at=_state["docs"].get(key, {}).get("created_at", _now()))
    _queue.put({"kind": "index", "path": str(path), "key": key, "track_id": track_id, "queued_at": _now()})
    return track_id


def enqueue_remove(file_name: str) -> None:
    _queue.put({"kind": "remove", "file_name": file_name, "key": doc_key(file_name), "queued_at": _now()})


def doc_status(file_name: str) -> dict | None:
    return _state["docs"].get(doc_key(file_name))


def track_status(track_id: str) -> dict | None:
    for k, d in _state["docs"].items():
        if d.get("track_id") == track_id:
            return {"id": k, **d}
    return None


def pipeline_status() -> dict:
    cur = _state.get("current_job")
    busy = cur is not None or not _queue.empty() or not _ready.is_set()
    msg = "MindMesh index is up to date"
    if not _ready.is_set():
        msg = "Preparing the live index"
    elif cur:
        d = _state["docs"].get(cur.get("key"), {})
        msg = f"{cur.get('kind', 'index').capitalize()}: {cur.get('file_name') or Path(cur.get('path', '')).name} - {d.get('stage', '')}"
    return {"busy": busy, "job_name": msg if busy else "idle", "job_start": (cur or {}).get("started_at"),
            "queued": _queue.qsize(), "latest_message": msg, "active_version": _state.get("active"),
            "history_messages": [f"{h.get('finished_at')} {h.get('kind')} "
                                 f"{h.get('file_name') or Path(h.get('path', '')).name}: {h.get('outcome')}"
                                 for h in _state.get("history", [])[-10:]]}


def graph_extractions_path() -> Path | None:
    d = active_dir()
    return d / "extractions.jsonl" if d and (d / "extractions.jsonl").exists() else None


def startup() -> None:
    """Load or seed the live index, activate it, start the worker, queue unindexed uploads. Never raises."""
    global _worker, _state
    if disabled():
        log.info("Live index disabled (LIVE_INDEX_DISABLED=1)")
        return
    try:
        LIVE_ROOT.mkdir(parents=True, exist_ok=True)
        if STATE.exists():
            _state.update(json.loads(STATE.read_text(encoding="utf-8")))
            _state["current_job"] = None
        if _worker is None:
            _worker = threading.Thread(target=_worker_loop, name="live-index-worker", daemon=True)
            _worker.start()
        a = _state.get("active")
        if a and (LIVE_ROOT / a / "version.json").exists():
            _activate(LIVE_ROOT / a)
            log.info("Live index active: %s", a)
        else:
            log.info("Seeding the live index from the canonical index (one-time)...")
            info = _publish(_seed_from_canonical())
            log.info("Live index seeded: %s %s", _state["active"], info)
        # interrupted jobs and uploads that were never indexed
        indexed = {r.get("file_name") for r in _load_records() if r.get("file_name")}
        if RAW_DIR.exists():
            for f in sorted(RAW_DIR.iterdir()):
                if f.is_file() and f.suffix.lower() in INDEXABLE and not f.name.startswith("."):
                    if f.name not in indexed:          # new, interrupted or previously failed (e.g. full disk)
                        enqueue_file(f)
    except Exception:  # noqa: BLE001 -- the app keeps working on the canonical stores
        log.exception("Live index startup failed; queries use the canonical index")
        _state["active"] = None
    finally:
        _ready.set()
