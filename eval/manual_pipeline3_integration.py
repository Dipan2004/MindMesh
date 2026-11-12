#!/usr/bin/env python
"""
manual_pipeline3_integration.py - MANUAL, MODEL-BACKED validation of genuine
upstream LightRAG (Pipeline 3) through the MindMesh Pipeline 3 adapter.

WARNING
-------
This script performs REAL Ollama embedding and LLM inference during document
insertion and querying.  It is intentionally NOT part of the automated test
suite and must only be run manually after explicit authorization.

What it validates
-----------------
  1.  The genuine upstream LightRAG package (LightRAG/lightrag, lightrag-hku)
      is imported - not the MindMesh native LightRAG-style implementation.
  2.  The MindMesh Pipeline 3 adapter builds the LightRAG instance with the
      configured LLM and embedding functions.
  3.  Storage is initialized before insertion and finalized afterwards.
  4.  Real MindMesh source documents are inserted with genuine upstream
      `ainsert`, producing a LightRAG graph/vector store.
  5.  A real query returns both retrieval context and a generated answer.
  6.  The retrieved context can be traced back to the indexed documents.
  7.  Pipeline 3 storage is isolated from Pipeline A/B and production paths.

Safety
------
  * Uses only qwen2.5:1.5b (qwen3:4b is explicitly refused).
  * Never pulls or installs models/dependencies.
  * Never touches eval/gold_test_set_frozen.json.
  * Never deletes existing MindMesh data.
  * Writes only under data/experiments/pipeline3_manual_integration/.
  * Refuses to reuse a non-empty run directory.
  * Optional --cleanup-on-failure removes only the dedicated run directory.

Usage
-----
  # CPU-only prerequisite check (no model calls)
  python eval/manual_pipeline3_integration.py --check-only

  # Genuine indexing + one query (MODEL-BACKED)
  python eval/manual_pipeline3_integration.py \
      --model qwen2.5:1.5b --embed-model nomic-embed-text --mode hybrid

  # Optional dev-set evaluation (MODEL-BACKED; indexes all 5 docs)
  python eval/manual_pipeline3_integration.py \
      --model qwen2.5:1.5b --embed-model nomic-embed-text --mode hybrid \
      --all-docs --dev-set
"""
from __future__ import annotations

import argparse
import asyncio
import enum
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT / "src"), str(ROOT / "LightRAG"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DEFAULT_MODEL = "qwen2.5:1.5b"
ALLOWED_MODELS = {DEFAULT_MODEL}
FORBIDDEN_MODELS = {"qwen3:4b"}
DEFAULT_EMBED_MODEL = "nomic-embed-text"
EXPECTED_EMBED_DIM = 768
DEFAULT_LLM_HOST = "http://localhost:11434"
DEFAULT_MODE = "hybrid"
DEFAULT_TOP_K = 6
DEFAULT_QUERY = (
    "What is hybrid search and how does it combine vector search with "
    "keyword search?"
)
DEFAULT_DOCS = ("vector_databases.md", "retrieval_augmented_generation.md")
DEDICATED_PARENT = ROOT / "data" / "experiments" / "pipeline3_manual_integration"
DEV_SET_PATH = ROOT / "eval" / "gold_dev_set.json"

STOPWORDS = {
    "about", "above", "after", "against", "also", "among", "and", "another",
    "are", "because", "been", "before", "being", "between", "both", "but",
    "can", "could", "does", "during", "each", "from", "further", "have",
    "having", "here", "into", "its", "more", "most", "must", "not", "only",
    "other", "over", "same", "should", "some", "such", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those",
    "through", "under", "until", "using", "was", "were", "what", "when",
    "where", "which", "while", "will", "with", "within", "would",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, enum.Enum):
        return str(value)
    return value


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(data), handle, indent=2, ensure_ascii=False)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _ngram_overlap(context: str, document: str, n: int = 4) -> dict:
    ctx = _tokenize(context)
    doc = _tokenize(document)
    if len(ctx) < n or len(doc) < n:
        return {"overlap_ngrams": 0, "examples": [], "n": n}
    ctx_grams = {tuple(ctx[i:i + n]) for i in range(len(ctx) - n + 1)}
    doc_grams = {tuple(doc[i:i + n]) for i in range(len(doc) - n + 1)}
    common = ctx_grams & doc_grams
    examples = [" ".join(gram) for gram in sorted(common)[:5]]
    return {"overlap_ngrams": len(common), "examples": examples, "n": n}


def _shared_technical_terms(context: str, document: str) -> list[str]:
    ctx = set(_tokenize(context))
    doc = set(_tokenize(document))
    shared = {
        term for term in (ctx & doc)
        if len(term) >= 5 and term not in STOPWORDS and term.isalpha()
    }
    return sorted(shared)[:20]


def _trace_context(
    context: str, documents: dict[str, str], query: str = ""
) -> dict:
    """Trace retrieval context back to indexed documents.

    The fallback term-overlap check excludes tokens that already appear in the
    query, so an echoed query cannot masquerade as retrieved document content.
    """
    query_terms = set(_tokenize(query))
    per_doc = {}
    for name, text in documents.items():
        overlap = _ngram_overlap(context, text)
        shared_terms = _shared_technical_terms(context, text)
        relevant_terms = [
            term for term in shared_terms if term not in query_terms
        ]
        per_doc[name] = {
            **overlap,
            "shared_technical_terms": shared_terms,
            "shared_terms_not_in_query": relevant_terms,
            "traced": bool(
                overlap["overlap_ngrams"] > 0 or len(relevant_terms) >= 3
            ),
        }
    traced_docs = [name for name, info in per_doc.items() if info["traced"]]
    return {
        "traced": bool(traced_docs),
        "traced_documents": traced_docs,
        "per_document": per_doc,
    }


def _storage_inventory(workdir: Path) -> list[dict]:
    if not workdir.exists():
        return []
    files = []
    for path in sorted(workdir.rglob("*")):
        if path.is_file():
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            files.append({
                "path": str(path.relative_to(workdir)),
                "size_bytes": size,
            })
    return files


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _model_present(installed: list[str], wanted: str) -> bool:
    if wanted in installed:
        return True
    if f"{wanted}:latest" in installed:
        return True
    wanted_base = wanted.split(":", 1)[0]
    return any(name.split(":", 1)[0] == wanted_base for name in installed)


def _ollama_models(host: str) -> tuple[list[str], str]:
    host = host.rstrip("/")
    http_error = None
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = [item["name"] for item in payload.get("models", []) if item.get("name")]
        return names, "http"
    except Exception as exc:
        http_error = exc

    try:
        proc = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=30
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or f"ollama list exit {proc.returncode}")
        names = []
        for line in proc.stdout.strip().splitlines()[1:]:
            parts = line.split()
            if parts:
                names.append(parts[0])
        return names, "cli"
    except Exception as cli_error:
        raise RuntimeError(
            f"Could not query Ollama via HTTP ({http_error}) or CLI ({cli_error})"
        )


def _resolve_documents(args: argparse.Namespace) -> list[Path]:
    raw_dir = ROOT / "data" / "raw"
    if args.all_docs:
        paths = sorted(
            p for p in raw_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".md", ".txt")
        )
    else:
        names = [n.strip() for n in args.docs.split(",") if n.strip()]
        paths = [raw_dir / name for name in names]
    return paths


def _create_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        run_dir = run_dir.resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(
                f"Refusing to use non-empty run directory: {run_dir}\n"
                "Choose a new --run-dir or remove it manually."
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    DEDICATED_PARENT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = DEDICATED_PARENT / f"run_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = DEDICATED_PARENT / f"run_{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _validate_run_paths(run_dir: Path, workdir: Path) -> dict:
    run_resolved = run_dir.resolve()
    work_resolved = workdir.resolve()
    cache_dir = (ROOT / "data" / "cache").resolve()
    p3_default = (ROOT / "data" / "pipeline3_workdir").resolve()

    work_under_run = work_resolved == run_resolved or run_resolved in work_resolved.parents
    under_cache = work_resolved == cache_dir or cache_dir in work_resolved.parents
    is_p3_default = work_resolved == p3_default

    return {
        "ok": bool(work_under_run and not under_cache and not is_p3_default),
        "run_dir": str(run_resolved),
        "workdir": str(work_resolved),
        "workdir_under_run_dir": work_under_run,
        "workdir_under_data_cache": under_cache,
        "workdir_is_default_pipeline3_workdir": is_p3_default,
        "canonical_cache": str(cache_dir),
        "default_pipeline3_workdir": str(p3_default),
    }


def _query_mode_map(top_k: int) -> dict:
    from lightrag import QueryParam
    from mindmesh.pipelines.pipeline3_lightrag.adapter import LIGHTRAG_MODES

    result = {}
    for mode in LIGHTRAG_MODES:
        param = QueryParam(
            mode=mode,
            top_k=top_k,
            only_need_context=True,
            enable_rerank=False,
        )
        result[mode] = {
            "ok": param.mode == mode and param.top_k == top_k,
            "mode": param.mode,
            "top_k": param.top_k,
            "only_need_context": param.only_need_context,
            "enable_rerank": param.enable_rerank,
        }
    return result


# ---------------------------------------------------------------------------
# Prerequisite checks (CPU-only)
# ---------------------------------------------------------------------------

def _run_prerequisites(args: argparse.Namespace) -> dict:
    checks: dict[str, dict] = {}
    errors: list[str] = []

    def run_check(name: str, func) -> Any:
        try:
            detail = func()
            checks[name] = {"ok": True, "detail": detail}
            return detail
        except Exception as exc:
            checks[name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            return None

    run_check("python", lambda: {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    })

    run_check("model_guard", lambda: _check_model_guard(args.model))

    lightrag_detail = run_check("lightrag_import", _import_lightrag_detail)
    run_check("adapter_import", _import_adapter_detail)

    def _ollama_detail() -> dict:
        names, source = _ollama_models(args.llm_host)
        return {"models": names, "source": source}

    installed_models = run_check("ollama_models", _ollama_detail)
    if installed_models is not None:
        names = installed_models["models"]
        run_check(
            "llm_model_available",
            lambda: _model_availability_detail(names, args.model, "LLM"),
        )
        run_check(
            "embedding_model_available",
            lambda: _model_availability_detail(
                names, args.embed_model, "Embedding"
            ),
        )

    doc_paths = _resolve_documents(args)
    run_check(
        "documents",
        lambda: _check_documents(doc_paths),
    )

    run_check(
        "query_mode_mapping",
        lambda: _assert_all_ok(_query_mode_map(args.top_k), "query mode mapping"),
    )

    tmp_run = DEDICATED_PARENT / "_check_only_run"
    check_workdir = tmp_run / "lightrag_storage"
    run_check(
        "storage_isolation_paths",
        lambda: _assert_isolation(
            _validate_run_paths(tmp_run, check_workdir)
        ),
    )

    run_check("adapter_dry_run", lambda: _adapter_dry_run(args, check_workdir))

    if args.dev_set:
        run_check("gold_dev_set", lambda: _check_dev_set(args))

    return {
        "ok": bool(checks) and all(item["ok"] for item in checks.values()),
        "checks": checks,
        "errors": errors,
        "lightrag_version": (lightrag_detail or {}).get("version"),
    }


def _assert_all_ok(detail: dict, label: str) -> dict:
    failed = [name for name, item in detail.items() if not item.get("ok")]
    if failed:
        raise RuntimeError(f"{label} failed for: {', '.join(failed)}")
    return detail


def _assert_isolation(detail: dict) -> dict:
    if not detail.get("ok"):
        raise RuntimeError(f"Storage isolation check failed: {detail}")
    return detail


def _model_availability_detail(
    installed: list[str], wanted: str, kind: str
) -> dict:
    if not _model_present(installed, wanted):
        raise RuntimeError(
            f"{kind} model {wanted!r} is not installed locally. "
            f"Installed: {', '.join(installed)}. Do not pull it automatically."
        )
    return {
        "requested": wanted,
        "present": True,
        "installed": installed,
    }


def _check_model_guard(model: str) -> dict:
    if model.strip().lower() in FORBIDDEN_MODELS:
        raise ValueError(
            "qwen3:4b is forbidden for this experiment. Use qwen2.5:1.5b only."
        )
    if model not in ALLOWED_MODELS:
        raise ValueError(
            f"Model {model!r} is not permitted. This experiment uses only "
            f"{', '.join(sorted(ALLOWED_MODELS))}."
        )
    return {"model": model, "allowed": sorted(ALLOWED_MODELS)}


def _import_lightrag_detail() -> dict:
    import lightrag
    from lightrag import LightRAG, QueryParam

    version = getattr(lightrag, "__version__", None)
    if version is None:
        try:
            from lightrag._version import __version__ as version
        except Exception:
            version = "unknown"
    return {
        "version": version,
        "module_file": getattr(lightrag, "__file__", None),
        "has_LightRAG": LightRAG is not None,
        "has_QueryParam": QueryParam is not None,
    }


def _import_adapter_detail() -> dict:
    from mindmesh.pipelines.pipeline3_lightrag.adapter import (
        LIGHTRAG_MODES,
        Pipeline3LightRAGAdapter,
        _make_ollama_embed_func,
    )

    return {
        "adapter_class": Pipeline3LightRAGAdapter.__name__,
        "supports_modes": list(LIGHTRAG_MODES),
        "embedding_func_factory": _make_ollama_embed_func.__name__,
    }


def _check_documents(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("No documents selected.")
    details = []
    for path in paths:
        resolved = path.resolve()
        raw_dir = (ROOT / "data" / "raw").resolve()
        if raw_dir not in resolved.parents:
            raise ValueError(f"Document is outside data/raw: {resolved}")
        if not resolved.exists():
            raise FileNotFoundError(f"Document not found: {resolved}")
        text = resolved.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Document is empty: {resolved}")
        details.append({
            "name": resolved.name,
            "doc_id": resolved.stem,
            "path": str(resolved),
            "chars": len(text),
        })
    return {"count": len(details), "documents": details}


def _adapter_dry_run(args: argparse.Namespace, workdir: Path) -> dict:
    from mindmesh.pipelines.pipeline3_lightrag.adapter import Pipeline3LightRAGAdapter

    adapter = Pipeline3LightRAGAdapter(
        model=args.model,
        top_k=args.top_k,
        workdir=workdir,
        llm_host=args.llm_host,
        embed_model=args.embed_model,
        embedding_dim=args.expected_embedding_dim,
    )
    dry = adapter.dry_run()
    if not dry.get("lightrag_importable"):
        raise RuntimeError(f"LightRAG not importable: {dry.get('lightrag_import_error')}")
    if dry.get("pipeline") != "pipeline_3_lightrag":
        raise RuntimeError(f"Unexpected adapter pipeline name: {dry.get('pipeline')}")
    return {
        "pipeline": dry.get("pipeline"),
        "workdir": dry.get("workdir"),
        "is_indexed": dry.get("is_indexed"),
        "lightrag_importable": dry.get("lightrag_importable"),
    }


def _check_dev_set(args: argparse.Namespace) -> dict:
    if not DEV_SET_PATH.exists():
        raise FileNotFoundError(f"Dev set not found: {DEV_SET_PATH}")
    data = _read_json(DEV_SET_PATH)
    queries = data.get("queries", [])
    if not queries:
        raise ValueError("Dev set contains no queries.")
    if not args.all_docs:
        raise ValueError(
            "--dev-set requires --all-docs because gold_dev_set.json covers "
            "all five documents in data/raw."
        )
    return {
        "path": str(DEV_SET_PATH),
        "set_name": data.get("set_name"),
        "n_queries": len(queries),
        "query_ids": [q.get("query_id") for q in queries],
    }


# ---------------------------------------------------------------------------
# Genuine model-backed experiment (single event loop)
# ---------------------------------------------------------------------------

def _set_stage(report: dict, name: str, ok: bool, started: float, **details) -> None:
    report["stages"][name] = {
        "ok": bool(ok),
        "elapsed_s": round(time.monotonic() - started, 3),
        **details,
    }


def _record_error(report: dict, stage: str, exc: Exception) -> None:
    _append_error(
        report,
        stage,
        type(exc).__name__,
        str(exc),
        traceback.format_exc(),
    )


def _append_error(
    report: dict,
    stage: str,
    error_type: str,
    message: str,
    traceback_text: str = "",
) -> None:
    report["errors"].append({
        "stage": stage,
        "type": error_type,
        "message": message,
        "traceback": traceback_text,
    })
    print(f"[ERROR] {stage}: {error_type}: {message}", file=sys.stderr, flush=True)


_PLACEHOLDER_LABEL_RE = re.compile(r"^<[^<>]+>$")


def _is_placeholder_label(label: Any) -> bool:
    return bool(_PLACEHOLDER_LABEL_RE.match(str(label).strip()))


def _storage_flush_evidence(
    inventory: list[dict],
    *,
    expect_entities: bool = True,
    expect_relationships: bool = True,
    expect_chunks: bool = True,
) -> dict:
    """Check that the vector indexes with data were actually persisted."""
    expected = []
    if expect_entities:
        expected.append("vdb_entities.json")
    if expect_relationships:
        expected.append("vdb_relationships.json")
    if expect_chunks:
        expected.append("vdb_chunks.json")

    names = {Path(item["path"]).name for item in inventory}
    present = {name: name in names for name in expected}
    return {
        "ok": all(present.values()),
        "files": present,
        "missing": [name for name, ok in present.items() if not ok],
        "expected": expected,
    }


def _status_text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").lower()


def _status_buckets(statuses: dict[str, dict]) -> dict[str, list[str]]:
    buckets = {
        "processed": [],
        "failed": [],
        "pending_or_active": [],
        "missing": [],
    }
    for doc_id, info in statuses.items():
        if not info.get("present"):
            buckets["missing"].append(doc_id)
            continue
        status = _status_text(info.get("status"))
        if status == "processed":
            buckets["processed"].append(doc_id)
        elif status == "failed":
            buckets["failed"].append(doc_id)
        else:
            buckets["pending_or_active"].append(doc_id)
    return buckets


def _insertion_success_criteria(
    doc_results: list[dict], statuses_all_processed: bool
) -> bool:
    """Insertion is successful only if every document reached PROCESSED."""
    return bool(
        doc_results
        and all(item.get("ok") for item in doc_results)
        and statuses_all_processed
    )


def _indexing_success_criteria(
    *,
    docs_present: bool,
    statuses_all_processed: bool,
    flush_ok: bool,
    graph_nonempty: bool,
    non_placeholder_count: int,
) -> dict:
    failures = []
    if not docs_present:
        failures.append("full_docs_empty")
    if not statuses_all_processed:
        failures.append("document_status_incomplete")
    if not flush_ok:
        failures.append("storage_flush_missing")
    if not graph_nonempty:
        failures.append("graph_empty")
    if non_placeholder_count <= 0:
        failures.append("no_non_placeholder_graph_labels")
    return {"ok": not failures, "failures": failures}


async def _document_statuses(rag, doc_ids: list[str]) -> dict[str, dict]:
    raw = await _maybe_await(rag.doc_status.get_by_ids(list(doc_ids)))
    statuses: dict[str, dict] = {}
    for doc_id, record in zip(doc_ids, raw or []):
        if not isinstance(record, dict):
            statuses[doc_id] = {
                "present": False,
                "status": None,
                "error_msg": None,
            }
            continue
        statuses[doc_id] = {
            "present": True,
            "status": record.get("status"),
            "error_msg": record.get("error_msg") or record.get("error"),
        }
    return statuses


def _adapter_metadata(adapter) -> dict:
    return {
        "class": type(adapter).__name__,
        "name": adapter.name,
        "model": adapter.model,
        "top_k": adapter.top_k,
        "workdir": str(adapter.workdir),
        "llm_host": adapter.llm_host,
        "embed_model": adapter.embed_model,
        "embedding_dim": adapter.embedding_dim,
        "supports_modes": list(adapter.supports_modes),
    }


async def _run_experiment(args, run_dir: Path, report: dict) -> None:
    workdir = run_dir / "lightrag_storage"
    report["paths"]["workdir"] = str(workdir)
    report["paths"]["storage_inventory_before"] = _storage_inventory(workdir)

    started = time.monotonic()
    try:
        from mindmesh.pipelines.pipeline3_lightrag.adapter import (
            Pipeline3LightRAGAdapter,
        )

        adapter = Pipeline3LightRAGAdapter(
            model=args.model,
            top_k=args.top_k,
            workdir=workdir,
            llm_host=args.llm_host,
            embed_model=args.embed_model,
            embedding_dim=args.expected_embedding_dim,
        )
        report["adapter"] = _adapter_metadata(adapter)
        report["adapter"]["lifecycle_note"] = (
            "The adapter's public index_documents()/query() methods do not call "
            "LightRAG.initialize_storages()/finalize_storages(). This manual "
            "script therefore drives the documented LightRAG lifecycle on the "
            "adapter-created instance."
        )
        rag = adapter._get_rag()
        report["initialization_reference"] = {
            "workspace": rag.workspace,
            "working_dir": rag.working_dir,
        }
    except Exception as exc:
        _set_stage(report, "initialization", False, started)
        _record_error(report, "initialization", exc)
        return

    initialized = False
    insertion_ok = False
    query_ok = False
    try:
        try:
            await rag.initialize_storages()
            initialized = True
            status = await _pipeline_status(rag)
            _set_stage(
                report, "initialization", True, started,
                workspace=rag.workspace,
                working_dir=rag.working_dir,
                pipeline_status_present=status.get("present"),
                pipeline_status_keys=status.get("keys"),
            )
        except Exception as exc:
            _set_stage(report, "initialization", False, started)
            _record_error(report, "initialization", exc)
            return

        # ---- Embedding dimension check (real embedding call) --------------
        started = time.monotonic()
        expected_dim = args.expected_embedding_dim
        try:
            import numpy as np
            probe = await rag.embedding_func(["embedding dimension probe"])
            shape = tuple(np.asarray(probe).shape)
            actual_dim = shape[-1] if shape else None
            ok = bool(shape == (1, expected_dim))
            _set_stage(
                report, "embedding_dimension", ok, started,
                embed_model=args.embed_model,
                expected_dim=expected_dim,
                actual_shape=shape,
                actual_dim=actual_dim,
            )
            if not ok:
                report["errors"].append({
                    "stage": "embedding_dimension",
                    "type": "DimensionMismatch",
                    "message": (
                        f"Embedding model {args.embed_model!r} returned shape "
                        f"{shape}; expected (1, {expected_dim})."
                    ),
                    "traceback": "",
                })
        except Exception as exc:
            _set_stage(report, "embedding_dimension", False, started)
            _record_error(report, "embedding_dimension", exc)

        # ---- Document insertion (real upstream ainsert) ------------------
        started = time.monotonic()
        doc_paths = _resolve_documents(args)
        documents: dict[str, str] = {}
        doc_results = []
        for path in doc_paths:
            name = path.name
            doc_id = path.stem
            text = path.read_text(encoding="utf-8")
            documents[doc_id] = text
            doc_started = time.monotonic()
            try:
                print(f"[insert] {name} ({len(text)} chars)...", flush=True)
                track_id = await rag.ainsert([text], ids=[doc_id])
                doc_results.append({
                    "name": name,
                    "doc_id": doc_id,
                    "ok": True,
                    "chars": len(text),
                    "track_id": track_id,
                    "elapsed_s": round(time.monotonic() - doc_started, 3),
                })
                print(
                    f"[insert] {name}: completed in "
                    f"{time.monotonic() - doc_started:.1f}s (pending status check)",
                    flush=True,
                )
            except Exception as exc:
                doc_results.append({
                    "name": name,
                    "doc_id": doc_id,
                    "ok": False,
                    "chars": len(text),
                    "elapsed_s": round(time.monotonic() - doc_started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                _record_error(report, f"document_insertion:{name}", exc)
        selected_doc_ids = [path.stem for path in doc_paths]
        statuses: dict[str, dict] = {}
        try:
            statuses = await _document_statuses(rag, selected_doc_ids)
        except Exception as exc:
            _append_error(
                report,
                "document_insertion",
                "DocumentStatusReadError",
                f"Could not read doc_status after insertion: {type(exc).__name__}: {exc}",
                traceback.format_exc(),
            )

        buckets = _status_buckets(statuses)
        for item in doc_results:
            info = statuses.get(item["doc_id"], {})
            item["status"] = info.get("status")
            item["status_error"] = info.get("error_msg")
            if not info.get("present"):
                item["ok"] = False
                item["status_error"] = "doc_status record missing"
            elif _status_text(info.get("status")) != "processed":
                item["ok"] = False

        for doc_id, info in statuses.items():
            if not info.get("present"):
                _append_error(
                    report,
                    f"document_insertion:{doc_id}",
                    "MissingDocumentStatus",
                    f"No doc_status record was written for {doc_id}",
                )
                continue
            status = _status_text(info.get("status"))
            if status == "failed":
                _append_error(
                    report,
                    f"document_insertion:{doc_id}",
                    "DocumentFailed",
                    f"{doc_id} failed: {info.get('error_msg') or 'no error message'}",
                )
            elif status != "processed":
                _append_error(
                    report,
                    f"document_insertion:{doc_id}",
                    "DocumentNotProcessed",
                    f"{doc_id} ended in non-terminal status {status!r}",
                )

        statuses_all_processed = (
            bool(selected_doc_ids)
            and len(buckets["processed"]) == len(selected_doc_ids)
            and not buckets["failed"]
            and not buckets["pending_or_active"]
            and not buckets["missing"]
        )
        insertion_ok = _insertion_success_criteria(
            doc_results, statuses_all_processed
        )
        _set_stage(
            report, "document_insertion", insertion_ok, started,
            documents=doc_results,
            inserted_count=sum(1 for item in doc_results if item["ok"]),
            doc_statuses=statuses,
            status_buckets=buckets,
            statuses_all_processed=statuses_all_processed,
        )
        report["documents"] = {
            name: {"chars": len(text)} for name, text in documents.items()
        }

        # ---- Indexing validation (in-memory + on-disk) -------------------
        started = time.monotonic()
        indexing_ok = False
        try:
            evidence = await _indexing_evidence(rag)
            inventory = _storage_inventory(workdir)
            try:
                statuses = await _document_statuses(rag, selected_doc_ids)
            except Exception as exc:
                _append_error(
                    report,
                    "indexing_validation",
                    "DocumentStatusReadError",
                    f"Could not read doc_status during indexing validation: "
                    f"{type(exc).__name__}: {exc}",
                    traceback.format_exc(),
                )
                statuses = {}
            buckets = _status_buckets(statuses)
            statuses_all_processed = (
                bool(selected_doc_ids)
                and len(buckets["processed"]) == len(selected_doc_ids)
                and not buckets["failed"]
                and not buckets["pending_or_active"]
                and not buckets["missing"]
            )
            docs_present = evidence.get("full_docs_empty") is False
            graph_labels_count = evidence.get("graph_labels_count") or 0
            non_placeholder_count = (
                evidence.get("non_placeholder_graph_labels_count") or 0
            )
            graph_nonempty = graph_labels_count > 0
            graph_edges_count = evidence.get("graph_edges_count") or 0
            text_chunks_empty = evidence.get("text_chunks_empty")
            flush = _storage_flush_evidence(
                inventory,
                expect_entities=graph_nonempty,
                expect_relationships=graph_edges_count > 0,
                expect_chunks=text_chunks_empty is False,
            )

            if not statuses_all_processed:
                _append_error(
                    report,
                    "indexing_validation",
                    "DocumentStatusIncomplete",
                    f"Indexing requires every selected document to be PROCESSED; "
                    f"buckets={buckets}",
                )
            if not flush["ok"]:
                _append_error(
                    report,
                    "indexing_validation",
                    "StorageFlushMissing",
                    f"Vector storage flush did not persist files: {flush['missing']}",
                )
            if graph_nonempty and non_placeholder_count == 0:
                _append_error(
                    report,
                    "indexing_validation",
                    "PlaceholderExtraction",
                    "All extracted graph labels are prompt placeholders; "
                    f"sample={evidence.get('graph_labels_sample')}",
                )

            criteria = _indexing_success_criteria(
                docs_present=docs_present,
                statuses_all_processed=statuses_all_processed,
                flush_ok=flush["ok"],
                graph_nonempty=graph_nonempty,
                non_placeholder_count=non_placeholder_count,
            )
            indexing_ok = criteria["ok"]
            _set_stage(
                report, "indexing_validation", indexing_ok, started,
                full_docs_empty=evidence.get("full_docs_empty"),
                graph_labels_count=graph_labels_count,
                graph_edges_count=graph_edges_count,
                text_chunks_empty=text_chunks_empty,
                graph_labels_sample=evidence.get("graph_labels_sample"),
                placeholder_graph_labels=evidence.get("placeholder_graph_labels"),
                placeholder_graph_labels_count=evidence.get(
                    "placeholder_graph_labels_count"
                ),
                non_placeholder_graph_labels_count=non_placeholder_count,
                doc_status_counts=evidence.get("doc_status_counts"),
                doc_statuses=statuses,
                status_buckets=buckets,
                storage_flush=flush,
                criteria_failures=criteria["failures"],
                storage_files=inventory,
            )
        except Exception as exc:
            _set_stage(report, "indexing_validation", False, started)
            _record_error(report, "indexing_validation", exc)

        # ---- Query execution (real upstream aquery) ----------------------
        started = time.monotonic()
        try:
            from lightrag import QueryParam

            context_result = await rag.aquery(
                args.query,
                param=QueryParam(
                    mode=args.mode,
                    top_k=args.top_k,
                    only_need_context=True,
                    enable_rerank=False,
                ),
            )
            context = "" if context_result is None else str(context_result)
            trace = _trace_context(context, documents, query=args.query)

            answer = None
            answer_error = None
            if not args.no_generate:
                try:
                    answer_result = await rag.aquery(
                        args.query,
                        param=QueryParam(
                            mode=args.mode,
                            top_k=args.top_k,
                            only_need_context=False,
                            enable_rerank=False,
                        ),
                    )
                    answer = "" if answer_result is None else str(answer_result)
                except Exception as exc:
                    answer_error = f"{type(exc).__name__}: {exc}"
                    _record_error(report, "query_answer_generation", exc)

            def _is_no_context_response(text: str) -> bool:
                if not text:
                    return True
                t = text.strip().lower()
                if "[no-context]" in t:
                    return True
                if "not able to provide an answer" in t and ("context" in t):
                    return True
                return False

            context_ok = bool(context.strip()) and not _is_no_context_response(context)
            answer_ok = args.no_generate or (bool((answer or "").strip()) and not _is_no_context_response(answer or ""))
            query_ok = bool(context_ok and answer_ok and trace["traced"])
            _set_stage(
                report, "query_execution", query_ok, started,
                question=args.query,
                mode=args.mode,
                top_k=args.top_k,
                generate_answer=not args.no_generate,
                context_chars=len(context),
                context_preview=context[:1000],
                answer=answer,
                answer_error=answer_error,
                context_nonempty=context_ok,
                answer_nonempty=answer_ok,
                trace=trace,
            )
            report["query"] = {
                "question": args.query,
                "mode": args.mode,
                "top_k": args.top_k,
                "context": context,
                "answer": answer,
                "trace": trace,
            }
        except Exception as exc:
            _set_stage(report, "query_execution", False, started)
            _record_error(report, "query_execution", exc)
        # ---- Optional dev-set evaluation (model-backed) ------------------
        if args.dev_set:
            started = time.monotonic()
            try:
                if not indexing_ok:
                    raise RuntimeError(
                        "Dev-set evaluation skipped because indexing validation "
                        "failed."
                    )
                dev_result = await _run_dev_set(args, rag, documents)
                _set_stage(
                    report, "dev_set_evaluation", dev_result["ok"], started,
                    **{k: v for k, v in dev_result.items() if k != "ok"},
                )
                dev_path = run_dir / "dev_set_results.json"
                _write_json(dev_path, dev_result)
                report["paths"]["dev_set_results"] = str(dev_path)
            except Exception as exc:
                _set_stage(report, "dev_set_evaluation", False, started)
                _record_error(report, "dev_set_evaluation", exc)

    finally:
        started = time.monotonic()
        if initialized:
            try:
                if rag is not None and hasattr(rag, "_flush_pipeline_status") and not insertion_ok:
                    try:
                        await rag._flush_pipeline_status()
                    except Exception:
                        pass
                await rag.finalize_storages()
                _set_stage(report, "finalization", True, started)
            except Exception as exc:
                _set_stage(report, "finalization", False, started)
                _record_error(report, "finalization", exc)
        else:
            _set_stage(
                report, "finalization", False, started,
                skipped="storages were never initialized",
            )
        report["paths"]["storage_inventory_after"] = _storage_inventory(workdir)

async def _pipeline_status(rag) -> dict:
    try:
        from lightrag.kg.shared_storage import get_namespace_data
        status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        return {"present": "busy" in status, "keys": sorted(status)[:20]}
    except Exception as exc:
        return {"present": False, "error": f"{type(exc).__name__}: {exc}"}


async def _indexing_evidence(rag) -> dict:
    evidence: dict[str, Any] = {}
    try:
        evidence["full_docs_empty"] = bool(
            await _maybe_await(rag.full_docs.is_empty())
        )
    except Exception as exc:
        evidence["full_docs_empty_error"] = f"{type(exc).__name__}: {exc}"
    try:
        labels = list(
            await _maybe_await(rag.chunk_entity_relation_graph.get_all_labels())
            or []
        )
        placeholder_labels = [label for label in labels if _is_placeholder_label(label)]
        evidence["graph_labels_count"] = len(labels)
        evidence["graph_labels_sample"] = labels[:20]
        evidence["placeholder_graph_labels"] = placeholder_labels[:20]
        evidence["placeholder_graph_labels_count"] = len(placeholder_labels)
        evidence["non_placeholder_graph_labels_count"] = (
            len(labels) - len(placeholder_labels)
        )
    except Exception as exc:
        evidence["graph_labels_error"] = f"{type(exc).__name__}: {exc}"
    try:
        evidence["text_chunks_empty"] = bool(
            await _maybe_await(rag.text_chunks.is_empty())
        )
    except Exception as exc:
        evidence["text_chunks_empty_error"] = f"{type(exc).__name__}: {exc}"
    try:
        edges = await _maybe_await(
            rag.chunk_entity_relation_graph.get_all_edges()
        )
        evidence["graph_edges_count"] = len(edges or [])
    except Exception as exc:
        evidence["graph_edges_error"] = f"{type(exc).__name__}: {exc}"
    try:
        counts = await _maybe_await(rag.doc_status.get_all_status_counts())
        evidence["doc_status_counts"] = {
            str(k): int(v) for k, v in (counts or {}).items()
        }
    except Exception as exc:
        evidence["doc_status_error"] = f"{type(exc).__name__}: {exc}"
    return evidence


async def _run_dev_set(args, rag, documents: dict[str, str]) -> dict:
    from lightrag import QueryParam

    data = _read_json(DEV_SET_PATH)
    queries = data.get("queries", [])
    results = []
    errors = []

    for index, item in enumerate(queries, 1):
        qid = item.get("query_id")
        question = item.get("query", "")
        started = time.monotonic()
        entry = {
            "query_id": qid,
            "type": item.get("type"),
            "question": question,
            "source_docs": item.get("source_docs", []),
            "gold_answer": item.get("gold_answer"),
            "elapsed_s": None,
        }
        try:
            print(f"[dev-set {index}/{len(queries)}] {qid}...", flush=True)
            raw_context = await rag.aquery(
                question,
                param=QueryParam(
                    mode=args.mode,
                    top_k=args.top_k,
                    only_need_context=True,
                    enable_rerank=False,
                ),
            )
            context = "" if raw_context is None else str(raw_context)
            trace = _trace_context(context, documents, query=question)
            required_terms = []
            for gold in item.get("gold", []) or []:
                required_terms.extend(gold.get("required_terms", []) or [])
            matched_terms = [
                term for term in required_terms
                if term.lower() in context.lower()
            ]
            entry.update({
                "context_chars": len(context),
                "context_preview": context[:800],
                "context_nonempty": bool(context.strip()),
                "trace": trace,
                "required_terms": required_terms,
                "required_terms_matched": matched_terms,
                "source_docs_traced": [
                    doc for doc in item.get("source_docs", [])
                    if trace["per_document"].get(doc, {}).get("traced")
                ],
            })
            if args.dev_set_generate:
                answer_raw = await rag.aquery(
                    question,
                    param=QueryParam(
                        mode=args.mode,
                        top_k=args.top_k,
                        only_need_context=False,
                        enable_rerank=False,
                    ),
                )
                entry["answer"] = None if answer_raw is None else str(answer_raw)
            entry["ok"] = True
        except Exception as exc:
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
            errors.append({
                "query_id": qid,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"[dev-set] {qid} ERROR: {exc}", file=sys.stderr, flush=True)
        entry["elapsed_s"] = round(time.monotonic() - started, 3)
        results.append(entry)

    ok = bool(results) and all(item.get("ok") for item in results)
    return {
        "ok": ok,
        "context_nonempty_count": sum(
            1 for item in results if item.get("context_nonempty")
        ),
        "dev_set_path": str(DEV_SET_PATH),
        "mode": args.mode,
        "top_k": args.top_k,
        "n_queries": len(queries),
        "metrics_note": (
            "gold_chunk_ids are MindMesh chunk IDs and cannot be compared "
            "directly to LightRAG's internal chunk IDs. These results record "
            "context, traceability, and required-term matches only."
        ),
        "results": results,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Report / CLI
# ---------------------------------------------------------------------------

def _build_report(args, run_dir: Path, prereq: dict) -> dict:
    run_dir = run_dir.resolve()
    workdir = run_dir / "lightrag_storage"
    return {
        "script": "eval/manual_pipeline3_integration.py",
        "timestamp": _now_iso(),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "config": {
            "model": args.model,
            "embed_model": args.embed_model,
            "llm_host": args.llm_host,
            "mode": args.mode,
            "top_k": args.top_k,
            "expected_embedding_dim": args.expected_embedding_dim,
            "workspace": args.workspace,
            "run_dir": str(run_dir),
            "workdir": str(workdir),
            "query": args.query,
            "documents": (
                "ALL data/raw documents"
                if args.all_docs
                else [n.strip() for n in args.docs.split(",") if n.strip()]
            ),
            "all_docs": args.all_docs,
            "dev_set": args.dev_set,
            "dev_set_generate": args.dev_set_generate,
            "generate_answer": not args.no_generate,
        },
        "prerequisites": prereq,
        "paths": {
            "run_dir": str(run_dir),
            "workdir": str(workdir),
            "storage_inventory_before": _storage_inventory(workdir),
        },
        "stages": {},
        "errors": [],
        "warnings": [],
        "success_criteria": {
            "requires_initialization": True,
            "requires_embedding_dimension": True,
            "requires_document_insertion": True,
            "requires_indexing_validation": True,
            "requires_query_execution": True,
            "requires_traceable_retrieval": True,
            "requires_isolated_storage": True,
            "requires_clean_finalization": True,
            "initialization_alone_is_not_success": True,
        },
        "overall_success": False,
    }


def _summarize(report: dict) -> None:
    print("\n" + "=" * 72)
    print("  PIPELINE 3 MANUAL INTEGRATION")
    print("=" * 72)
    print(f"  Model:        {report['config']['model']}")
    print(f"  Embedding:    {report['config']['embed_model']}")
    print(f"  Mode:         {report['config']['mode']}  (top_k={report['config']['top_k']})")
    print(f"  Run dir:      {report['paths']['run_dir']}")
    print(f"  Storage:      {report['paths']['workdir']}")
    print("-" * 72)
    for name, stage in report.get("stages", {}).items():
        state = "OK" if stage.get("ok") else "FAIL"
        print(f"  {name:<26} {state}  ({stage.get('elapsed_s', '?')}s)")
    print("-" * 72)
    print(f"  Overall success: {report.get('overall_success')}")
    if report.get("errors"):
        print(f"  Errors: {len(report['errors'])} (see JSON report)")
    print("=" * 72)


def _maybe_cleanup(args, run_dir: Path, report: dict, report_path: Path) -> None:
    if not args.cleanup_on_failure or report.get("overall_success"):
        return
    resolved = run_dir.resolve()
    parent = DEDICATED_PARENT.resolve()
    try:
        resolved.relative_to(parent)
    except ValueError:
        print(
            f"Refusing cleanup: run dir is outside {parent}",
            file=sys.stderr,
        )
        return
    if resolved == parent or resolved.is_symlink():
        print("Refusing cleanup: unsafe run dir target.", file=sys.stderr)
        return
    parent.mkdir(parents=True, exist_ok=True)
    preserved = parent / f"failed_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        shutil.copy2(report_path, preserved)
    except OSError as exc:
        print(f"Could not preserve failed report: {exc}", file=sys.stderr)
    try:
        shutil.rmtree(resolved)
        print(f"Removed failed run dir: {resolved}")
        print(f"Failed report preserved at: {preserved}")
    except OSError as exc:
        print(f"Cleanup failed for {resolved}: {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manual, model-backed validation of genuine upstream LightRAG "
            "through the MindMesh Pipeline 3 adapter."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"LLM model (only {DEFAULT_MODEL} permitted).")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                        help="Ollama embedding model (not the LLM).")
    parser.add_argument("--llm-host", default=DEFAULT_LLM_HOST,
                        help="Ollama host URL.")
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=[
        "local", "global", "hybrid", "naive", "mix", "bypass"
    ], help="Upstream LightRAG query mode.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="Retrieval top_k.")
    parser.add_argument("--query", default=DEFAULT_QUERY,
                        help="Single validation query.")
    parser.add_argument("--docs", default=",".join(DEFAULT_DOCS),
                        help="Comma-separated data/raw document names.")
    parser.add_argument("--all-docs", action="store_true",
                        help="Index every .md/.txt file in data/raw.")
    parser.add_argument("--run-dir", default=None,
                        help="Dedicated run directory (must be new/empty).")
    parser.add_argument("--workspace", default="pipeline3_manual_integration",
                        help="LightRAG workspace name.")
    parser.add_argument("--expected-embedding-dim", type=int,
                        default=EXPECTED_EMBED_DIM,
                        help="Expected embedding dimension (nomic-embed-text=768).")
    parser.add_argument("--dev-set", action="store_true",
                        help="After indexing, run eval/gold_dev_set.json "
                             "(requires --all-docs; model-backed).")
    parser.add_argument("--dev-set-generate", action="store_true",
                        help="With --dev-set, also generate answers (extra LLM calls).")
    parser.add_argument("--no-generate", action="store_true",
                        help="Retrieve context only; skip answer generation.")
    parser.add_argument("--check-only", action="store_true",
                        help="Run CPU-only prerequisite checks and exit.")
    parser.add_argument("--cleanup-on-failure", action="store_true",
                        help="Remove only the dedicated run dir if the experiment "
                             "fails; preserve the report in the parent directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.check_only:
        prereq = _run_prerequisites(args)
        print(json.dumps(_jsonable(prereq), indent=2, ensure_ascii=False))
        return 0 if prereq["ok"] else 1

    run_dir = _create_run_dir(args)
    workdir = run_dir / "lightrag_storage"

    isolation = _validate_run_paths(run_dir, workdir)
    report = _build_report(args, run_dir, prereq={})
    report["stages"]["storage_isolation"] = {
        "ok": isolation["ok"],
        "elapsed_s": 0.0,
        **isolation,
    }

    old_workspace = os.environ.get("WORKSPACE")
    os.environ["WORKSPACE"] = args.workspace
    try:
        prereq = _run_prerequisites(args)
        report["prerequisites"] = prereq
        lightrag_check = prereq.get("checks", {}).get("lightrag_import", {})
        report["lightrag_version"] = lightrag_check.get("detail", {}).get("version")

        if not prereq["ok"]:
            report["warnings"].append(
                "Prerequisite checks failed; no model-backed work was attempted."
            )
        elif not isolation["ok"]:
            report["warnings"].append(
                "Storage isolation check failed; no model-backed work was attempted."
            )
        else:
            print(f"[run] Dedicated experiment dir: {run_dir}", flush=True)
            print(f"[run] Pipeline 3 storage:      {workdir}", flush=True)
            print(
                "[run] WARNING: this performs real Ollama embedding and LLM calls.",
                flush=True,
            )
            try:
                asyncio.run(_run_experiment(args, run_dir, report))
            except Exception as exc:
                report["errors"].append({
                    "stage": "experiment_top_level",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                })
                report["warnings"].append(
                    "An unhandled exception escaped the experiment runner; "
                    "see errors."
                )
                print(f"[run] FATAL: {exc}", file=sys.stderr, flush=True)
    finally:
        if old_workspace is None:
            os.environ.pop("WORKSPACE", None)
        else:
            os.environ["WORKSPACE"] = old_workspace

    required_stages = [
        "storage_isolation",
        "initialization",
        "embedding_dimension",
        "document_insertion",
        "indexing_validation",
        "query_execution",
        "finalization",
    ]
    if args.dev_set:
        required_stages.append("dev_set_evaluation")

    stages_ok = all(
        report.get("stages", {}).get(name, {}).get("ok") is True
        for name in required_stages
    )
    prereq_ok = bool(report.get("prerequisites", {}).get("ok"))
    report["overall_success"] = bool(stages_ok and prereq_ok and not report["errors"])
    if report["overall_success"]:
        report["conclusion"] = (
            "Genuine upstream LightRAG indexing and retrieval succeeded on the "
            "selected documents: insertion completed, a real query returned "
            "non-empty context and answer, and the context was traceable to "
            "the indexed source documents."
        )
    else:
        report["conclusion"] = (
            "Integration is NOT proven. Initialization or import success alone "
            "does not count; see failed stages and errors."
        )

    report_path = run_dir / "report.json"
    _write_json(report_path, report)
    _summarize(report)
    print(f"Report: {report_path}")
    print(f"Conclusion: {report['conclusion']}")
    _maybe_cleanup(args, run_dir, report, report_path)
    return 0 if report["overall_success"] else 1


if __name__ == "__main__":
    sys.exit(main())
