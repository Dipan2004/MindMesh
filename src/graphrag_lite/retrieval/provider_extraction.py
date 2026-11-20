"""
provider_extraction.py -- graph-extraction provider abstraction:
OpenRouter (primary, only when explicitly selected) with local Ollama as fallback.

    from graphrag_lite.retrieval.provider_extraction import (
        load_provider_settings, OpenRouterExtractionCache, ProviderExtractor)

Scope: graph extraction only. Answer generation, embeddings and retrieval are not touched.

Methodology is unchanged:
  * same prompt   -- extractor._EXTRACTION_PROMPT_TEMPLATE, sent verbatim as one user message;
  * same schema   -- the MindMesh entity / relationship fields (name, type, description,
                     source_sentence / source, target, type, description, source_sentence),
                     sent as a strict JSON Schema (response_format type=json_schema);
  * same post-processing -- extractor._parse_extraction (normalisation, endpoint checks).
The returned JSON is also validated locally against the same schema; a response with
missing or empty required fields is rejected, never silently accepted.

Differences that cannot be avoided (recorded in every cache's metadata):
  * frequency_penalty is not sent: the OpenRouter endpoints for
    google/gemini-2.5-flash-lite do not list it as supported, and with
    require_parameters=true the request would be refused. The local retry ladder
    (frequency-penalty escalation) is therefore replaced by plain retries with the same
    parameters (max_retries) for transient HTTP errors and invalid / schema-violating JSON.
    A response cut off at max_tokens (finish_reason=length) is NOT retried: the request is
    deterministic, so a retry repeats the same truncated output and is billed again. The
    entry is recorded as failed_model and ProviderExtractor falls back to local Ollama.
  * provider.require_parameters=true makes OpenRouter route only to endpoints that support
    every parameter sent (response_format / structured outputs, seed, temperature, max_tokens),
    so a structured-output requirement is never silently dropped.

Caching (no duplicate extraction on retry or resume):
  * OpenRouter entries: <root>/openrouter/<version>/cache.jsonl, version =
    sha256(provider, exact model id, prompt hash, schema hash, config hash)[:12];
    entry key = sha256(paragraph text).
  * Ollama fallback entries: <root>/ollama/<version>/ via the existing ExtractionCache
    (its version hashes the Ollama model tag + digest). Different directories, so the two
    providers never share a namespace, and old Qwen records can never be read as Gemini ones.
  * failed entries are final for their cache version (as in ExtractionCache).

The API key is read from the environment (or the git-ignored .env file) and is never
logged, printed or written anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
PROVIDER_FILE = ROOT / "config" / "extraction_provider.json"

# Strict JSON Schema for the existing MindMesh extraction fields (all required, no extras).
_ENTITY = {"type": "object", "additionalProperties": False,
           "properties": {"name": {"type": "string"}, "type": {"type": "string"},
                          "description": {"type": "string"}, "source_sentence": {"type": "string"}},
           "required": ["name", "type", "description", "source_sentence"]}
_RELATION = {"type": "object", "additionalProperties": False,
             "properties": {"source": {"type": "string"}, "target": {"type": "string"},
                            "type": {"type": "string"}, "description": {"type": "string"},
                            "source_sentence": {"type": "string"}},
             "required": ["source", "target", "type", "description", "source_sentence"]}
EXTRACTION_SCHEMA = {"type": "object", "additionalProperties": False,
                     "properties": {"entities": {"type": "array", "items": _ENTITY},
                                    "relationships": {"type": "array", "items": _RELATION}},
                     "required": ["entities", "relationships"]}
_NON_EMPTY = {"entities": ("name", "type", "source_sentence"),
              "relationships": ("source", "target", "type", "source_sentence")}

# HTTP statuses worth retrying (transient); anything else 4xx is a hard provider error.
_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


def _sha(s: str | bytes) -> str:
    return hashlib.sha256(s.encode("utf-8") if isinstance(s, str) else s).hexdigest()


SCHEMA_SHA256 = _sha(json.dumps(EXTRACTION_SCHEMA, sort_keys=True))


# ── settings ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class OpenRouterSettings:
    model: str = "google/gemini-2.5-flash-lite"
    base_url: str = "https://openrouter.ai/api/v1"
    endpoint: str = "/chat/completions"
    api_key_env: str = "OPENROUTER_API_KEY"
    require_parameters: bool = True
    max_tokens: int = 4096
    temperature: float = 0.0
    seed: int = 42
    reasoning_enabled: bool = False
    timeout_s: float = 120.0
    max_retries: int = 2


@dataclass(frozen=True)
class ProviderSettings:
    provider: str                     # "openrouter" | "ollama"
    openrouter: OpenRouterSettings
    fallback_enabled: bool
    fallback_model: str               # Ollama tag
    source: str = field(default="")


def load_provider_settings(path: Path | None = None) -> ProviderSettings:
    p = path or PROVIDER_FILE
    d = json.loads(p.read_text(encoding="utf-8"))
    o = {k: v for k, v in d.get("openrouter", {}).items() if not k.startswith("_")}
    if os.environ.get("OPENROUTER_MODEL"):
        o["model"] = os.environ["OPENROUTER_MODEL"]
    if os.environ.get("OPENROUTER_BASE_URL"):
        o["base_url"] = os.environ["OPENROUTER_BASE_URL"]
    fb = d.get("fallback", {})
    provider = (os.environ.get("EXTRACTION_PROVIDER") or d.get("provider", "ollama")).strip().lower()
    if provider not in ("openrouter", "ollama"):
        raise ValueError(f"unknown extraction provider {provider!r} (expected 'openrouter' or 'ollama')")
    return ProviderSettings(provider=provider, openrouter=OpenRouterSettings(**o),
                            fallback_enabled=bool(fb.get("enabled", True)),
                            fallback_model=os.environ.get("EXTRACTION_FALLBACK_MODEL") or fb.get("model", "qwen2.5:3b"),
                            source=str(p))


def _api_key(env_name: str) -> str:
    """Key from the process environment, else from the git-ignored .env. Never logged."""
    key = os.environ.get(env_name, "").strip()
    if not key:
        try:
            from dotenv import dotenv_values
            key = (dotenv_values(ROOT / ".env").get(env_name) or "").strip()
        except Exception:  # noqa: BLE001
            key = ""
    return key


def api_key_available(s: OpenRouterSettings) -> bool:
    return bool(_api_key(s.api_key_env))


def endpoint_support(s: OpenRouterSettings) -> dict:
    """Public model metadata: which upstream endpoints support each parameter we send."""
    r = requests.get(f"{s.base_url}/models/{s.model}/endpoints", timeout=30)
    r.raise_for_status()
    eps = r.json()["data"]["endpoints"]
    need = ["structured_outputs", "response_format", "seed", "temperature", "max_tokens"]
    rows = [{"provider_name": e.get("provider_name"), "tag": e.get("tag"),
             "supported_parameters": e.get("supported_parameters", []),
             "pricing_prompt": e.get("pricing", {}).get("prompt"),
             "pricing_completion": e.get("pricing", {}).get("completion"),
             "missing": [n for n in need if n not in e.get("supported_parameters", [])],
             "frequency_penalty_supported": "frequency_penalty" in e.get("supported_parameters", [])}
            for e in eps]
    return {"model": s.model, "required_parameters": need, "endpoints": rows,
            "any_endpoint_supports_all": any(not r["missing"] for r in rows),
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def validate_payload(obj) -> list[str]:
    """Local validation against EXTRACTION_SCHEMA plus non-empty required strings."""
    import jsonschema
    errs = [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in jsonschema.Draft7Validator(EXTRACTION_SCHEMA).iter_errors(obj)]
    if errs:
        return errs[:20]
    for kind, fields in _NON_EMPTY.items():
        for i, item in enumerate(obj[kind]):
            for f in fields:
                if not item[f].strip():
                    errs.append(f"{kind}/{i}/{f}: empty")
    return errs[:20]


# ── OpenRouter cache ─────────────────────────────────────────────────────────
class OpenRouterExtractionCache:
    PROVIDER = "openrouter"

    def __init__(self, root: Path, settings: OpenRouterSettings):
        from graphrag_lite.config import get_config
        from graphrag_lite.ingestion import extractor as X
        self.X, self.cfg, self.s = X, get_config(), settings
        self.prompt_hash = _sha(X._EXTRACTION_PROMPT_TEMPLATE)
        self.config = {"temperature": settings.temperature, "seed": settings.seed,
                       "max_tokens": settings.max_tokens, "reasoning_enabled": settings.reasoning_enabled,
                       "require_parameters": settings.require_parameters, "max_retries": settings.max_retries,
                       "length_cutoff_policy": "no retry (deterministic request)",
                       "frequency_penalty": "not sent (unsupported by the endpoints)",
                       "response_format": "json_schema strict", "messages": "single user message = prompt",
                       "gleanings": 0, "stub_endpoints": self.cfg.EXTRACTION_STUB_ENDPOINTS}
        self.config_hash = _sha(json.dumps(self.config, sort_keys=True))
        self.version = _sha(json.dumps({"provider": self.PROVIDER, "model": settings.model,
                                        "prompt": self.prompt_hash, "schema": SCHEMA_SHA256,
                                        "config": self.config_hash}, sort_keys=True))[:12]
        self.dir = Path(root) / self.PROVIDER / self.version
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "cache.jsonl"
        meta = {"version": self.version, "provider": self.PROVIDER, "model": settings.model,
                "base_url": settings.base_url, "endpoint": settings.endpoint,
                "prompt_template_sha256": self.prompt_hash, "prompt_version": "mindmesh-extraction-original",
                "schema_sha256": SCHEMA_SHA256, "schema": EXTRACTION_SCHEMA,
                "config": self.config, "config_sha256": self.config_hash,
                "key": "sha256(paragraph text) within this version"}
        (self.dir / "cache_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.entries: dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    self.entries[e["content_sha256"]] = e

    def get(self, text: str) -> dict | None:
        return self.entries.get(_sha(text))

    def _put(self, entry: dict) -> None:
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.entries[entry["content_sha256"]] = entry

    def _body(self, prompt: str) -> dict:
        s = self.s
        body = {"model": s.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": s.temperature, "seed": s.seed, "max_tokens": s.max_tokens,
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "mindmesh_extraction", "strict": True,
                                                    "schema": EXTRACTION_SCHEMA}},
                "provider": {"require_parameters": s.require_parameters},
                "usage": {"include": True}}
        if not s.reasoning_enabled:
            body["reasoning"] = {"enabled": False}
        return body

    def extract(self, para: dict, question_id: str = "") -> dict:
        """Extract one paragraph (or return its cached entry). Never raises for model/provider errors."""
        hit = self.get(para["text"])
        if hit is not None:
            return hit
        key = _api_key(self.s.api_key_env)
        prompt = self.X._EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=para["text"])
        url = self.s.base_url.rstrip("/") + self.s.endpoint
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                   "X-Title": "MindMesh extraction"}
        log, status, err, ents, rels = [], "failed_infra", "", [], []
        json_valid, val_errors = False, []
        t0 = time.monotonic()
        if not key:
            err = f"{self.s.api_key_env} is not set"
        for attempt in range(1, (self.s.max_retries + 1) + 1 if key else 1):
            a = {"attempt": attempt}
            ta = time.monotonic()
            try:
                r = requests.post(url, headers=headers, json=self._body(prompt), timeout=self.s.timeout_s)
                a["wall_s"] = round(time.monotonic() - ta, 3)
                a["http_status"] = r.status_code
                data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            except Exception as exc:  # noqa: BLE001  (network / timeout)
                a["wall_s"] = round(time.monotonic() - ta, 3)
                a["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                log.append(a); status, err = "failed_infra", a["error"]
                time.sleep(min(2 ** attempt, 10)); continue
            u = data.get("usage") or {}
            a.update({"generation_id": data.get("id"), "actual_model": data.get("model"),
                      "upstream_provider": data.get("provider"),
                      "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                      "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                      "cost_usd": u.get("cost")})
            if r.status_code != 200 or "error" in data:
                e = data.get("error") or {}
                a["error"] = f"HTTP {r.status_code}: {str(e.get('message') or r.text)[:300]}"
                log.append(a); status, err = "failed_infra", a["error"]
                if r.status_code in _RETRYABLE or (r.status_code == 200 and "error" in data):
                    time.sleep(min(2 ** attempt, 10)); continue
                break                               # hard provider error (400/401/402/403/404): no retry
            choice = (data.get("choices") or [{}])[0]
            a["finish_reason"] = choice.get("finish_reason")
            content = ((choice.get("message") or {}).get("content")) or ""
            if a["finish_reason"] == "length":
                # Output hit max_tokens. The request is deterministic (temperature 0, seed), so an
                # identical retry reproduces the same truncated output and is billed again: stop here
                # and let the caller fall back.
                a["error"] = f"output truncated at max_tokens={self.s.max_tokens} (finish_reason=length); not retried"
                log.append(a); status, err = "failed_model", a["error"]
                break
            try:
                obj = json.loads(content)
            except Exception as exc:  # noqa: BLE001
                a["error"] = f"invalid JSON: {type(exc).__name__}: {str(exc)[:200]}"
                log.append(a); status, err = "failed_model", a["error"]; continue
            val_errors = validate_payload(obj)
            if val_errors:
                a["error"] = "schema validation failed: " + "; ".join(val_errors[:5])
                log.append(a); status, err = "failed_model", a["error"]; continue
            json_valid = True
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = self.X._parse_extraction(obj, para["chunk_id"], para["doc_id"], chunk_text=para["text"],
                                                   allow_stub_endpoints=self.cfg.EXTRACTION_STUB_ENDPOINTS)
                status, err, ents, rels = "ok", "", res["entities"], res["relationships"]
                log.append(a); break
            except Exception as exc:  # noqa: BLE001
                a["error"] = f"post-processing: {type(exc).__name__}: {str(exc)[:200]}"
                log.append(a); status, err = "failed_model", a["error"]
        last = log[-1] if log else {}
        tot = lambda k: sum((x.get(k) or 0) for x in log)  # noqa: E731
        costs = [x.get("cost_usd") for x in log if x.get("cost_usd") is not None]
        entry = {
            "content_sha256": _sha(para["text"]), "cache_version": self.version, "status": status,
            "provider": self.PROVIDER, "requested_model": self.s.model,
            "actual_model": last.get("actual_model"), "upstream_provider": last.get("upstream_provider"),
            "generation_id": last.get("generation_id"), "fallback": False,
            "chunk_id": para["chunk_id"], "doc_id": para["doc_id"], "chunk_index": 0, "title": para.get("title"),
            "text": para["text"], "entities": ents, "relationships": rels, "error": err,
            "json_valid": json_valid, "validation_errors": val_errors,
            "prompt_template_sha256": self.prompt_hash, "schema_sha256": SCHEMA_SHA256,
            "config_sha256": self.config_hash, "triggered_by_question": question_id,
            "attempts": len(log), "retries": max(0, len(log) - 1), "attempt_log": log,
            "request_wall_s": round(sum(x.get("wall_s", 0) for x in log), 3),
            "extraction_wall_s": round(time.monotonic() - t0, 3),
            "prompt_tokens": tot("prompt_tokens"), "completion_tokens": tot("completion_tokens"),
            "reasoning_tokens": tot("reasoning_tokens"),
            # Ollama-compatible fields used by shared reporting code (eval_s: no decode-only time for an API call)
            "eval_tokens": tot("completion_tokens"), "eval_s": 0.0,
            "cost_usd": round(sum(costs), 8) if costs else None,
            "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._put(entry)
        return entry


# ── primary + fallback ───────────────────────────────────────────────────────
class ProviderExtractor:
    """OpenRouter first; Ollama fallback only when the OpenRouter entry is not ok.

    Results are never mixed: each returned record says which provider produced it, and the
    two providers write to separate cache namespaces. Fallback events are appended to
    <root>/fallback_events.jsonl with the OpenRouter failure that caused them."""

    def __init__(self, root: Path, settings: ProviderSettings | None = None):
        self.settings = settings or load_provider_settings()
        if self.settings.provider != "openrouter":
            raise ValueError("ProviderExtractor is for provider=openrouter; use ExtractionCache for local Ollama")
        self.root = Path(root)
        self.primary = OpenRouterExtractionCache(self.root, self.settings.openrouter)
        self._fallback = None

    def _fallback_cache(self):
        if self._fallback is None:
            from graphrag_lite.retrieval.extraction_cache import ExtractionCache
            from mindmesh.benchmarks.hotpotqa.runtime import resolve_local_model
            tag, info = resolve_local_model(self.settings.fallback_model)   # refuses non-local / missing models
            self._fallback = ExtractionCache(self.root / "ollama", tag, info.get("digest", ""))
        return self._fallback

    def extract(self, para: dict, question_id: str = "") -> dict:
        p = self.primary.extract(para, question_id)
        if p["status"] == "ok" or not self.settings.fallback_enabled:
            return {"used_provider": p["provider"] if p["status"] == "ok" else None, "record": p if p["status"] == "ok" else None,
                    "primary": p, "fallback": None}
        fc = self._fallback_cache()
        cached = fc.get(para["text"])
        f = cached or fc.extract(para, question_id)
        f = dict(f, provider="ollama", fallback=True)
        ev = {"chunk_id": para["chunk_id"], "content_sha256": p["content_sha256"],
              "from_cache": cached is not None,   # True on resume: no new extraction was run
              "openrouter_status": p["status"], "openrouter_error": p["error"],
              "fallback_model": f.get("model"), "fallback_digest": f.get("model_digest"),
              "fallback_cache_version": f.get("cache_version"), "fallback_status": f["status"],
              "fallback_wall_s": f.get("extraction_wall_s"), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(self.root / "fallback_events.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev) + "\n")
        return {"used_provider": "ollama" if f["status"] == "ok" else None,
                "record": f if f["status"] == "ok" else None, "primary": p, "fallback": f}
