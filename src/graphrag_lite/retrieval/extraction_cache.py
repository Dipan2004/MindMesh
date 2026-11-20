"""
extraction_cache.py -- persistent, versioned per-paragraph extraction cache with on-demand extraction.

Cache version = sha256(model tag, model digest, extraction-prompt hash, extraction config)[:12].
Each version lives in its own folder:  <root>/<version>/cache.jsonl  (+ cache_meta.json).
Entry key     = sha256(paragraph text)  (content hash), within that version.

Every entry records provenance (chunk_id, doc_id, title, content hash, question that triggered it,
model tag + digest, prompt hash, config hash, timestamp) and a status:
  ok              extraction parsed -> usable for graphs
  failed_model    the production retry ladder (frequency-penalty escalation) was exhausted;
                  final for this cache version (decoding is deterministic, so no retry)
  failed_infra    the model call itself errored (connection / CUDA / server); retried up to
                  INFRA_RETRIES times within the call, then recorded and excluded
Failed entries are never returned as records. Entries are appended, flushed and fsynced one
at a time, so each paragraph is extracted at most once per cache version.

Extraction mirrors the production extractor exactly (same prompt template, same OllamaClient
generation options, same retry ladder, same _extract_json / _parse_extraction post-processing)
but calls Ollama directly, without the shared LLM response cache, so every miss is real,
measured work (wall time + Ollama eval tokens / duration).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from pathlib import Path

INFRA_RETRIES = 2


def _sha(s: str | bytes) -> str:
    return hashlib.sha256(s.encode("utf-8") if isinstance(s, str) else s).hexdigest()


class ExtractionCache:
    def __init__(self, root: Path, model: str, digest: str):
        from graphrag_lite.config import get_config
        from graphrag_lite.ingestion import extractor as X
        from graphrag_lite.llm.ollama_client import OllamaClient
        from mindmesh.model_config import think_for
        cfg = get_config()
        self.X = X
        self.cfg = cfg
        self.client = OllamaClient(model=model, temperature=cfg.EXTRACTION_TEMPERATURE, seed=cfg.EXTRACTION_SEED,
                                   timeout=cfg.LLM_TIMEOUT_S, num_predict=cfg.LLM_NUM_PREDICT, num_ctx=cfg.LLM_NUM_CTX,
                                   think=think_for(model), frequency_penalty=cfg.LLM_FREQUENCY_PENALTY)
        self.ladder = [None] + [float(p) for p in cfg.LLM_RETRY_FREQUENCY_PENALTIES
                                if float(p) != float(cfg.LLM_FREQUENCY_PENALTY)]
        self.model, self.digest = model, digest
        self.prompt_hash = _sha(X._EXTRACTION_PROMPT_TEMPLATE)
        self.config = {"temperature": cfg.EXTRACTION_TEMPERATURE, "seed": cfg.EXTRACTION_SEED,
                       "num_predict": cfg.LLM_NUM_PREDICT, "num_ctx": cfg.LLM_NUM_CTX,
                       "frequency_penalty": cfg.LLM_FREQUENCY_PENALTY, "retry_penalties": self.ladder[1:],
                       "think": self.client.think, "gleanings": 0,
                       "stub_endpoints": cfg.EXTRACTION_STUB_ENDPOINTS}
        self.config_hash = _sha(json.dumps(self.config, sort_keys=True))
        self.version = _sha(json.dumps({"model": model, "digest": digest, "prompt": self.prompt_hash,
                                        "config": self.config_hash}, sort_keys=True))[:12]
        self.dir = Path(root) / self.version
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "cache.jsonl"
        meta = {"version": self.version, "model": model, "digest": digest, "prompt_template_sha256": self.prompt_hash,
                "prompt_version": "mindmesh-extraction-original", "config": self.config, "config_sha256": self.config_hash,
                "key": "sha256(paragraph text) within this version"}
        (self.dir / "cache_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    self.entries[e["content_sha256"]] = e

    # ── lookup / store ──────────────────────────────────────────────────────
    def get(self, text: str) -> dict | None:
        return self.entries.get(_sha(text))

    def _put(self, entry: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.entries[entry["content_sha256"]] = entry

    # ── on-demand extraction ────────────────────────────────────────────────
    def extract(self, para: dict, question_id: str) -> dict:
        import ollama
        X = self.X
        prompt = X._EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=para["text"])
        t0 = time.monotonic()
        tok = {"eval_count": 0, "eval_s": 0.0, "prompt_count": 0}
        attempts = 0
        status, err, ents, rels = "failed_model", "", [], []
        infra_failures = 0
        for pen in self.ladder:
            client = self.client if pen is None else self.client.with_overrides(frequency_penalty=pen)
            raw = None
            for _ in range(INFRA_RETRIES + 1):
                attempts += 1
                try:
                    r = ollama.Client(timeout=self.cfg.LLM_TIMEOUT_S).generate(**client._kwargs(prompt))
                    raw = r.response or ""
                    tok["eval_count"] += r.eval_count or 0
                    tok["eval_s"] += (r.eval_duration or 0) / 1e9
                    tok["prompt_count"] += r.prompt_eval_count or 0
                    break
                except Exception as exc:   # model call failed: infrastructure, not model output
                    infra_failures += 1
                    err = f"{type(exc).__name__}: {str(exc)[:300]}"
                    time.sleep(5)
            if raw is None:
                status = "failed_infra"
                break
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = X._parse_extraction(X._extract_json(raw), para["chunk_id"], para["doc_id"],
                                              chunk_text=para["text"],
                                              allow_stub_endpoints=self.cfg.EXTRACTION_STUB_ENDPOINTS)
                status, err, ents, rels = "ok", "", res["entities"], res["relationships"]
                break
            except Exception as exc:
                status, err = "failed_model", f"{type(exc).__name__}: {str(exc)[:300]}"
        entry = {
            "content_sha256": _sha(para["text"]), "cache_version": self.version, "status": status,
            "chunk_id": para["chunk_id"], "doc_id": para["doc_id"], "chunk_index": 0, "title": para["title"],
            "text": para["text"], "entities": ents, "relationships": rels, "error": err,
            "model": self.model, "model_digest": self.digest, "prompt_template_sha256": self.prompt_hash,
            "config_sha256": self.config_hash, "triggered_by_question": question_id,
            "attempts": attempts, "infra_failures": infra_failures,
            "extraction_wall_s": round(time.monotonic() - t0, 3),
            "eval_tokens": tok["eval_count"], "eval_s": round(tok["eval_s"], 3), "prompt_tokens": tok["prompt_count"],
            "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._put(entry)
        return entry
