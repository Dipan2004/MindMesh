"""
embedder.py — Batch embedding via the configured embedding model.

Supports three models per PRD §5:
  - nomic-embed-text  (primary, Ollama-native)
  - Qwen3-Embedding-0.6B  (ablation, Ollama-native)
  - all-MiniLM-L6-v2  (ablation, sentence-transformers)

CRITICAL DESIGN CONSTRAINT
---------------------------
EMBEDDING_DIM in Config is NEVER the authoritative source for vector length.
The actual dimension is measured from the model's real runtime output on the
first embed call. Config stores what was measured; this module populates it.

A table of EXPECTED dimensions per model exists here ONLY as a sanity-check
guard: after measuring the real dimension we assert it matches the expected
value for the configured model. If they don't match (wrong model pulled,
API change, quantisation artifact) the function raises loudly rather than
silently producing wrong-length vectors that corrupt downstream Zvec collections.

Usage:
    from graphrag_lite.embeddings.embedder import embed_batch
    vectors = embed_batch(["text1", "text2"], model="nomic-embed-text")
"""

from __future__ import annotations

import os
from typing import Any

import ollama as _ollama

from graphrag_lite.config import get_config


# ── Expected-dimension guard table ─────────────────────────────────────────────
# Purpose: validation ONLY. If the measured runtime dimension doesn't match the
# model's known dimension, raise immediately rather than producing bad vectors.
_EXPECTED_DIM: dict[str, int] = {
    "nomic-embed-text":      768,
    "Qwen3-Embedding-0.6B":  1024,   # verify when Ollama tag is finalised
    "all-MiniLM-L6-v2":     384,
}


def _embed_ollama(texts: list[str], model: str) -> list[list[float]]:
    """Call ollama.embed for each text and return vectors."""
    client = _ollama.Client()
    vectors = []
    for text in texts:
        resp = client.embed(model=model, input=text)
        # ollama SDK ≥0.4: response is an EmbedResponse with .embeddings list
        emb = resp.embeddings[0] if hasattr(resp, "embeddings") else resp["embeddings"][0]
        vectors.append(emb)
    return vectors


def _embed_sentence_transformers(texts: list[str], model: str) -> list[list[float]]:
    """Use sentence-transformers for all-MiniLM-L6-v2 (non-Ollama path)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for all-MiniLM-L6-v2. "
            "Install it with: pip install sentence-transformers"
        ) from exc
    st_model = SentenceTransformer(model)
    embeddings = st_model.encode(texts, convert_to_numpy=True)
    return [list(map(float, vec)) for vec in embeddings]


def embed_batch(
    texts: list[str],
    model: str | None = None,
) -> list[list[float]]:
    """
    Embed a batch of texts and return a list of float vectors.

    The embedding dimension is measured from the ACTUAL model output on
    the first call. If the dimension doesn't match the expected value for
    the configured model, a ValueError is raised immediately.

    After the first successful call, ``get_config().EMBEDDING_DIM`` is set
    to the measured dimension so downstream components (Zvec collections)
    can read it without re-embedding.

    Parameters
    ----------
    texts:
        Non-empty list of strings to embed.
    model:
        Model name. Defaults to ``get_config().EMBEDDING_MODEL`` if None.

    Returns
    -------
    list[list[float]]
        One vector per input text. All vectors have the same length.

    Raises
    ------
    ValueError
        If ``texts`` is empty, or if the measured dimension doesn't match
        the expected dimension for the configured model.
    RuntimeError
        If the Ollama server is unreachable or the model is not pulled.
    """
    if not texts:
        raise ValueError("embed_batch: texts must not be empty")

    cfg = get_config()
    if model is None:
        model = cfg.EMBEDDING_MODEL

    # ── Route to appropriate backend ──────────────────────────────────────────
    ollama_models = {"nomic-embed-text", "Qwen3-Embedding-0.6B"}
    st_models     = {"all-MiniLM-L6-v2"}

    if model in ollama_models:
        vectors = _embed_ollama(texts, model)
    elif model in st_models:
        vectors = _embed_sentence_transformers(texts, model)
    else:
        # Unknown model — try Ollama first, fall back to sentence-transformers
        try:
            vectors = _embed_ollama(texts, model)
        except Exception:
            vectors = _embed_sentence_transformers(texts, model)

    # ── Measure real dimension from first vector ───────────────────────────────
    measured_dim = len(vectors[0])

    # ── Validate against expected dimension ───────────────────────────────────
    if model in _EXPECTED_DIM:
        expected = _EXPECTED_DIM[model]
        if measured_dim != expected:
            raise ValueError(
                f"embed_batch: model={model!r} returned dimension {measured_dim} "
                f"but expected {expected}. "
                f"Check the Ollama pull tag or the sentence-transformers version."
            )

    # ── Assert all vectors have the same length ────────────────────────────────
    for i, vec in enumerate(vectors):
        if len(vec) != measured_dim:
            raise ValueError(
                f"embed_batch: vector[{i}] has length {len(vec)}, "
                f"expected {measured_dim}. Model returned inconsistent dimensions."
            )

    # ── Persist measured dimension into config (runtime population) ───────────
    # Uses object.__setattr__ to bypass pydantic's immutability on the
    # cached singleton — this is the ONE place we write EMBEDDING_DIM.
    try:
        object.__setattr__(cfg, "EMBEDDING_DIM", measured_dim)
    except Exception:
        pass  # Config frozen — dimension already set from a prior call

    return vectors
