"""
keywords.py -- low/high-level keyword extraction for LightRAG-style retrieval.

LightRAG proper asks the LLM for low-level (entity) and high-level (theme)
keywords.  MindMesh defers that (GPU_DEFERRED_TASKS.md, G6): both functions
return the query unchanged, so retrieval needs no LLM call and the
dense+BM25 hybrid search handles matching.
"""

from __future__ import annotations


def extract_low_level_keywords(query: str) -> str:
    return query


def extract_high_level_keywords(query: str) -> str:
    return query
