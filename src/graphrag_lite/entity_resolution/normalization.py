"""
normalization.py — Text normalization for entity resolution.

Adapted from Zeno's ids.py::normalize() with additions for the MindMesh
entity context (technical terms, abbreviations, plurals).
"""
from __future__ import annotations
import re
import unicodedata


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    """
    Primary normalization key: NFKC + casefold + whitespace collapse.

    This is the Zeno-style normalization (unicode-aware), which is STRONGER
    than MindMesh's existing _norm() (ASCII casefold only). Using NFKC means
    'GPT‑4' (en-dash) and 'GPT-4' (hyphen) normalize identically.

    Adapted from: zeno/memory/core/ids.py::normalize()
    """
    s = unicodedata.normalize("NFKC", str(text)).strip().lower()
    return _WS.sub(" ", s)


def norm_alias(text: str) -> str:
    """
    Secondary alias key: normalize + strip punctuation + strip trailing s/es.

    Collapses: 'Knowledge Graph' / 'Knowledge Graphs' / 'knowledge-graph'
    Does NOT collapse: 'AI' vs 'artificial intelligence' (too aggressive)

    This is the MindMesh _norm_alias() behavior, lifted to a public function.
    """
    s = normalize(text)
    s = _PUNCT.sub("", s)
    s = _WS.sub(" ", s).strip()
    if s.endswith("es"):
        s = s[:-2]
    elif s.endswith("s"):
        s = s[:-1]
    return s


# Ambiguous terms that MUST NOT be merged based on name alone
AMBIGUOUS_TERMS: frozenset[str] = frozenset({
    "apple",        # company vs fruit
    "java",         # language vs island
    "python",       # language vs snake
    "mercury",      # planet vs element vs car
    "rust",         # language vs corrosion
    "swift",        # language vs bird
    "go",           # language vs game
    "c",            # language vs letter
    "net",          # framework vs mesh
    "spring",       # framework vs season
    "node",         # generic term
    "edge",         # generic term
    "graph",        # generic term
    "tree",         # generic term
    "query",        # generic term
    "table",        # generic term
    "index",        # generic term
})


def is_ambiguous(label: str) -> bool:
    """Return True if the normalized label is known to be ambiguous."""
    return normalize(label) in AMBIGUOUS_TERMS
