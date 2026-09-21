"""
loader.py — Load source documents from a directory into Document dicts.

Supports .txt and .md files. Every document retains its source path and a
stable doc_id derived from the filename so that all downstream components
(chunks, entities, relationships) can trace back to the original file.

Usage:
    from graphrag_lite.ingestion.loader import load_documents
    docs = load_documents("data/raw")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict


class Document(TypedDict):
    """Minimal representation of a loaded source document."""
    doc_id: str       # stable identifier — stem of the filename, e.g. "transformer_architecture"
    text: str         # full raw text content
    source_path: str  # absolute path to the original file


_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md"})


def load_documents(dir_path: str) -> list[Document]:
    """
    Load all .txt and .md files from ``dir_path`` into a list of Documents.

    Parameters
    ----------
    dir_path:
        Path to the directory containing source documents. May be relative
        or absolute.

    Returns
    -------
    list[Document]
        One Document per file, sorted by filename for deterministic ordering.

    Raises
    ------
    FileNotFoundError
        If ``dir_path`` does not exist or is not a directory.
    PermissionError / OSError
        If a file exists but cannot be read — propagated, not silently skipped.
        Every readable file is included; nothing is silently dropped.
    ValueError
        If ``dir_path`` is empty string.
    """
    if not dir_path:
        raise ValueError("dir_path must not be empty")

    directory = Path(dir_path).resolve()

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    # Collect matching files, sorted for deterministic ordering
    candidates = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )

    documents: list[Document] = []
    for file_path in candidates:
        # Raises PermissionError / UnicodeDecodeError / OSError on unreadable file
        # — we let it propagate per TASKS.md rule 4 ("must raise, not silently skip")
        text = file_path.read_text(encoding="utf-8")

        doc: Document = {
            "doc_id": file_path.stem,          # filename without extension
            "text": text,
            "source_path": str(file_path),
        }
        documents.append(doc)

    return documents
