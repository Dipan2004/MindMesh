"""
chunker.py — Interchangeable chunking strategies.

All four strategies share the same ``chunk_document(doc, cfg)`` entry
point so callers don't need to know which strategy is active.

Strategies (selected by cfg.CHUNKING_STRATEGY):
  "sliding_window"   — token-based sliding window (original baseline)
  "recursive_char"   — recursive character splitting (LangChain default order)
  "markdown_aware"   — split at ## / ### headers, then sliding window within
  "small_window"     — same as sliding_window but CHUNK_SIZE_EXTRACTION=250

Usage:
    from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
    from graphrag_lite.config import get_config

    cfg = get_config()
    tokenizer = get_tokenizer(cfg.TOKENIZER)
    chunks = chunk_document(doc, cfg, tokenizer)
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, TypedDict

import tiktoken

from graphrag_lite.ingestion.loader import Document


class Chunk(TypedDict):
    """One chunk derived from a source Document."""
    chunk_id: str        # sha256(doc_id::chunk_index)[:16] — stable
    doc_id: str
    text: str
    token_count: int     # real token count from tokenizer
    start_offset: int    # token index (inclusive) in full doc token list
    end_offset: int      # token index (exclusive)
    chunk_index: int     # 0-based within document
    strategy: str        # which chunking strategy produced this chunk


def get_tokenizer(tokenizer_name: str = "qwen") -> Any:
    """Return tiktoken cl100k_base for 'qwen' (closest public approximation)."""
    supported = {"qwen"}
    if tokenizer_name.lower() not in supported:
        raise ValueError(
            f"Unsupported tokenizer: {tokenizer_name!r}. Supported: {supported}"
        )
    return tiktoken.get_encoding("cl100k_base")


def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    raw = f"{doc_id}::{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _make_chunks_from_texts(
    texts: list[str],
    doc_id: str,
    tokenizer: Any,
    strategy: str,
    start_index: int = 0,
) -> list[Chunk]:
    """
    Convert a list of text segments into Chunk objects, computing real token
    counts. start_offset/end_offset are approximated as cumulative token
    positions (exact for sliding_window, approximate for text-split strategies
    since they don't operate in token space).
    """
    chunks: list[Chunk] = []
    cursor = 0
    for i, text in enumerate(texts):
        token_ids = tokenizer.encode(text)
        tc = len(token_ids)
        chunk: Chunk = {
            "chunk_id": _make_chunk_id(doc_id, start_index + i),
            "doc_id": doc_id,
            "text": text,
            "token_count": tc,
            "start_offset": cursor,
            "end_offset": cursor + tc,
            "chunk_index": start_index + i,
            "strategy": strategy,
        }
        chunks.append(chunk)
        cursor += tc
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Sliding window (original baseline)
# ─────────────────────────────────────────────────────────────────────────────

def _sliding_window(
    doc: Document,
    size: int,
    overlap_pct: float,
    tokenizer: Any,
    strategy: str = "sliding_window",
) -> list[Chunk]:
    token_ids: list[int] = tokenizer.encode(doc["text"])
    total_tokens = len(token_ids)
    overlap_tokens = max(0, int(size * overlap_pct))
    stride = max(1, size - overlap_tokens)

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < total_tokens:
        end = min(start + size, total_tokens)
        window_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(window_ids)
        chunks.append(Chunk(
            chunk_id=_make_chunk_id(doc["doc_id"], chunk_index),
            doc_id=doc["doc_id"],
            text=chunk_text,
            token_count=len(window_ids),
            start_offset=start,
            end_offset=end,
            chunk_index=chunk_index,
            strategy=strategy,
        ))
        chunk_index += 1
        start += stride
        if start >= total_tokens:
            break

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Recursive character splitting
# ─────────────────────────────────────────────────────────────────────────────

def _recursive_char_split(
    text: str,
    target_tokens: int,
    overlap_pct: float,
    tokenizer: Any,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Recursively split ``text`` on a priority list of separators until all
    pieces are <= target_tokens. Overlap is applied as a character-level
    suffix/prefix of roughly overlap_pct * target_tokens tokens.

    This matches LangChain's RecursiveCharacterTextSplitter default separator
    order: ["\n\n", "\n", ". ", " ", ""].
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    def _token_len(t: str) -> int:
        return len(tokenizer.encode(t))

    def _split_on(t: str, sep: str) -> list[str]:
        if sep == "":
            # character-level fallback — split into individual chars,
            # then re-merge up to target size (handled below)
            return list(t)
        parts = t.split(sep)
        # Re-attach separator to all but last piece to preserve punctuation
        return [p + sep for p in parts[:-1]] + [parts[-1]] if parts else []

    def _merge(pieces: list[str], tgt: int, ovl_pct: float) -> list[str]:
        """Greedily merge short pieces into target-size chunks with overlap."""
        overlap_chars = int(ovl_pct * tgt * 4)  # rough char budget for overlap
        chunks_out: list[str] = []
        current = ""
        tail = ""  # overlap tail from previous chunk

        for piece in pieces:
            candidate = current + piece
            if _token_len(candidate) <= tgt:
                current = candidate
            else:
                if current:
                    chunks_out.append(current)
                    # carry overlap tail into next chunk
                    tail = current[-overlap_chars:] if overlap_chars > 0 else ""
                current = tail + piece
                tail = ""

        if current:
            chunks_out.append(current)
        return [c for c in chunks_out if c.strip()]

    def _recurse(t: str, seps: list[str]) -> list[str]:
        if _token_len(t) <= target_tokens:
            return [t] if t.strip() else []
        if not seps:
            # Last resort: hard-split by characters into target-size token windows
            token_ids = tokenizer.encode(t)
            stride = max(1, target_tokens - int(target_tokens * overlap_pct))
            results = []
            s = 0
            while s < len(token_ids):
                results.append(tokenizer.decode(token_ids[s: s + target_tokens]))
                s += stride
                if s >= len(token_ids):
                    break
            return results

        sep = seps[0]
        pieces = _split_on(t, sep)
        merged = _merge(pieces, target_tokens, overlap_pct)
        # Any piece still too large gets recursed with remaining separators
        result = []
        for piece in merged:
            if _token_len(piece) > target_tokens:
                result.extend(_recurse(piece, seps[1:]))
            else:
                if piece.strip():
                    result.append(piece)
        return result

    return _recurse(text, separators)


def _chunk_recursive_char(
    doc: Document,
    size: int,
    overlap_pct: float,
    tokenizer: Any,
) -> list[Chunk]:
    texts = _recursive_char_split(doc["text"], size, overlap_pct, tokenizer)
    return _make_chunks_from_texts(texts, doc["doc_id"], tokenizer, "recursive_char")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Markdown-structure-aware
# ─────────────────────────────────────────────────────────────────────────────

_MARKDOWN_MIN_TOKENS: int = 100  # sections below this are merged into the next one


def _split_markdown_sections(text: str) -> list[str]:
    """
    Split on ## and ### headers (ATX style). Each section includes its header
    line. Returns at least one section (the whole text if no headers found).

    Any section whose token count falls below _MARKDOWN_MIN_TOKENS is merged
    forward into the next section. This prevents bare header lines and very
    short subsections from being sent to the LLM as standalone chunks.
    """
    # Match lines starting with ## or ### (but not ####+ to avoid over-splitting)
    pattern = re.compile(r"^(#{2,3} .+)$", re.MULTILINE)
    positions = [m.start() for m in pattern.finditer(text)]

    if not positions:
        return [text]

    raw_sections: list[str] = []
    if positions[0] > 0:
        preamble = text[:positions[0]].strip()
        if preamble:
            raw_sections.append(preamble)

    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        section = text[pos:end].strip()
        if section:
            raw_sections.append(section)

    # Merge-forward pass: any section under _MARKDOWN_MIN_TOKENS joins the next
    _tok = tiktoken.get_encoding("cl100k_base")  # fast, cached by tiktoken
    merged: list[str] = []
    carry = ""
    for section in raw_sections:
        combined = (carry + "\n\n" + section).strip() if carry else section
        if _tok.encode(combined).__len__() < _MARKDOWN_MIN_TOKENS:
            carry = combined  # still too small — carry forward
        else:
            merged.append(combined)
            carry = ""
    if carry:
        # Remaining carry — either append to last or keep as-is
        if merged:
            merged[-1] = merged[-1] + "\n\n" + carry
        else:
            merged.append(carry)

    return [s for s in merged if s.strip()]


def _chunk_markdown_aware(
    doc: Document,
    size: int,
    overlap_pct: float,
    tokenizer: Any,
) -> list[Chunk]:
    sections = _split_markdown_sections(doc["text"])
    all_chunks: list[Chunk] = []
    running_index = 0

    for section in sections:
        token_ids = tokenizer.encode(section)
        if len(token_ids) <= size:
            # Section fits in one chunk — keep whole
            all_chunks.append(Chunk(
                chunk_id=_make_chunk_id(doc["doc_id"], running_index),
                doc_id=doc["doc_id"],
                text=section,
                token_count=len(token_ids),
                start_offset=0,   # within-section offset
                end_offset=len(token_ids),
                chunk_index=running_index,
                strategy="markdown_aware",
            ))
            running_index += 1
        else:
            # Section too large — apply sliding window within it
            section_doc: Document = {
                "doc_id": doc["doc_id"],
                "text": section,
                "source_path": doc["source_path"],
            }
            sub_chunks = _sliding_window(
                section_doc, size, overlap_pct, tokenizer, strategy="markdown_aware"
            )
            for sc in sub_chunks:
                sc["chunk_id"] = _make_chunk_id(doc["doc_id"], running_index)
                sc["chunk_index"] = running_index
                all_chunks.append(sc)
                running_index += 1

    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4: Small window (250 tokens, same overlap)
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_small_window(
    doc: Document,
    overlap_pct: float,
    tokenizer: Any,
) -> list[Chunk]:
    return _sliding_window(doc, size=250, overlap_pct=overlap_pct,
                           tokenizer=tokenizer, strategy="small_window")


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def chunk_document(doc: Document, cfg: Any, tokenizer: Any) -> list[Chunk]:
    """
    Dispatch to the configured chunking strategy.

    Parameters
    ----------
    doc:   A Document from load_documents().
    cfg:   The Config singleton (get_config()).  Reads:
           - cfg.CHUNKING_STRATEGY  ("sliding_window" | "recursive_char" |
                                     "markdown_aware" | "small_window")
           - cfg.CHUNK_SIZE_EXTRACTION  (target token budget)
           - cfg.CHUNK_OVERLAP_PCT      (overlap fraction 0–1)
    tokenizer: tiktoken Encoding from get_tokenizer().

    Raises
    ------
    ValueError  on unrecognised strategy or empty document.
    """
    if not doc.get("text"):
        raise ValueError(f"Document '{doc.get('doc_id')}' has empty text")

    strategy = getattr(cfg, "CHUNKING_STRATEGY", "sliding_window")
    size = cfg.CHUNK_SIZE_EXTRACTION
    overlap = cfg.CHUNK_OVERLAP_PCT

    if strategy == "sliding_window":
        return _sliding_window(doc, size, overlap, tokenizer)
    elif strategy == "recursive_char":
        return _chunk_recursive_char(doc, size, overlap, tokenizer)
    elif strategy == "markdown_aware":
        return _chunk_markdown_aware(doc, size, overlap, tokenizer)
    elif strategy == "small_window":
        return _chunk_small_window(doc, overlap, tokenizer)
    else:
        raise ValueError(
            f"Unknown CHUNKING_STRATEGY: {strategy!r}. "
            "Choose from: sliding_window, recursive_char, markdown_aware, small_window"
        )
