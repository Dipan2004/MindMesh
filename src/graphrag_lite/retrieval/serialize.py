"""
serialize.py -- evidence serialization for the LLM layer (Task 4.4).

Two formats, one data model:

  to_toon(evidence)   TOON (Token-Oriented Object Notation): a tabular
                      header naming the columns once, then one row per
                      record. The repeated keys of JSON are dropped; the
                      values stay.
  to_json(evidence)   ordinary JSON, the baseline the TOON ablation is
                      measured against.

Both are LOSSLESS: from_toon(to_toon(e)) == e and
from_json(to_json(e)) == e, field for field.  The round trip is part of the
test, not an afterthought -- a serializer that silently drops a field is
worse than no serializer at all, because the LLM never sees what was lost.

TOON layout produced here:

    evidence[2]{chunk_id,source_doc,chunk_index,score,text}:
      6d878b7f56a9b0c1,knowledge_graphs,0,0.032789,"# Knowledge Graphs\\n\\n..."
      33d32892fc640b3f,retrieval_augmented_generation,1,0.031800,"..."

A field is quoted only when it has to be (it contains a comma, a quote, a
newline, or leading/trailing whitespace).  A quoted field uses JSON string
escaping, so the parser is json.loads for that field and a plain read for
the rest.

`include_text=False` drops the `text` column.  That variant exists because
the TOON-vs-JSON token comparison is dominated by the evidence text, which
is identical in both formats -- the structural saving can only be seen on
the metadata.  Both numbers are reported; neither is presented alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Iterable

from graphrag_lite.config import get_config
from graphrag_lite.ingestion.chunker import get_tokenizer
from graphrag_lite.retrieval.pipeline_a import Evidence

# Column order of the TOON table.  `text` is last so the fixed-size columns
# stay scannable.
TOON_FIELDS = ("chunk_id", "source_doc", "chunk_index", "score", "source", "text")


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #

def _needs_quoting(value: str) -> bool:
    return (
        value == ""
        or value != value.strip()
        or any(ch in value for ch in (",", '"', "\n", "\r", "\t"))
    )


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _unquote(token: str) -> str:
    token = token.strip()
    if token.startswith('"'):
        return json.loads(token)
    return token


def _split_row(row: str) -> list[str]:
    """Split one TOON row on the commas that are NOT inside a quoted field."""
    fields: list[str] = []
    buf: list[str] = []
    in_quotes = False
    escaped = False
    for ch in row:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_quotes:
            buf.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_quotes = not in_quotes
            buf.append(ch)
            continue
        if ch == "," and not in_quotes:
            fields.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    fields.append("".join(buf))
    return fields


def _evidence_fields(ev: Evidence, fields: list[str]) -> list[str]:
    d = asdict(ev)
    out: list[str] = []
    for name in fields:
        value = d[name]
        if name == "score":
            out.append(repr(float(value)))
        elif name == "chunk_index":
            out.append(str(int(value)))
        else:
            text = "" if value is None else str(value)
            out.append(_quote(text) if _needs_quoting(text) else text)
    return out


# ------------------------------------------------------------------ #
# TOON
# ------------------------------------------------------------------ #

def to_toon(evidence: Iterable[Evidence], include_text: bool = True) -> str:
    """Serialize evidence as TOON. See the module docstring for the layout."""
    items = list(evidence)
    fields = [f for f in TOON_FIELDS if include_text or f != "text"]
    header = f"evidence[{len(items)}]" + "{" + ",".join(fields) + "}:"
    if not items:
        return header + "\n"
    lines = [header]
    for ev in items:
        lines.append("  " + ",".join(_evidence_fields(ev, fields)))
    return "\n".join(lines) + "\n"


def from_toon(text: str) -> list[Evidence]:
    """Parse TOON back into Evidence. Inverse of to_toon."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].strip()
    if not header.startswith("evidence[") or not header.endswith("}:"):
        raise ValueError(f"from_toon: bad header {header!r}")
    inner = header[len("evidence["):-2]
    count_s, _, cols_s = inner.partition("]{")
    count = int(count_s)
    cols = cols_s.split(",")
    out: list[Evidence] = []
    for row in lines[1:]:
        parts = _split_row(row.strip())
        if len(parts) != len(cols):
            raise ValueError(
                f"from_toon: row has {len(parts)} fields, header has {len(cols)}")
        d: dict[str, Any] = {}
        for name, raw in zip(cols, parts):
            if name == "score":
                d[name] = float(raw)
            elif name == "chunk_index":
                d[name] = int(raw)
            else:
                d[name] = _unquote(raw)
        d.setdefault("source", "chunk")
        d.setdefault("text", "")
        out.append(Evidence(**d))
    if len(out) != count:
        raise ValueError(f"from_toon: header says {count} records, found {len(out)}")
    return out


# ------------------------------------------------------------------ #
# JSON
# ------------------------------------------------------------------ #

def to_json(evidence: Iterable[Evidence], include_text: bool = True,
            indent: int | None = 2) -> str:
    """Serialize evidence as JSON -- the baseline of the TOON ablation."""
    items = []
    for ev in evidence:
        d = asdict(ev)
        if not include_text:
            d.pop("text", None)
        items.append(d)
    return json.dumps(items, ensure_ascii=False, indent=indent)


def from_json(text: str) -> list[Evidence]:
    """Parse JSON back into Evidence. Inverse of to_json."""
    data = json.loads(text)
    return [Evidence(**d) for d in data]


# ------------------------------------------------------------------ #
# token counting
# ------------------------------------------------------------------ #

def count_tokens(text: str, tokenizer: Any | None = None) -> int:
    """Count tokens with the REAL tokenizer from config (not char count)."""
    tok = tokenizer if tokenizer is not None else get_tokenizer(get_config().TOKENIZER)
    return len(tok.encode(text, disallowed_special=()))


def compare_formats(evidence: Iterable[Evidence], tokenizer: Any | None = None) -> dict:
    """Token counts for both formats, with and without the evidence text."""
    items = list(evidence)
    out: dict[str, Any] = {"n_evidence": len(items)}
    for include_text in (True, False):
        toon = to_toon(items, include_text=include_text)
        js = to_json(items, include_text=include_text)
        t_toon = count_tokens(toon, tokenizer)
        t_json = count_tokens(js, tokenizer)
        out["with_text" if include_text else "metadata_only"] = {
            "toon_tokens": t_toon,
            "json_tokens": t_json,
            "saved_tokens": t_json - t_toon,
            "saved_pct": (100.0 * (t_json - t_toon) / t_json) if t_json else 0.0,
            "toon_chars": len(toon),
            "json_chars": len(js),
        }
    return out
