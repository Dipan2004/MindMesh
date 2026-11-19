"""
generation.py -- HotpotQA answer + supporting-fact generation with a local Ollama model.

Every pipeline's retrieved paragraphs go through this SAME generator with the
SAME settings, so pipeline comparisons isolate retrieval.

The model sees the top ``context_k`` retrieved paragraphs with numbered
sentences and must return JSON:
    {"answer": "<short answer>", "supporting_facts": [[passage_no, sentence_no], ...]}
Passage/sentence numbers are mapped back to (title, sentence_index) pairs for
the official supporting-fact metric.
"""

from __future__ import annotations

import json
import re
import time

PROMPT_VERSION = "hotpotqa-json-v2-schema"

# Structured-output schema passed to Ollama as ``format`` (freeze amendment 3).
# With the generic ``format="json"`` grammar, mistral:latest wrote
# ``"supporting_facts": [4, 1], `` and then tried to open a second pair; the
# generic grammar only allowed whitespace there, the model emitted tabs until
# Ollama aborted with "token repeat limit reached", and the answer was lost.
# The schema fixes the nesting (a list of [passage, sentence] integer pairs).
# The prompt text is unchanged.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "supporting_facts": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
    },
    "required": ["answer", "supporting_facts"],
}

INSTRUCTIONS = (
    "Answer the question using ONLY the passages below. The question may need "
    "facts from two passages.\n"
    "Rules:\n"
    "- The answer must be as short as possible: an entity name, a date, a number, "
    "or 'yes' / 'no'. Do not write a sentence.\n"
    "- List every sentence you used as supporting facts, as [passage_number, sentence_number].\n"
    "- Reply with JSON only, exactly in this form:\n"
    '{"answer": "...", "supporting_facts": [[1, 0], [3, 2]]}'
)


def build_prompt(question: str, paragraphs: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(paragraphs, 1):
        lines = [f"Passage {i}: {p['title']}"]
        for j, s in enumerate(p["sentences"]):
            lines.append(f"  ({j}) {s.strip()}")
        blocks.append("\n".join(lines))
    return f"{INSTRUCTIONS}\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nJSON:"


def _parse(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def generate(model: str, question: str, paragraphs: list[dict], options: dict) -> dict:
    """Return {answer, sp_pred, raw, parse_ok, latency_s, token stats}."""
    import ollama
    prompt = build_prompt(question, paragraphs)
    t0 = time.monotonic()
    r = ollama.Client(timeout=600).generate(
        model=model, prompt=prompt, format=ANSWER_SCHEMA, options=options, keep_alive="10m",
    )
    latency = time.monotonic() - t0
    raw = r.response or ""
    obj = _parse(raw)
    answer = obj.get("answer", "")
    if not isinstance(answer, str):
        answer = json.dumps(answer) if answer is not None else ""
    sp_pred = []
    raw_sp = obj.get("supporting_facts", []) or []
    # Accept the requested [[p, s], ...] form, and also a flat list of
    # integers ([p, s] or [p1, s1, p2, s2]) read as consecutive pairs --
    # mistral:latest emits the flat form (freeze amendment 2).
    if isinstance(raw_sp, list) and raw_sp and all(isinstance(x, (int, float)) for x in raw_sp):
        raw_sp = [raw_sp[i:i + 2] for i in range(0, len(raw_sp) - 1, 2)]
    for item in raw_sp if isinstance(raw_sp, list) else []:
        try:
            pi, si = int(item[0]), int(item[1])
        except Exception:
            continue
        if 1 <= pi <= len(paragraphs):
            p = paragraphs[pi - 1]
            if 0 <= si < len(p["sentences"]):
                pair = [p["title"], si]
                if pair not in sp_pred:
                    sp_pred.append(pair)
    eval_s = (r.eval_duration or 0) / 1e9
    return {
        "answer": answer.strip(),
        "sp_pred": sp_pred,
        "raw": raw,
        "parse_ok": bool(obj) and "answer" in obj,
        "latency_s": latency,
        "prompt_tokens": r.prompt_eval_count,
        "output_tokens": r.eval_count,
        "tokens_per_s": round(r.eval_count / eval_s, 2) if eval_s else None,
        "load_s": round((r.load_duration or 0) / 1e9, 3),
    }
