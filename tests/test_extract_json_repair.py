"""
test_extract_json_repair.py -- the trailing-comma repair in _extract_json.

Context (2026-10-22): chunk a4a679c4 (retrieval_augmented_generation/7) run on
qwen2.5:1.5b with num_predict=4096 produced a COMPLETE extraction -- 20
entities, 20 relationships -- whose only defect was one trailing comma before
the closing "]". Strict json.loads rejected it, the chunk was recorded as a
parse failure, and the whole extraction was lost.

These tests are offline: they use the recorded raw response when it is
available, and hand-written minimal cases otherwise.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.ingestion.extractor import _extract_json  # noqa: E402


def test_trailing_comma_in_array_is_repaired():
    raw = '{"entities": [{"name": "A"},], "relationships": []}'
    out = _extract_json(raw)
    assert out["entities"][0]["name"] == "A"
    assert out["relationships"] == []


def test_trailing_comma_in_object_is_repaired():
    raw = '{"entities": [], "relationships": [],}'
    out = _extract_json(raw)
    assert out == {"entities": [], "relationships": []}


def test_multiple_trailing_commas_are_repaired():
    raw = '{\n  "entities": [\n    {"name": "A", "type": "X",},\n  ],\n  "relationships": [\n    {"source": "A", "target": "B", "type": "R",},\n  ],\n}'
    out = _extract_json(raw)
    assert len(out["entities"]) == 1
    assert len(out["relationships"]) == 1
    assert out["relationships"][0]["source"] == "A"


def test_valid_json_is_unchanged():
    raw = '{"entities": [{"name": "A", "type": "X"}], "relationships": []}'
    out = _extract_json(raw)
    assert out == json.loads(raw)


def test_comma_inside_a_string_is_not_touched():
    # A comma followed by whitespace and a quote is data, not a dangling comma.
    raw = '{"entities": [{"name": "A, B", "type": "X"}], "relationships": []}'
    out = _extract_json(raw)
    assert out["entities"][0]["name"] == "A, B"


def test_fenced_block_with_trailing_comma_and_trailing_prose():
    raw = (
        "```json\n"
        '{"entities": [{"name": "A", "type": "X"},], "relationships": []}\n'
        "```\n\nThis JSON object contains the following information:\n"
        "- **Entities**: a list of entities.\n"
    )
    out = _extract_json(raw)
    assert out["entities"][0]["name"] == "A"


def test_truncated_json_still_raises():
    # The repair must NOT rescue a genuinely truncated response -- the whole
    # point of the _parse_error record is that it stays visible.
    raw = '{"entities": [{"name": "Ashish Vaswani", "type": "PER'
    try:
        _extract_json(raw)
    except ValueError as exc:
        assert "No JSON object found" in str(exc) or "JSON parse error" in str(exc)
    else:
        raise AssertionError("truncated JSON must still raise ValueError")


def test_repaired_real_response_recovers_the_lost_chunk():
    """The actual recorded response, if the experiment output is present."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "data", "cache",
        "exp_timeout_chunks.jsonl",
    )
    if not os.path.exists(path):
        return  # experiment output not present; nothing to assert against
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if not rec["chunk_id"].startswith("a4a679c4"):
                continue
            if rec["num_predict"] != 4096:
                continue
            out = _extract_json(rec["raw_response"])
            assert len(out["entities"]) == 20, len(out["entities"])
            assert len(out["relationships"]) == 20, len(out["relationships"])
            return
    raise AssertionError("a4a679c4 np=4096 record not found in exp_timeout_chunks.jsonl")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
