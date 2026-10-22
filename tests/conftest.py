"""
conftest.py — pytest collection rules for tests/.

Two files in this directory are **manual verification scripts**, not unit
tests: they execute at import time, call the local LLM, and print their
output for a human to read.  `pytest tests/` must not collect them — it
would fire LLM calls (and take minutes) as a side effect of a test run.

They are still runnable by hand:

    python tests/test_extractor_task13.py
    python tests/test_gleaning_task14.py
"""

collect_ignore = [
    "test_extractor_task13.py",
    "test_gleaning_task14.py",
]
