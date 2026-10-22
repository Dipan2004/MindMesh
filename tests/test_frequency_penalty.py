"""
test_frequency_penalty.py -- LLM_FREQUENCY_PENALTY config flag and the
OllamaClient `frequency_penalty` plumbing.

Why this exists (2026-10-22). Chunk 4af3dc8c of transformer_architecture
loops degenerately on qwen2.5:1.5b -- the same entity line is emitted over
and over until the num_predict cap is reached, and the response never
parses. A flat `repeat_penalty` does NOT fix it (measured: 83x at no
penalty, 37x at 1.05, 48x at 1.15; every setting still truncated and
unparseable, and the response was non-monotonic in the penalty).

A `frequency_penalty` DOES fix it, because the penalty scales with the
COUNT of a token in the context rather than being a flat penalty on every
token ever seen. Measured on that chunk, num_predict=4096, 3 runs per
setting:

    no penalty   eval=4096  done=length  parse=FAIL   0/3
    0.25         eval=2583  done=stop    parse=ok     3/3  21 entities / 15 relationships
    0.5          eval=489   done=stop    parse=ok     3/3  10 entities /  8 relationships
    0.75         eval=353   done=stop    parse=ok     3/3   8 entities /  3 relationships
    1.0          eval=398   done=stop    parse=FAIL        (record cut off mid-field)
    1.5          eval=98    done=stop    parse=FAIL        (record cut off mid-field)

Too high a penalty breaks the extraction a different way -- the model
abandons a record mid-field and the JSON is left unterminated.  Too high
in the other direction (0.5, 0.75) the penalty breaks the loop but
UNDER-extracts: the chunk names ~22 real entities and only 0.25 recovers
them.  So the value matters, and 0.25 is the default.

These tests are pure/offline: no Ollama server, no network, no cache writes.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.config import get_config
from graphrag_lite.llm.ollama_client import OllamaClient


# ── Config: LLM_FREQUENCY_PENALTY ───────────────────────────────────────────

def test_config_frequency_penalty_default():
    """The default must be the measured sweet spot, 0.25.

    0.25 is the only setting that both breaks the repetition loop (the model
    stops before the num_predict cap) AND recovers the full entity set --
    21 of the chunk's ~22 real entities, with zero repetition. 0.5 and 0.75
    also stop the loop but under-extract (10 and 8 entities); 1.0 and 1.5
    truncate the extraction mid-field.
    """
    cfg = get_config()
    assert cfg.LLM_FREQUENCY_PENALTY == 0.25


def test_frequency_penalty_is_a_plain_float():
    cfg = get_config()
    assert isinstance(cfg.LLM_FREQUENCY_PENALTY, float)
    assert 0.0 <= cfg.LLM_FREQUENCY_PENALTY <= 2.0


# ── Client: the knob is sent inside options, like num_ctx ───────────────────

def test_frequency_penalty_is_sent_in_options():
    """frequency_penalty lives INSIDE options -- like num_ctx, unlike think."""
    kwargs = OllamaClient("qwen2.5:1.5b", frequency_penalty=0.5)._kwargs()
    assert kwargs["options"]["frequency_penalty"] == 0.5
    assert "frequency_penalty" not in kwargs


def test_zero_frequency_penalty_is_not_sent():
    """0.0 is llama.cpp's own default (a no-op), so do not send it.

    Omitting the parameter keeps the request byte-identical to the
    pre-penalty behaviour, which is what makes the knob safe to add.
    """
    kwargs = OllamaClient("qwen2.5:1.5b", frequency_penalty=0.0)._kwargs()
    assert "frequency_penalty" not in kwargs["options"]


def test_negative_frequency_penalty_is_not_sent():
    """A negative penalty would REWARD repetition. Refuse to send it."""
    kwargs = OllamaClient("qwen2.5:1.5b", frequency_penalty=-1.0)._kwargs()
    assert "frequency_penalty" not in kwargs["options"]


def test_frequency_penalty_composes_with_num_ctx_and_think():
    """All three knobs at once: think top-level, the other two in options."""
    kwargs = OllamaClient(
        "qwen3:4b",
        think=False,
        num_ctx=8192,
        num_predict=4096,
        frequency_penalty=0.5,
    )._kwargs()
    assert kwargs["think"] is False
    assert kwargs["options"]["num_ctx"] == 8192
    assert kwargs["options"]["num_predict"] == 4096
    assert kwargs["options"]["frequency_penalty"] == 0.5


def test_other_options_are_untouched_by_the_penalty():
    """The penalty must not disturb temperature/seed/num_predict."""
    kwargs = OllamaClient("qwen2.5:1.5b", temperature=0.0, seed=42,
                          num_predict=4096, frequency_penalty=0.5)._kwargs()
    assert kwargs["options"]["temperature"] == 0.0
    assert kwargs["options"]["seed"] == 42
    assert kwargs["options"]["num_predict"] == 4096


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("All frequency-penalty tests passed.")
