"""
test_think_flag.py -- LLM_THINK config flag and OllamaClient `think` plumbing.

Why this exists (2026-10-22): qwen3-family models are reasoning models. Their
chain-of-thought arrives in a SEPARATE `thinking` field, which consumes the
whole num_predict budget and leaves the visible `response` empty -- so
extraction returns nothing that parses. The fix is Ollama's top-level
`think` parameter, which is a SIBLING of `options`, not a key inside it.

These tests are pure/offline: no Ollama server, no network, no cache writes.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.config import get_config
from graphrag_lite.llm.ollama_client import OllamaClient


# ── Config: LLM_THINK -> effective_think() ───────────────────────────────────

def test_llm_think_default_is_auto():
    """Default is 'auto' -> effective_think() returns None (don't send the param)."""
    cfg = get_config()
    assert cfg.LLM_THINK == "auto"
    assert cfg.effective_think() is None


def test_effective_think_off():
    cfg = get_config()
    cfg.LLM_THINK = "off"
    try:
        assert cfg.effective_think() is False
    finally:
        cfg.LLM_THINK = "auto"


def test_effective_think_on():
    cfg = get_config()
    cfg.LLM_THINK = "on"
    try:
        assert cfg.effective_think() is True
    finally:
        cfg.LLM_THINK = "auto"


# ── Client: the `think` kwarg is top-level, a sibling of `options` ──────────

def test_think_none_omits_the_parameter():
    """auto -> the parameter is NOT sent, so the model keeps its own default."""
    kwargs = OllamaClient("qwen3:4b")._kwargs()
    assert "think" not in kwargs


def test_think_false_is_top_level():
    """`think` must be a sibling of `options`, never a key inside it."""
    kwargs = OllamaClient("qwen3:4b", think=False)._kwargs()
    assert kwargs["think"] is False
    assert "think" not in kwargs["options"]


def test_think_true_is_top_level():
    kwargs = OllamaClient("qwen3:4b", think=True)._kwargs()
    assert kwargs["think"] is True
    assert "think" not in kwargs["options"]


def test_options_still_carries_temperature_seed_and_num_predict():
    kwargs = OllamaClient("qwen3:4b", temperature=0.0, seed=42,
                          num_predict=4096)._kwargs()
    assert kwargs["options"]["temperature"] == 0.0
    assert kwargs["options"]["seed"] == 42
    assert kwargs["options"]["num_predict"] == 4096


# ── Client: num_ctx lifts the generation wall ───────────────────────────────

def test_num_ctx_none_omits_the_parameter():
    kwargs = OllamaClient("qwen3:4b")._kwargs()
    assert "num_ctx" not in kwargs["options"]


def test_num_ctx_is_sent_in_options():
    """num_ctx lives INSIDE options -- unlike think, which is top-level."""
    kwargs = OllamaClient("qwen3:4b", num_ctx=8192)._kwargs()
    assert kwargs["options"]["num_ctx"] == 8192


def test_num_ctx_and_think_can_be_combined():
    """think is top-level, num_ctx is inside options -- both at once."""
    kwargs = OllamaClient("qwen3.5:2b", think=False, num_ctx=8192,
                          num_predict=4096)._kwargs()
    assert kwargs["think"] is False
    assert kwargs["options"]["num_ctx"] == 8192
    assert kwargs["options"]["num_predict"] == 4096


def test_config_num_ctx_default():
    """The default must exceed num_predict + the measured prompt size.

    The extraction prompt costs 403 prompt tokens (measured 2026-10-22).
    With Ollama's default num_ctx of 4096, generation is capped at 3693
    tokens regardless of num_predict -- the response comes back with
    done_reason='length' and an eval_count BELOW num_predict.
    """
    cfg = get_config()
    assert cfg.LLM_NUM_CTX == 8192
    assert cfg.LLM_NUM_CTX > cfg.LLM_NUM_PREDICT


def test_num_predict_is_reachable_within_num_ctx():
    """num_predict + prompt must fit inside num_ctx, or the cap is a lie."""
    cfg = get_config()
    measured_prompt_tokens = 403
    assert cfg.LLM_NUM_PREDICT + measured_prompt_tokens <= cfg.LLM_NUM_CTX


# ── Client: discarded thinking text is surfaced, not swallowed ───────────────

def test_warns_when_thinking_is_discarded():
    """A non-empty `thinking` field with think not disabled must warn."""
    import warnings

    client = OllamaClient("qwen3.5:2b")           # think=None -> model default
    response = {"response": "", "thinking": "x" * 100}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client._warn_if_thinking(response)
    assert len(caught) == 1
    assert "thinking" in str(caught[0].message)


def test_no_warning_when_thinking_was_explicitly_disabled():
    """think=False is the intended configuration -- do not nag about it."""
    import warnings

    client = OllamaClient("qwen3.5:2b", think=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client._warn_if_thinking({"response": "{}", "thinking": "x" * 100})
    assert len(caught) == 0


def test_no_warning_when_thinking_is_empty():
    import warnings

    client = OllamaClient("qwen3.5:2b")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client._warn_if_thinking({"response": "{}", "thinking": ""})
    assert len(caught) == 0


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("All think-flag tests passed.")
