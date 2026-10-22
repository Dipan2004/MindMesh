"""
test_ollama_client_prompt.py — the prompt must actually reach the model.

Regression test for a bug that was introduced on 2026-10-22 while adding
``frequency_penalty`` / ``cache_tag`` / ``with_overrides`` to
``OllamaClient``.

The refactor replaced an explicit call

    client.generate(model=..., prompt=prompt, think=..., options=...)

with ``client.generate(**self._kwargs())``, and ``_kwargs()`` built only
``model`` / ``options`` / ``think`` — **the prompt was never included**.
The client therefore sent a prompt-less request, Ollama answered with an
empty response, and every extraction failed to parse. The symptom was
silent: no exception, no warning, just an empty string that the parser
rejected with "No JSON object found in LLM response. Raw response:".

It was caught by a 42-chunk corpus run that failed 42/42 in 9.6 seconds —
every chunk failing in 0.2 s is the tell: a model cannot generate a real
extraction that fast.

These tests are offline: the Ollama SDK's ``Client.generate`` is stubbed,
so nothing is sent anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag_lite.llm.ollama_client import OllamaClient  # noqa: E402


def test_kwargs_includes_the_prompt_when_given():
    kwargs = OllamaClient("qwen2.5:1.5b")._kwargs("the prompt text")
    assert kwargs["prompt"] == "the prompt text"


def test_kwargs_omits_the_prompt_when_not_given():
    # The other tests (test_frequency_penalty, test_think_flag) call
    # _kwargs() with no arguments, so the no-prompt form must stay valid.
    kwargs = OllamaClient("qwen2.5:1.5b")._kwargs()
    assert "prompt" not in kwargs
    assert kwargs["model"] == "qwen2.5:1.5b"


def test_prompt_is_a_top_level_field_not_an_option():
    # Ollama takes `prompt` alongside `model` and `think`, NOT inside
    # `options` -- options carries the sampling parameters.
    kwargs = OllamaClient("qwen2.5:1.5b")._kwargs("P")
    assert "prompt" not in kwargs["options"]


def test_generate_sends_the_prompt(monkeypatch):
    """The end-to-end check: what reaches the SDK is what the caller passed."""
    import ollama

    seen = {}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, **kwargs):
            seen.update(kwargs)
            return {"response": "PONG", "thinking": "", "done_reason": "stop"}

    monkeypatch.setattr(ollama, "Client", _FakeClient)

    client = OllamaClient("qwen2.5:1.5b", think=False, num_predict=4096)
    out = client.generate("Reply with exactly: PONG")

    assert out == "PONG"
    assert seen.get("prompt") == "Reply with exactly: PONG"
    assert seen.get("model") == "qwen2.5:1.5b"
    assert seen.get("think") is False
    assert seen["options"]["num_predict"] == 4096


def test_generate_does_not_send_an_empty_prompt(monkeypatch):
    """
    A prompt-less request comes back with an empty response and no
    exception -- the failure is silent. Guard against it explicitly.
    """
    import ollama

    seen = {}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, **kwargs):
            seen.update(kwargs)
            return {"response": "", "thinking": "", "done_reason": "load"}

    monkeypatch.setattr(ollama, "Client", _FakeClient)

    client = OllamaClient("qwen2.5:1.5b", think=False)
    client.generate("a real prompt")

    assert seen.get("prompt") == "a real prompt"
    assert seen.get("prompt") != ""
