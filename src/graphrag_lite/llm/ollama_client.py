"""
ollama_client.py — Thin wrapper around the ollama Python SDK.

Returns a simple object with a .generate(prompt) -> str interface
that matches the LLMClient protocol used by extractor.py and generate.py.

Temperature and seed are pinned at construction time so extraction/gleaning
calls are deterministic across re-runs (temperature=0.0, seed=42 by default
per CONFIG.md §2).

A request timeout is set so a stuck Ollama generation fails loudly with a
TimeoutError rather than blocking the calling process indefinitely. Without
this, an abandoned in-flight request from a previous process keeps Ollama's
single-threaded generation queue blocked, causing all subsequent calls to
hang silently until the server is restarted.
"""

from __future__ import annotations

import ollama


class OllamaClient:
    """Synchronous Ollama client implementing the LLMClient protocol."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        seed: int = 42,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout  # seconds before raising TimeoutError

    def generate(self, prompt: str) -> str:
        """
        Send ``prompt`` to the local Ollama model and return the response text.

        Temperature and seed are pinned for deterministic output.

        Retries once on ReadTimeout because Ollama's single-threaded generation
        queue can be temporarily blocked by a leftover in-flight request from a
        previously killed process. The first call drains that stuck request;
        the retry then completes normally. A second consecutive timeout is a
        genuine failure and is re-raised.

        Raises
        ------
        ollama.ResponseError
            If the model is not available or the server is unreachable.
        httpx.ReadTimeout / httpcore.ReadTimeout
            If two consecutive attempts both time out (genuine stuck queue or
            model too slow for the configured timeout).
        """
        import httpx

        client = ollama.Client(timeout=self.timeout)
        try:
            response = client.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": self.temperature, "seed": self.seed},
            )
            return response["response"]
        except httpx.ReadTimeout:
            # First timeout: Ollama queue may have been blocked by a leftover
            # request from a previous killed process. Wait briefly, then retry once.
            import time
            import warnings
            warnings.warn(
                f"[OllamaClient] ReadTimeout on first attempt — retrying once "
                f"(queue may have been draining a leftover request). "
                f"model={self.model}",
                UserWarning,
                stacklevel=2,
            )
            time.sleep(5)  # give Ollama a moment after the queue drains
            client2 = ollama.Client(timeout=self.timeout)
            response = client2.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": self.temperature, "seed": self.seed},
            )
            return response["response"]
