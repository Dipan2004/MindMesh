"""
client_factory.py -- Factory for LLM clients supporting local and API modes.
"""
from __future__ import annotations

from typing import Literal, Protocol

from graphrag_lite.config import get_config
from graphrag_lite.llm.ollama_client import OllamaClient

class LLMClient(Protocol):
    """Protocol defining the interface for LLM clients."""
    def generate(self, prompt: str) -> str: ...
    @property
    def model(self) -> str: ...

class GrokClientStub:
    """
    Stub for the Grok API client.
    As per requirements: an interface only. No hosted calls are made.
    The API key is passed at construction but never stored in Config or logged.
    """
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._model = "grok-1"

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: str) -> str:
        # Interface only: return a constant indicating it's a stub
        return "[Grok API Stub] Request received. (Interface only - no hosted call made)"

def get_llm_client(mode: Literal["local", "api"], api_key: str | None = None) -> LLMClient:
    """
    Factory to return the appropriate LLM client based on the mode.

    - 'local': Returns an OllamaClient using the SLM_MODEL from config.
    - 'api': Returns a GrokClientStub.
    """
    cfg = get_config()

    if mode == "local":
        return OllamaClient(
            model=cfg.SLM_MODEL,
            temperature=0.0,
            seed=42,
            num_predict=2048, # Default for extraction, can be overridden by generator
        )
    elif mode == "api":
        return GrokClientStub(api_key=api_key)
    else:
        raise ValueError(f"Unsupported LLM mode: {mode}. Use 'local' or 'api'.")
