"""
ollama_client.py -- Thin wrapper around the ollama Python SDK.

Temperature, seed, and num_predict are pinned at construction time.
num_predict caps runaway generation (default 2048 from config) so that
entity-rich chunks cannot cause ReadTimeout by generating indefinitely.

THINKING MODELS (2026-10-22): qwen3-family models are reasoning models and
return their chain-of-thought in a SEPARATE ``thinking`` field, not in
``response``.  The thinking text consumes the whole ``num_predict`` budget
and leaves the visible ``response`` empty, so extraction returns nothing
that parses.  The Ollama API disables this with a top-level ``think``
parameter -- a SIBLING of ``options``, not a key inside it:

    client.generate(model=..., prompt=..., think=False, options={...})

``think=None`` (the default here) means "do not send the parameter", which
leaves the model's own default in place.  Pass ``think=False`` for
extraction and ``think=True`` only when the reasoning trace is wanted.
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
        timeout: float = 300.0,
        num_predict: int = 2048,
        think: bool | None = None,
        num_ctx: int | None = None,
        frequency_penalty: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.num_predict = num_predict
        # None  -> do not send the parameter; the model decides (default)
        # False -> explicitly disable reasoning
        # True  -> explicitly request reasoning
        self.think = think
        # Runtime context window. Ollama's default of 4096 is NOT a hard cap
        # on generation -- Ollama grows the runtime context to fit
        # num_predict (measured 2026-10-22: 8750 tokens in one request,
        # prompt_eval=558 + eval=8192, on a runner 'ollama ps' showed at
        # CONTEXT=4096). But a request served by a runner still at 4096 can
        # only generate (4096 - prompt_tokens), so a 403-token prompt leaves
        # only 3693 and the response comes back with done_reason='length'
        # even though num_predict was not reached. Raising num_ctx to 8192
        # moved that same request to the full num_predict.
        # None -> do not send the parameter.
        self.num_ctx = num_ctx
        # Frequency penalty (llama.cpp ``frequency_penalty``): the penalty
        # applied to a token scales with HOW MANY TIMES it has already
        # appeared, unlike ``repeat_penalty`` which applies the same flat
        # penalty to every token that has ever been seen.
        #
        # It is the lever for degenerate repetition loops. Measured
        # 2026-10-22 on qwen2.5:1.5b, chunk 4af3dc8c (transformer_architecture
        # chunk 4, num_predict=4096, 3 runs per setting):
        #
        #   no penalty   eval=4096  done=length  parse=FAIL   0/3
        #   0.25         eval=2583  done=stop    parse=ok     3/3  21 ent / 15 rel
        #   0.5          eval=489   done=stop    parse=ok     3/3  10 ent /  8 rel
        #   0.75         eval=353   done=stop    parse=ok     3/3   8 ent /  3 rel
        #   1.0          eval=398   done=stop    parse=FAIL        (cut off mid-field)
        #   1.5          eval=98    done=stop    parse=FAIL        (cut off mid-field)
        #
        # Too high and the model abandons a record mid-field, leaving
        # unterminated JSON -- the penalty makes the tokens needed to FINISH
        # a field it has already started expensive too.  Too low and it does
        # not break the loop at all.  0.25 is the default: the chunk names
        # ~22 real entities and 0.25 recovers 21 of them with zero
        # repetition, while 0.5 and 0.75 have traded the repetition for
        # under-extraction.
        #
        # 0.0 -> do not send the parameter (llama.cpp's own default; a no-op).
        self.frequency_penalty = frequency_penalty

    def _kwargs(self, prompt: str | None = None) -> dict:
        """
        Build the kwargs for ollama.Client.generate().

        ``prompt`` is included only when it is not None. It is a separate
        argument rather than part of ``options`` because Ollama takes it as
        a top-level field, alongside ``model`` and ``think``.
        """
        options: dict = {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_predict": self.num_predict,
        }
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.frequency_penalty and self.frequency_penalty > 0:
            options["frequency_penalty"] = self.frequency_penalty
        kwargs: dict = {"model": self.model, "options": options}
        if self.think is not None:
            kwargs["think"] = self.think
        if prompt is not None:
            kwargs["prompt"] = prompt
        return kwargs

    @property
    def cache_tag(self) -> str:
        """
        A short string identifying the generation settings this client uses.

        The LLM cache key is (prompt, model). Two calls with the same prompt
        and model but DIFFERENT settings -- a different num_predict, a
        different num_ctx, a different frequency_penalty -- are not the
        same call, and a cache that cannot tell them apart will hand back a
        response generated under the wrong settings. Folding this tag into
        the cache key keeps those apart.

        It is deliberately derived from the settings that actually change
        the response: temperature, seed, num_predict, num_ctx,
        frequency_penalty and think.
        """
        return (
            f"t={self.temperature};seed={self.seed};"
            f"np={self.num_predict};"
            f"ctx={self.num_ctx};"
            f"fp={self.frequency_penalty};"
            f"think={self.think}"
        )

    def with_overrides(self, **overrides) -> "OllamaClient":
        """
        Return a NEW client with the same settings, except those overridden.

        Used to escalate a failed call -- e.g. retry an extraction that
        fell into a repetition loop with a higher ``frequency_penalty``,
        which is a different decode and therefore a different cache entry.
        The original client is not modified.
        """
        settings = {
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "timeout": self.timeout,
            "num_predict": self.num_predict,
            "think": self.think,
            "num_ctx": self.num_ctx,
            "frequency_penalty": self.frequency_penalty,
        }
        settings.update(overrides)
        return OllamaClient(**settings)

    def generate(self, prompt: str) -> str:
        """
        Send prompt to the Ollama model and return the response text.

        Retries once on ReadTimeout (leftover queue drain). Second timeout
        is genuine failure and is re-raised.

        If the model returned a non-empty ``thinking`` field and thinking
        was not explicitly disabled, a warning is emitted: the reasoning
        trace was discarded and it may have consumed the whole
        ``num_predict`` budget, leaving ``response`` empty.
        """
        import httpx, time, warnings

        client = ollama.Client(timeout=self.timeout)
        try:
            response = client.generate(**self._kwargs(prompt))
            self._warn_if_thinking(response)
            return response["response"]
        except httpx.ReadTimeout:
            warnings.warn(
                f"[OllamaClient] ReadTimeout on first attempt -- retrying once. "
                f"model={self.model}",
                UserWarning,
                stacklevel=2,
            )
            time.sleep(5)
            client2 = ollama.Client(timeout=self.timeout)
            response = client2.generate(**self._kwargs(prompt))
            self._warn_if_thinking(response)
            return response["response"]

    def _warn_if_thinking(self, response: dict) -> None:
        """Warn when a reasoning model's thinking text was discarded."""
        import warnings

        thinking = (response or {}).get("thinking") or ""
        if thinking and self.think is not False:
            warnings.warn(
                f"[OllamaClient] model={self.model} returned a non-empty "
                f"'thinking' field ({len(thinking)} chars) that was "
                f"discarded. Reasoning models (qwen3 family) spend the "
                f"num_predict budget on it, which can leave 'response' "
                f"empty and the output unparseable. Pass think=False to "
                f"disable thinking.",
                UserWarning,
                stacklevel=3,
            )
