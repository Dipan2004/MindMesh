"""
tracing.py -- LangSmith tracing toggle (Task 6.1).

Sets the LANGCHAIN_TRACING_V2 environment variable according to the run
mode, as specified in CONFIG.md §8:

  mode="local"  ->  LANGCHAIN_TRACING_V2 = "true"   (tracing on)
  mode="api"    ->  LANGCHAIN_TRACING_V2 = "false"  (tracing off)

The assignment must be called ONCE at startup, before any pipeline or
LLM call is made. Calling it per-request would be a race condition in a
concurrent server context and is explicitly prohibited by the TASKS.md
Task 6.1 spec.

LANGCHAIN_PROJECT is also set from config so all local runs are grouped
together in the LangSmith dashboard under a consistent project name.

Design notes
------------
- This function intentionally writes to ``os.environ`` directly rather than
  to a .env file. The variable must be live for the current process; writing
  to disk would affect only future processes.
- The function never reads an existing value and never silently downgrades a
  previously set "true" to "false". It always overwrites, which is the
  correct startup-time behaviour.
- API keys (LANGCHAIN_API_KEY) are NEVER written here — the caller is
  responsible for having the key in the environment before calling
  set_tracing. This function only flips the on/off toggle.
- The function is idempotent: calling it twice with the same mode produces
  the same environment state and emits a DEBUG log on the second call.

Usage:
    from graphrag_lite.observability.tracing import set_tracing
    set_tracing("local")   # at process startup
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from graphrag_lite.config import get_config

log = logging.getLogger(__name__)

# The two canonical string values expected by the LangChain SDK.
_TRACING_ON = "true"
_TRACING_OFF = "false"


def set_tracing(mode: Literal["local", "api"]) -> None:
    """
    Set LANGCHAIN_TRACING_V2 for the current process based on run mode.

    Parameters
    ----------
    mode : "local" | "api"
        "local"  -- tracing ON  (LANGCHAIN_TRACING_V2="true")
        "api"    -- tracing OFF (LANGCHAIN_TRACING_V2="false")

    Raises
    ------
    ValueError
        If ``mode`` is not "local" or "api".

    Side effects
    ------------
    Writes to os.environ:
      - LANGCHAIN_TRACING_V2   ("true" or "false")
      - LANGCHAIN_PROJECT      (from cfg.LANGCHAIN_PROJECT, if non-empty)

    Does NOT read, write, or delete any other environment variable.
    Does NOT write to disk.
    Does NOT log API keys or other secrets.
    """
    if mode not in ("local", "api"):
        raise ValueError(
            f"set_tracing: mode must be 'local' or 'api', got {mode!r}"
        )

    cfg = get_config()

    tracing_value = _TRACING_ON if mode == "local" else _TRACING_OFF

    previous = os.environ.get("LANGCHAIN_TRACING_V2")
    if previous == tracing_value:
        log.debug(
            "set_tracing: LANGCHAIN_TRACING_V2 already %r for mode=%r — no change",
            tracing_value, mode,
        )
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = tracing_value
        log.debug(
            "set_tracing: LANGCHAIN_TRACING_V2 set to %r (mode=%r)",
            tracing_value, mode,
        )

    # Set the project name so eval runs stay grouped in the dashboard.
    project = cfg.LANGCHAIN_PROJECT
    if project:
        os.environ["LANGCHAIN_PROJECT"] = project
        log.debug("set_tracing: LANGCHAIN_PROJECT set to %r", project)


def current_tracing_state() -> dict[str, str | None]:
    """
    Return the current tracing environment state (read-only).

    Returns a dict with:
      - "LANGCHAIN_TRACING_V2": current value or None if not set
      - "LANGCHAIN_PROJECT": current value or None if not set

    Intentionally does NOT return LANGCHAIN_API_KEY — that is a secret
    and must never be surfaced in logs or diagnostic output.
    """
    return {
        "LANGCHAIN_TRACING_V2": os.environ.get("LANGCHAIN_TRACING_V2"),
        "LANGCHAIN_PROJECT": os.environ.get("LANGCHAIN_PROJECT"),
    }
