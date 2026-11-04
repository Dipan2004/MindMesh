"""
test_phase6_tracing.py — Unit tests for Task 6.1 (tracing toggle).

Covers every acceptance criterion in TASKS.md Task 6.1:
  - mode="local"  sets LANGCHAIN_TRACING_V2="true"
  - mode="api"    sets LANGCHAIN_TRACING_V2="false"
  - Both values are the exact lowercase strings the LangChain SDK reads.
  - LANGCHAIN_PROJECT is set from config when non-empty.
  - Invalid mode raises ValueError.
  - set_tracing is idempotent.
  - No API key or secret is written to the environment.
  - current_tracing_state() never returns LANGCHAIN_API_KEY.
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch


# ── helpers ──────────────────────────────────────────────────────────────────

def _clean_env():
    """Context manager: remove tracing env vars before a test, restore after."""
    KEYS = ("LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT", "LANGCHAIN_API_KEY")
    old = {k: os.environ.pop(k, None) for k in KEYS}
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# Use as a fixture too
@pytest.fixture(autouse=False)
def clean_tracing_env():
    KEYS = ("LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT")
    old = {k: os.environ.pop(k, None) for k in KEYS}
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ── Task 6.1 DoD test — exact values required by the spec ────────────────────

def test_set_tracing_local_sets_true(clean_tracing_env):
    """mode='local' must set LANGCHAIN_TRACING_V2 to the exact string 'true'."""
    from graphrag_lite.observability.tracing import set_tracing
    set_tracing("local")
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true", (
        f"Expected 'true', got {os.environ.get('LANGCHAIN_TRACING_V2')!r}"
    )


def test_set_tracing_api_sets_false(clean_tracing_env):
    """mode='api' must set LANGCHAIN_TRACING_V2 to the exact string 'false'."""
    from graphrag_lite.observability.tracing import set_tracing
    set_tracing("api")
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false", (
        f"Expected 'false', got {os.environ.get('LANGCHAIN_TRACING_V2')!r}"
    )


def test_set_tracing_local_then_api(clean_tracing_env):
    """Calling local then api must end on 'false'."""
    from graphrag_lite.observability.tracing import set_tracing
    set_tracing("local")
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    set_tracing("api")
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_set_tracing_api_then_local(clean_tracing_env):
    """Calling api then local must end on 'true'."""
    from graphrag_lite.observability.tracing import set_tracing
    set_tracing("api")
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    set_tracing("local")
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"


# ── Idempotency ───────────────────────────────────────────────────────────────

def test_set_tracing_idempotent_local(clean_tracing_env):
    """Calling set_tracing('local') twice must leave the same value."""
    from graphrag_lite.observability.tracing import set_tracing
    set_tracing("local")
    set_tracing("local")
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"


def test_set_tracing_idempotent_api(clean_tracing_env):
    """Calling set_tracing('api') twice must leave the same value."""
    from graphrag_lite.observability.tracing import set_tracing
    set_tracing("api")
    set_tracing("api")
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


# ── LANGCHAIN_PROJECT ─────────────────────────────────────────────────────────

def test_set_tracing_sets_project_from_config(clean_tracing_env):
    """set_tracing must also set LANGCHAIN_PROJECT from cfg.LANGCHAIN_PROJECT."""
    from graphrag_lite.observability.tracing import set_tracing
    from graphrag_lite.config import get_config
    cfg = get_config()
    set_tracing("local")
    if cfg.LANGCHAIN_PROJECT:
        assert os.environ.get("LANGCHAIN_PROJECT") == cfg.LANGCHAIN_PROJECT
    # If project is empty, the key should either be absent or empty — no assertion
    # needed on an empty-project config.


# ── Invalid mode ──────────────────────────────────────────────────────────────

def test_set_tracing_invalid_mode_raises():
    """An invalid mode must raise ValueError."""
    from graphrag_lite.observability.tracing import set_tracing
    with pytest.raises(ValueError, match="mode must be 'local' or 'api'"):
        set_tracing("cloud")  # type: ignore[arg-type]


def test_set_tracing_empty_mode_raises():
    from graphrag_lite.observability.tracing import set_tracing
    with pytest.raises(ValueError):
        set_tracing("")  # type: ignore[arg-type]


# ── Secret redaction ──────────────────────────────────────────────────────────

def test_set_tracing_does_not_write_api_key(clean_tracing_env):
    """set_tracing must never write LANGCHAIN_API_KEY to the environment."""
    # Pre-condition: key not in env
    os.environ.pop("LANGCHAIN_API_KEY", None)
    from graphrag_lite.observability.tracing import set_tracing
    set_tracing("local")
    assert "LANGCHAIN_API_KEY" not in os.environ, (
        "set_tracing must not add LANGCHAIN_API_KEY to the environment"
    )


def test_set_tracing_does_not_overwrite_existing_api_key():
    """If LANGCHAIN_API_KEY was already in the env, set_tracing must not change it."""
    sentinel = "SENTINEL_KEY_DO_NOT_CLEAR"
    os.environ["LANGCHAIN_API_KEY"] = sentinel
    try:
        from graphrag_lite.observability.tracing import set_tracing
        set_tracing("local")
        assert os.environ.get("LANGCHAIN_API_KEY") == sentinel, (
            "set_tracing must not overwrite an existing LANGCHAIN_API_KEY"
        )
    finally:
        os.environ.pop("LANGCHAIN_API_KEY", None)


def test_current_tracing_state_excludes_api_key(clean_tracing_env):
    """current_tracing_state() must never return LANGCHAIN_API_KEY."""
    os.environ["LANGCHAIN_API_KEY"] = "secret_key_value"
    try:
        from graphrag_lite.observability.tracing import set_tracing, current_tracing_state
        set_tracing("local")
        state = current_tracing_state()
        assert "LANGCHAIN_API_KEY" not in state, (
            "current_tracing_state must not surface LANGCHAIN_API_KEY"
        )
        assert "secret_key_value" not in str(state), (
            "current_tracing_state must not expose the API key value"
        )
    finally:
        os.environ.pop("LANGCHAIN_API_KEY", None)


# ── current_tracing_state() ───────────────────────────────────────────────────

def test_current_tracing_state_reflects_set_tracing(clean_tracing_env):
    """current_tracing_state() must reflect the value just set by set_tracing."""
    from graphrag_lite.observability.tracing import set_tracing, current_tracing_state
    set_tracing("local")
    state = current_tracing_state()
    assert state["LANGCHAIN_TRACING_V2"] == "true"

    set_tracing("api")
    state = current_tracing_state()
    assert state["LANGCHAIN_TRACING_V2"] == "false"


def test_current_tracing_state_returns_none_when_unset(clean_tracing_env):
    """Before any set_tracing call, current_tracing_state returns None for the key."""
    os.environ.pop("LANGCHAIN_TRACING_V2", None)
    from graphrag_lite.observability.tracing import current_tracing_state
    state = current_tracing_state()
    assert state["LANGCHAIN_TRACING_V2"] is None
