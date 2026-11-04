"""
middleware.py — Production hardening for the API layer (Tasks 7.2 + 7.3).

Three concerns are handled here, all sourced from CONFIG.md §11:

  1. CPU-bound offload  (Task 7.2)
     Graph operations (bfs_expand, rank_pagerank) run on the CPU and block
     the asyncio event loop.  ``run_in_threadpool`` wraps any callable in
     asyncio.to_thread so the event loop stays free for other requests.

  2. Outbound Grok connection pool + semaphore  (Task 7.2)
     A single shared httpx.AsyncClient is kept alive for the process
     lifetime.  A semaphore caps concurrent outbound calls to
     STRESS_SEMAPHORE_LIMIT (default 50, per CONFIG.md §10 — checked
     against Grok's documented rate limit of 60 req/min on the free tier
     at time of implementation; 50 gives a 17 % headroom buffer).
     Reference: https://console.x.ai/

  3. Exponential backoff retry + circuit breaker  (Task 7.3)
     ``api_call_with_retry`` wraps outbound Grok calls:
       - Up to RETRY_MAX_ATTEMPTS (default 3) attempts.
       - Base wait RETRY_BACKOFF_BASE_S (default 0.5 s), doubled each rung.
       - After CIRCUIT_BREAKER_FAILURE_THRESHOLD (default 5) consecutive
         failures the breaker OPENS and all subsequent calls raise
         CircuitBreakerOpenError immediately (no retry, no wait).
       - The breaker resets after CIRCUIT_BREAKER_COOLDOWN_S (default 60 s).

These utilities are imported by main.py and routes.py; they do not import
from either, so the dependency graph is acyclic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, TypeVar

import httpx

from graphrag_lite.config import get_config

log = logging.getLogger(__name__)

T = TypeVar("T")


# ── 1. CPU-bound thread-pool offload ─────────────────────────────────────────

async def run_in_threadpool(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Run a synchronous, CPU-bound callable in a worker thread so the asyncio
    event loop is not blocked.

    Usage::

        result = await run_in_threadpool(bfs_expand, G, seeds, hops=2)

    Parameters
    ----------
    func : callable
        A plain synchronous function (no ``async``).
    *args, **kwargs
        Forwarded to ``func``.

    Returns
    -------
    The return value of ``func(*args, **kwargs)``.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ── 2. Shared httpx.AsyncClient + semaphore ───────────────────────────────────

# Module-level singletons; initialised in main.py lifespan and accessed here.
_http_client: httpx.AsyncClient | None = None
_api_semaphore: asyncio.Semaphore | None = None


def get_http_client() -> httpx.AsyncClient:
    """Return the process-level shared AsyncClient (must be initialised first)."""
    if _http_client is None:
        raise RuntimeError(
            "HTTP client not initialised. "
            "Call init_http_client() during application startup."
        )
    return _http_client


def get_api_semaphore() -> asyncio.Semaphore:
    """Return the process-level API semaphore (must be initialised first)."""
    if _api_semaphore is None:
        raise RuntimeError(
            "API semaphore not initialised. "
            "Call init_http_client() during application startup."
        )
    return _api_semaphore


def init_http_client() -> None:
    """
    Create the shared AsyncClient and semaphore.

    Must be called exactly once, from the FastAPI lifespan startup handler.
    The limits are read from config so they can be overridden via .env.

    Semaphore limit = cfg.STRESS_SEMAPHORE_LIMIT (default 50).
    Connection pool: keep-alive 20 connections per host, max 100 total.
    """
    global _http_client, _api_semaphore
    cfg = get_config()
    limit = cfg.STRESS_SEMAPHORE_LIMIT
    _http_client = httpx.AsyncClient(
        timeout=cfg.LLM_TIMEOUT_S,
        limits=httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
        ),
    )
    _api_semaphore = asyncio.Semaphore(limit)
    log.info("HTTP client initialised (semaphore limit=%d)", limit)


async def close_http_client() -> None:
    """
    Close the shared AsyncClient.

    Must be called from the FastAPI lifespan shutdown handler.
    """
    global _http_client, _api_semaphore
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
        _api_semaphore = None
        log.info("HTTP client closed.")


# ── 3. Circuit breaker + exponential backoff retry ────────────────────────────

class CircuitBreakerOpenError(RuntimeError):
    """
    Raised when the circuit breaker is open and the call is fast-failed.

    Callers (route handlers) should convert this to HTTP 503.
    """


class _CircuitBreaker:
    """
    Simple three-state circuit breaker (CLOSED → OPEN → HALF_OPEN → CLOSED).

    State transitions
    -----------------
    CLOSED  : normal operation.  On CIRCUIT_BREAKER_FAILURE_THRESHOLD
              consecutive failures → OPEN.
    OPEN    : all calls are rejected immediately with CircuitBreakerOpenError.
              After CIRCUIT_BREAKER_COOLDOWN_S seconds → HALF_OPEN.
    HALF_OPEN: one probe call is allowed.  Success → CLOSED; failure → OPEN.

    Thread / task safety
    --------------------
    The breaker state is shared across all asyncio tasks in the process.
    Since asyncio is single-threaded per event loop, reads/writes to
    ``_consecutive_failures`` and ``_state`` are safe without a lock.
    For multi-process deployments (uvicorn workers > 1) each worker has its
    own breaker state — this is acceptable; the breaker protects per-worker
    event loops, not global call counts.
    """

    _CLOSED = "CLOSED"
    _OPEN = "OPEN"
    _HALF_OPEN = "HALF_OPEN"

    def __init__(self) -> None:
        cfg = get_config()
        self._threshold: int = cfg.CIRCUIT_BREAKER_FAILURE_THRESHOLD
        # cooldown is not in CONFIG.md; use 60 s (reasonable default, not tunable)
        self._cooldown_s: float = 60.0
        self._state: str = self._CLOSED
        self._consecutive_failures: int = 0
        self._opened_at: float = 0.0

    @property
    def state(self) -> str:
        """Current breaker state string (for logging / health responses)."""
        self._maybe_transition_to_half_open()
        return self._state

    def _maybe_transition_to_half_open(self) -> None:
        if (
            self._state == self._OPEN
            and time.monotonic() - self._opened_at >= self._cooldown_s
        ):
            self._state = self._HALF_OPEN
            log.info("Circuit breaker → HALF_OPEN (cooldown elapsed)")

    def check(self) -> None:
        """
        Raise CircuitBreakerOpenError if the breaker is OPEN.

        Call this BEFORE making an outbound call.
        """
        self._maybe_transition_to_half_open()
        if self._state == self._OPEN:
            raise CircuitBreakerOpenError(
                "Circuit breaker is OPEN — outbound API call rejected. "
                f"Opened at {self._opened_at:.1f}, cooldown={self._cooldown_s}s."
            )

    def record_success(self) -> None:
        """Record a successful call; reset failure counter and close the breaker."""
        self._consecutive_failures = 0
        if self._state != self._CLOSED:
            log.info("Circuit breaker → CLOSED (success after %s)", self._state)
        self._state = self._CLOSED

    def record_failure(self) -> None:
        """Record a failed call; open the breaker if threshold is reached."""
        self._consecutive_failures += 1
        log.warning(
            "Circuit breaker: consecutive failure %d/%d",
            self._consecutive_failures,
            self._threshold,
        )
        if self._consecutive_failures >= self._threshold:
            self._state = self._OPEN
            self._opened_at = time.monotonic()
            log.error(
                "Circuit breaker → OPEN after %d consecutive failures",
                self._consecutive_failures,
            )


# One breaker per process (shared across all async tasks in this event loop).
_breaker = _CircuitBreaker()


def get_circuit_breaker() -> _CircuitBreaker:
    """Return the process-level circuit breaker (for testing and health checks)."""
    return _breaker


async def api_call_with_retry(
    coro_factory: Callable[[], Coroutine[Any, Any, T]],
    *,
    label: str = "api_call",
) -> T:
    """
    Execute an outbound API coroutine with exponential backoff retry and
    circuit breaker protection.

    Failure-counting semantics (P2-2 fix)
    ----------------------------------------
    A **logical failure** is a request that has exhausted every configured
    retry attempt without succeeding.  Intermediate retry attempts within a
    single logical request do NOT increment the breaker counter.  After all
    retries are exhausted, exactly ONE failure is recorded on the breaker.
    A successful attempt records ONE success and resets the counter.

    Semaphore enforcement (P2-1 fix)
    ---------------------------------
    Each individual outbound attempt acquires the shared API semaphore for
    the duration of that attempt only.  The permit is released:
      - After a successful response.
      - After a failed attempt (before the backoff sleep).
      - After cancellation (``async with`` guarantees release on any exit).
    The permit is NOT held during retry backoff sleeps so other requests can
    proceed while this one waits.

    Parameters
    ----------
    coro_factory : () -> Coroutine
        A callable that produces a fresh coroutine on each attempt.
    label : str
        Human-readable label for log messages.

    Returns
    -------
    The return value of the coroutine.

    Raises
    ------
    CircuitBreakerOpenError
        If the circuit breaker is open at the time of the call.
    The last exception from the coroutine
        If all retry attempts are exhausted.
    """
    cfg = get_config()
    max_attempts: int = cfg.RETRY_MAX_ATTEMPTS
    backoff_base: float = cfg.RETRY_BACKOFF_BASE_S

    # Fast-fail if breaker is open before we start.
    _breaker.check()

    sem = get_api_semaphore()
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            # Acquire semaphore for this individual attempt only.
            # Released automatically on any exit (success, exception, cancel).
            async with sem:
                result = await coro_factory()
            # Logical success — record once and return.
            _breaker.record_success()
            return result
        except CircuitBreakerOpenError:
            # Propagate immediately — never retry a breaker rejection.
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                wait = backoff_base * (2 ** (attempt - 1))
                log.warning(
                    "%s: attempt %d/%d failed (%s). Retrying in %.2fs.",
                    label, attempt, max_attempts, type(exc).__name__, wait,
                )
                # Sleep outside the semaphore — permit already released above.
                await asyncio.sleep(wait)
            else:
                log.error(
                    "%s: all %d attempts exhausted. Last error: %s",
                    label, max_attempts, exc,
                )

    # All retries exhausted → record exactly ONE logical failure.
    _breaker.record_failure()
    assert last_exc is not None
    raise last_exc
