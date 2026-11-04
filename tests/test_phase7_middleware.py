"""
test_phase7_middleware.py — Unit tests for Task 7.2 + 7.3 middleware.

Covers:
  - run_in_threadpool: CPU-bound calls offloaded to thread pool
  - init/close_http_client: singleton creation and teardown
  - get_http_client / get_api_semaphore: raise before init, succeed after
  - _CircuitBreaker: CLOSED → OPEN on threshold, HALF_OPEN after cooldown
  - api_call_with_retry: success, retry, exhaustion, breaker fast-fail
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── run_in_threadpool ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_in_threadpool_executes_sync_function():
    from graphrag_lite.api.middleware import run_in_threadpool
    result = await run_in_threadpool(lambda x, y: x + y, 3, 4)
    assert result == 7


@pytest.mark.asyncio
async def test_run_in_threadpool_returns_correct_type():
    from graphrag_lite.api.middleware import run_in_threadpool
    result = await run_in_threadpool(str.upper, "hello")
    assert result == "HELLO"


@pytest.mark.asyncio
async def test_run_in_threadpool_propagates_exception():
    from graphrag_lite.api.middleware import run_in_threadpool
    with pytest.raises(ValueError, match="boom"):
        await run_in_threadpool(lambda: (_ for _ in ()).throw(ValueError("boom")))


@pytest.mark.asyncio
async def test_run_in_threadpool_does_not_block_event_loop():
    """Two threadpool tasks should run concurrently (wall time < sum of sleeps)."""
    import time as _time
    from graphrag_lite.api.middleware import run_in_threadpool

    def slow():
        _time.sleep(0.1)
        return 1

    t0 = _time.monotonic()
    results = await asyncio.gather(run_in_threadpool(slow), run_in_threadpool(slow))
    elapsed = _time.monotonic() - t0
    assert results == [1, 1]
    # If truly concurrent they finish in ~0.1s; serial would take ~0.2s
    assert elapsed < 0.18, f"Tasks appear serialised: elapsed={elapsed:.3f}s"


# ── HTTP client lifecycle ─────────────────────────────────────────────────────

def test_get_http_client_raises_before_init():
    """get_http_client() must raise RuntimeError if init_http_client() was not called."""
    import graphrag_lite.api.middleware as mw
    # Save and temporarily clear the singleton
    original = mw._http_client
    mw._http_client = None
    try:
        with pytest.raises(RuntimeError, match="not initialised"):
            mw.get_http_client()
    finally:
        mw._http_client = original


def test_get_api_semaphore_raises_before_init():
    import graphrag_lite.api.middleware as mw
    original = mw._api_semaphore
    mw._api_semaphore = None
    try:
        with pytest.raises(RuntimeError, match="not initialised"):
            mw.get_api_semaphore()
    finally:
        mw._api_semaphore = original


@pytest.mark.asyncio
async def test_init_and_close_http_client():
    """init_http_client creates client + semaphore; close_http_client tears down."""
    import graphrag_lite.api.middleware as mw
    # Save originals
    orig_client = mw._http_client
    orig_sem = mw._api_semaphore
    try:
        mw._http_client = None
        mw._api_semaphore = None
        mw.init_http_client()
        assert mw._http_client is not None
        assert mw._api_semaphore is not None
        # semaphore limit from config
        from graphrag_lite.config import get_config
        cfg = get_config()
        # asyncio.Semaphore._value is the implementation detail; use internal
        assert mw._api_semaphore._value == cfg.STRESS_SEMAPHORE_LIMIT
        await mw.close_http_client()
        assert mw._http_client is None
        assert mw._api_semaphore is None
    finally:
        mw._http_client = orig_client
        mw._api_semaphore = orig_sem


# ── shared fixture: semaphore for api_call_with_retry tests ──────────────────
# api_call_with_retry now calls get_api_semaphore() on every invocation.
# Tests that don't go through the full lifespan must install a real semaphore.

@pytest.fixture()
def install_semaphore():
    """Install a test-scoped asyncio.Semaphore as the module singleton."""
    import graphrag_lite.api.middleware as mw
    orig = mw._api_semaphore
    # Create a real semaphore with a generous limit so tests don't deadlock
    mw._api_semaphore = asyncio.Semaphore(100)
    yield mw._api_semaphore
    mw._api_semaphore = orig



def _fresh_breaker(threshold: int = 3, cooldown: float = 60.0):
    """Return a fresh _CircuitBreaker with a small threshold for testing."""
    from graphrag_lite.api.middleware import _CircuitBreaker
    b = _CircuitBreaker.__new__(_CircuitBreaker)
    b._threshold = threshold
    b._cooldown_s = cooldown
    b._state = _CircuitBreaker._CLOSED
    b._consecutive_failures = 0
    b._opened_at = 0.0
    return b


def test_breaker_starts_closed():
    b = _fresh_breaker()
    assert b.state == "CLOSED"


def test_breaker_opens_after_threshold():
    b = _fresh_breaker(threshold=3)
    b.record_failure()
    b.record_failure()
    assert b.state == "CLOSED"
    b.record_failure()   # 3rd — should open
    assert b.state == "OPEN"


def test_breaker_check_raises_when_open():
    from graphrag_lite.api.middleware import CircuitBreakerOpenError
    b = _fresh_breaker(threshold=1)
    b.record_failure()
    assert b.state == "OPEN"
    with pytest.raises(CircuitBreakerOpenError):
        b.check()


def test_breaker_success_resets_counter():
    b = _fresh_breaker(threshold=3)
    b.record_failure()
    b.record_failure()
    b.record_success()
    assert b._consecutive_failures == 0
    assert b.state == "CLOSED"


def test_breaker_transitions_to_half_open_after_cooldown():
    # Use a real cooldown so the breaker stays OPEN until we manually expire it.
    b = _fresh_breaker(threshold=1, cooldown=9999.0)
    b.record_failure()
    # With cooldown=9999s it should stay OPEN immediately after failure
    assert b._state == "OPEN"
    # Simulate cooldown expiry by back-dating _opened_at
    b._opened_at = time.monotonic() - 10000.0
    # Now accessing .state should trigger HALF_OPEN transition
    assert b.state == "HALF_OPEN"


def test_breaker_closes_after_success_in_half_open():
    b = _fresh_breaker(threshold=1, cooldown=0.0)
    b.record_failure()
    b._opened_at = time.monotonic() - 1.0
    assert b.state == "HALF_OPEN"
    b.record_success()
    assert b.state == "CLOSED"


def test_breaker_reopens_on_failure_in_half_open():
    b = _fresh_breaker(threshold=1, cooldown=9999.0)
    b.record_failure()                              # opens with fresh cooldown
    assert b._state == "OPEN"
    b._opened_at = time.monotonic() - 10000.0       # cooldown elapsed → HALF_OPEN
    assert b.state == "HALF_OPEN"
    b.record_failure()                              # fails in HALF_OPEN → re-opens
    assert b._state == "OPEN"


def test_breaker_check_passes_when_closed():
    b = _fresh_breaker()
    b.check()  # should not raise


# ── api_call_with_retry ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retry_succeeds_on_first_attempt(install_semaphore):
    from graphrag_lite.api.middleware import api_call_with_retry, _breaker, _CircuitBreaker
    # Reset breaker
    _breaker._state = _CircuitBreaker._CLOSED
    _breaker._consecutive_failures = 0

    call_count = 0
    async def coro():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await api_call_with_retry(lambda: coro())
    assert result == "ok"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_retries_on_failure_and_succeeds(install_semaphore):
    """Fails twice then succeeds on 3rd attempt."""
    from graphrag_lite.api.middleware import api_call_with_retry, _breaker, _CircuitBreaker
    _breaker._state = _CircuitBreaker._CLOSED
    _breaker._consecutive_failures = 0

    call_count = 0
    async def coro():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("transient")
        return "recovered"

    # Patch sleep so the test is fast
    with patch("graphrag_lite.api.middleware.asyncio.sleep", new_callable=AsyncMock):
        result = await api_call_with_retry(lambda: coro())
    assert result == "recovered"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_raises_after_exhaustion(install_semaphore):
    """All RETRY_MAX_ATTEMPTS attempts fail → raises the last exception."""
    from graphrag_lite.api.middleware import api_call_with_retry, _breaker, _CircuitBreaker, get_config
    from graphrag_lite.config import get_config
    _breaker._state = _CircuitBreaker._CLOSED
    _breaker._consecutive_failures = 0

    cfg = get_config()
    call_count = 0

    async def coro():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("always fails")

    with patch("graphrag_lite.api.middleware.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(RuntimeError, match="always fails"):
            await api_call_with_retry(lambda: coro())
    assert call_count == cfg.RETRY_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_retry_fast_fails_when_breaker_open():
    """When circuit breaker is OPEN, api_call_with_retry must raise immediately."""
    from graphrag_lite.api.middleware import (
        api_call_with_retry, _breaker, _CircuitBreaker, CircuitBreakerOpenError
    )
    _breaker._state = _CircuitBreaker._OPEN
    _breaker._opened_at = time.monotonic()   # freshly opened
    _breaker._cooldown_s = 9999.0            # won't expire

    call_count = 0
    async def coro():
        nonlocal call_count
        call_count += 1
        return "should not be called"

    with pytest.raises(CircuitBreakerOpenError):
        await api_call_with_retry(lambda: coro())
    assert call_count == 0    # never called


@pytest.mark.asyncio
async def test_retry_opens_breaker_after_threshold_failures(install_semaphore):
    """Exactly threshold consecutive failed logical requests opens the breaker.
    Each api_call_with_retry call that exhausts all retries counts as ONE failure."""
    from graphrag_lite.api.middleware import (
        api_call_with_retry, _breaker, _CircuitBreaker, CircuitBreakerOpenError
    )
    from graphrag_lite.config import get_config
    cfg = get_config()
    _breaker._state = _CircuitBreaker._CLOSED
    _breaker._consecutive_failures = 0

    # Each call exhausts all retries → records one logical failure.
    # After exactly threshold calls the breaker opens.
    async def always_fail():
        raise ConnectionError("hard fail")

    with patch("graphrag_lite.api.middleware.asyncio.sleep", new_callable=AsyncMock):
        for _ in range(cfg.CIRCUIT_BREAKER_FAILURE_THRESHOLD):
            try:
                await api_call_with_retry(lambda: always_fail())
            except Exception:
                pass

    assert _breaker.state == "OPEN"


# ── P2-1 regression: semaphore enforcement ────────────────────────────────────

@pytest.mark.asyncio
async def test_semaphore_acquired_during_attempt(install_semaphore):
    """api_call_with_retry must hold the semaphore while the coroutine runs."""
    import graphrag_lite.api.middleware as mw
    sem = mw._api_semaphore
    # Count how many times the semaphore was held at the moment of execution
    acquired_counts: list[int] = []

    async def probe_coro():
        # Semaphore value decrements by 1 when held; 100-1=99 means it's acquired
        acquired_counts.append(100 - sem._value)
        return "ok"

    from graphrag_lite.api.middleware import api_call_with_retry, _breaker, _CircuitBreaker
    _breaker._state = _CircuitBreaker._CLOSED
    _breaker._consecutive_failures = 0

    await api_call_with_retry(lambda: probe_coro())
    assert acquired_counts == [1], f"Semaphore should be held during execution: {acquired_counts}"


@pytest.mark.asyncio
async def test_semaphore_released_after_exception(install_semaphore):
    """The semaphore permit must be released even when the coroutine raises."""
    import graphrag_lite.api.middleware as mw
    sem = mw._api_semaphore
    initial_value = sem._value  # should be 100

    async def failing_coro():
        raise RuntimeError("boom")

    from graphrag_lite.api.middleware import api_call_with_retry, _breaker, _CircuitBreaker
    _breaker._state = _CircuitBreaker._CLOSED
    _breaker._consecutive_failures = 0

    with patch("graphrag_lite.api.middleware.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(RuntimeError):
            await api_call_with_retry(lambda: failing_coro())

    assert sem._value == initial_value, (
        f"Semaphore not fully released after exception: "
        f"expected {initial_value}, got {sem._value}"
    )


@pytest.mark.asyncio
async def test_semaphore_not_held_during_backoff_sleep(install_semaphore):
    """
    The semaphore must be released before the backoff sleep so other
    concurrent calls can proceed during the wait.
    """
    import graphrag_lite.api.middleware as mw
    sem = mw._api_semaphore
    initial_value = sem._value

    sleep_calls_semaphore_value: list[int] = []

    async def mock_sleep(delay: float):
        # Capture the semaphore value AT THE MOMENT of sleep
        sleep_calls_semaphore_value.append(sem._value)

    attempt_count = 0

    async def fail_then_succeed():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 2:
            raise ConnectionError("first attempt fails")
        return "ok"

    from graphrag_lite.api.middleware import api_call_with_retry, _breaker, _CircuitBreaker
    _breaker._state = _CircuitBreaker._CLOSED
    _breaker._consecutive_failures = 0

    with patch("graphrag_lite.api.middleware.asyncio.sleep", side_effect=mock_sleep):
        result = await api_call_with_retry(lambda: fail_then_succeed())

    assert result == "ok"
    # During the sleep the semaphore should be fully released (back to initial)
    assert sleep_calls_semaphore_value, "asyncio.sleep was never called"
    for val in sleep_calls_semaphore_value:
        assert val == initial_value, (
            f"Semaphore still held during backoff sleep: "
            f"value={val}, expected={initial_value}"
        )


@pytest.mark.asyncio
async def test_concurrent_calls_respect_semaphore_limit():
    """
    When the semaphore limit is N, at most N concurrent attempts can execute
    simultaneously. Excess attempts must wait until a permit is free.

    We set limit=2 and fire 5 concurrent requests to prove the constraint.
    """
    import graphrag_lite.api.middleware as mw
    orig_sem = mw._api_semaphore
    limit = 2
    mw._api_semaphore = asyncio.Semaphore(limit)

    max_concurrent_seen = 0
    in_flight = 0

    async def counting_coro():
        nonlocal in_flight, max_concurrent_seen
        in_flight += 1
        if in_flight > max_concurrent_seen:
            max_concurrent_seen = in_flight
        await asyncio.sleep(0.01)  # brief real sleep to allow overlap
        in_flight -= 1
        return "ok"

    from graphrag_lite.api.middleware import api_call_with_retry, _breaker, _CircuitBreaker
    _breaker._state = _CircuitBreaker._CLOSED
    _breaker._consecutive_failures = 0

    try:
        tasks = [
            asyncio.create_task(api_call_with_retry(lambda: counting_coro()))
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks)
    finally:
        mw._api_semaphore = orig_sem

    assert all(r == "ok" for r in results)
    assert max_concurrent_seen <= limit, (
        f"Semaphore violated: {max_concurrent_seen} concurrent calls observed, "
        f"limit is {limit}"
    )
    print(f"\n✓ Max concurrent in-flight: {max_concurrent_seen} (limit={limit})")


# ── P2-2 regression: per-logical-request breaker semantics ───────────────────

@pytest.mark.asyncio
async def test_one_logical_request_counts_as_one_failure(install_semaphore):
    """
    A request that exhausts all retry attempts must record exactly ONE
    failure on the circuit breaker (not one per attempt).
    """
    from graphrag_lite.api.middleware import api_call_with_retry, _breaker, _CircuitBreaker
    _breaker._state = _CircuitBreaker._CLOSED
    _breaker._consecutive_failures = 0

    async def always_fail():
        raise ConnectionError("fail")

    with patch("graphrag_lite.api.middleware.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(ConnectionError):
            await api_call_with_retry(lambda: always_fail())

    assert _breaker._consecutive_failures == 1, (
        f"Expected 1 failure after one exhausted request, "
        f"got {_breaker._consecutive_failures}"
    )


@pytest.mark.asyncio
async def test_four_failed_requests_leave_breaker_closed(install_semaphore):
    """
    With threshold=5, four consecutive failed logical requests must leave
    the breaker CLOSED (the fifth would open it).
    """
    from graphrag_lite.api.middleware import api_call_with_retry, _breaker, _CircuitBreaker
    from graphrag_lite.config import get_config
    cfg = get_config()
    assert cfg.CIRCUIT_BREAKER_FAILURE_THRESHOLD == 5, "Test assumes threshold=5"
    _breaker._state = _CircuitBreaker._CLOSED
    _breaker._consecutive_failures = 0

    async def always_fail():
        raise ConnectionError("fail")

    with patch("graphrag_lite.api.middleware.asyncio.sleep", new_callable=AsyncMock):
        for _ in range(4):
            with pytest.raises(ConnectionError):
                await api_call_with_retry(lambda: always_fail())

    assert _breaker._consecutive_failures == 4
    assert _breaker._state == _CircuitBreaker._CLOSED


@pytest.mark.asyncio
async def test_fifth_failed_request_opens_breaker(install_semaphore):
    """The fifth consecutive failed logical request opens the breaker."""
    from graphrag_lite.api.middleware import api_call_with_retry, _breaker, _CircuitBreaker
    from graphrag_lite.config import get_config
    cfg = get_config()
    _breaker._state = _CircuitBreaker._CLOSED
    _breaker._consecutive_failures = 0

    async def always_fail():
        raise ConnectionError("fail")

    with patch("graphrag_lite.api.middleware.asyncio.sleep", new_callable=AsyncMock):
        for _ in range(cfg.CIRCUIT_BREAKER_FAILURE_THRESHOLD):
            with pytest.raises(ConnectionError):
                await api_call_with_retry(lambda: always_fail())

    assert _breaker._state == _CircuitBreaker._OPEN


@pytest.mark.asyncio
async def test_success_resets_failure_counter(install_semaphore):
    """A successful logical request resets the consecutive-failure count."""
    from graphrag_lite.api.middleware import api_call_with_retry, _breaker, _CircuitBreaker
    _breaker._state = _CircuitBreaker._CLOSED
    _breaker._consecutive_failures = 3  # pre-load some failures

    async def succeed():
        return "ok"

    result = await api_call_with_retry(lambda: succeed())
    assert result == "ok"
    assert _breaker._consecutive_failures == 0


@pytest.mark.asyncio
async def test_open_breaker_fast_fails_without_outbound_call(install_semaphore):
    """An OPEN breaker must reject the call immediately, never touching the coro."""
    from graphrag_lite.api.middleware import (
        api_call_with_retry, _breaker, _CircuitBreaker, CircuitBreakerOpenError,
    )
    _breaker._state = _CircuitBreaker._OPEN
    _breaker._opened_at = time.monotonic()
    _breaker._cooldown_s = 9999.0

    called = False

    async def should_not_run():
        nonlocal called
        called = True
        return "reached"

    with pytest.raises(CircuitBreakerOpenError):
        await api_call_with_retry(lambda: should_not_run())

    assert not called, "Coro must not execute when breaker is OPEN"


@pytest.mark.asyncio
async def test_half_open_success_closes_breaker(install_semaphore):
    """A successful probe in HALF_OPEN state closes the breaker."""
    from graphrag_lite.api.middleware import api_call_with_retry, _breaker, _CircuitBreaker
    _breaker._state = _CircuitBreaker._HALF_OPEN
    _breaker._consecutive_failures = 5

    async def succeed():
        return "probe ok"

    result = await api_call_with_retry(lambda: succeed())
    assert result == "probe ok"
    assert _breaker._state == _CircuitBreaker._CLOSED
    assert _breaker._consecutive_failures == 0


@pytest.mark.asyncio
async def test_half_open_failure_reopens_breaker(install_semaphore):
    """A failed probe in HALF_OPEN state re-opens the breaker."""
    from graphrag_lite.api.middleware import (
        api_call_with_retry, _breaker, _CircuitBreaker,
    )
    _breaker._state = _CircuitBreaker._HALF_OPEN
    _breaker._consecutive_failures = 5  # already at threshold

    async def fail_probe():
        raise ConnectionError("probe fail")

    with patch("graphrag_lite.api.middleware.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(ConnectionError):
            await api_call_with_retry(lambda: fail_probe())

    assert _breaker._state == _CircuitBreaker._OPEN


@pytest.mark.asyncio
async def test_concurrent_requests_preserve_threshold(install_semaphore):
    """
    Under concurrent load the breaker threshold is counted in whole logical
    requests, not in individual attempt noise.  Five concurrent requests all
    failing must produce exactly five breaker failures (threshold=5), opening
    the breaker — not more, not fewer.
    """
    from graphrag_lite.api.middleware import api_call_with_retry, _breaker, _CircuitBreaker
    from graphrag_lite.config import get_config
    cfg = get_config()
    _breaker._state = _CircuitBreaker._CLOSED
    _breaker._consecutive_failures = 0

    async def always_fail():
        raise ConnectionError("concurrent fail")

    with patch("graphrag_lite.api.middleware.asyncio.sleep", new_callable=AsyncMock):
        tasks = [
            asyncio.create_task(api_call_with_retry(lambda: always_fail()))
            for _ in range(cfg.CIRCUIT_BREAKER_FAILURE_THRESHOLD)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # All should have raised
    assert all(isinstance(r, Exception) for r in results)
    # Breaker must be open — exactly threshold failures recorded
    assert _breaker.state == "OPEN", (
        f"Breaker should be OPEN after {cfg.CIRCUIT_BREAKER_FAILURE_THRESHOLD} "
        f"concurrent failed requests, state={_breaker.state}"
    )
