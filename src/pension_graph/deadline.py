"""Cooperative request deadlines; no detached worker is created on timeout.

Every blocking boundary must use the remaining budget and check again afterward.
This cannot forcibly interrupt a non-cooperative C extension or an already-running
thread. HTTP read timeouts are per chunk, not an end-to-end wall-clock guarantee.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from functools import wraps
from threading import Event
from time import monotonic, sleep
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


class RequestDeadlineExceeded(TimeoutError):
    """A safe public exception type; contains no prompt, secret or provider body."""


@dataclass
class Deadline:
    expires_at: float
    parent: Deadline | None = field(default=None, repr=False)
    _cancelled: Event = field(default_factory=Event, repr=False)

    def remaining_seconds(self) -> float:
        if self._cancelled.is_set():
            raise RequestDeadlineExceeded("Request budget expired or cancelled")
        remaining = self.expires_at - monotonic()
        if self.parent is not None:
            remaining = min(remaining, self.parent.remaining_seconds())
        if remaining <= 0:
            raise RequestDeadlineExceeded("Request budget expired or cancelled")
        return remaining

    def cancel(self) -> None:
        self._cancelled.set()


_current_deadline: ContextVar[Deadline | None] = ContextVar("request_deadline", default=None)


def get_deadline() -> Deadline | None:
    return _current_deadline.get()


@contextmanager
def request_deadline(seconds: float) -> Iterator[Deadline]:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Request budget must be a positive finite number")
    parent = get_deadline()
    deadline = Deadline(expires_at=monotonic() + seconds, parent=parent)
    token = _current_deadline.set(deadline)
    try:
        check_deadline()
        yield deadline
        check_deadline()
    finally:
        # Workers explicitly bound to this context must stop at their next boundary,
        # including if the caller exits early after an unrelated exception.
        deadline.cancel()
        _current_deadline.reset(token)


def check_deadline() -> None:
    deadline = get_deadline()
    if deadline is not None:
        deadline.remaining_seconds()


def remaining_timeout(default_seconds: float) -> float:
    """Cap a blocking operation's timeout without extending the request deadline."""
    if not math.isfinite(default_seconds) or default_seconds <= 0:
        raise ValueError("Operation timeout must be a positive finite number")
    deadline = get_deadline()
    return min(default_seconds, deadline.remaining_seconds()) if deadline else default_seconds


def sleep_with_deadline(seconds: float) -> None:
    """Skip a retry whose backoff alone exhausts the budget; never sleep past it."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Retry delay must be a non-negative finite number")
    deadline = get_deadline()
    if deadline is None:
        sleep(seconds)
        return
    if seconds >= deadline.remaining_seconds():
        raise RequestDeadlineExceeded("Retry delay exceeds remaining request budget")
    if deadline._cancelled.wait(timeout=seconds):
        raise RequestDeadlineExceeded("Request budget expired or cancelled")
    check_deadline()


def bind_deadline(function: Callable[P, T]) -> Callable[P, T]:
    """Capture context before executor.submit(); a fresh copy permits parallel calls."""
    context = copy_context()

    @wraps(function)
    def bound(*args: P.args, **kwargs: P.kwargs) -> T:
        def invoke() -> T:
            check_deadline()
            result = function(*args, **kwargs)
            check_deadline()
            return result

        return context.copy().run(invoke)

    return bound
