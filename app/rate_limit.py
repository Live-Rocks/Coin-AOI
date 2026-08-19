"""Small, process-local request rate limiter for the public portfolio demo."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock


class FixedWindowRateLimiter:
    """Allow a bounded number of requests per client in a rolling time window."""

    def __init__(self, limit: int, window_seconds: float) -> None:
        if limit < 1:
            raise ValueError("limit must be at least one.")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, client_key: str) -> bool:
        """Record a request when within the window limit."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            requests = self._requests[client_key]
            while requests and requests[0] <= cutoff:
                requests.popleft()
            if len(requests) >= self.limit:
                return False
            requests.append(now)
            return True
