from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request

from app.config import Settings


class InMemoryRateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = max(1, requests_per_minute)
        self.window_seconds = 60.0
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= self.requests_per_minute:
                retry_after = max(1, int(self.window_seconds - (now - bucket[0])))
                return False, retry_after

            bucket.append(now)
            return True, 0


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip() or "unknown"
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


def enforce_api_key(provided_api_key: str | None, settings: Settings) -> None:
    expected = settings.app_api_key.strip()
    if not expected:
        return

    provided = (provided_api_key or "").strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def enforce_rate_limit(
    request: Request,
    limiter: InMemoryRateLimiter,
    provided_api_key: str | None = None,
) -> None:
    api_key_part = (provided_api_key or "").strip() or "anonymous"
    key = f"{request.url.path}:{_client_ip(request)}:{api_key_part}"
    allowed, retry_after = limiter.allow(key)
    if allowed:
        return

    raise HTTPException(
        status_code=429,
        detail="Rate limit exceeded. Try again later.",
        headers={"Retry-After": str(retry_after)},
    )
