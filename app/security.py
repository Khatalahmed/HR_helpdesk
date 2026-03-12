from __future__ import annotations

import hmac
import logging
import sqlite3
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from fastapi import HTTPException, Request

from app.config import Settings


logger = logging.getLogger("snailcloud.security")


class RateLimiterProtocol(Protocol):
    def allow(self, key: str) -> tuple[bool, int]:
        ...


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


class SQLiteRateLimiter:
    """Shared limiter across multiple workers on the same host via SQLite."""

    def __init__(self, requests_per_minute: int, db_path: str):
        self.requests_per_minute = max(1, requests_per_minute)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=5, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL;")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rate_limit_hits (
                    client_key TEXT NOT NULL,
                    minute_bucket INTEGER NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY (client_key, minute_bucket)
                )
                """
            )

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.time()
        minute_bucket = int(now // 60)
        retry_after = max(1, int(60 - (now % 60)))

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM rate_limit_hits WHERE minute_bucket < ?",
                (minute_bucket - 1,),
            )
            row = connection.execute(
                "SELECT count FROM rate_limit_hits WHERE client_key = ? AND minute_bucket = ?",
                (key, minute_bucket),
            ).fetchone()

            current_count = int(row[0]) if row else 0
            if current_count >= self.requests_per_minute:
                connection.execute("COMMIT")
                return False, retry_after

            if row:
                connection.execute(
                    "UPDATE rate_limit_hits SET count = count + 1 WHERE client_key = ? AND minute_bucket = ?",
                    (key, minute_bucket),
                )
            else:
                connection.execute(
                    "INSERT INTO rate_limit_hits (client_key, minute_bucket, count) VALUES (?, ?, 1)",
                    (key, minute_bucket),
                )
            connection.execute("COMMIT")

        return True, 0


class RedisRateLimiter:
    """Distributed fixed-window limiter backed by Redis INCR + EXPIRE."""

    def __init__(self, requests_per_minute: int, redis_url: str, prefix: str):
        self.requests_per_minute = max(1, requests_per_minute)
        self.prefix = (prefix or "snailcloud:ratelimit").strip()

        try:
            import redis  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Redis backend selected but 'redis' package is not installed. "
                "Install dependencies from requirements.txt."
            ) from exc

        self._client = redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=1.0)
        self._client.ping()

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.time()
        minute_bucket = int(now // 60)
        retry_after = max(1, int(60 - (now % 60)))
        redis_key = f"{self.prefix}:{minute_bucket}:{key}"

        count_raw: Any = self._client.incr(redis_key)
        count = int(count_raw)
        if count == 1:
            self._client.expire(redis_key, 61)

        if count > self.requests_per_minute:
            return False, retry_after
        return True, 0


def build_rate_limiter(settings: Settings) -> RateLimiterProtocol:
    backend = settings.rate_limit_backend
    if backend == "memory":
        return InMemoryRateLimiter(settings.rate_limit_per_minute)
    if backend == "sqlite":
        return SQLiteRateLimiter(settings.rate_limit_per_minute, settings.rate_limit_storage_path)
    if backend == "redis":
        return RedisRateLimiter(
            settings.rate_limit_per_minute,
            settings.rate_limit_redis_url,
            settings.rate_limit_redis_prefix,
        )

    logger.warning("Unknown RATE_LIMIT_BACKEND=%s. Falling back to memory limiter.", backend)
    return InMemoryRateLimiter(settings.rate_limit_per_minute)


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
    limiter: RateLimiterProtocol,
    provided_api_key: str | None = None,
) -> None:
    api_key_part = (provided_api_key or "").strip() or "anonymous"
    key = f"{request.url.path}:{_client_ip(request)}:{api_key_part}"

    try:
        allowed, retry_after = limiter.allow(key)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Rate limiter unavailable.") from exc

    if allowed:
        return

    raise HTTPException(
        status_code=429,
        detail="Rate limit exceeded. Try again later.",
        headers={"Retry-After": str(retry_after)},
    )
