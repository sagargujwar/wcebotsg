"""
app/cache.py
────────────
Redis cache wrapper for query responses.

Design decisions:
  - Query keys are SHA-256 hashes of a *normalised* query string (lowercase,
    stripped, whitespace-collapsed) so semantically identical queries hit the
    same cache entry even if the user typed slightly differently.
  - All Redis errors are caught and logged; the app degrades gracefully to a
    cache-miss rather than crashing the request.
  - Rate-limit counters live in the same Redis instance under a separate key
    prefix ("rl:") so there is no extra dependency.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Optional

import redis

from app.config import get_settings

logger = logging.getLogger(__name__)

_CACHE_KEY_PREFIX = "wcebot:cache:"
_RL_KEY_PREFIX = "wcebot:rl:"


def _normalise(query: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", query.strip().lower())


def _hash(normalised: str) -> str:
    return hashlib.sha256(normalised.encode()).hexdigest()


class RedisCache:
    """Thin wrapper around a Redis connection for response caching."""

    def __init__(self) -> None:
        settings = get_settings()
        self._ttl = settings.cache_ttl_seconds
        self._client: Optional[redis.Redis] = None
        try:
            self._client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._client.ping()
            logger.info("Redis connected: %s", settings.redis_url)
        except Exception as exc:
            logger.warning("Redis unavailable — cache disabled: %s", exc)
            self._client = None

    # ── Connectivity ──────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Return True if Redis is reachable."""
        if self._client is None:
            return False
        try:
            return self._client.ping()
        except Exception:
            return False

    # ── Response cache ────────────────────────────────────────────────────────

    def get(self, query: str) -> Optional[dict]:
        """Return cached response dict, or None on miss / Redis unavailable."""
        if self._client is None:
            return None
        key = _CACHE_KEY_PREFIX + _hash(_normalise(query))
        try:
            raw = self._client.get(key)
            if raw:
                return json.loads(raw)
        except Exception as exc:
            logger.warning("Cache GET error: %s", exc)
        return None

    def set(self, query: str, response: dict, ttl: Optional[int] = None) -> None:
        """Store a response in Redis with the configured TTL."""
        if self._client is None:
            return
        key = _CACHE_KEY_PREFIX + _hash(_normalise(query))
        try:
            self._client.setex(key, ttl or self._ttl, json.dumps(response))
        except Exception as exc:
            logger.warning("Cache SET error: %s", exc)

    # ── Rate limiting ─────────────────────────────────────────────────────────

    def check_rate_limit(self, ip: str, limit: int, window: int) -> bool:
        """
        Increment and check the per-IP rate-limit counter.

        Returns True if the request is allowed, False if the limit is exceeded.
        Fails open (returns True) if Redis is unavailable.
        """
        if self._client is None:
            return True  # degrade gracefully — allow the request
        key = _RL_KEY_PREFIX + ip
        try:
            pipe = self._client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            results = pipe.execute()
            count = results[0]
            return count <= limit
        except Exception as exc:
            logger.warning("Rate-limit check error: %s", exc)
            return True  # fail open
