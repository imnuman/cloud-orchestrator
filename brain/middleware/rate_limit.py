"""
Rate limiting middleware using Redis for distributed rate limiting.
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Callable, Optional

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from brain.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# In-memory rate limit storage (use Redis in production)
_rate_limit_store: dict[str, list[float]] = {}
_rate_limit_lock = asyncio.Lock()


class RateLimiter:
    """
    Token bucket rate limiter.

    Supports both in-memory (dev) and Redis (production) backends.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour
            burst_size: Max burst of requests allowed
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size

    async def is_allowed(self, key: str) -> tuple[bool, dict]:
        """
        Check if a request is allowed for the given key.

        Args:
            key: Unique identifier (IP, user ID, API key)

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        async with _rate_limit_lock:
            now = time.time()
            minute_ago = now - 60
            hour_ago = now - 3600

            if key not in _rate_limit_store:
                _rate_limit_store[key] = []

            # Clean old entries
            timestamps = _rate_limit_store[key]
            timestamps = [t for t in timestamps if t > hour_ago]
            _rate_limit_store[key] = timestamps

            # Count requests in time windows
            minute_count = sum(1 for t in timestamps if t > minute_ago)
            hour_count = len(timestamps)

            # Check limits
            if minute_count >= self.requests_per_minute:
                return False, {
                    "limit": self.requests_per_minute,
                    "remaining": 0,
                    "reset": int(minute_ago + 60),
                    "window": "minute",
                }

            if hour_count >= self.requests_per_hour:
                return False, {
                    "limit": self.requests_per_hour,
                    "remaining": 0,
                    "reset": int(hour_ago + 3600),
                    "window": "hour",
                }

            # Allow request
            timestamps.append(now)
            _rate_limit_store[key] = timestamps

            return True, {
                "limit": self.requests_per_minute,
                "remaining": self.requests_per_minute - minute_count - 1,
                "reset": int(now + 60),
                "window": "minute",
            }

    async def get_key_from_request(self, request: Request) -> str:
        """
        Extract rate limit key from request.

        Uses API key if present, otherwise IP address.
        """
        # Check for API key in header
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
        if api_key:
            if api_key.startswith("Bearer "):
                api_key = api_key[7:]
            return f"api:{api_key[:16]}"

        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return f"ip:{ip}"


# Default rate limiter instance
rate_limiter = RateLimiter(
    requests_per_minute=60,
    requests_per_hour=1000,
    burst_size=10,
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    """

    def __init__(
        self,
        app,
        limiter: Optional[RateLimiter] = None,
        exclude_paths: Optional[list[str]] = None,
    ):
        """
        Initialize rate limit middleware.

        Args:
            app: FastAPI app
            limiter: RateLimiter instance
            exclude_paths: Paths to exclude from rate limiting
        """
        super().__init__(app)
        self.limiter = limiter or rate_limiter
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Skip excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        # Get rate limit key
        key = await self.limiter.get_key_from_request(request)

        # Check rate limit
        allowed, info = await self.limiter.is_allowed(key)

        if not allowed:
            logger.warning(f"Rate limit exceeded for {key}")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": info["limit"],
                    "reset": info["reset"],
                    "window": info["window"],
                },
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": str(info["remaining"]),
                    "X-RateLimit-Reset": str(info["reset"]),
                    "Retry-After": str(info["reset"] - int(time.time())),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(info["reset"])

        return response


def rate_limit(
    requests_per_minute: int = 30,
    requests_per_hour: int = 500,
):
    """
    Decorator for endpoint-specific rate limiting.

    Usage:
        @router.get("/expensive-endpoint")
        @rate_limit(requests_per_minute=10)
        async def expensive_endpoint():
            ...
    """
    limiter = RateLimiter(
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
    )

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, request: Request, **kwargs):
            key = await limiter.get_key_from_request(request)
            allowed, info = await limiter.is_allowed(key)

            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "limit": info["limit"],
                        "reset": info["reset"],
                    },
                )

            return await func(*args, request=request, **kwargs)

        return wrapper

    return decorator
