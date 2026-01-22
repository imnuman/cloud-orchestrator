"""
Security middleware for hardening API responses.
"""

import logging
import re
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds security headers to all responses.

    Headers added:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Strict-Transport-Security: max-age=31536000; includeSubDomains
    - Content-Security-Policy: default-src 'self'
    - Referrer-Policy: strict-origin-when-cross-origin
    - Permissions-Policy: geolocation=(), microphone=(), camera=()
    """

    def __init__(
        self,
        app,
        enable_hsts: bool = True,
        enable_csp: bool = True,
        csp_policy: Optional[str] = None,
    ):
        """
        Initialize security headers middleware.

        Args:
            app: FastAPI app
            enable_hsts: Enable HTTP Strict Transport Security
            enable_csp: Enable Content Security Policy
            csp_policy: Custom CSP policy (default: self only)
        """
        super().__init__(app)
        self.enable_hsts = enable_hsts
        self.enable_csp = enable_csp
        self.csp_policy = csp_policy or "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # XSS protection (legacy but still useful)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), payment=()"
        )

        # HSTS (only for HTTPS)
        if self.enable_hsts:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        # Content Security Policy
        if self.enable_csp:
            response.headers["Content-Security-Policy"] = self.csp_policy

        return response


class InputSanitizer:
    """
    Utility class for sanitizing user input.
    """

    # Patterns for potentially dangerous input
    SQL_INJECTION_PATTERNS = [
        r"(\s|^)(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\s",
        r"--",
        r";\s*(SELECT|INSERT|UPDATE|DELETE|DROP)",
        r"'.*OR.*'",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
    ]

    @classmethod
    def is_safe_string(cls, value: str) -> bool:
        """
        Check if a string is safe from common injection attacks.

        Args:
            value: String to check

        Returns:
            True if safe, False if potentially dangerous
        """
        value_upper = value.upper()

        # Check SQL injection patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_upper, re.IGNORECASE):
                logger.warning(f"Potential SQL injection detected: {value[:50]}...")
                return False

        # Check XSS patterns
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Potential XSS detected: {value[:50]}...")
                return False

        return True

    @classmethod
    def sanitize_html(cls, value: str) -> str:
        """
        Remove HTML tags from a string.

        Args:
            value: String to sanitize

        Returns:
            Sanitized string
        """
        # Remove HTML tags
        clean = re.sub(r"<[^>]+>", "", value)

        # Remove common XSS vectors
        clean = re.sub(r"javascript:", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"on\w+\s*=", "", clean, flags=re.IGNORECASE)

        return clean

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize a filename to prevent directory traversal.

        Args:
            filename: Filename to sanitize

        Returns:
            Sanitized filename
        """
        # Remove path separators
        filename = filename.replace("/", "_").replace("\\", "_")

        # Remove .. to prevent directory traversal
        filename = filename.replace("..", "_")

        # Only allow safe characters
        filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)

        return filename


class RequestValidator:
    """
    Utility class for validating incoming requests.
    """

    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB default
    ALLOWED_CONTENT_TYPES = [
        "application/json",
        "application/x-www-form-urlencoded",
        "multipart/form-data",
    ]

    @classmethod
    def validate_content_length(cls, request: Request, max_length: Optional[int] = None) -> bool:
        """
        Validate that content length is within limits.

        Args:
            request: FastAPI request
            max_length: Maximum allowed content length

        Returns:
            True if valid, False if too large
        """
        max_len = max_length or cls.MAX_CONTENT_LENGTH
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                length = int(content_length)
                if length > max_len:
                    logger.warning(
                        f"Content length {length} exceeds max {max_len}"
                    )
                    return False
            except ValueError:
                return False

        return True

    @classmethod
    def validate_content_type(
        cls,
        request: Request,
        allowed_types: Optional[list[str]] = None,
    ) -> bool:
        """
        Validate content type is allowed.

        Args:
            request: FastAPI request
            allowed_types: List of allowed content types

        Returns:
            True if valid content type
        """
        allowed = allowed_types or cls.ALLOWED_CONTENT_TYPES
        content_type = request.headers.get("content-type", "")

        # Extract base content type (without charset, etc.)
        base_type = content_type.split(";")[0].strip().lower()

        return base_type in allowed or not content_type
