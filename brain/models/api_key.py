"""
API Key model for managing multiple API keys per user.

Features:
- Multiple keys per user with custom names
- Scoped permissions (read, write, admin)
- Key expiration and rotation
- Usage tracking (last used, request count)
"""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import String, Boolean, Integer, ForeignKey, DateTime, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from brain.models.base import Base

if TYPE_CHECKING:
    from brain.models.user import User


class APIKeyScope(str, Enum):
    """API key permission scopes."""

    # Read-only access
    READ = "read"

    # Read and write access (deploy pods, manage deployments)
    WRITE = "write"

    # Full access including billing and account management
    ADMIN = "admin"

    # Model inference only (for deployed models)
    INFERENCE = "inference"


class APIKey(Base):
    """
    API Key for programmatic access.

    Users can create multiple API keys with different permissions
    and track their usage.
    """

    __tablename__ = "api_keys"

    # Key identification
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # The key itself (hashed for storage, only shown once on creation)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Key prefix for identification (first 8 chars, e.g., "sk_live_a1b2c3d4")
    key_prefix: Mapped[str] = mapped_column(String(20), nullable=False)

    # Owner
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Permissions
    scopes: Mapped[list] = mapped_column(JSON, default=["read", "write"])

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Expiration (optional)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Usage tracking
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_used_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    request_count: Mapped[int] = mapped_column(Integer, default=0)

    # Rate limiting (optional per-key limits)
    rate_limit_per_minute: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rate_limit_per_hour: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Restrictions (optional)
    allowed_ips: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    allowed_origins: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="api_keys", lazy="selectin")

    def __repr__(self) -> str:
        return f"<APIKey {self.key_prefix}... ({self.name})>"

    def has_scope(self, scope: str) -> bool:
        """Check if key has a specific scope."""
        if "admin" in self.scopes:
            return True  # Admin has all scopes
        return scope in self.scopes

    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """Check if key is valid (active and not expired)."""
        return self.is_active and not self.is_expired()

    def is_ip_allowed(self, ip: str) -> bool:
        """Check if IP is allowed for this key."""
        if self.allowed_ips is None or len(self.allowed_ips) == 0:
            return True  # No restrictions
        return ip in self.allowed_ips
