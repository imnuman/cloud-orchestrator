"""
User model for authentication and account management.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import String, Float, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from brain.models.base import Base

if TYPE_CHECKING:
    from brain.models.pod import Pod
    from brain.models.billing import Transaction
    from brain.models.provider import Provider
    from brain.models.api_key import APIKey


class User(Base):
    """User account model."""

    __tablename__ = "users"

    # Authentication
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Account status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    is_provider: Mapped[bool] = mapped_column(Boolean, default=False)

    # Billing
    balance: Mapped[float] = mapped_column(Float, default=0.0)
    total_spent: Mapped[float] = mapped_column(Float, default=0.0)
    credit_limit: Mapped[float] = mapped_column(Float, default=0.0)

    # Stripe integration
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(
        String(255), unique=True, nullable=True, index=True
    )

    # Auto-refill settings
    auto_refill_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    auto_refill_threshold: Mapped[float] = mapped_column(Float, default=10.0)
    auto_refill_amount: Mapped[float] = mapped_column(Float, default=50.0)

    # Provider info (Phase 2)
    provider_earnings: Mapped[float] = mapped_column(Float, default=0.0)
    provider_payout_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # API access
    api_key: Mapped[Optional[str]] = mapped_column(String(64), unique=True, nullable=True, index=True)

    # Metadata
    last_login_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    verification_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Relationships
    pods: Mapped[list["Pod"]] = relationship("Pod", back_populates="user", lazy="selectin")
    transactions: Mapped[list["Transaction"]] = relationship(
        "Transaction", back_populates="user", lazy="selectin"
    )
    provider: Mapped[Optional["Provider"]] = relationship(
        "Provider", back_populates="user", uselist=False, lazy="selectin"
    )
    api_keys: Mapped[list["APIKey"]] = relationship(
        "APIKey", back_populates="user", lazy="selectin", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User {self.email}>"

    @property
    def available_balance(self) -> float:
        """Balance available for spending (balance + credit limit)."""
        return self.balance + self.credit_limit

    def can_afford(self, amount: float) -> bool:
        """Check if user can afford a charge."""
        return self.available_balance >= amount
