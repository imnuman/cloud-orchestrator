"""
Billing models for transactions and usage tracking.
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import String, Float, Integer, Enum, ForeignKey, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from brain.models.base import Base

if TYPE_CHECKING:
    from brain.models.user import User


class TransactionType(str, PyEnum):
    """Types of billing transactions."""
    DEPOSIT = "deposit"  # User adds funds
    CHARGE = "charge"  # Usage charge
    REFUND = "refund"  # Refund to user
    CREDIT = "credit"  # Promotional credit
    PAYOUT = "payout"  # Provider payout (Phase 2)
    ADJUSTMENT = "adjustment"  # Manual adjustment


class PaymentMethod(str, PyEnum):
    """Payment methods."""
    STRIPE = "stripe"
    CRYPTO = "crypto"
    MANUAL = "manual"
    PROMOTIONAL = "promotional"


class Transaction(Base):
    """Financial transaction record."""

    __tablename__ = "transactions"

    # User
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)

    # Transaction details
    type: Mapped[TransactionType] = mapped_column(Enum(TransactionType), index=True)
    amount: Mapped[float] = mapped_column(Float)  # Positive = credit, negative = debit
    balance_after: Mapped[float] = mapped_column(Float)  # Balance after transaction

    # Description
    description: Mapped[str] = mapped_column(String(512))
    reference_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )  # External reference (Stripe ID, etc.)

    # Related entities
    pod_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("pods.id"),
        nullable=True,
    )

    # Payment info
    payment_method: Mapped[Optional[PaymentMethod]] = mapped_column(
        Enum(PaymentMethod),
        nullable=True,
    )

    # Extra data
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="transactions")

    def __repr__(self) -> str:
        return f"<Transaction {self.type.value}: {self.amount}>"


class UsageRecord(Base):
    """Detailed usage record for billing."""

    __tablename__ = "usage_records"

    # References
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    pod_id: Mapped[str] = mapped_column(ForeignKey("pods.id"), index=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id"), index=True)

    # Resource info
    gpu_type: Mapped[str] = mapped_column(String(255))
    gpu_count: Mapped[int] = mapped_column(Integer, default=1)

    # Timing
    period_start: Mapped[datetime] = mapped_column(index=True)
    period_end: Mapped[datetime] = mapped_column()
    duration_seconds: Mapped[int] = mapped_column(Integer)

    # Pricing
    hourly_rate: Mapped[float] = mapped_column(Float)
    amount_charged: Mapped[float] = mapped_column(Float)
    provider_cost: Mapped[float] = mapped_column(Float, default=0.0)

    # Status
    is_billed: Mapped[bool] = mapped_column(default=False)
    transaction_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("transactions.id"),
        nullable=True,
    )

    def __repr__(self) -> str:
        return f"<UsageRecord {self.pod_id}: {self.duration_seconds}s>"
