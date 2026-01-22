"""
Provider model for GPU providers (community members, crypto miners, etc.).
Phase 2: Community GPU Onboarding
"""

import secrets
from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import String, Float, Boolean, Integer, Enum, ForeignKey, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from brain.models.base import Base

if TYPE_CHECKING:
    from brain.models.user import User
    from brain.models.node import Node


class PayoutMethod(str, PyEnum):
    """Payout methods for providers."""
    CRYPTO = "crypto"  # USDC/USDT wallet address
    PAYPAL = "paypal"  # PayPal email


class PayoutStatus(str, PyEnum):
    """Status of a payout request."""
    PENDING = "pending"  # Requested, awaiting processing
    PROCESSING = "processing"  # Being processed
    COMPLETED = "completed"  # Payout sent
    FAILED = "failed"  # Failed to process
    CANCELLED = "cancelled"  # Cancelled by admin or provider


class VerificationLevel(int, PyEnum):
    """Provider verification levels."""
    NONE = 0  # Unverified
    EMAIL = 1  # Email verified
    IDENTITY = 2  # Identity verified
    BUSINESS = 3  # Business verified (highest tier)


def generate_provider_key() -> str:
    """Generate a unique provider key for install scripts."""
    return f"pk_{secrets.token_urlsafe(24)}"


class Provider(Base):
    """
    GPU Provider model for community/marketplace providers.

    A Provider is a user who lists their GPU hardware on the platform.
    Providers can have multiple nodes (GPU machines) and earn revenue
    when their nodes are used.

    Revenue Split Tiers:
    - Basic (0-1): Platform 20%, Provider 80%
    - Verified (2): Platform 15%, Provider 85%
    - Pro/Business (3): Platform 10%, Provider 90%
    """

    __tablename__ = "providers"

    # Link to User account
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id"),
        unique=True,
        index=True,
    )

    # Provider identity
    company_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    display_name: Mapped[str] = mapped_column(String(255))

    # Provider key for install scripts
    provider_key: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        default=generate_provider_key,
    )

    # Payout configuration
    payout_method: Mapped[PayoutMethod] = mapped_column(
        Enum(PayoutMethod),
        default=PayoutMethod.CRYPTO,
    )
    payout_address: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )  # Wallet address or PayPal email
    minimum_payout: Mapped[float] = mapped_column(Float, default=50.0)

    # Earnings tracking
    total_earnings: Mapped[float] = mapped_column(Float, default=0.0)
    pending_earnings: Mapped[float] = mapped_column(Float, default=0.0)  # Not yet settled
    available_balance: Mapped[float] = mapped_column(Float, default=0.0)  # Ready for payout
    total_paid_out: Mapped[float] = mapped_column(Float, default=0.0)

    # Usage statistics
    total_gpu_hours: Mapped[float] = mapped_column(Float, default=0.0)
    total_jobs_completed: Mapped[int] = mapped_column(Integer, default=0)

    # Verification
    verification_level: Mapped[VerificationLevel] = mapped_column(
        Enum(VerificationLevel),
        default=VerificationLevel.NONE,
    )
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verified_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_suspended: Mapped[bool] = mapped_column(Boolean, default=False)
    suspension_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Platform fee override (None = use tier default)
    custom_platform_fee_percent: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    # Contact info
    contact_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    contact_phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Extra data (custom settings, notes, etc.)
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="provider")
    nodes: Mapped[list["Node"]] = relationship(
        "Node",
        back_populates="provider",
        foreign_keys="Node.provider_user_id",
    )
    payouts: Mapped[list["Payout"]] = relationship("Payout", back_populates="provider")

    def __repr__(self) -> str:
        return f"<Provider {self.display_name} ({self.user_id})>"

    @property
    def platform_fee_percent(self) -> float:
        """
        Get the platform fee percentage based on verification level.

        Returns:
            Platform fee as a percentage (e.g., 20.0 for 20%)
        """
        if self.custom_platform_fee_percent is not None:
            return self.custom_platform_fee_percent

        # Fee tiers based on verification level
        fee_tiers = {
            VerificationLevel.NONE: 20.0,
            VerificationLevel.EMAIL: 20.0,
            VerificationLevel.IDENTITY: 15.0,
            VerificationLevel.BUSINESS: 10.0,
        }
        return fee_tiers.get(self.verification_level, 20.0)

    @property
    def provider_share_percent(self) -> float:
        """Get the provider's share percentage (100 - platform fee)."""
        return 100.0 - self.platform_fee_percent

    @property
    def can_request_payout(self) -> bool:
        """Check if provider can request a payout."""
        return (
            self.is_active
            and not self.is_suspended
            and self.available_balance >= self.minimum_payout
            and self.payout_address is not None
        )

    @property
    def node_count(self) -> int:
        """Get the number of nodes owned by this provider."""
        return len(self.nodes) if self.nodes else 0

    def calculate_provider_earnings(self, gross_amount: float) -> tuple[float, float]:
        """
        Calculate provider earnings and platform fee from a gross amount.

        Args:
            gross_amount: Total amount charged to customer

        Returns:
            Tuple of (provider_earnings, platform_fee)
        """
        platform_fee = gross_amount * (self.platform_fee_percent / 100.0)
        provider_earnings = gross_amount - platform_fee
        return provider_earnings, platform_fee


class Payout(Base):
    """
    Payout record for provider earnings.

    Tracks requests for payouts and their status.
    """

    __tablename__ = "payouts"

    # Provider reference
    provider_id: Mapped[str] = mapped_column(
        ForeignKey("providers.id"),
        index=True,
    )

    # Payout details
    amount: Mapped[float] = mapped_column(Float)
    payout_method: Mapped[PayoutMethod] = mapped_column(Enum(PayoutMethod))
    payout_address: Mapped[str] = mapped_column(String(255))

    # Status
    status: Mapped[PayoutStatus] = mapped_column(
        Enum(PayoutStatus),
        default=PayoutStatus.PENDING,
        index=True,
    )
    status_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Processing info
    processed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    transaction_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )  # External transaction reference (blockchain tx, PayPal ID, etc.)

    # Admin notes
    admin_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    processed_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )  # Admin who processed the payout

    # Extra data
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    provider: Mapped["Provider"] = relationship("Provider", back_populates="payouts")

    def __repr__(self) -> str:
        return f"<Payout {self.id}: ${self.amount} ({self.status.value})>"


class ProviderEarning(Base):
    """
    Individual earning record for provider revenue tracking.

    Each time a provider's GPU is used and billed, an earning record is created.
    This allows for detailed analytics and audit trails.
    """

    __tablename__ = "provider_earnings"

    # References
    provider_id: Mapped[str] = mapped_column(
        ForeignKey("providers.id"),
        index=True,
    )
    node_id: Mapped[str] = mapped_column(
        ForeignKey("nodes.id"),
        index=True,
    )
    pod_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("pods.id"),
        nullable=True,
    )
    usage_record_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("usage_records.id"),
        nullable=True,
    )

    # Amounts
    gross_amount: Mapped[float] = mapped_column(Float)  # Total charged to customer
    platform_fee: Mapped[float] = mapped_column(Float)  # Platform's cut
    provider_earnings: Mapped[float] = mapped_column(Float)  # Provider's earnings

    # Usage details
    gpu_hours: Mapped[float] = mapped_column(Float)
    hourly_rate: Mapped[float] = mapped_column(Float)

    # Settlement status
    is_settled: Mapped[bool] = mapped_column(Boolean, default=False)
    settled_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Period
    period_start: Mapped[datetime] = mapped_column(index=True)
    period_end: Mapped[datetime] = mapped_column()

    def __repr__(self) -> str:
        return f"<ProviderEarning {self.provider_id}: ${self.provider_earnings}>"
