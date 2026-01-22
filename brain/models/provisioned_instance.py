"""
ProvisionedInstance model for tracking rented GPU instances from providers.
"""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import String, Float, Integer, Enum as SQLEnum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from brain.models.base import Base
from shared.schemas import ProviderType

if TYPE_CHECKING:
    from brain.models.node import Node


class ProvisioningStatus(str, Enum):
    """Status of a provisioned instance lifecycle."""
    PENDING = "pending"  # Request created, not yet submitted to provider
    CREATING = "creating"  # Instance creation request sent to provider
    STARTING = "starting"  # Provider is starting the instance
    INSTALLING = "installing"  # Instance is running, agent installing
    WAITING_REGISTRATION = "waiting_registration"  # Waiting for agent to register
    ACTIVE = "active"  # Agent registered and linked
    FAILED = "failed"  # Provisioning failed
    TERMINATING = "terminating"  # Termination request sent
    TERMINATED = "terminated"  # Instance terminated


class ProvisionedInstance(Base):
    """
    Tracks a GPU instance rented from an external provider.

    Lifecycle:
    1. PENDING - We decided to provision this offer
    2. CREATING - Instance creation API call made
    3. STARTING - Provider is booting the instance
    4. INSTALLING - Agent install script running
    5. WAITING_REGISTRATION - Waiting for agent to call /nodes/register
    6. ACTIVE - Agent registered, linked to a Node
    7. TERMINATED - Instance destroyed
    """

    __tablename__ = "provisioned_instances"

    # Provider identification
    provider_type: Mapped[ProviderType] = mapped_column(
        SQLEnum(ProviderType),
        default=ProviderType.VAST_AI,
        index=True,
    )
    provider_instance_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )
    provider_offer_id: Mapped[str] = mapped_column(String(255))

    # Status
    status: Mapped[ProvisioningStatus] = mapped_column(
        SQLEnum(ProvisioningStatus),
        default=ProvisioningStatus.PENDING,
        index=True,
    )
    status_message: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    last_status_check_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Connection info (populated when instance is running)
    ssh_host: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    ssh_port: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    public_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)

    # GPU Information
    gpu_type: Mapped[str] = mapped_column(String(255), index=True)
    gpu_count: Mapped[int] = mapped_column(Integer, default=1)
    gpu_vram_mb: Mapped[int] = mapped_column(Integer)

    # Pricing and costs
    hourly_cost: Mapped[float] = mapped_column(Float)  # What we pay the provider
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)  # Accumulated cost
    started_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    terminated_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Link to Node (established when agent registers)
    node_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("nodes.id"),
        nullable=True,
        index=True,
    )
    node: Mapped[Optional["Node"]] = relationship(
        "Node",
        back_populates="provisioned_instance",
        foreign_keys=[node_id],
    )

    # Configuration used for provisioning
    docker_image: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    onstart_script: Mapped[Optional[str]] = mapped_column(String(4096), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<ProvisionedInstance {self.id} "
            f"({self.provider_type.value}:{self.provider_instance_id}) "
            f"[{self.status.value}]>"
        )

    @property
    def is_active(self) -> bool:
        """Check if instance is currently active and billable."""
        return self.status in (
            ProvisioningStatus.STARTING,
            ProvisioningStatus.INSTALLING,
            ProvisioningStatus.WAITING_REGISTRATION,
            ProvisioningStatus.ACTIVE,
        )

    @property
    def is_provisioning(self) -> bool:
        """Check if instance is in the provisioning process."""
        return self.status in (
            ProvisioningStatus.PENDING,
            ProvisioningStatus.CREATING,
            ProvisioningStatus.STARTING,
            ProvisioningStatus.INSTALLING,
            ProvisioningStatus.WAITING_REGISTRATION,
        )

    @property
    def runtime_hours(self) -> float:
        """Calculate runtime in hours."""
        if not self.started_at:
            return 0.0

        end = self.terminated_at or datetime.utcnow()
        delta = end - self.started_at
        return delta.total_seconds() / 3600

    def calculate_current_cost(self) -> float:
        """Calculate current cost based on runtime."""
        return self.runtime_hours * self.hourly_cost
