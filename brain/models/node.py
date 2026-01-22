"""
Node model for GPU worker machines.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import String, Float, Boolean, Integer, Enum, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from brain.models.base import Base
from shared.schemas import NodeStatus, ProviderType

if TYPE_CHECKING:
    from brain.models.pod import Pod
    from brain.models.provisioned_instance import ProvisionedInstance


class Node(Base):
    """GPU Node (worker machine) model."""

    __tablename__ = "nodes"

    # Identity
    hostname: Mapped[str] = mapped_column(String(255))
    ip_address: Mapped[str] = mapped_column(String(45), index=True)  # VPN IP
    public_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)

    # Status
    status: Mapped[NodeStatus] = mapped_column(
        Enum(NodeStatus),
        default=NodeStatus.OFFLINE,
        index=True,
    )
    last_heartbeat_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    consecutive_missed_heartbeats: Mapped[int] = mapped_column(Integer, default=0)

    # Provider info
    provider_type: Mapped[ProviderType] = mapped_column(
        Enum(ProviderType),
        default=ProviderType.COMMUNITY,
    )
    provider_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )  # External provider's node/instance ID
    owner_user_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
    )  # For Phase 2: Community providers

    # GPU Information
    gpu_model: Mapped[str] = mapped_column(String(255), index=True)
    gpu_count: Mapped[int] = mapped_column(Integer, default=1)
    total_vram_mb: Mapped[int] = mapped_column(Integer)
    gpu_details: Mapped[dict] = mapped_column(JSON, default=dict)

    # System Information
    cpu_model: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    cpu_cores: Mapped[int] = mapped_column(Integer, default=1)
    ram_total_mb: Mapped[int] = mapped_column(Integer, default=0)
    disk_total_gb: Mapped[float] = mapped_column(Float, default=0.0)
    os_info: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Pricing
    hourly_price: Mapped[float] = mapped_column(Float, default=0.0)
    provider_cost: Mapped[float] = mapped_column(Float, default=0.0)  # Our cost from provider

    # Agent info
    agent_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    agent_api_key: Mapped[Optional[str]] = mapped_column(
        String(64),
        unique=True,
        nullable=True,
        index=True,
    )

    # Capacity
    max_pods: Mapped[int] = mapped_column(Integer, default=1)
    current_pod_count: Mapped[int] = mapped_column(Integer, default=0)

    # Metrics (updated via heartbeat)
    current_gpu_utilization: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    current_vram_used_mb: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    current_temperature_c: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    pods: Mapped[list["Pod"]] = relationship("Pod", back_populates="node", lazy="selectin")
    provisioned_instance: Mapped[Optional["ProvisionedInstance"]] = relationship(
        "ProvisionedInstance",
        back_populates="node",
        uselist=False,
        foreign_keys="ProvisionedInstance.node_id",
    )

    def __repr__(self) -> str:
        return f"<Node {self.hostname} ({self.gpu_model})>"

    @property
    def is_available(self) -> bool:
        """Check if node can accept new pods."""
        return (
            self.status == NodeStatus.ONLINE
            and self.current_pod_count < self.max_pods
        )

    @property
    def available_vram_mb(self) -> int:
        """Estimated available VRAM."""
        used = self.current_vram_used_mb or 0
        return self.total_vram_mb - used
