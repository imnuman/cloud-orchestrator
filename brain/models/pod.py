"""
Pod model for user containers/instances.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import String, Float, Integer, Enum, JSON, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from brain.models.base import Base
from shared.schemas import PodStatus

if TYPE_CHECKING:
    from brain.models.user import User
    from brain.models.node import Node


class Pod(Base):
    """User Pod (container instance) model."""

    __tablename__ = "pods"

    # Ownership
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    node_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("nodes.id"),
        nullable=True,
        index=True,
    )

    # Status
    status: Mapped[PodStatus] = mapped_column(
        Enum(PodStatus),
        default=PodStatus.PENDING,
        index=True,
    )
    status_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Configuration
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    docker_image: Mapped[str] = mapped_column(String(512))
    gpu_type: Mapped[str] = mapped_column(String(255), index=True)
    gpu_count: Mapped[int] = mapped_column(Integer, default=1)
    gpu_indices: Mapped[list] = mapped_column(JSON, default=list)

    # Networking
    port_mappings: Mapped[dict] = mapped_column(JSON, default=dict)
    ssh_port: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ssh_host: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    jupyter_port: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Container info
    container_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    environment_variables: Mapped[dict] = mapped_column(JSON, default=dict)
    startup_command: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    volume_mounts: Mapped[list] = mapped_column(JSON, default=list)

    # Pricing
    hourly_price: Mapped[float] = mapped_column(Float)
    provider_cost: Mapped[float] = mapped_column(Float, default=0.0)

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    stopped_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    last_billed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Billing
    total_runtime_seconds: Mapped[int] = mapped_column(Integer, default=0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)

    # Termination
    termination_reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    auto_stop_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="pods")
    node: Mapped[Optional["Node"]] = relationship("Node", back_populates="pods")

    def __repr__(self) -> str:
        return f"<Pod {self.id} ({self.gpu_type})>"

    @property
    def is_running(self) -> bool:
        """Check if pod is currently running."""
        return self.status == PodStatus.RUNNING

    @property
    def connection_string(self) -> Optional[str]:
        """Get SSH connection string if available."""
        if self.ssh_host and self.ssh_port:
            return f"ssh root@{self.ssh_host} -p {self.ssh_port}"
        return None

    def calculate_current_cost(self) -> float:
        """Calculate cost for current session."""
        if not self.started_at:
            return 0.0

        end_time = self.stopped_at or datetime.utcnow()
        runtime_seconds = (end_time - self.started_at).total_seconds()
        hourly_rate = self.hourly_price
        return (runtime_seconds / 3600) * hourly_rate
