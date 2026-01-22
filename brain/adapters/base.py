"""
Abstract base class for GPU provider adapters.
Defines the interface that all provider integrations must implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class InstanceStatus(str, Enum):
    """Status of a provider instance."""
    PENDING = "pending"
    LOADING = "loading"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class GpuOffer(BaseModel):
    """Normalized GPU offer from any provider."""
    offer_id: str = Field(..., description="Provider's unique offer ID")
    provider: str = Field(..., description="Provider name (vast_ai, runpod, etc.)")
    gpu_name: str = Field(..., description="GPU model name (e.g., 'RTX 4090')")
    gpu_count: int = Field(1, description="Number of GPUs")
    gpu_vram_mb: int = Field(..., description="VRAM per GPU in MB")
    total_vram_mb: int = Field(..., description="Total VRAM across all GPUs")
    cpu_cores: int = Field(..., description="Number of CPU cores")
    ram_mb: int = Field(..., description="RAM in MB")
    disk_gb: float = Field(..., description="Disk space in GB")
    hourly_price: float = Field(..., description="Price per hour in USD")
    reliability_score: float = Field(1.0, description="Reliability score 0-1")
    location: Optional[str] = Field(None, description="Geographic location")
    internet_speed_mbps: Optional[float] = Field(None, description="Internet speed")
    cuda_version: Optional[str] = Field(None, description="CUDA version")
    driver_version: Optional[str] = Field(None, description="NVIDIA driver version")


class Instance(BaseModel):
    """Normalized instance from any provider."""
    instance_id: str = Field(..., description="Provider's unique instance ID")
    provider: str = Field(..., description="Provider name")
    offer_id: Optional[str] = Field(None, description="Original offer ID")
    status: InstanceStatus = Field(..., description="Current status")
    gpu_name: str = Field(..., description="GPU model name")
    gpu_count: int = Field(1, description="Number of GPUs")
    hourly_price: float = Field(..., description="Price per hour")
    ssh_host: Optional[str] = Field(None, description="SSH hostname")
    ssh_port: Optional[int] = Field(None, description="SSH port")
    ssh_user: Optional[str] = Field("root", description="SSH username")
    public_ip: Optional[str] = Field(None, description="Public IP address")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None, description="When instance started running")


class OfferFilters(BaseModel):
    """Filters for searching GPU offers."""
    gpu_name: Optional[str] = Field(None, description="GPU model name filter")
    min_gpu_count: int = Field(1, description="Minimum number of GPUs")
    min_vram_mb: Optional[int] = Field(None, description="Minimum VRAM per GPU")
    max_hourly_price: Optional[float] = Field(None, description="Maximum hourly price")
    min_reliability: float = Field(0.0, description="Minimum reliability score")
    min_disk_gb: float = Field(10.0, description="Minimum disk space")
    cuda_version: Optional[str] = Field(None, description="Required CUDA version")


class InstanceConfig(BaseModel):
    """Configuration for creating an instance."""
    docker_image: str = Field(..., description="Docker image to run")
    disk_gb: float = Field(20.0, description="Disk space to allocate")
    onstart_script: Optional[str] = Field(None, description="Script to run on startup")
    env_vars: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    ssh_key: Optional[str] = Field(None, description="SSH public key")
    label: Optional[str] = Field(None, description="Instance label")


class BaseProviderAdapter(ABC):
    """Abstract base class for GPU provider adapters."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'vast_ai')."""
        pass

    @abstractmethod
    async def list_offers(self, filters: Optional[OfferFilters] = None) -> list[GpuOffer]:
        """
        List available GPU offers from the provider.

        Args:
            filters: Optional filters to apply

        Returns:
            List of normalized GPU offers
        """
        pass

    @abstractmethod
    async def create_instance(
        self, offer_id: str, config: InstanceConfig
    ) -> Instance:
        """
        Create a new instance from an offer.

        Args:
            offer_id: The offer ID to instantiate
            config: Instance configuration

        Returns:
            The created instance
        """
        pass

    @abstractmethod
    async def get_instance(self, instance_id: str) -> Optional[Instance]:
        """
        Get information about an instance.

        Args:
            instance_id: The instance ID

        Returns:
            Instance info or None if not found
        """
        pass

    @abstractmethod
    async def terminate_instance(self, instance_id: str) -> bool:
        """
        Terminate/destroy an instance.

        Args:
            instance_id: The instance ID to terminate

        Returns:
            True if termination was successful
        """
        pass

    async def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass
