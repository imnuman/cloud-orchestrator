"""
RunPod-specific Pydantic schemas.
Maps RunPod API responses to our normalized format.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from brain.adapters.base import GpuOffer, Instance, InstanceStatus


class RunPodGpu(BaseModel):
    """RunPod GPU information."""

    id: str = Field(..., description="GPU type ID")
    displayName: str = Field(..., description="Display name (e.g., 'NVIDIA RTX 4090')")
    memoryInGb: int = Field(..., description="VRAM in GB")


class RunPodGpuOffer(BaseModel):
    """
    RunPod GPU offer (machine/pod type).
    Based on RunPod's GPU types API.
    """

    id: str = Field(..., description="GPU type ID")
    display_name: str = Field(..., description="GPU display name")
    memory_in_gb: int = Field(..., description="VRAM per GPU in GB")
    secure_price: Optional[float] = Field(None, description="Secure cloud hourly price")
    community_price: Optional[float] = Field(None, description="Community cloud hourly price")
    community_spot_price: Optional[float] = Field(None, description="Community spot price")
    secure_spot_price: Optional[float] = Field(None, description="Secure spot price")
    lowest_price: Optional[float] = Field(None, description="Lowest available price")
    stock_status: str = Field("unavailable", description="Availability status")
    max_gpu_count: int = Field(1, description="Maximum GPUs per instance")
    cuda_version: Optional[str] = Field(None, description="CUDA version")

    # Additional computed fields
    cpu_cores: int = Field(8, description="Estimated CPU cores")
    ram_mb: int = Field(32768, description="Estimated RAM in MB")
    disk_gb: float = Field(100.0, description="Default disk space")
    reliability_score: float = Field(0.99, description="RunPod reliability score")
    location: Optional[str] = Field(None, description="Data center location")

    def to_normalized(self) -> GpuOffer:
        """Convert to normalized GpuOffer format."""
        # Use the lowest available price
        hourly_price = self.lowest_price
        if hourly_price is None:
            hourly_price = self.secure_price or self.community_price or 0.0

        return GpuOffer(
            offer_id=self.id,
            provider="runpod",
            gpu_name=self.display_name,
            gpu_count=1,
            gpu_vram_mb=self.memory_in_gb * 1024,
            total_vram_mb=self.memory_in_gb * 1024,
            cpu_cores=self.cpu_cores,
            ram_mb=self.ram_mb,
            disk_gb=self.disk_gb,
            hourly_price=hourly_price,
            reliability_score=self.reliability_score,
            location=self.location,
            cuda_version=self.cuda_version,
        )


class RunPodInstance(BaseModel):
    """
    RunPod Pod instance.
    Based on RunPod's instances API.
    """

    id: str = Field(..., description="Pod ID")
    name: Optional[str] = Field(None, description="Pod name")
    desiredStatus: str = Field(..., description="Desired status")
    runtime: Optional[dict] = Field(None, description="Runtime info with ports/SSH")
    machine: Optional[dict] = Field(None, description="Machine/GPU info")
    imageName: str = Field(..., description="Docker image")
    gpuCount: int = Field(1, description="Number of GPUs")
    volumeInGb: float = Field(0, description="Volume size")
    containerDiskInGb: float = Field(0, description="Container disk size")
    costPerHr: float = Field(0, description="Hourly cost")
    uptimeSeconds: Optional[int] = Field(None, description="Uptime in seconds")

    def _map_status(self) -> InstanceStatus:
        """Map RunPod status to our InstanceStatus."""
        status_map = {
            "CREATED": InstanceStatus.LOADING,
            "RUNNING": InstanceStatus.RUNNING,
            "EXITED": InstanceStatus.STOPPED,
            "TERMINATED": InstanceStatus.STOPPED,
        }
        return status_map.get(self.desiredStatus, InstanceStatus.PENDING)

    def to_normalized(self) -> Instance:
        """Convert to normalized Instance format."""
        # Extract SSH connection info from runtime
        ssh_host = None
        ssh_port = None
        public_ip = None

        if self.runtime:
            # RunPod provides ports in runtime object
            ports = self.runtime.get("ports", [])
            for port in ports:
                if port.get("privatePort") == 22:
                    ssh_host = port.get("ip")
                    ssh_port = port.get("publicPort")
                    break

            public_ip = self.runtime.get("gpus", [{}])[0].get("gpuId") if self.runtime.get("gpus") else None

        # Get GPU name from machine info
        gpu_name = "Unknown"
        if self.machine:
            gpu_name = self.machine.get("gpuDisplayName", "Unknown")

        return Instance(
            instance_id=self.id,
            provider="runpod",
            offer_id=self.id,
            status=self._map_status(),
            gpu_name=gpu_name,
            gpu_count=self.gpuCount,
            hourly_price=self.costPerHr,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            public_ip=public_ip,
            created_at=datetime.utcnow(),
        )


class RunPodInstanceConfig(BaseModel):
    """Configuration for creating a RunPod Pod."""

    name: str = Field(..., description="Pod name")
    image_name: str = Field(..., description="Docker image")
    gpu_type_id: str = Field(..., description="GPU type ID")
    gpu_count: int = Field(1, description="Number of GPUs")
    volume_in_gb: float = Field(20.0, description="Persistent volume size")
    container_disk_in_gb: float = Field(20.0, description="Container disk size")
    ports: str = Field("22/tcp,8888/http", description="Port mappings")
    volume_mount_path: str = Field("/workspace", description="Volume mount path")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    docker_args: str = Field("", description="Docker run arguments")
    cloud_type: str = Field("ALL", description="SECURE, COMMUNITY, or ALL")


class RunPodOfferFilters(BaseModel):
    """Filters for searching RunPod GPU offers."""

    gpu_type_id: Optional[str] = Field(None, description="Specific GPU type ID")
    min_memory_in_gb: Optional[int] = Field(None, description="Minimum VRAM in GB")
    max_price_per_hour: Optional[float] = Field(None, description="Maximum hourly price")
    cloud_type: str = Field("ALL", description="SECURE, COMMUNITY, or ALL")
    cuda_version: Optional[str] = Field(None, description="Required CUDA version")
