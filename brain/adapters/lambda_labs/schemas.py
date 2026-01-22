"""
Lambda Labs-specific Pydantic schemas.
Maps Lambda Labs API responses to our normalized format.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from brain.adapters.base import GpuOffer, Instance, InstanceStatus


class LambdaInstanceType(BaseModel):
    """Lambda Labs instance type (GPU configuration)."""

    name: str = Field(..., description="Instance type name (e.g., 'gpu_1x_a100')")
    description: str = Field("", description="Human-readable description")
    price_cents_per_hour: int = Field(..., description="Price in cents per hour")
    specs: dict = Field(default_factory=dict, description="Hardware specifications")


class LambdaGpuOffer(BaseModel):
    """
    Lambda Labs GPU offer (instance type with availability).
    Based on Lambda Labs' instance-types API.
    """

    instance_type: str = Field(..., description="Instance type name")
    region: str = Field(..., description="Region name")
    price_cents_per_hour: int = Field(..., description="Price in cents per hour")
    description: str = Field("", description="Instance description")
    gpu_name: str = Field(..., description="GPU model name")
    gpu_count: int = Field(1, description="Number of GPUs")
    vram_gb: int = Field(..., description="VRAM per GPU in GB")
    vcpus: int = Field(8, description="Number of vCPUs")
    memory_gb: int = Field(32, description="RAM in GB")
    storage_gb: int = Field(100, description="Storage in GB")
    available: bool = Field(True, description="Whether available for launch")

    def to_normalized(self) -> GpuOffer:
        """Convert to normalized GpuOffer format."""
        return GpuOffer(
            offer_id=f"{self.instance_type}:{self.region}",
            provider="lambda_labs",
            gpu_name=self.gpu_name,
            gpu_count=self.gpu_count,
            gpu_vram_mb=self.vram_gb * 1024,
            total_vram_mb=self.vram_gb * self.gpu_count * 1024,
            cpu_cores=self.vcpus,
            ram_mb=self.memory_gb * 1024,
            disk_gb=float(self.storage_gb),
            hourly_price=self.price_cents_per_hour / 100.0,
            reliability_score=0.99,  # Lambda Labs has high reliability
            location=self.region,
        )


class LambdaInstance(BaseModel):
    """
    Lambda Labs instance.
    Based on Lambda Labs' instances API.
    """

    id: str = Field(..., description="Instance ID")
    name: Optional[str] = Field(None, description="Instance name")
    ip: Optional[str] = Field(None, description="Public IP address")
    status: str = Field(..., description="Instance status")
    ssh_key_names: list[str] = Field(default_factory=list, description="SSH key names")
    file_system_names: list[str] = Field(default_factory=list, description="Filesystem names")
    region: dict = Field(default_factory=dict, description="Region info")
    instance_type: dict = Field(default_factory=dict, description="Instance type info")
    hostname: Optional[str] = Field(None, description="Instance hostname")
    jupyter_token: Optional[str] = Field(None, description="JupyterLab token")
    jupyter_url: Optional[str] = Field(None, description="JupyterLab URL")

    def _map_status(self) -> InstanceStatus:
        """Map Lambda Labs status to our InstanceStatus."""
        status_map = {
            "booting": InstanceStatus.LOADING,
            "active": InstanceStatus.RUNNING,
            "unhealthy": InstanceStatus.ERROR,
            "terminated": InstanceStatus.STOPPED,
        }
        return status_map.get(self.status.lower(), InstanceStatus.PENDING)

    def to_normalized(self) -> Instance:
        """Convert to normalized Instance format."""
        # Extract GPU info from instance_type
        gpu_name = "Unknown"
        gpu_count = 1
        hourly_price = 0.0

        if self.instance_type:
            desc = self.instance_type.get("description", "")
            gpu_name = desc.split(",")[0] if desc else "Unknown"
            price_cents = self.instance_type.get("price_cents_per_hour", 0)
            hourly_price = price_cents / 100.0
            # Parse GPU count from instance type name (e.g., "gpu_8x_a100")
            type_name = self.instance_type.get("name", "")
            if "x_" in type_name:
                try:
                    gpu_count = int(type_name.split("x_")[0].split("_")[-1])
                except (ValueError, IndexError):
                    gpu_count = 1

        return Instance(
            instance_id=self.id,
            provider="lambda_labs",
            offer_id=self.instance_type.get("name"),
            status=self._map_status(),
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            hourly_price=hourly_price,
            ssh_host=self.ip or self.hostname,
            ssh_port=22,
            public_ip=self.ip,
            created_at=datetime.utcnow(),
        )


class LambdaInstanceConfig(BaseModel):
    """Configuration for creating a Lambda Labs instance."""

    instance_type_name: str = Field(..., description="Instance type (e.g., 'gpu_1x_a100')")
    region_name: str = Field(..., description="Region (e.g., 'us-west-1')")
    ssh_key_names: list[str] = Field(..., description="SSH key names to attach")
    file_system_names: list[str] = Field(
        default_factory=list, description="Filesystem names to attach"
    )
    quantity: int = Field(1, description="Number of instances to launch")
    name: Optional[str] = Field(None, description="Instance name")


class LambdaOfferFilters(BaseModel):
    """Filters for searching Lambda Labs offers."""

    instance_type: Optional[str] = Field(None, description="Specific instance type")
    region: Optional[str] = Field(None, description="Specific region")
    min_gpus: int = Field(1, description="Minimum number of GPUs")
    max_price_cents: Optional[int] = Field(None, description="Maximum price in cents/hour")
    gpu_name: Optional[str] = Field(None, description="GPU model name filter")
