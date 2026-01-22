"""
Pydantic schemas for Vast.ai API interactions.
These models represent the raw Vast.ai API data structures.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class VastInstanceStatus(str, Enum):
    """Vast.ai instance status values."""
    LOADING = "loading"
    RUNNING = "running"
    EXITED = "exited"
    CREATED = "created"
    OFFLINE = "offline"


class VastGpuOffer(BaseModel):
    """
    Vast.ai GPU offer/listing.
    Represents a machine available for rent on Vast.ai marketplace.
    """
    id: int = Field(..., description="Vast.ai offer ID")
    machine_id: int = Field(..., description="Machine ID")

    # GPU Information
    gpu_name: str = Field(..., description="GPU model name")
    num_gpus: int = Field(1, description="Number of GPUs")
    gpu_ram: int = Field(..., description="VRAM per GPU in MB")
    total_flops: float = Field(0.0, description="Total TFLOPS")
    cuda_max_good: float = Field(0.0, description="Max CUDA version supported")
    driver_version: Optional[str] = Field(None, description="NVIDIA driver version")

    # System specs
    cpu_cores: int = Field(..., description="Number of CPU cores")
    cpu_cores_effective: float = Field(..., description="Effective CPU cores")
    cpu_name: str = Field("", description="CPU model name")
    cpu_ram: int = Field(..., description="RAM in MB")
    disk_space: float = Field(..., description="Available disk in GB")
    disk_bw: float = Field(0.0, description="Disk bandwidth MB/s")

    # Network
    inet_up: float = Field(0.0, description="Upload speed Mbps")
    inet_down: float = Field(0.0, description="Download speed Mbps")

    # Pricing
    dph_total: float = Field(..., description="Price per hour (dollars)")
    min_bid: float = Field(0.0, description="Minimum bid price")

    # Reliability/Quality
    reliability2: float = Field(1.0, description="Reliability score 0-1")
    dlperf: float = Field(0.0, description="Deep learning performance score")
    dlperf_per_dphtotal: float = Field(0.0, description="DL perf per dollar")
    duration: float = Field(0.0, description="Average rental duration hours")
    verification: str = Field("", description="Verification status")

    # Location
    geolocation: Optional[str] = Field(None, description="Geographic location")
    country: Optional[str] = Field(None, description="Country code")

    # Status
    rentable: bool = Field(True, description="Whether currently available")
    rented: bool = Field(False, description="Whether currently rented")

    def to_normalized(self) -> "brain.adapters.base.GpuOffer":
        """Convert to normalized GpuOffer."""
        from brain.adapters.base import GpuOffer

        return GpuOffer(
            offer_id=str(self.id),
            provider="vast_ai",
            gpu_name=self.gpu_name,
            gpu_count=self.num_gpus,
            gpu_vram_mb=self.gpu_ram,
            total_vram_mb=self.gpu_ram * self.num_gpus,
            cpu_cores=self.cpu_cores,
            ram_mb=self.cpu_ram,
            disk_gb=self.disk_space,
            hourly_price=self.dph_total,
            reliability_score=self.reliability2,
            location=self.geolocation or self.country,
            internet_speed_mbps=self.inet_down,
            cuda_version=str(self.cuda_max_good) if self.cuda_max_good else None,
            driver_version=self.driver_version,
        )


class VastInstance(BaseModel):
    """
    Vast.ai rented instance.
    Represents an active rental on Vast.ai.
    """
    id: int = Field(..., description="Instance ID")
    machine_id: int = Field(..., description="Machine ID")

    # Status
    actual_status: VastInstanceStatus = Field(..., description="Current status")
    intended_status: str = Field("running", description="Intended status")
    status_msg: Optional[str] = Field(None, description="Status message")

    # GPU Info
    gpu_name: str = Field(..., description="GPU model")
    num_gpus: int = Field(1, description="Number of GPUs")
    gpu_ram: int = Field(..., description="VRAM per GPU in MB")

    # Connection info
    ssh_host: Optional[str] = Field(None, description="SSH hostname")
    ssh_port: Optional[int] = Field(None, description="SSH port")
    ssh_idx: Optional[str] = Field(None, description="SSH index")
    public_ipaddr: Optional[str] = Field(None, description="Public IP")

    # Pricing
    dph_total: float = Field(..., description="Price per hour")

    # Docker
    image_uuid: Optional[str] = Field(None, description="Docker image")
    docker_image: Optional[str] = Field(None, description="Docker image name")
    onstart_cmd: Optional[str] = Field(None, description="Startup command")

    # Timing
    start_date: Optional[float] = Field(None, description="Start timestamp")
    end_date: Optional[float] = Field(None, description="End timestamp")

    # Disk
    disk_space: float = Field(0.0, description="Allocated disk GB")

    # Label
    label: Optional[str] = Field(None, description="Instance label")

    def to_normalized(self) -> "brain.adapters.base.Instance":
        """Convert to normalized Instance."""
        from brain.adapters.base import Instance, InstanceStatus

        status_map = {
            VastInstanceStatus.LOADING: InstanceStatus.LOADING,
            VastInstanceStatus.RUNNING: InstanceStatus.RUNNING,
            VastInstanceStatus.EXITED: InstanceStatus.STOPPED,
            VastInstanceStatus.CREATED: InstanceStatus.PENDING,
            VastInstanceStatus.OFFLINE: InstanceStatus.ERROR,
        }

        return Instance(
            instance_id=str(self.id),
            provider="vast_ai",
            offer_id=str(self.machine_id),
            status=status_map.get(self.actual_status, InstanceStatus.PENDING),
            gpu_name=self.gpu_name,
            gpu_count=self.num_gpus,
            hourly_price=self.dph_total,
            ssh_host=self.ssh_host,
            ssh_port=self.ssh_port,
            ssh_user="root",
            public_ip=self.public_ipaddr,
            created_at=datetime.utcnow(),
            started_at=(
                datetime.fromtimestamp(self.start_date)
                if self.start_date
                else None
            ),
        )


class VastOfferFilters(BaseModel):
    """
    Filters for searching Vast.ai offers.
    Maps to Vast.ai search API parameters.
    """
    gpu_name: Optional[str] = Field(None, description="GPU model filter")
    num_gpus: int = Field(1, description="Minimum number of GPUs")
    min_gpu_ram: Optional[int] = Field(None, description="Min VRAM per GPU in MB")
    max_dph: Optional[float] = Field(None, description="Max price per hour")
    min_reliability: float = Field(0.0, description="Min reliability score (0-1)")
    min_disk: float = Field(10.0, description="Min disk space GB")
    cuda_vers: Optional[float] = Field(None, description="Required CUDA version")
    rentable: bool = Field(True, description="Only show available")
    order: str = Field("dph_total", description="Sort field")
    limit: int = Field(100, description="Max results")


class VastInstanceConfig(BaseModel):
    """
    Configuration for creating a Vast.ai instance.
    Maps to Vast.ai create instance API.
    """
    client_id: str = Field("me", description="Client ID (usually 'me')")
    image: str = Field(..., description="Docker image")
    disk: float = Field(20.0, description="Disk space GB")
    onstart: Optional[str] = Field(None, description="Startup script")
    env: dict[str, str] = Field(default_factory=dict, description="Environment vars")
    label: Optional[str] = Field(None, description="Instance label")
    runtype: str = Field("ssh", description="Run type (ssh, jupyter, etc)")
    python_utf8: bool = Field(True, description="Use UTF-8 locale")
    lang_utf8: bool = Field(True, description="Use UTF-8 language")


# Type alias for import convenience
import brain.adapters.base
