"""
Shared Pydantic schemas for Brain-Agent communication.
These schemas define the contract between the Control Plane and Data Plane.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class NodeStatus(str, Enum):
    """Status of a GPU node in the system."""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    PROVISIONING = "provisioning"


class PodStatus(str, Enum):
    """Status of a user's pod/container."""
    PENDING = "pending"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    TERMINATED = "terminated"


class ProviderType(str, Enum):
    """Upstream GPU provider type (Phase 1: Arbitrage)."""
    RUNPOD = "runpod"
    LAMBDA_LABS = "lambda_labs"
    VAST_AI = "vast_ai"
    COMMUNITY = "community"  # Phase 2: Direct provider
    INTERNAL = "internal"  # Self-owned hardware


# ============================================================================
# GPU Information
# ============================================================================

class GpuInfo(BaseModel):
    """Information about a single GPU."""
    index: int = Field(..., description="GPU index (0, 1, 2, etc.)")
    name: str = Field(..., description="GPU model name (e.g., 'NVIDIA GeForce RTX 4090')")
    memory_total_mb: int = Field(..., description="Total VRAM in MB")
    memory_used_mb: int = Field(0, description="Currently used VRAM in MB")
    memory_free_mb: int = Field(0, description="Free VRAM in MB")
    temperature_c: Optional[int] = Field(None, description="GPU temperature in Celsius")
    utilization_percent: Optional[int] = Field(None, description="GPU utilization percentage")
    power_draw_w: Optional[float] = Field(None, description="Current power draw in Watts")
    driver_version: Optional[str] = Field(None, description="NVIDIA driver version")
    cuda_version: Optional[str] = Field(None, description="CUDA version")


class SystemInfo(BaseModel):
    """System information for a node."""
    hostname: str
    os_name: str = Field(..., description="e.g., 'Ubuntu 22.04'")
    kernel_version: str
    cpu_model: str
    cpu_cores: int
    ram_total_mb: int
    ram_available_mb: int
    disk_total_gb: float
    disk_available_gb: float
    docker_version: Optional[str] = None
    nvidia_driver_version: Optional[str] = None


# ============================================================================
# Node Schemas (Agent -> Brain)
# ============================================================================

class NodeRegistrationRequest(BaseModel):
    """Request from Agent to register with the Brain."""
    hostname: str
    ip_address: str = Field(..., description="Internal VPN IP (Tailscale/WireGuard)")
    public_ip: Optional[str] = Field(None, description="Public IP if available")
    gpus: list[GpuInfo] = Field(..., description="List of GPUs on this node")
    system_info: SystemInfo
    agent_version: str
    provider_type: ProviderType = ProviderType.COMMUNITY
    provider_id: Optional[str] = Field(None, description="Provider's instance ID (for auto-provisioned nodes)")
    hourly_price: Optional[float] = Field(None, description="Price set by provider (Phase 2)")
    api_key: Optional[str] = Field(None, description="Agent API key for authentication")
    provider_key: Optional[str] = Field(None, description="Provider key for community GPU providers (Phase 2)")


class NodeRegistrationResponse(BaseModel):
    """Response from Brain after successful registration."""
    node_id: UUID
    api_key: str = Field(..., description="API key for future communication")
    heartbeat_interval_seconds: int = Field(30, description="How often to send heartbeats")
    message: str = "Registration successful"


class NodeHeartbeatRequest(BaseModel):
    """Periodic heartbeat from Agent to Brain."""
    node_id: UUID
    gpus: list[GpuInfo] = Field(..., description="Current GPU status")
    running_pods: list[UUID] = Field(default_factory=list, description="Currently running pod IDs")
    system_info: SystemInfo
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class NodeHeartbeatResponse(BaseModel):
    """Response to heartbeat, may include commands."""
    acknowledged: bool = True
    commands: list[dict] = Field(default_factory=list, description="Commands for the agent to execute")


# ============================================================================
# Pod Schemas (User facing)
# ============================================================================

class PortMapping(BaseModel):
    """Port mapping configuration."""
    host_port: int
    container_port: int
    protocol: str = "tcp"


class PodCreateRequest(BaseModel):
    """Request to create a new pod."""
    gpu_type: str = Field(..., description="Requested GPU type (e.g., 'RTX 4090')")
    gpu_count: int = Field(1, ge=1, le=8, description="Number of GPUs")
    docker_image: str = Field(..., description="Docker image to run")
    port_mappings: list[PortMapping] = Field(default_factory=list)
    environment_variables: dict[str, str] = Field(default_factory=dict)
    startup_command: Optional[str] = Field(None, description="Custom startup command")
    volume_mounts: list[str] = Field(default_factory=list, description="Volume mount paths")
    min_vram_mb: Optional[int] = Field(None, description="Minimum VRAM required")
    max_hourly_price: Optional[float] = Field(None, description="Maximum price willing to pay")


class PodResponse(BaseModel):
    """Pod information returned to user."""
    id: UUID
    user_id: UUID
    node_id: Optional[UUID] = None
    status: PodStatus
    gpu_type: str
    gpu_count: int
    docker_image: str
    port_mappings: list[PortMapping] = Field(default_factory=list)
    ssh_connection_string: Optional[str] = Field(None, description="SSH connection info")
    jupyter_url: Optional[str] = Field(None, description="JupyterLab URL if available")
    web_terminal_url: Optional[str] = Field(None, description="Web terminal URL")
    hourly_price: float
    created_at: datetime
    started_at: Optional[datetime] = None
    total_cost: float = Field(0.0, description="Total cost incurred so far")


# ============================================================================
# Agent Command Schemas (Brain -> Agent)
# ============================================================================

class PodDeployCommand(BaseModel):
    """Command sent to Agent to deploy a pod."""
    pod_id: UUID
    docker_image: str
    gpu_indices: list[int] = Field(..., description="Which GPUs to assign")
    port_mappings: list[PortMapping]
    environment_variables: dict[str, str] = Field(default_factory=dict)
    startup_command: Optional[str] = None
    volume_mounts: list[str] = Field(default_factory=list)
    resource_limits: dict = Field(default_factory=dict)


class PodDeployResponse(BaseModel):
    """Response from Agent after pod deployment."""
    pod_id: UUID
    success: bool
    container_id: Optional[str] = None
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    mapped_ports: dict[int, int] = Field(default_factory=dict, description="container_port -> host_port")
    error_message: Optional[str] = None


class PodStopCommand(BaseModel):
    """Command to stop a running pod."""
    pod_id: UUID
    force: bool = False
    timeout_seconds: int = 30


class PodStopResponse(BaseModel):
    """Response after stopping a pod."""
    pod_id: UUID
    success: bool
    error_message: Optional[str] = None


# ============================================================================
# Billing Schemas
# ============================================================================

class UsageRecord(BaseModel):
    """Record of resource usage for billing."""
    pod_id: UUID
    user_id: UUID
    node_id: UUID
    gpu_type: str
    gpu_count: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: int = 0
    hourly_rate: float
    total_cost: float = 0.0


class BillingEvent(BaseModel):
    """A billing event (charge or credit)."""
    id: UUID
    user_id: UUID
    amount: float = Field(..., description="Positive = charge, negative = credit")
    description: str
    pod_id: Optional[UUID] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# User Schemas
# ============================================================================

class UserCreate(BaseModel):
    """Request to create a new user."""
    email: str
    password: str
    name: Optional[str] = None


class UserResponse(BaseModel):
    """User information (public)."""
    id: UUID
    email: str
    name: Optional[str] = None
    balance: float = Field(0.0, description="Account balance")
    created_at: datetime
    is_active: bool = True
    is_verified: bool = Field(False, description="Whether email is verified")
    is_provider: bool = Field(False, description="Whether user is also a GPU provider")


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Seconds until expiration")
