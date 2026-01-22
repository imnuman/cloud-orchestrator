"""
Model Catalog and Deployment models for AI model serving.
Phase 2B: Customer AI Model Serving
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import String, Float, Boolean, Integer, Enum, ForeignKey, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from brain.models.base import Base

if TYPE_CHECKING:
    from brain.models.user import User
    from brain.models.node import Node
    from brain.models.pod import Pod


class ModelCategory(str, PyEnum):
    """Categories of AI models."""
    TEXT = "text"  # LLMs for text generation (Llama, Mistral, etc.)
    IMAGE = "image"  # Image generation (SDXL, Flux, etc.)
    AUDIO = "audio"  # Audio processing (Whisper, TTS, etc.)
    VIDEO = "video"  # Video generation/processing
    MULTIMODAL = "multimodal"  # Vision-language models
    EMBEDDING = "embedding"  # Text embedding models
    CODE = "code"  # Code generation models


class ServingBackend(str, PyEnum):
    """Backend serving frameworks."""
    VLLM = "vllm"  # vLLM - high throughput LLM serving
    TGI = "tgi"  # Text Generation Inference (HuggingFace)
    OLLAMA = "ollama"  # Ollama - easy deployment
    COMFYUI = "comfyui"  # ComfyUI for Stable Diffusion
    A1111 = "a1111"  # Automatic1111 WebUI
    WHISPER = "whisper"  # Whisper/faster-whisper
    TTS = "tts"  # TTS backends (Coqui, Bark, etc.)
    CUSTOM = "custom"  # Custom container


class DeploymentStatus(str, PyEnum):
    """Status of a model deployment."""
    PENDING = "pending"  # Waiting for resources
    PROVISIONING = "provisioning"  # Creating container
    STARTING = "starting"  # Container starting
    LOADING = "loading"  # Loading model weights
    READY = "ready"  # Ready to serve
    UNHEALTHY = "unhealthy"  # Health check failing
    STOPPING = "stopping"  # Stopping container
    STOPPED = "stopped"  # Container stopped
    FAILED = "failed"  # Deployment failed


class ModelTemplate(Base):
    """
    Model template defining how to deploy an AI model.

    Each template includes all configuration needed to deploy the model:
    - Docker image and startup command
    - Resource requirements (VRAM, GPU type)
    - Default environment variables
    - API endpoints and health checks
    """

    __tablename__ = "model_templates"

    # Identity
    name: Mapped[str] = mapped_column(String(255), index=True)  # "Llama 3.1 70B"
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True)  # "llama-3.1-70b"
    version: Mapped[str] = mapped_column(String(64), default="latest")

    # Classification
    category: Mapped[ModelCategory] = mapped_column(Enum(ModelCategory), index=True)
    serving_backend: Mapped[ServingBackend] = mapped_column(Enum(ServingBackend))

    # Resource requirements
    min_vram_gb: Mapped[int] = mapped_column(Integer)  # Minimum VRAM in GB
    recommended_vram_gb: Mapped[int] = mapped_column(Integer)  # Recommended VRAM
    min_gpu_count: Mapped[int] = mapped_column(Integer, default=1)
    recommended_gpu: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # "A100 80GB"
    supports_quantization: Mapped[bool] = mapped_column(Boolean, default=False)

    # Container configuration
    docker_image: Mapped[str] = mapped_column(String(512))
    docker_tag: Mapped[str] = mapped_column(String(128), default="latest")
    startup_command: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    default_env: Mapped[dict] = mapped_column(JSON, default=dict)
    default_ports: Mapped[dict] = mapped_column(JSON, default=dict)  # {"8000": "API", "7860": "UI"}

    # Health checking
    health_check_url: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # "/health"
    health_check_interval_seconds: Mapped[int] = mapped_column(Integer, default=30)
    startup_timeout_seconds: Mapped[int] = mapped_column(Integer, default=600)  # 10 min for large models

    # Model source
    model_id: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)  # HuggingFace model ID
    model_source: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # "huggingface", "custom"

    # API compatibility
    api_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # "openai", "custom"
    openai_compatible: Mapped[bool] = mapped_column(Boolean, default=False)

    # Metadata
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    documentation_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    license: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Display
    is_featured: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    display_order: Mapped[int] = mapped_column(Integer, default=0)

    # Usage tracking
    deployment_count: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    deployments: Mapped[list["Deployment"]] = relationship(
        "Deployment", back_populates="model_template"
    )

    def __repr__(self) -> str:
        return f"<ModelTemplate {self.slug} ({self.category.value})>"

    @property
    def full_docker_image(self) -> str:
        """Get full Docker image name with tag."""
        return f"{self.docker_image}:{self.docker_tag}"


class Deployment(Base):
    """
    A user's deployment of a model.

    Represents an active instance of a model running on a GPU node.
    """

    __tablename__ = "deployments"

    # Ownership
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)

    # Model reference
    model_template_id: Mapped[str] = mapped_column(
        ForeignKey("model_templates.id"),
        index=True,
    )

    # Node/Pod reference (when deployed)
    node_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("nodes.id"),
        nullable=True,
    )
    pod_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("pods.id"),
        nullable=True,
    )

    # Deployment name (user-friendly)
    name: Mapped[str] = mapped_column(String(255))

    # Status
    status: Mapped[DeploymentStatus] = mapped_column(
        Enum(DeploymentStatus),
        default=DeploymentStatus.PENDING,
        index=True,
    )
    status_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_health_check_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    health_check_failures: Mapped[int] = mapped_column(Integer, default=0)

    # Configuration overrides
    custom_env: Mapped[dict] = mapped_column(JSON, default=dict)
    gpu_count: Mapped[int] = mapped_column(Integer, default=1)
    gpu_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Access
    api_endpoint: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    api_key: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    ui_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # Billing
    hourly_price: Mapped[float] = mapped_column(Float, default=0.0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    total_runtime_seconds: Mapped[int] = mapped_column(Integer, default=0)
    last_billed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Timestamps
    started_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    stopped_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Usage metrics
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens_out: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    user: Mapped["User"] = relationship("User")
    model_template: Mapped["ModelTemplate"] = relationship(
        "ModelTemplate", back_populates="deployments"
    )
    node: Mapped[Optional["Node"]] = relationship("Node")
    pod: Mapped[Optional["Pod"]] = relationship("Pod")

    def __repr__(self) -> str:
        return f"<Deployment {self.name} ({self.status.value})>"

    @property
    def is_running(self) -> bool:
        """Check if deployment is running."""
        return self.status == DeploymentStatus.READY

    @property
    def is_active(self) -> bool:
        """Check if deployment is in an active state."""
        return self.status in [
            DeploymentStatus.PENDING,
            DeploymentStatus.PROVISIONING,
            DeploymentStatus.STARTING,
            DeploymentStatus.LOADING,
            DeploymentStatus.READY,
            DeploymentStatus.UNHEALTHY,
        ]

    def calculate_current_cost(self) -> float:
        """Calculate current cost based on runtime."""
        if not self.started_at:
            return self.total_cost

        end_time = self.stopped_at or datetime.utcnow()
        runtime_seconds = (end_time - self.started_at).total_seconds()
        runtime_hours = runtime_seconds / 3600.0
        return runtime_hours * self.hourly_price


class ModelUsageLog(Base):
    """
    Usage log for tracking model requests and billing.

    Records individual API calls or batches for analytics and billing.
    """

    __tablename__ = "model_usage_logs"

    # References
    deployment_id: Mapped[str] = mapped_column(
        ForeignKey("deployments.id"),
        index=True,
    )
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)

    # Request info
    request_type: Mapped[str] = mapped_column(String(64))  # "chat", "completion", "image", etc.
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)

    # Status
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<ModelUsageLog {self.deployment_id}: {self.request_type}>"
