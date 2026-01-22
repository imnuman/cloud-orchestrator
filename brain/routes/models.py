"""
Model catalog and deployment routes for AI model serving.
Phase 2B: Customer AI Model Serving
"""

import secrets
from datetime import datetime
from typing import Annotated, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from brain.config import get_settings
from brain.models.base import get_db
from brain.models.user import User
from brain.models.model_catalog import (
    ModelTemplate,
    Deployment,
    ModelUsageLog,
    ModelCategory,
    ServingBackend,
    DeploymentStatus,
)
from brain.routes.auth import get_current_active_user

router = APIRouter(prefix="/models", tags=["Models"])
settings = get_settings()


def escape_like(value: str) -> str:
    """Escape special LIKE pattern characters to prevent SQL injection."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# =============================================================================
# Request/Response Schemas
# =============================================================================

class ModelTemplateResponse(BaseModel):
    """Model template information."""
    id: UUID
    name: str
    slug: str
    version: str
    category: ModelCategory
    serving_backend: ServingBackend
    min_vram_gb: int
    recommended_vram_gb: int
    min_gpu_count: int
    recommended_gpu: Optional[str]
    supports_quantization: bool
    docker_image: str
    openai_compatible: bool
    description: Optional[str]
    documentation_url: Optional[str]
    license: Optional[str]
    is_featured: bool
    deployment_count: int


class ModelTemplateDetailResponse(ModelTemplateResponse):
    """Detailed model template information."""
    default_env: dict
    default_ports: dict
    health_check_url: Optional[str]
    startup_timeout_seconds: int
    model_id: Optional[str]
    api_type: Optional[str]


class DeployModelRequest(BaseModel):
    """Request to deploy a model."""
    name: Optional[str] = Field(None, max_length=255, description="Custom deployment name")
    gpu_type: Optional[str] = Field(None, description="Preferred GPU type")
    gpu_count: int = Field(1, ge=1, le=8, description="Number of GPUs")
    custom_env: dict = Field(default_factory=dict, description="Custom environment variables")
    max_hourly_price: Optional[float] = Field(None, ge=0.01, description="Maximum hourly price")


class DeploymentResponse(BaseModel):
    """Deployment information."""
    id: UUID
    user_id: UUID
    model_template_id: UUID
    model_name: str
    model_slug: str
    name: str
    status: DeploymentStatus
    status_message: Optional[str]
    api_endpoint: Optional[str]
    api_key: Optional[str]
    ui_url: Optional[str]
    gpu_type: Optional[str]
    gpu_count: int
    hourly_price: float
    total_cost: float
    total_runtime_seconds: int
    started_at: Optional[datetime]
    stopped_at: Optional[datetime]
    created_at: datetime


class DeploymentDetailResponse(DeploymentResponse):
    """Detailed deployment information including metrics."""
    total_requests: int
    total_tokens_in: int
    total_tokens_out: int
    last_health_check_at: Optional[datetime]
    health_check_failures: int
    custom_env: dict


class DeploymentLogsResponse(BaseModel):
    """Deployment logs (placeholder)."""
    deployment_id: UUID
    logs: list[str]


# =============================================================================
# Helper Functions
# =============================================================================

def generate_deployment_api_key() -> str:
    """Generate a unique API key for a deployment."""
    return f"deploy_{secrets.token_urlsafe(32)}"


def model_template_to_response(template: ModelTemplate) -> ModelTemplateResponse:
    """Convert ModelTemplate to response schema."""
    return ModelTemplateResponse(
        id=UUID(template.id),
        name=template.name,
        slug=template.slug,
        version=template.version,
        category=template.category,
        serving_backend=template.serving_backend,
        min_vram_gb=template.min_vram_gb,
        recommended_vram_gb=template.recommended_vram_gb,
        min_gpu_count=template.min_gpu_count,
        recommended_gpu=template.recommended_gpu,
        supports_quantization=template.supports_quantization,
        docker_image=template.full_docker_image,
        openai_compatible=template.openai_compatible,
        description=template.description,
        documentation_url=template.documentation_url,
        license=template.license,
        is_featured=template.is_featured,
        deployment_count=template.deployment_count,
    )


def deployment_to_response(
    deployment: Deployment,
    model_template: ModelTemplate,
) -> DeploymentResponse:
    """Convert Deployment to response schema."""
    return DeploymentResponse(
        id=UUID(deployment.id),
        user_id=UUID(deployment.user_id),
        model_template_id=UUID(deployment.model_template_id),
        model_name=model_template.name,
        model_slug=model_template.slug,
        name=deployment.name,
        status=deployment.status,
        status_message=deployment.status_message,
        api_endpoint=deployment.api_endpoint,
        api_key=deployment.api_key,
        ui_url=deployment.ui_url,
        gpu_type=deployment.gpu_type,
        gpu_count=deployment.gpu_count,
        hourly_price=deployment.hourly_price,
        total_cost=deployment.calculate_current_cost(),
        total_runtime_seconds=deployment.total_runtime_seconds,
        started_at=deployment.started_at,
        stopped_at=deployment.stopped_at,
        created_at=deployment.created_at,
    )


# =============================================================================
# Model Catalog Endpoints
# =============================================================================

@router.get("", response_model=list[ModelTemplateResponse])
async def list_models(
    db: Annotated[AsyncSession, Depends(get_db)],
    category: Optional[ModelCategory] = None,
    featured_only: bool = False,
    search: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[ModelTemplateResponse]:
    """
    List available model templates.

    Filter by category, featured status, or search by name.
    """
    query = (
        select(ModelTemplate)
        .where(ModelTemplate.is_active == True)  # noqa: E712
        .order_by(ModelTemplate.display_order, ModelTemplate.name)
    )

    if category:
        query = query.where(ModelTemplate.category == category)
    if featured_only:
        query = query.where(ModelTemplate.is_featured == True)  # noqa: E712
    if search:
        escaped_search = escape_like(search)
        query = query.where(
            ModelTemplate.name.ilike(f"%{escaped_search}%")
            | ModelTemplate.description.ilike(f"%{escaped_search}%")
        )

    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    templates = result.scalars().all()

    return [model_template_to_response(t) for t in templates]


@router.get("/categories")
async def list_categories() -> list[dict]:
    """List all model categories with descriptions."""
    categories = [
        {
            "category": ModelCategory.TEXT.value,
            "name": "Text Generation",
            "description": "Large Language Models for text generation, chat, and completion",
            "examples": ["Llama 3.1", "Mistral", "Qwen", "DeepSeek"],
        },
        {
            "category": ModelCategory.IMAGE.value,
            "name": "Image Generation",
            "description": "Models for generating images from text prompts",
            "examples": ["SDXL", "Flux", "Stable Diffusion 3"],
        },
        {
            "category": ModelCategory.AUDIO.value,
            "name": "Audio Processing",
            "description": "Speech-to-text, text-to-speech, and audio processing",
            "examples": ["Whisper", "Coqui TTS", "Bark", "XTTS"],
        },
        {
            "category": ModelCategory.VIDEO.value,
            "name": "Video Generation",
            "description": "Models for video generation and processing",
            "examples": ["CogVideoX", "Mochi"],
        },
        {
            "category": ModelCategory.MULTIMODAL.value,
            "name": "Multimodal",
            "description": "Vision-language and multimodal models",
            "examples": ["LLaVA", "Qwen-VL", "PaliGemma"],
        },
        {
            "category": ModelCategory.EMBEDDING.value,
            "name": "Embeddings",
            "description": "Text embedding models for search and similarity",
            "examples": ["BGE", "E5", "GTE"],
        },
        {
            "category": ModelCategory.CODE.value,
            "name": "Code Generation",
            "description": "Specialized models for code generation",
            "examples": ["CodeLlama", "StarCoder", "DeepSeek-Coder"],
        },
    ]
    return categories


@router.get("/featured", response_model=list[ModelTemplateResponse])
async def list_featured_models(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ModelTemplateResponse]:
    """List featured models."""
    result = await db.execute(
        select(ModelTemplate)
        .where(ModelTemplate.is_active == True)  # noqa: E712
        .where(ModelTemplate.is_featured == True)  # noqa: E712
        .order_by(ModelTemplate.display_order, ModelTemplate.name)
        .limit(10)
    )
    templates = result.scalars().all()
    return [model_template_to_response(t) for t in templates]


@router.get("/{slug}", response_model=ModelTemplateDetailResponse)
async def get_model(
    slug: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelTemplateDetailResponse:
    """Get detailed information about a model template."""
    result = await db.execute(
        select(ModelTemplate).where(ModelTemplate.slug == slug)
    )
    template = result.scalar_one_or_none()

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{slug}' not found",
        )

    return ModelTemplateDetailResponse(
        id=UUID(template.id),
        name=template.name,
        slug=template.slug,
        version=template.version,
        category=template.category,
        serving_backend=template.serving_backend,
        min_vram_gb=template.min_vram_gb,
        recommended_vram_gb=template.recommended_vram_gb,
        min_gpu_count=template.min_gpu_count,
        recommended_gpu=template.recommended_gpu,
        supports_quantization=template.supports_quantization,
        docker_image=template.full_docker_image,
        openai_compatible=template.openai_compatible,
        description=template.description,
        documentation_url=template.documentation_url,
        license=template.license,
        is_featured=template.is_featured,
        deployment_count=template.deployment_count,
        default_env=template.default_env,
        default_ports=template.default_ports,
        health_check_url=template.health_check_url,
        startup_timeout_seconds=template.startup_timeout_seconds,
        model_id=template.model_id,
        api_type=template.api_type,
    )


# =============================================================================
# Deployment Endpoints
# =============================================================================

@router.post("/{slug}/deploy", response_model=DeploymentResponse, status_code=status.HTTP_201_CREATED)
async def deploy_model(
    slug: str,
    request: DeployModelRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DeploymentResponse:
    """
    Deploy a model.

    Creates a new deployment of the specified model.
    The deployment will be provisioned on an available GPU node.
    """
    # Get model template
    result = await db.execute(
        select(ModelTemplate).where(ModelTemplate.slug == slug)
    )
    template = result.scalar_one_or_none()

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{slug}' not found",
        )

    if not template.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This model is not currently available for deployment",
        )

    # Validate GPU count
    if request.gpu_count < template.min_gpu_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"This model requires at least {template.min_gpu_count} GPU(s)",
        )

    # Check user balance
    estimated_hourly = 0.50 * request.gpu_count  # Default estimate
    if not current_user.can_afford(estimated_hourly):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient balance. Please add funds to deploy.",
        )

    # Generate deployment name
    deployment_name = request.name or f"{template.name} - {datetime.utcnow().strftime('%Y%m%d-%H%M')}"

    # Create deployment record
    deployment = Deployment(
        user_id=current_user.id,
        model_template_id=template.id,
        name=deployment_name,
        status=DeploymentStatus.PENDING,
        gpu_count=request.gpu_count,
        gpu_type=request.gpu_type,
        custom_env=request.custom_env,
        api_key=generate_deployment_api_key(),
    )

    db.add(deployment)

    # Increment deployment count
    template.deployment_count += 1

    await db.flush()
    await db.refresh(deployment)

    # Note: The actual provisioning happens asynchronously via Celery task
    # The deployment service will pick this up and provision a pod

    return deployment_to_response(deployment, template)


@router.get("/deployments", response_model=list[DeploymentResponse])
async def list_deployments(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    status_filter: Optional[DeploymentStatus] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[DeploymentResponse]:
    """List user's deployments."""
    query = (
        select(Deployment)
        .where(Deployment.user_id == current_user.id)
        .options(selectinload(Deployment.model_template))
        .order_by(Deployment.created_at.desc())
    )

    if status_filter:
        query = query.where(Deployment.status == status_filter)

    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    deployments = result.scalars().all()

    return [
        deployment_to_response(d, d.model_template)
        for d in deployments
    ]


@router.get("/deployments/active", response_model=list[DeploymentResponse])
async def list_active_deployments(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[DeploymentResponse]:
    """List user's active deployments."""
    active_statuses = [
        DeploymentStatus.PENDING,
        DeploymentStatus.PROVISIONING,
        DeploymentStatus.STARTING,
        DeploymentStatus.LOADING,
        DeploymentStatus.READY,
        DeploymentStatus.UNHEALTHY,
    ]

    result = await db.execute(
        select(Deployment)
        .where(Deployment.user_id == current_user.id)
        .where(Deployment.status.in_(active_statuses))
        .options(selectinload(Deployment.model_template))
        .order_by(Deployment.created_at.desc())
    )
    deployments = result.scalars().all()

    return [
        deployment_to_response(d, d.model_template)
        for d in deployments
    ]


@router.get("/deployments/{deployment_id}", response_model=DeploymentDetailResponse)
async def get_deployment(
    deployment_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DeploymentDetailResponse:
    """Get detailed deployment information."""
    result = await db.execute(
        select(Deployment)
        .where(Deployment.id == str(deployment_id))
        .where(Deployment.user_id == current_user.id)
        .options(selectinload(Deployment.model_template))
    )
    deployment = result.scalar_one_or_none()

    if not deployment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deployment not found",
        )

    template = deployment.model_template

    return DeploymentDetailResponse(
        id=UUID(deployment.id),
        user_id=UUID(deployment.user_id),
        model_template_id=UUID(deployment.model_template_id),
        model_name=template.name,
        model_slug=template.slug,
        name=deployment.name,
        status=deployment.status,
        status_message=deployment.status_message,
        api_endpoint=deployment.api_endpoint,
        api_key=deployment.api_key,
        ui_url=deployment.ui_url,
        gpu_type=deployment.gpu_type,
        gpu_count=deployment.gpu_count,
        hourly_price=deployment.hourly_price,
        total_cost=deployment.calculate_current_cost(),
        total_runtime_seconds=deployment.total_runtime_seconds,
        started_at=deployment.started_at,
        stopped_at=deployment.stopped_at,
        created_at=deployment.created_at,
        total_requests=deployment.total_requests,
        total_tokens_in=deployment.total_tokens_in,
        total_tokens_out=deployment.total_tokens_out,
        last_health_check_at=deployment.last_health_check_at,
        health_check_failures=deployment.health_check_failures,
        custom_env=deployment.custom_env,
    )


@router.delete("/deployments/{deployment_id}", response_model=DeploymentResponse)
async def stop_deployment(
    deployment_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DeploymentResponse:
    """Stop and terminate a deployment."""
    result = await db.execute(
        select(Deployment)
        .where(Deployment.id == str(deployment_id))
        .where(Deployment.user_id == current_user.id)
        .options(selectinload(Deployment.model_template))
    )
    deployment = result.scalar_one_or_none()

    if not deployment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deployment not found",
        )

    if deployment.status in [DeploymentStatus.STOPPED, DeploymentStatus.STOPPING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Deployment is already stopped or stopping",
        )

    # Mark as stopping
    deployment.status = DeploymentStatus.STOPPING
    deployment.stopped_at = datetime.utcnow()
    deployment.total_cost = deployment.calculate_current_cost()

    # Note: The actual container termination happens via Celery task

    await db.flush()

    return deployment_to_response(deployment, deployment.model_template)


@router.get("/deployments/{deployment_id}/logs", response_model=DeploymentLogsResponse)
async def get_deployment_logs(
    deployment_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    lines: int = Query(100, ge=10, le=1000),
) -> DeploymentLogsResponse:
    """Get deployment container logs."""
    result = await db.execute(
        select(Deployment)
        .where(Deployment.id == str(deployment_id))
        .where(Deployment.user_id == current_user.id)
    )
    deployment = result.scalar_one_or_none()

    if not deployment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deployment not found",
        )

    # In production, this would fetch logs from the container
    # For now, return placeholder logs
    logs = [
        f"[{datetime.utcnow().isoformat()}] Deployment status: {deployment.status.value}",
        f"[{datetime.utcnow().isoformat()}] Model: {deployment.name}",
    ]

    if deployment.status_message:
        logs.append(f"[{datetime.utcnow().isoformat()}] {deployment.status_message}")

    return DeploymentLogsResponse(
        deployment_id=deployment_id,
        logs=logs,
    )


@router.get("/deployments/{deployment_id}/metrics")
async def get_deployment_metrics(
    deployment_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Get deployment metrics (GPU utilization, requests/sec, etc.)."""
    result = await db.execute(
        select(Deployment)
        .where(Deployment.id == str(deployment_id))
        .where(Deployment.user_id == current_user.id)
        .options(selectinload(Deployment.node))
    )
    deployment = result.scalar_one_or_none()

    if not deployment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deployment not found",
        )

    # Calculate runtime
    runtime_seconds = 0
    if deployment.started_at:
        end_time = deployment.stopped_at or datetime.utcnow()
        runtime_seconds = int((end_time - deployment.started_at).total_seconds())

    # Get GPU metrics if node is available
    gpu_utilization = None
    gpu_memory_used = None
    if deployment.node:
        gpu_utilization = deployment.node.current_gpu_utilization
        gpu_memory_used = deployment.node.current_vram_used_mb

    return {
        "deployment_id": str(deployment_id),
        "status": deployment.status.value,
        "runtime_seconds": runtime_seconds,
        "runtime_hours": runtime_seconds / 3600.0,
        "total_requests": deployment.total_requests,
        "total_tokens_in": deployment.total_tokens_in,
        "total_tokens_out": deployment.total_tokens_out,
        "total_cost": deployment.calculate_current_cost(),
        "gpu_utilization_percent": gpu_utilization,
        "gpu_memory_used_mb": gpu_memory_used,
        "health_check_failures": deployment.health_check_failures,
    }
