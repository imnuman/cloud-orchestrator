"""
Provider management routes for community GPU providers.
Phase 2: Community GPU Onboarding
"""

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
from brain.models.node import Node
from brain.models.provider import (
    Provider,
    Payout,
    ProviderEarning,
    PayoutMethod,
    PayoutStatus,
    VerificationLevel,
    generate_provider_key,
)
from brain.routes.auth import get_current_active_user
from shared.schemas import NodeStatus

router = APIRouter(prefix="/providers", tags=["Providers"])
settings = get_settings()


# =============================================================================
# Request/Response Schemas
# =============================================================================

class ProviderRegisterRequest(BaseModel):
    """Request to register as a GPU provider."""
    display_name: str = Field(..., min_length=2, max_length=255)
    company_name: Optional[str] = Field(None, max_length=255)
    payout_method: PayoutMethod = PayoutMethod.CRYPTO
    payout_address: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None, max_length=255)
    minimum_payout: float = Field(50.0, ge=10.0, le=1000.0)


class ProviderUpdateRequest(BaseModel):
    """Request to update provider settings."""
    display_name: Optional[str] = Field(None, min_length=2, max_length=255)
    company_name: Optional[str] = Field(None, max_length=255)
    payout_method: Optional[PayoutMethod] = None
    payout_address: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None, max_length=255)
    contact_phone: Optional[str] = Field(None, max_length=50)
    minimum_payout: Optional[float] = Field(None, ge=10.0, le=1000.0)


class ProviderResponse(BaseModel):
    """Provider information response."""
    id: UUID
    user_id: UUID
    display_name: str
    company_name: Optional[str]
    payout_method: PayoutMethod
    payout_address: Optional[str]
    minimum_payout: float
    total_earnings: float
    pending_earnings: float
    available_balance: float
    total_paid_out: float
    total_gpu_hours: float
    total_jobs_completed: int
    verification_level: VerificationLevel
    is_verified: bool
    is_active: bool
    platform_fee_percent: float
    provider_share_percent: float
    node_count: int
    created_at: datetime


class ProviderDashboardResponse(BaseModel):
    """Provider dashboard with earnings and stats."""
    provider: ProviderResponse
    nodes_online: int
    nodes_offline: int
    total_nodes: int
    active_pods: int
    today_earnings: float
    week_earnings: float
    month_earnings: float
    can_request_payout: bool
    install_command: str


class ProviderNodeResponse(BaseModel):
    """Node information for provider dashboard."""
    id: UUID
    hostname: str
    gpu_model: str
    gpu_count: int
    total_vram_mb: int
    status: str
    hourly_price: float
    current_pod_count: int
    max_pods: int
    last_heartbeat_at: Optional[datetime]
    total_earnings: float
    utilization_percent: Optional[int]


class NodePricingRequest(BaseModel):
    """Request to update node pricing."""
    hourly_price: float = Field(..., ge=0.01, le=100.0)


class PayoutRequestCreate(BaseModel):
    """Request to create a payout."""
    amount: Optional[float] = Field(None, ge=10.0)  # None = request all available


class PayoutResponse(BaseModel):
    """Payout information response."""
    id: UUID
    provider_id: UUID
    amount: float
    payout_method: PayoutMethod
    payout_address: str
    status: PayoutStatus
    status_message: Optional[str]
    created_at: datetime
    processed_at: Optional[datetime]
    transaction_id: Optional[str]


# =============================================================================
# Helper Functions
# =============================================================================

async def get_current_provider(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Provider:
    """Get the current user's provider account."""
    result = await db.execute(
        select(Provider)
        .where(Provider.user_id == current_user.id)
        .options(selectinload(Provider.nodes))
    )
    provider = result.scalar_one_or_none()

    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="You are not registered as a provider. Use POST /providers/register first.",
        )
    return provider


async def require_active_provider(
    provider: Annotated[Provider, Depends(get_current_provider)],
) -> Provider:
    """Require an active, non-suspended provider."""
    if not provider.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Provider account is inactive",
        )
    if provider.is_suspended:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Provider account is suspended: {provider.suspension_reason or 'Contact support'}",
        )
    return provider


def provider_to_response(provider: Provider) -> ProviderResponse:
    """Convert Provider model to response schema."""
    return ProviderResponse(
        id=UUID(provider.id),
        user_id=UUID(provider.user_id),
        display_name=provider.display_name,
        company_name=provider.company_name,
        payout_method=provider.payout_method,
        payout_address=provider.payout_address,
        minimum_payout=provider.minimum_payout,
        total_earnings=provider.total_earnings,
        pending_earnings=provider.pending_earnings,
        available_balance=provider.available_balance,
        total_paid_out=provider.total_paid_out,
        total_gpu_hours=provider.total_gpu_hours,
        total_jobs_completed=provider.total_jobs_completed,
        verification_level=provider.verification_level,
        is_verified=provider.is_verified,
        is_active=provider.is_active,
        platform_fee_percent=provider.platform_fee_percent,
        provider_share_percent=provider.provider_share_percent,
        node_count=provider.node_count,
        created_at=provider.created_at,
    )


# =============================================================================
# Provider Registration & Profile
# =============================================================================

@router.post("/register", response_model=ProviderResponse, status_code=status.HTTP_201_CREATED)
async def register_provider(
    request: ProviderRegisterRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProviderResponse:
    """
    Register as a GPU provider.

    After registration, you'll receive a provider key to use with the install script
    to connect your GPU machines to the platform.
    """
    # Check if already a provider
    result = await db.execute(
        select(Provider).where(Provider.user_id == current_user.id)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Already registered as a provider",
        )

    # Create provider account
    provider = Provider(
        user_id=current_user.id,
        display_name=request.display_name,
        company_name=request.company_name,
        payout_method=request.payout_method,
        payout_address=request.payout_address,
        contact_email=request.contact_email or current_user.email,
        minimum_payout=request.minimum_payout,
        provider_key=generate_provider_key(),
    )

    # Mark user as provider
    current_user.is_provider = True

    db.add(provider)
    await db.flush()
    await db.refresh(provider)

    return provider_to_response(provider)


@router.get("/me", response_model=ProviderResponse)
async def get_provider_profile(
    provider: Annotated[Provider, Depends(get_current_provider)],
) -> ProviderResponse:
    """Get current provider profile."""
    return provider_to_response(provider)


@router.patch("/me", response_model=ProviderResponse)
async def update_provider_profile(
    request: ProviderUpdateRequest,
    provider: Annotated[Provider, Depends(get_current_provider)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProviderResponse:
    """Update provider profile settings."""
    update_data = request.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(provider, field, value)

    await db.flush()
    await db.refresh(provider)

    return provider_to_response(provider)


@router.post("/me/regenerate-key", response_model=dict)
async def regenerate_provider_key(
    provider: Annotated[Provider, Depends(require_active_provider)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """
    Regenerate provider key.

    WARNING: This will invalidate the old key. You'll need to update
    your install scripts with the new key.
    """
    new_key = generate_provider_key()
    provider.provider_key = new_key

    await db.flush()

    return {
        "provider_key": new_key,
        "message": "Provider key regenerated. Update your install scripts with the new key.",
    }


# =============================================================================
# Provider Dashboard
# =============================================================================

@router.get("/dashboard", response_model=ProviderDashboardResponse)
async def get_provider_dashboard(
    provider: Annotated[Provider, Depends(get_current_provider)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProviderDashboardResponse:
    """
    Get provider dashboard with earnings, stats, and node status.
    """
    # Get node counts
    result = await db.execute(
        select(Node).where(Node.provider_user_id == provider.id)
    )
    nodes = result.scalars().all()

    nodes_online = sum(1 for n in nodes if n.status == NodeStatus.ONLINE)
    nodes_offline = len(nodes) - nodes_online
    active_pods = sum(n.current_pod_count for n in nodes)

    # Calculate earnings for different periods
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Simple week calculation (last 7 days)
    from datetime import timedelta
    week_start = week_start - timedelta(days=now.weekday())
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # Get today's earnings
    today_result = await db.execute(
        select(func.sum(ProviderEarning.provider_earnings))
        .where(ProviderEarning.provider_id == provider.id)
        .where(ProviderEarning.period_start >= today_start)
    )
    today_earnings = today_result.scalar() or 0.0

    # Get week's earnings
    week_result = await db.execute(
        select(func.sum(ProviderEarning.provider_earnings))
        .where(ProviderEarning.provider_id == provider.id)
        .where(ProviderEarning.period_start >= week_start)
    )
    week_earnings = week_result.scalar() or 0.0

    # Get month's earnings
    month_result = await db.execute(
        select(func.sum(ProviderEarning.provider_earnings))
        .where(ProviderEarning.provider_id == provider.id)
        .where(ProviderEarning.period_start >= month_start)
    )
    month_earnings = month_result.scalar() or 0.0

    # Generate install command
    brain_url = settings.brain_public_url.rstrip("/")
    install_command = f'curl -sSL {brain_url}/api/v1/nodes/install.sh | bash -s -- --provider-key={provider.provider_key}'

    return ProviderDashboardResponse(
        provider=provider_to_response(provider),
        nodes_online=nodes_online,
        nodes_offline=nodes_offline,
        total_nodes=len(nodes),
        active_pods=active_pods,
        today_earnings=today_earnings,
        week_earnings=week_earnings,
        month_earnings=month_earnings,
        can_request_payout=provider.can_request_payout,
        install_command=install_command,
    )


# =============================================================================
# Provider Nodes
# =============================================================================

@router.get("/nodes", response_model=list[ProviderNodeResponse])
async def list_provider_nodes(
    provider: Annotated[Provider, Depends(get_current_provider)],
    db: Annotated[AsyncSession, Depends(get_db)],
    status: Optional[NodeStatus] = None,
) -> list[ProviderNodeResponse]:
    """List all nodes owned by this provider."""
    query = select(Node).where(Node.provider_user_id == provider.id)

    if status:
        query = query.where(Node.status == status)

    result = await db.execute(query)
    nodes = result.scalars().all()

    # Get earnings per node
    earnings_result = await db.execute(
        select(ProviderEarning.node_id, func.sum(ProviderEarning.provider_earnings))
        .where(ProviderEarning.provider_id == provider.id)
        .group_by(ProviderEarning.node_id)
    )
    earnings_by_node = {row[0]: row[1] for row in earnings_result.all()}

    return [
        ProviderNodeResponse(
            id=UUID(node.id),
            hostname=node.hostname,
            gpu_model=node.gpu_model,
            gpu_count=node.gpu_count,
            total_vram_mb=node.total_vram_mb,
            status=node.status.value,
            hourly_price=node.hourly_price,
            current_pod_count=node.current_pod_count,
            max_pods=node.max_pods,
            last_heartbeat_at=node.last_heartbeat_at,
            total_earnings=earnings_by_node.get(node.id, 0.0),
            utilization_percent=node.current_gpu_utilization,
        )
        for node in nodes
    ]


@router.get("/nodes/{node_id}", response_model=ProviderNodeResponse)
async def get_provider_node(
    node_id: UUID,
    provider: Annotated[Provider, Depends(get_current_provider)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProviderNodeResponse:
    """Get details of a specific node owned by this provider."""
    result = await db.execute(
        select(Node)
        .where(Node.id == str(node_id))
        .where(Node.provider_user_id == provider.id)
    )
    node = result.scalar_one_or_none()

    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Node not found or not owned by you",
        )

    # Get earnings for this node
    earnings_result = await db.execute(
        select(func.sum(ProviderEarning.provider_earnings))
        .where(ProviderEarning.node_id == str(node_id))
        .where(ProviderEarning.provider_id == provider.id)
    )
    total_earnings = earnings_result.scalar() or 0.0

    return ProviderNodeResponse(
        id=UUID(node.id),
        hostname=node.hostname,
        gpu_model=node.gpu_model,
        gpu_count=node.gpu_count,
        total_vram_mb=node.total_vram_mb,
        status=node.status.value,
        hourly_price=node.hourly_price,
        current_pod_count=node.current_pod_count,
        max_pods=node.max_pods,
        last_heartbeat_at=node.last_heartbeat_at,
        total_earnings=total_earnings,
        utilization_percent=node.current_gpu_utilization,
    )


@router.patch("/nodes/{node_id}/pricing", response_model=ProviderNodeResponse)
async def update_node_pricing(
    node_id: UUID,
    request: NodePricingRequest,
    provider: Annotated[Provider, Depends(require_active_provider)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProviderNodeResponse:
    """Update pricing for a node."""
    result = await db.execute(
        select(Node)
        .where(Node.id == str(node_id))
        .where(Node.provider_user_id == provider.id)
    )
    node = result.scalar_one_or_none()

    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Node not found or not owned by you",
        )

    node.hourly_price = request.hourly_price
    await db.flush()

    # Get earnings
    earnings_result = await db.execute(
        select(func.sum(ProviderEarning.provider_earnings))
        .where(ProviderEarning.node_id == str(node_id))
    )
    total_earnings = earnings_result.scalar() or 0.0

    return ProviderNodeResponse(
        id=UUID(node.id),
        hostname=node.hostname,
        gpu_model=node.gpu_model,
        gpu_count=node.gpu_count,
        total_vram_mb=node.total_vram_mb,
        status=node.status.value,
        hourly_price=node.hourly_price,
        current_pod_count=node.current_pod_count,
        max_pods=node.max_pods,
        last_heartbeat_at=node.last_heartbeat_at,
        total_earnings=total_earnings,
        utilization_percent=node.current_gpu_utilization,
    )


# =============================================================================
# Payouts
# =============================================================================

@router.get("/payouts", response_model=list[PayoutResponse])
async def list_payouts(
    provider: Annotated[Provider, Depends(get_current_provider)],
    db: Annotated[AsyncSession, Depends(get_db)],
    status: Optional[PayoutStatus] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[PayoutResponse]:
    """List payout history."""
    query = (
        select(Payout)
        .where(Payout.provider_id == provider.id)
        .order_by(Payout.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    if status:
        query = query.where(Payout.status == status)

    result = await db.execute(query)
    payouts = result.scalars().all()

    return [
        PayoutResponse(
            id=UUID(p.id),
            provider_id=UUID(p.provider_id),
            amount=p.amount,
            payout_method=p.payout_method,
            payout_address=p.payout_address,
            status=p.status,
            status_message=p.status_message,
            created_at=p.created_at,
            processed_at=p.processed_at,
            transaction_id=p.transaction_id,
        )
        for p in payouts
    ]


@router.post("/payouts/request", response_model=PayoutResponse, status_code=status.HTTP_201_CREATED)
async def request_payout(
    request: PayoutRequestCreate,
    provider: Annotated[Provider, Depends(require_active_provider)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PayoutResponse:
    """
    Request a payout of available earnings.

    If amount is not specified, requests all available balance.
    """
    if not provider.payout_address:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please set a payout address first",
        )

    # Determine amount
    amount = request.amount if request.amount else provider.available_balance

    if amount > provider.available_balance:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Requested amount (${amount:.2f}) exceeds available balance (${provider.available_balance:.2f})",
        )

    if amount < provider.minimum_payout:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Minimum payout amount is ${provider.minimum_payout:.2f}",
        )

    # Check for pending payouts
    pending_result = await db.execute(
        select(Payout)
        .where(Payout.provider_id == provider.id)
        .where(Payout.status.in_([PayoutStatus.PENDING, PayoutStatus.PROCESSING]))
    )
    if pending_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You already have a pending payout request",
        )

    # Create payout request
    payout = Payout(
        provider_id=provider.id,
        amount=amount,
        payout_method=provider.payout_method,
        payout_address=provider.payout_address,
        status=PayoutStatus.PENDING,
    )

    # Reduce available balance
    provider.available_balance -= amount

    db.add(payout)
    await db.flush()
    await db.refresh(payout)

    return PayoutResponse(
        id=UUID(payout.id),
        provider_id=UUID(payout.provider_id),
        amount=payout.amount,
        payout_method=payout.payout_method,
        payout_address=payout.payout_address,
        status=payout.status,
        status_message=payout.status_message,
        created_at=payout.created_at,
        processed_at=payout.processed_at,
        transaction_id=payout.transaction_id,
    )


@router.delete("/payouts/{payout_id}", response_model=PayoutResponse)
async def cancel_payout(
    payout_id: UUID,
    provider: Annotated[Provider, Depends(get_current_provider)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PayoutResponse:
    """Cancel a pending payout request."""
    result = await db.execute(
        select(Payout)
        .where(Payout.id == str(payout_id))
        .where(Payout.provider_id == provider.id)
    )
    payout = result.scalar_one_or_none()

    if not payout:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payout not found",
        )

    if payout.status != PayoutStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel payout with status: {payout.status.value}",
        )

    # Cancel and refund
    payout.status = PayoutStatus.CANCELLED
    payout.status_message = "Cancelled by provider"
    provider.available_balance += payout.amount

    await db.flush()

    return PayoutResponse(
        id=UUID(payout.id),
        provider_id=UUID(payout.provider_id),
        amount=payout.amount,
        payout_method=payout.payout_method,
        payout_address=payout.payout_address,
        status=payout.status,
        status_message=payout.status_message,
        created_at=payout.created_at,
        processed_at=payout.processed_at,
        transaction_id=payout.transaction_id,
    )


# =============================================================================
# Provider Key Validation (for install script)
# =============================================================================

@router.get("/validate-key/{provider_key}")
async def validate_provider_key(
    provider_key: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """
    Validate a provider key (called by install script).

    Returns provider info if valid, used to link nodes during registration.
    """
    result = await db.execute(
        select(Provider).where(Provider.provider_key == provider_key)
    )
    provider = result.scalar_one_or_none()

    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid provider key",
        )

    if not provider.is_active or provider.is_suspended:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Provider account is not active",
        )

    return {
        "valid": True,
        "provider_id": provider.id,
        "display_name": provider.display_name,
    }


# =============================================================================
# Suggested Pricing (for providers)
# =============================================================================

@router.get("/suggested-pricing")
async def get_suggested_pricing(
    gpu_model: str = Query(..., description="GPU model name (e.g., 'RTX 4090')"),
) -> dict:
    """
    Get suggested pricing for a GPU model based on market rates.

    This helps providers set competitive prices.
    """
    # Market rate estimates (these would typically come from provider APIs)
    suggested_rates = {
        "RTX 4090": {"low": 0.35, "mid": 0.55, "high": 0.75},
        "RTX 3090": {"low": 0.25, "mid": 0.40, "high": 0.55},
        "RTX 3080": {"low": 0.18, "mid": 0.28, "high": 0.38},
        "A100 40GB": {"low": 1.20, "mid": 1.80, "high": 2.40},
        "A100 80GB": {"low": 1.80, "mid": 2.50, "high": 3.20},
        "H100": {"low": 2.50, "mid": 3.50, "high": 4.50},
        "A6000": {"low": 0.60, "mid": 0.90, "high": 1.20},
        "A5000": {"low": 0.40, "mid": 0.60, "high": 0.80},
    }

    # Find matching GPU
    gpu_lower = gpu_model.lower()
    for gpu_name, rates in suggested_rates.items():
        if gpu_name.lower() in gpu_lower or gpu_lower in gpu_name.lower():
            return {
                "gpu_model": gpu_model,
                "matched_gpu": gpu_name,
                "suggested_hourly_price": rates["mid"],
                "price_range": rates,
                "note": "Prices in USD per hour. Mid-range pricing is recommended for new providers.",
            }

    # Default for unknown GPUs
    return {
        "gpu_model": gpu_model,
        "matched_gpu": None,
        "suggested_hourly_price": 0.50,
        "price_range": {"low": 0.30, "mid": 0.50, "high": 0.70},
        "note": "GPU not in our database. Using default pricing. Adjust based on performance.",
    }
