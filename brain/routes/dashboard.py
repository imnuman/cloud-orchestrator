"""
Dashboard API routes.

Provides unified endpoints for the dashboard UI including:
- Overview statistics
- GPU availability across providers
- Cost analytics
- Provider health monitoring
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from brain.models.base import get_async_session
from brain.models.node import Node
from brain.models.pod import Pod
from brain.models.user import User
from brain.models.billing import Transaction
from brain.models.provisioned_instance import ProvisionedInstance, ProvisioningStatus
from brain.routes.auth import get_current_user
from brain.services.multi_provider import (
    MultiProviderService,
    PricingStrategy,
    get_multi_provider_service,
)
from brain.services.sourcing import get_sourcing_service
from shared.schemas import NodeStatus, PodStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dashboard", tags=["dashboard"])


# ==============================================================================
# Response Models
# ==============================================================================

class ProviderHealthResponse(BaseModel):
    """Provider health status."""

    provider: str
    is_healthy: bool
    consecutive_failures: int
    avg_response_time_ms: float
    error_message: Optional[str] = None


class GpuAvailabilityResponse(BaseModel):
    """GPU availability across providers."""

    gpu_type: str
    providers: list[dict] = Field(default_factory=list)
    lowest_price: Optional[float] = None
    lowest_price_provider: Optional[str] = None
    total_available: int = 0


class OverviewStatsResponse(BaseModel):
    """Dashboard overview statistics."""

    # User stats
    total_users: int = 0
    active_users: int = 0

    # Node stats
    total_nodes: int = 0
    online_nodes: int = 0
    total_gpus: int = 0

    # Pod stats
    total_pods: int = 0
    running_pods: int = 0

    # Financial stats
    total_revenue: float = 0.0
    total_provider_costs: float = 0.0
    gross_profit: float = 0.0

    # Instance stats
    provisioned_instances: int = 0
    active_instances: int = 0


class CostAnalyticsResponse(BaseModel):
    """Cost analytics data."""

    period: str
    total_revenue: float = 0.0
    total_costs: float = 0.0
    gross_profit: float = 0.0
    profit_margin: float = 0.0
    by_gpu_type: dict = Field(default_factory=dict)
    by_provider: dict = Field(default_factory=dict)


class GpuOfferResponse(BaseModel):
    """GPU offer details."""

    offer_id: str
    provider: str
    gpu_name: str
    gpu_count: int
    gpu_vram_mb: int
    hourly_price: float
    our_price: float  # With markup
    reliability_score: float
    location: Optional[str] = None
    value_score: float = 0.0


# ==============================================================================
# Dashboard Endpoints
# ==============================================================================

@router.get("/overview", response_model=OverviewStatsResponse)
async def get_overview_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
) -> OverviewStatsResponse:
    """
    Get dashboard overview statistics.

    Requires authentication. Admin users see all stats,
    regular users see only their own.
    """
    stats = OverviewStatsResponse()

    # User stats (admin only)
    if current_user.is_admin:
        result = await db.execute(select(func.count(User.id)))
        stats.total_users = result.scalar() or 0

        result = await db.execute(
            select(func.count(User.id)).where(User.is_active == True)
        )
        stats.active_users = result.scalar() or 0

    # Node stats
    result = await db.execute(select(func.count(Node.id)))
    stats.total_nodes = result.scalar() or 0

    result = await db.execute(
        select(func.count(Node.id)).where(Node.status == NodeStatus.ONLINE)
    )
    stats.online_nodes = result.scalar() or 0

    result = await db.execute(select(func.coalesce(func.sum(Node.gpu_count), 0)))
    stats.total_gpus = result.scalar() or 0

    # Pod stats
    if current_user.is_admin:
        result = await db.execute(select(func.count(Pod.id)))
        stats.total_pods = result.scalar() or 0

        result = await db.execute(
            select(func.count(Pod.id)).where(Pod.status == PodStatus.RUNNING)
        )
        stats.running_pods = result.scalar() or 0
    else:
        # User's own pods
        result = await db.execute(
            select(func.count(Pod.id)).where(Pod.user_id == str(current_user.id))
        )
        stats.total_pods = result.scalar() or 0

        result = await db.execute(
            select(func.count(Pod.id)).where(
                Pod.user_id == str(current_user.id),
                Pod.status == PodStatus.RUNNING,
            )
        )
        stats.running_pods = result.scalar() or 0

    # Financial stats (admin only)
    if current_user.is_admin:
        result = await db.execute(
            select(func.coalesce(func.sum(Pod.total_cost), 0.0))
        )
        stats.total_revenue = result.scalar() or 0.0

        result = await db.execute(
            select(func.coalesce(func.sum(ProvisionedInstance.total_cost), 0.0))
        )
        stats.total_provider_costs = result.scalar() or 0.0

        stats.gross_profit = stats.total_revenue - stats.total_provider_costs

    # Provisioned instances
    result = await db.execute(select(func.count(ProvisionedInstance.id)))
    stats.provisioned_instances = result.scalar() or 0

    result = await db.execute(
        select(func.count(ProvisionedInstance.id)).where(
            ProvisionedInstance.status == ProvisioningStatus.ACTIVE
        )
    )
    stats.active_instances = result.scalar() or 0

    return stats


@router.get("/provider-health", response_model=list[ProviderHealthResponse])
async def get_provider_health(
    current_user: User = Depends(get_current_user),
) -> list[ProviderHealthResponse]:
    """
    Get health status for all GPU providers.
    """
    service = get_multi_provider_service()
    health_data = service.get_provider_health()

    return [
        ProviderHealthResponse(
            provider=name,
            is_healthy=health.is_healthy,
            consecutive_failures=health.consecutive_failures,
            avg_response_time_ms=health.avg_response_time_ms,
            error_message=health.error_message,
        )
        for name, health in health_data.items()
    ]


@router.get("/gpu-availability", response_model=list[GpuAvailabilityResponse])
async def get_gpu_availability(
    current_user: User = Depends(get_current_user),
    gpu_types: Optional[str] = Query(
        None,
        description="Comma-separated list of GPU types to check",
    ),
) -> list[GpuAvailabilityResponse]:
    """
    Get GPU availability across all providers.
    """
    service = get_multi_provider_service()

    # Parse GPU types if provided
    target_types = None
    if gpu_types:
        target_types = [t.strip() for t in gpu_types.split(",")]

    # Get best offers from all providers
    all_offers = await service.list_all_offers()

    # Group by GPU type
    by_type: dict[str, list] = {}
    for offer in all_offers:
        # Filter by target types if specified
        if target_types:
            if not any(t.lower() in offer.gpu_name.lower() for t in target_types):
                continue

        gpu_type = offer.gpu_name
        if gpu_type not in by_type:
            by_type[gpu_type] = []
        by_type[gpu_type].append(offer)

    # Build response
    result = []
    for gpu_type, offers in by_type.items():
        # Group by provider
        by_provider = {}
        for offer in offers:
            if offer.provider not in by_provider:
                by_provider[offer.provider] = []
            by_provider[offer.provider].append(offer)

        # Find lowest price
        lowest = min(offers, key=lambda o: o.hourly_price)

        result.append(
            GpuAvailabilityResponse(
                gpu_type=gpu_type,
                providers=[
                    {
                        "provider": provider,
                        "count": len(provider_offers),
                        "lowest_price": min(o.hourly_price for o in provider_offers),
                    }
                    for provider, provider_offers in by_provider.items()
                ],
                lowest_price=lowest.hourly_price,
                lowest_price_provider=lowest.provider,
                total_available=len(offers),
            )
        )

    # Sort by GPU type
    result.sort(key=lambda r: r.gpu_type)

    return result


@router.get("/gpu-offers", response_model=list[GpuOfferResponse])
async def get_gpu_offers(
    current_user: User = Depends(get_current_user),
    gpu_type: Optional[str] = Query(None, description="Filter by GPU type"),
    max_price: Optional[float] = Query(None, description="Maximum hourly price"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    strategy: str = Query("balanced", description="Pricing strategy"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results"),
) -> list[GpuOfferResponse]:
    """
    Get available GPU offers with pricing optimization.
    """
    from brain.config import get_settings

    settings = get_settings()
    markup = 1 + (settings.default_markup_percent / 100)

    # Map strategy string to enum
    strategy_map = {
        "lowest_price": PricingStrategy.LOWEST_PRICE,
        "best_value": PricingStrategy.BEST_VALUE,
        "reliability": PricingStrategy.HIGHEST_RELIABILITY,
        "balanced": PricingStrategy.BALANCED,
    }
    pricing_strategy = strategy_map.get(strategy, PricingStrategy.BALANCED)

    service = get_multi_provider_service()

    offers = await service.find_best_offers(
        gpu_type=gpu_type,
        max_price=max_price,
        strategy=pricing_strategy,
        limit=limit,
    )

    # Filter by provider if specified
    if provider:
        offers = [o for o in offers if o.offer.provider == provider]

    return [
        GpuOfferResponse(
            offer_id=agg.offer.offer_id,
            provider=agg.offer.provider,
            gpu_name=agg.offer.gpu_name,
            gpu_count=agg.offer.gpu_count,
            gpu_vram_mb=agg.offer.gpu_vram_mb,
            hourly_price=agg.offer.hourly_price,
            our_price=round(agg.offer.hourly_price * markup, 2),
            reliability_score=agg.offer.reliability_score,
            location=agg.offer.location,
            value_score=round(agg.value_score, 4),
        )
        for agg in offers
    ]


@router.get("/cost-analytics", response_model=CostAnalyticsResponse)
async def get_cost_analytics(
    current_user: User = Depends(get_current_user),
    period: str = Query("day", description="Period: day, week, month"),
    db: AsyncSession = Depends(get_async_session),
) -> CostAnalyticsResponse:
    """
    Get cost analytics for the specified period.

    Admin-only endpoint.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    # Calculate time range
    now = datetime.utcnow()
    if period == "day":
        start_time = now - timedelta(days=1)
    elif period == "week":
        start_time = now - timedelta(weeks=1)
    elif period == "month":
        start_time = now - timedelta(days=30)
    else:
        start_time = now - timedelta(days=1)

    # Get revenue from pods
    result = await db.execute(
        select(
            func.coalesce(func.sum(Pod.total_cost), 0.0),
            Pod.gpu_type,
        )
        .where(Pod.created_at >= start_time)
        .group_by(Pod.gpu_type)
    )
    revenue_by_gpu = {row[1]: row[0] for row in result.all()}
    total_revenue = sum(revenue_by_gpu.values())

    # Get costs from provisioned instances
    result = await db.execute(
        select(
            func.coalesce(func.sum(ProvisionedInstance.total_cost), 0.0),
            ProvisionedInstance.provider_type,
        )
        .where(ProvisionedInstance.created_at >= start_time)
        .group_by(ProvisionedInstance.provider_type)
    )
    costs_by_provider = {str(row[1].value): row[0] for row in result.all()}
    total_costs = sum(costs_by_provider.values())

    gross_profit = total_revenue - total_costs
    profit_margin = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0

    return CostAnalyticsResponse(
        period=period,
        total_revenue=round(total_revenue, 2),
        total_costs=round(total_costs, 2),
        gross_profit=round(gross_profit, 2),
        profit_margin=round(profit_margin, 1),
        by_gpu_type=revenue_by_gpu,
        by_provider=costs_by_provider,
    )


@router.get("/price-comparison")
async def get_price_comparison(
    current_user: User = Depends(get_current_user),
    gpu_type: str = Query(..., description="GPU type to compare"),
) -> dict:
    """
    Get price comparison for a specific GPU across all providers.
    """
    sourcing_service = get_sourcing_service()

    comparison = await sourcing_service.get_provider_comparison(gpu_type)

    from brain.config import get_settings

    settings = get_settings()
    markup = 1 + (settings.default_markup_percent / 100)

    result = {
        "gpu_type": gpu_type,
        "markup_percent": settings.default_markup_percent,
        "providers": {},
    }

    for provider, offers in comparison.items():
        if offers:
            lowest = min(o.hourly_price for o in offers)
            highest = max(o.hourly_price for o in offers)
            avg = sum(o.hourly_price for o in offers) / len(offers)

            result["providers"][provider] = {
                "count": len(offers),
                "lowest_price": round(lowest, 2),
                "highest_price": round(highest, 2),
                "average_price": round(avg, 2),
                "our_lowest_price": round(lowest * markup, 2),
                "offers": [
                    {
                        "offer_id": o.offer_id,
                        "hourly_price": o.hourly_price,
                        "our_price": round(o.hourly_price * markup, 2),
                        "vram_mb": o.gpu_vram_mb,
                        "location": o.location,
                    }
                    for o in offers[:5]  # Top 5 per provider
                ],
            }

    return result
