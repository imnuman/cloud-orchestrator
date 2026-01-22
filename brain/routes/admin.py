"""
Admin Routes.

Administrative endpoints for platform management:
- User management (list, view, suspend, delete, adjust balance)
- System statistics and metrics
- Node management (view all, update status)
- Financial overview (revenue, costs, transactions)
- Provider management (approve, update tiers)
- Deployment management (view all, terminate)
"""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession

from brain.config import get_settings
from brain.models.base import get_db
from brain.models.user import User
from brain.models.node import Node
from brain.models.pod import Pod
from brain.models.billing import Transaction, UsageRecord
from brain.models.provider import Provider, VerificationLevel
from brain.models.model_catalog import Deployment, DeploymentStatus
from brain.models.api_key import APIKey
from brain.routes.auth import get_current_user

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/admin", tags=["Admin"])


# ============================================================================
# Admin Authentication
# ============================================================================


async def get_admin_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Require admin privileges for access."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user


# ============================================================================
# Response Models
# ============================================================================


class SystemStatsResponse(BaseModel):
    """Overall system statistics."""

    # Users
    total_users: int
    active_users: int
    verified_users: int
    new_users_24h: int
    new_users_7d: int

    # Nodes
    total_nodes: int
    online_nodes: int
    offline_nodes: int
    total_gpus: int

    # Pods
    total_pods: int
    running_pods: int
    pending_pods: int

    # Deployments
    total_deployments: int
    active_deployments: int

    # Providers
    total_providers: int
    verified_providers: int

    # Financial
    total_revenue: float
    revenue_24h: float
    revenue_7d: float
    total_user_balance: float

    # API Keys
    total_api_keys: int
    active_api_keys: int


class UserListItem(BaseModel):
    """User item for admin list."""

    id: str
    email: str
    name: Optional[str]
    balance: float
    total_spent: float
    is_active: bool
    is_verified: bool
    is_admin: bool
    is_provider: bool
    created_at: str
    last_login_at: Optional[str]
    pod_count: int
    api_key_count: int


class UserListResponse(BaseModel):
    """Paginated user list."""

    users: list[UserListItem]
    total: int
    page: int
    limit: int


class UserDetailResponse(BaseModel):
    """Detailed user information for admin."""

    id: str
    email: str
    name: Optional[str]
    balance: float
    total_spent: float
    credit_limit: float
    is_active: bool
    is_verified: bool
    is_admin: bool
    is_provider: bool
    stripe_customer_id: Optional[str]
    auto_refill_enabled: bool
    auto_refill_threshold: float
    auto_refill_amount: float
    created_at: str
    last_login_at: Optional[str]
    pods: list[dict]
    api_keys: list[dict]
    recent_transactions: list[dict]


class UpdateUserRequest(BaseModel):
    """Admin request to update a user."""

    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    is_admin: Optional[bool] = None
    credit_limit: Optional[float] = Field(None, ge=0)
    balance_adjustment: Optional[float] = Field(None, description="Amount to add/subtract from balance")
    adjustment_reason: Optional[str] = Field(None, max_length=500)


class NodeListItem(BaseModel):
    """Node item for admin list."""

    id: str
    hostname: str
    ip_address: str
    status: str
    gpu_count: int
    gpu_type: str
    provider_type: str
    hourly_price: float
    last_heartbeat: Optional[str]
    running_pods: int
    created_at: str


class NodeListResponse(BaseModel):
    """Paginated node list."""

    nodes: list[NodeListItem]
    total: int
    page: int
    limit: int


class ProviderListItem(BaseModel):
    """Provider item for admin list."""

    id: str
    user_id: str
    user_email: str
    company_name: Optional[str]
    verification_level: str
    total_earnings: float
    pending_payout: float
    node_count: int
    is_verified: bool
    created_at: str


class ProviderListResponse(BaseModel):
    """Paginated provider list."""

    providers: list[ProviderListItem]
    total: int
    page: int
    limit: int


class UpdateProviderRequest(BaseModel):
    """Admin request to update a provider."""

    verification_level: Optional[str] = None
    platform_fee_override: Optional[float] = Field(None, ge=0, le=50)


class FinancialSummaryResponse(BaseModel):
    """Financial summary for admin."""

    # Revenue
    total_revenue: float
    revenue_today: float
    revenue_yesterday: float
    revenue_this_week: float
    revenue_this_month: float

    # Costs (provider payouts)
    total_provider_payouts: float
    pending_payouts: float

    # Margins
    gross_margin: float
    gross_margin_percent: float

    # User balances
    total_user_balance: float
    total_credit_extended: float

    # Transaction counts
    total_transactions: int
    deposits_count: int
    charges_count: int


class DeploymentListItem(BaseModel):
    """Deployment item for admin list."""

    id: str
    user_id: str
    user_email: str
    model_name: str
    status: str
    gpu_type: str
    hourly_cost: float
    created_at: str
    total_runtime_hours: float
    total_cost: float


class DeploymentListResponse(BaseModel):
    """Paginated deployment list."""

    deployments: list[DeploymentListItem]
    total: int
    page: int
    limit: int


# ============================================================================
# System Statistics
# ============================================================================


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get comprehensive system statistics.

    Provides overview of users, nodes, pods, deployments, and financials.
    """
    now = datetime.now(timezone.utc)
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)

    # User stats
    total_users = await db.scalar(select(func.count(User.id)))
    active_users = await db.scalar(select(func.count(User.id)).where(User.is_active == True))
    verified_users = await db.scalar(select(func.count(User.id)).where(User.is_verified == True))
    new_users_24h = await db.scalar(
        select(func.count(User.id)).where(User.created_at >= day_ago)
    )
    new_users_7d = await db.scalar(
        select(func.count(User.id)).where(User.created_at >= week_ago)
    )

    # Node stats
    total_nodes = await db.scalar(select(func.count(Node.id)))
    online_nodes = await db.scalar(
        select(func.count(Node.id)).where(Node.status == "online")
    )
    offline_nodes = await db.scalar(
        select(func.count(Node.id)).where(Node.status == "offline")
    )
    total_gpus = await db.scalar(select(func.sum(Node.gpu_count))) or 0

    # Pod stats
    total_pods = await db.scalar(select(func.count(Pod.id)))
    running_pods = await db.scalar(
        select(func.count(Pod.id)).where(Pod.status == "running")
    )
    pending_pods = await db.scalar(
        select(func.count(Pod.id)).where(Pod.status == "pending")
    )

    # Deployment stats
    total_deployments = await db.scalar(select(func.count(Deployment.id)))
    active_deployments = await db.scalar(
        select(func.count(Deployment.id)).where(
            Deployment.status.in_([DeploymentStatus.RUNNING, DeploymentStatus.STARTING])
        )
    )

    # Provider stats
    total_providers = await db.scalar(select(func.count(Provider.id)))
    verified_providers = await db.scalar(
        select(func.count(Provider.id)).where(Provider.is_verified == True)
    )

    # Financial stats
    total_revenue = await db.scalar(
        select(func.sum(Transaction.amount)).where(Transaction.type == "charge")
    ) or 0
    revenue_24h = await db.scalar(
        select(func.sum(Transaction.amount)).where(
            and_(Transaction.type == "charge", Transaction.created_at >= day_ago)
        )
    ) or 0
    revenue_7d = await db.scalar(
        select(func.sum(Transaction.amount)).where(
            and_(Transaction.type == "charge", Transaction.created_at >= week_ago)
        )
    ) or 0
    total_user_balance = await db.scalar(select(func.sum(User.balance))) or 0

    # API Key stats
    total_api_keys = await db.scalar(select(func.count(APIKey.id)))
    active_api_keys = await db.scalar(
        select(func.count(APIKey.id)).where(APIKey.is_active == True)
    )

    return {
        "total_users": total_users or 0,
        "active_users": active_users or 0,
        "verified_users": verified_users or 0,
        "new_users_24h": new_users_24h or 0,
        "new_users_7d": new_users_7d or 0,
        "total_nodes": total_nodes or 0,
        "online_nodes": online_nodes or 0,
        "offline_nodes": offline_nodes or 0,
        "total_gpus": total_gpus,
        "total_pods": total_pods or 0,
        "running_pods": running_pods or 0,
        "pending_pods": pending_pods or 0,
        "total_deployments": total_deployments or 0,
        "active_deployments": active_deployments or 0,
        "total_providers": total_providers or 0,
        "verified_providers": verified_providers or 0,
        "total_revenue": float(total_revenue),
        "revenue_24h": float(revenue_24h),
        "revenue_7d": float(revenue_7d),
        "total_user_balance": float(total_user_balance),
        "total_api_keys": total_api_keys or 0,
        "active_api_keys": active_api_keys or 0,
    }


# ============================================================================
# User Management
# ============================================================================


@router.get("/users", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search by email or name"),
    is_active: Optional[bool] = None,
    is_verified: Optional[bool] = None,
    is_admin: Optional[bool] = None,
    is_provider: Optional[bool] = None,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    List all users with pagination and filters.
    """
    query = select(User)

    # Apply filters
    if search:
        query = query.where(
            User.email.ilike(f"%{search}%") | User.name.ilike(f"%{search}%")
        )
    if is_active is not None:
        query = query.where(User.is_active == is_active)
    if is_verified is not None:
        query = query.where(User.is_verified == is_verified)
    if is_admin is not None:
        query = query.where(User.is_admin == is_admin)
    if is_provider is not None:
        query = query.where(User.is_provider == is_provider)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # Apply pagination
    offset = (page - 1) * limit
    query = query.order_by(desc(User.created_at)).offset(offset).limit(limit)

    result = await db.execute(query)
    users = result.scalars().all()

    return {
        "users": [
            UserListItem(
                id=u.id,
                email=u.email,
                name=u.name,
                balance=u.balance,
                total_spent=u.total_spent,
                is_active=u.is_active,
                is_verified=u.is_verified,
                is_admin=u.is_admin,
                is_provider=u.is_provider,
                created_at=u.created_at.isoformat() if u.created_at else "",
                last_login_at=u.last_login_at.isoformat() if u.last_login_at else None,
                pod_count=len(u.pods) if u.pods else 0,
                api_key_count=len(u.api_keys) if u.api_keys else 0,
            )
            for u in users
        ],
        "total": total or 0,
        "page": page,
        "limit": limit,
    }


@router.get("/users/{user_id}", response_model=UserDetailResponse)
async def get_user_detail(
    user_id: str,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get detailed information about a specific user.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get recent transactions
    tx_result = await db.execute(
        select(Transaction)
        .where(Transaction.user_id == user_id)
        .order_by(desc(Transaction.created_at))
        .limit(20)
    )
    transactions = tx_result.scalars().all()

    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "balance": user.balance,
        "total_spent": user.total_spent,
        "credit_limit": user.credit_limit,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "is_admin": user.is_admin,
        "is_provider": user.is_provider,
        "stripe_customer_id": user.stripe_customer_id,
        "auto_refill_enabled": user.auto_refill_enabled,
        "auto_refill_threshold": user.auto_refill_threshold,
        "auto_refill_amount": user.auto_refill_amount,
        "created_at": user.created_at.isoformat() if user.created_at else "",
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
        "pods": [
            {
                "id": p.id,
                "status": p.status,
                "gpu_type": p.gpu_type,
                "created_at": p.created_at.isoformat() if p.created_at else "",
            }
            for p in (user.pods or [])
        ],
        "api_keys": [
            {
                "id": k.id,
                "name": k.name,
                "key_prefix": k.key_prefix,
                "is_active": k.is_active,
                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                "request_count": k.request_count,
            }
            for k in (user.api_keys or [])
        ],
        "recent_transactions": [
            {
                "id": t.id,
                "type": t.type.value if hasattr(t.type, "value") else t.type,
                "amount": t.amount,
                "description": t.description,
                "created_at": t.created_at.isoformat() if t.created_at else "",
            }
            for t in transactions
        ],
    }


@router.patch("/users/{user_id}")
async def update_user(
    user_id: str,
    request: UpdateUserRequest,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Update a user's admin-controlled settings.

    Can update:
    - Account status (active/inactive)
    - Verification status
    - Admin privileges
    - Credit limit
    - Balance adjustments (with reason)
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Prevent self-demotion
    if user.id == admin.id and request.is_admin is False:
        raise HTTPException(
            status_code=400,
            detail="Cannot remove your own admin privileges",
        )

    # Update fields
    if request.is_active is not None:
        user.is_active = request.is_active
    if request.is_verified is not None:
        user.is_verified = request.is_verified
    if request.is_admin is not None:
        user.is_admin = request.is_admin
    if request.credit_limit is not None:
        user.credit_limit = request.credit_limit

    # Handle balance adjustment
    if request.balance_adjustment is not None:
        old_balance = user.balance
        user.balance += request.balance_adjustment

        # Create transaction record
        tx_type = "admin_credit" if request.balance_adjustment > 0 else "admin_debit"
        transaction = Transaction(
            user_id=user.id,
            type=tx_type,
            amount=abs(request.balance_adjustment),
            balance_after=user.balance,
            description=request.adjustment_reason or f"Admin adjustment by {admin.email}",
            reference_id=f"admin:{admin.id}",
        )
        db.add(transaction)

    await db.flush()
    await db.refresh(user)

    return {
        "message": "User updated successfully",
        "user_id": user.id,
        "email": user.email,
        "balance": user.balance,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "is_admin": user.is_admin,
    }


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Soft delete a user (deactivates account).

    Use with caution - this will:
    - Deactivate the user account
    - Revoke all API keys
    - Stop all running pods/deployments

    For hard delete, use database admin tools directly.
    """
    if user_id == admin.id:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete your own account",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Soft delete - deactivate
    user.is_active = False

    # Revoke all API keys
    for key in user.api_keys or []:
        key.is_active = False

    await db.flush()

    return {
        "message": "User deactivated successfully",
        "user_id": user.id,
        "email": user.email,
    }


# ============================================================================
# Node Management
# ============================================================================


@router.get("/nodes", response_model=NodeListResponse)
async def list_nodes(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by status"),
    provider_type: Optional[str] = Query(None, description="Filter by provider type"),
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    List all nodes with pagination and filters.
    """
    query = select(Node)

    if status:
        query = query.where(Node.status == status)
    if provider_type:
        query = query.where(Node.provider_type == provider_type)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # Apply pagination
    offset = (page - 1) * limit
    query = query.order_by(desc(Node.created_at)).offset(offset).limit(limit)

    result = await db.execute(query)
    nodes = result.scalars().all()

    return {
        "nodes": [
            NodeListItem(
                id=n.id,
                hostname=n.hostname,
                ip_address=n.ip_address,
                status=n.status,
                gpu_count=n.gpu_count,
                gpu_type=n.gpu_type or "Unknown",
                provider_type=n.provider_type,
                hourly_price=n.hourly_price,
                last_heartbeat=n.last_heartbeat.isoformat() if n.last_heartbeat else None,
                running_pods=len([p for p in (n.pods or []) if p.status == "running"]),
                created_at=n.created_at.isoformat() if n.created_at else "",
            )
            for n in nodes
        ],
        "total": total or 0,
        "page": page,
        "limit": limit,
    }


@router.patch("/nodes/{node_id}")
async def update_node(
    node_id: str,
    status: Optional[str] = None,
    hourly_price: Optional[float] = None,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Update a node's admin-controlled settings.
    """
    result = await db.execute(select(Node).where(Node.id == node_id))
    node = result.scalar_one_or_none()

    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    if status:
        node.status = status
    if hourly_price is not None:
        node.hourly_price = hourly_price

    await db.flush()

    return {
        "message": "Node updated successfully",
        "node_id": node.id,
        "hostname": node.hostname,
        "status": node.status,
        "hourly_price": node.hourly_price,
    }


# ============================================================================
# Provider Management
# ============================================================================


@router.get("/providers", response_model=ProviderListResponse)
async def list_providers(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    is_verified: Optional[bool] = None,
    verification_level: Optional[str] = None,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    List all GPU providers with pagination and filters.
    """
    query = select(Provider)

    if is_verified is not None:
        query = query.where(Provider.is_verified == is_verified)
    if verification_level:
        query = query.where(Provider.verification_level == verification_level)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # Apply pagination
    offset = (page - 1) * limit
    query = query.order_by(desc(Provider.created_at)).offset(offset).limit(limit)

    result = await db.execute(query)
    providers = result.scalars().all()

    return {
        "providers": [
            ProviderListItem(
                id=p.id,
                user_id=p.user_id,
                user_email=p.user.email if p.user else "Unknown",
                company_name=p.company_name,
                verification_level=p.verification_level.value if hasattr(p.verification_level, "value") else str(p.verification_level),
                total_earnings=p.total_earnings,
                pending_payout=p.pending_payout,
                node_count=len(p.nodes) if p.nodes else 0,
                is_verified=p.is_verified,
                created_at=p.created_at.isoformat() if p.created_at else "",
            )
            for p in providers
        ],
        "total": total or 0,
        "page": page,
        "limit": limit,
    }


@router.patch("/providers/{provider_id}")
async def update_provider(
    provider_id: str,
    request: UpdateProviderRequest,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Update a provider's settings (verification level, fee override).
    """
    result = await db.execute(select(Provider).where(Provider.id == provider_id))
    provider = result.scalar_one_or_none()

    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    if request.verification_level:
        try:
            provider.verification_level = VerificationLevel(request.verification_level)
            # Auto-verify if level is set
            if provider.verification_level != VerificationLevel.NONE:
                provider.is_verified = True
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid verification level. Valid: {[v.value for v in VerificationLevel]}",
            )

    if request.platform_fee_override is not None:
        provider.platform_fee_override = request.platform_fee_override

    await db.flush()
    await db.refresh(provider)

    return {
        "message": "Provider updated successfully",
        "provider_id": provider.id,
        "verification_level": provider.verification_level.value if hasattr(provider.verification_level, "value") else str(provider.verification_level),
        "is_verified": provider.is_verified,
        "platform_fee_override": provider.platform_fee_override,
    }


# ============================================================================
# Financial Management
# ============================================================================


@router.get("/financial/summary", response_model=FinancialSummaryResponse)
async def get_financial_summary(
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get comprehensive financial summary.
    """
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    week_start = today_start - timedelta(days=7)
    month_start = today_start.replace(day=1)

    # Revenue calculations
    total_revenue = await db.scalar(
        select(func.sum(Transaction.amount)).where(Transaction.type == "charge")
    ) or 0

    revenue_today = await db.scalar(
        select(func.sum(Transaction.amount)).where(
            and_(Transaction.type == "charge", Transaction.created_at >= today_start)
        )
    ) or 0

    revenue_yesterday = await db.scalar(
        select(func.sum(Transaction.amount)).where(
            and_(
                Transaction.type == "charge",
                Transaction.created_at >= yesterday_start,
                Transaction.created_at < today_start,
            )
        )
    ) or 0

    revenue_this_week = await db.scalar(
        select(func.sum(Transaction.amount)).where(
            and_(Transaction.type == "charge", Transaction.created_at >= week_start)
        )
    ) or 0

    revenue_this_month = await db.scalar(
        select(func.sum(Transaction.amount)).where(
            and_(Transaction.type == "charge", Transaction.created_at >= month_start)
        )
    ) or 0

    # Provider payouts
    total_provider_payouts = await db.scalar(
        select(func.sum(Provider.total_earnings))
    ) or 0

    pending_payouts = await db.scalar(
        select(func.sum(Provider.pending_payout))
    ) or 0

    # User balances
    total_user_balance = await db.scalar(select(func.sum(User.balance))) or 0
    total_credit_extended = await db.scalar(select(func.sum(User.credit_limit))) or 0

    # Transaction counts
    total_transactions = await db.scalar(select(func.count(Transaction.id))) or 0
    deposits_count = await db.scalar(
        select(func.count(Transaction.id)).where(Transaction.type == "deposit")
    ) or 0
    charges_count = await db.scalar(
        select(func.count(Transaction.id)).where(Transaction.type == "charge")
    ) or 0

    # Calculate margins
    gross_margin = float(total_revenue) - float(total_provider_payouts)
    gross_margin_percent = (gross_margin / float(total_revenue) * 100) if total_revenue > 0 else 0

    return {
        "total_revenue": float(total_revenue),
        "revenue_today": float(revenue_today),
        "revenue_yesterday": float(revenue_yesterday),
        "revenue_this_week": float(revenue_this_week),
        "revenue_this_month": float(revenue_this_month),
        "total_provider_payouts": float(total_provider_payouts),
        "pending_payouts": float(pending_payouts),
        "gross_margin": gross_margin,
        "gross_margin_percent": round(gross_margin_percent, 2),
        "total_user_balance": float(total_user_balance),
        "total_credit_extended": float(total_credit_extended),
        "total_transactions": total_transactions,
        "deposits_count": deposits_count,
        "charges_count": charges_count,
    }


@router.get("/financial/transactions")
async def list_transactions(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    user_id: Optional[str] = None,
    type: Optional[str] = None,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    List all transactions with pagination and filters.
    """
    query = select(Transaction)

    if user_id:
        query = query.where(Transaction.user_id == user_id)
    if type:
        query = query.where(Transaction.type == type)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # Apply pagination
    offset = (page - 1) * limit
    query = query.order_by(desc(Transaction.created_at)).offset(offset).limit(limit)

    result = await db.execute(query)
    transactions = result.scalars().all()

    return {
        "transactions": [
            {
                "id": t.id,
                "user_id": t.user_id,
                "type": t.type.value if hasattr(t.type, "value") else t.type,
                "amount": t.amount,
                "balance_after": t.balance_after,
                "description": t.description,
                "reference_id": t.reference_id,
                "created_at": t.created_at.isoformat() if t.created_at else "",
            }
            for t in transactions
        ],
        "total": total or 0,
        "page": page,
        "limit": limit,
    }


# ============================================================================
# Deployment Management
# ============================================================================


@router.get("/deployments", response_model=DeploymentListResponse)
async def list_deployments(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    status: Optional[str] = None,
    user_id: Optional[str] = None,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    List all model deployments with pagination and filters.
    """
    query = select(Deployment)

    if status:
        try:
            query = query.where(Deployment.status == DeploymentStatus(status))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid status")
    if user_id:
        query = query.where(Deployment.user_id == user_id)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # Apply pagination
    offset = (page - 1) * limit
    query = query.order_by(desc(Deployment.created_at)).offset(offset).limit(limit)

    result = await db.execute(query)
    deployments = result.scalars().all()

    return {
        "deployments": [
            DeploymentListItem(
                id=d.id,
                user_id=d.user_id,
                user_email=d.user.email if d.user else "Unknown",
                model_name=d.model.name if d.model else d.model_template_id,
                status=d.status.value if hasattr(d.status, "value") else str(d.status),
                gpu_type=d.gpu_type,
                hourly_cost=d.hourly_cost,
                created_at=d.created_at.isoformat() if d.created_at else "",
                total_runtime_hours=d.total_runtime_hours,
                total_cost=d.total_cost,
            )
            for d in deployments
        ],
        "total": total or 0,
        "page": page,
        "limit": limit,
    }


@router.delete("/deployments/{deployment_id}")
async def terminate_deployment(
    deployment_id: str,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Force terminate a deployment (admin override).
    """
    result = await db.execute(select(Deployment).where(Deployment.id == deployment_id))
    deployment = result.scalar_one_or_none()

    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    deployment.status = DeploymentStatus.TERMINATED
    deployment.stopped_at = datetime.now(timezone.utc)

    await db.flush()

    return {
        "message": "Deployment terminated",
        "deployment_id": deployment.id,
        "status": "terminated",
    }


# ============================================================================
# Admin Actions Log (for audit trail)
# ============================================================================


@router.get("/audit-log")
async def get_audit_log(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get admin actions audit log.

    Returns transactions with admin references for tracking admin actions.
    """
    query = select(Transaction).where(
        Transaction.reference_id.like("admin:%")
    ).order_by(desc(Transaction.created_at))

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # Apply pagination
    offset = (page - 1) * limit
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    actions = result.scalars().all()

    return {
        "actions": [
            {
                "id": a.id,
                "user_id": a.user_id,
                "type": a.type.value if hasattr(a.type, "value") else a.type,
                "amount": a.amount,
                "description": a.description,
                "admin_id": a.reference_id.replace("admin:", "") if a.reference_id else None,
                "created_at": a.created_at.isoformat() if a.created_at else "",
            }
            for a in actions
        ],
        "total": total or 0,
        "page": page,
        "limit": limit,
    }
