"""
User management routes.
"""

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brain.config import get_settings
from brain.models.base import get_db
from brain.models.user import User
from brain.models.billing import Transaction, TransactionType
from brain.routes.auth import get_current_active_user
from shared.schemas import UserResponse

router = APIRouter(prefix="/users", tags=["Users"])
settings = get_settings()


@router.get("/me", response_model=dict)
async def get_current_user_info(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> dict:
    """Get current user information."""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "balance": current_user.balance,
        "total_spent": current_user.total_spent,
        "is_provider": current_user.is_provider,
        "provider_earnings": current_user.provider_earnings,
        "api_key": current_user.api_key,
        "created_at": current_user.created_at,
        "last_login_at": current_user.last_login_at,
    }


@router.post("/api-key")
async def generate_api_key(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Generate a new API key for the user."""
    new_key = f"gpu_user_{secrets.token_urlsafe(32)}"
    current_user.api_key = new_key
    await db.flush()

    return {
        "api_key": new_key,
        "message": "New API key generated. Store it securely - it won't be shown again.",
    }


@router.get("/balance")
async def get_balance(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> dict:
    """Get current balance information."""
    return {
        "balance": current_user.balance,
        "available_balance": current_user.available_balance,
        "credit_limit": current_user.credit_limit,
        "total_spent": current_user.total_spent,
    }


@router.post("/deposit")
async def add_deposit(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    amount: float,
) -> dict:
    """
    Add funds to user account (simulated for MVP).
    In production, this would integrate with Stripe.
    """
    if amount <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Amount must be positive",
        )

    if amount > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum single deposit is $1000 (MVP limit)",
        )

    # Update balance
    current_user.balance += amount

    # Create transaction record
    transaction = Transaction(
        user_id=current_user.id,
        type=TransactionType.DEPOSIT,
        amount=amount,
        balance_after=current_user.balance,
        description=f"Deposit of ${amount:.2f}",
    )
    db.add(transaction)
    await db.flush()

    return {
        "message": f"Successfully deposited ${amount:.2f}",
        "new_balance": current_user.balance,
        "transaction_id": transaction.id,
    }


@router.get("/transactions")
async def get_transactions(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Get transaction history."""
    result = await db.execute(
        select(Transaction)
        .where(Transaction.user_id == current_user.id)
        .order_by(Transaction.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    transactions = result.scalars().all()

    return [
        {
            "id": t.id,
            "type": t.type.value,
            "amount": t.amount,
            "balance_after": t.balance_after,
            "description": t.description,
            "pod_id": t.pod_id,
            "created_at": t.created_at,
        }
        for t in transactions
    ]


@router.get("/usage")
async def get_usage_summary(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Get usage summary for the current billing period."""
    # TODO: Calculate actual usage from pods and usage records
    return {
        "total_spent": current_user.total_spent,
        "current_month_spend": 0.0,  # TODO: Calculate
        "active_pods": len([p for p in current_user.pods if p.is_running]),
        "total_gpu_hours": 0.0,  # TODO: Calculate
    }
