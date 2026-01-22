"""
Billing Routes.

Handles payment operations:
- Add funds via Stripe Checkout
- Payment webhooks
- Auto-refill configuration
- Balance and transaction management
"""

import logging
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from brain.config import get_settings
from brain.models.base import get_db
from brain.models.billing import Transaction
from brain.models.user import User
from brain.routes.auth import get_current_user
from brain.services.stripe_billing import (
    get_stripe_billing_service,
    StripeNotConfiguredError,
    PaymentFailedError,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/billing", tags=["Billing"])


# Request/Response Models


class AddFundsRequest(BaseModel):
    """Request to add funds."""

    amount: float = Field(..., gt=0, le=10000, description="Amount in USD (min $1, max $10000)")
    success_url: str = Field(..., description="URL to redirect after successful payment")
    cancel_url: str = Field(..., description="URL to redirect if payment is cancelled")


class AddFundsResponse(BaseModel):
    """Response with checkout URL."""

    session_id: str
    checkout_url: str
    amount: float


class PaymentIntentRequest(BaseModel):
    """Request to create a payment intent."""

    amount: float = Field(..., gt=0, le=10000, description="Amount in USD")


class PaymentIntentResponse(BaseModel):
    """Response with client secret for Stripe Elements."""

    client_secret: str
    payment_intent_id: str
    amount: float


class AutoRefillRequest(BaseModel):
    """Request to configure auto-refill."""

    enabled: bool = Field(..., description="Enable or disable auto-refill")
    threshold: float = Field(10.0, ge=5, le=1000, description="Trigger when balance drops below")
    refill_amount: float = Field(50.0, ge=10, le=1000, description="Amount to add")


class AutoRefillResponse(BaseModel):
    """Auto-refill configuration."""

    enabled: bool
    threshold: Optional[float] = None
    refill_amount: Optional[float] = None


class BalanceResponse(BaseModel):
    """User balance information."""

    balance: float
    available_balance: float
    credit_limit: float
    auto_refill: AutoRefillResponse
    stripe_configured: bool


class TransactionResponse(BaseModel):
    """Transaction record."""

    id: str
    type: str
    amount: float
    balance_after: float
    description: Optional[str]
    reference_id: Optional[str]
    created_at: str


class StripeConfigResponse(BaseModel):
    """Stripe configuration for frontend."""

    enabled: bool
    publishable_key: Optional[str] = None


# Endpoints


@router.get("/config", response_model=StripeConfigResponse)
async def get_stripe_config():
    """
    Get Stripe configuration for frontend.

    Returns publishable key for Stripe.js initialization.
    """
    service = get_stripe_billing_service()
    return StripeConfigResponse(
        enabled=service.is_enabled,
        publishable_key=service.get_publishable_key(),
    )


@router.get("/balance", response_model=BalanceResponse)
async def get_balance(
    current_user: User = Depends(get_current_user),
):
    """
    Get current user's balance and billing info.
    """
    service = get_stripe_billing_service()

    return BalanceResponse(
        balance=current_user.balance,
        available_balance=current_user.available_balance,
        credit_limit=current_user.credit_limit,
        auto_refill=AutoRefillResponse(
            enabled=current_user.auto_refill_enabled,
            threshold=current_user.auto_refill_threshold if current_user.auto_refill_enabled else None,
            refill_amount=current_user.auto_refill_amount if current_user.auto_refill_enabled else None,
        ),
        stripe_configured=bool(current_user.stripe_customer_id),
    )


@router.post("/add-funds", response_model=AddFundsResponse)
async def add_funds(
    request: AddFundsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a Stripe Checkout session to add funds.

    Redirects user to Stripe-hosted payment page.
    """
    service = get_stripe_billing_service()

    if not service.is_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Payment processing is not configured",
        )

    try:
        result = await service.create_checkout_session(
            user=current_user,
            amount=Decimal(str(request.amount)),
            success_url=request.success_url,
            cancel_url=request.cancel_url,
            db=db,
        )
        await db.commit()

        return AddFundsResponse(
            session_id=result["session_id"],
            checkout_url=result["url"],
            amount=result["amount"],
        )

    except StripeNotConfiguredError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Payment processing is not configured",
        )
    except PaymentFailedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/payment-intent", response_model=PaymentIntentResponse)
async def create_payment_intent(
    request: PaymentIntentRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a Payment Intent for custom payment forms.

    Use with Stripe Elements for embedded payment forms.
    """
    service = get_stripe_billing_service()

    if not service.is_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Payment processing is not configured",
        )

    try:
        result = await service.create_payment_intent(
            user=current_user,
            amount=Decimal(str(request.amount)),
            db=db,
        )
        await db.commit()

        return PaymentIntentResponse(
            client_secret=result["client_secret"],
            payment_intent_id=result["payment_intent_id"],
            amount=result["amount"],
        )

    except StripeNotConfiguredError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Payment processing is not configured",
        )
    except PaymentFailedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/auto-refill", response_model=AutoRefillResponse)
async def configure_auto_refill(
    request: AutoRefillRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Configure automatic balance refill.

    When enabled, automatically adds funds when balance drops below threshold.
    Requires a saved payment method.
    """
    service = get_stripe_billing_service()

    if not service.is_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Payment processing is not configured",
        )

    try:
        if request.enabled:
            result = await service.setup_auto_refill(
                user=current_user,
                threshold=Decimal(str(request.threshold)),
                refill_amount=Decimal(str(request.refill_amount)),
                db=db,
            )
        else:
            result = await service.disable_auto_refill(
                user=current_user,
                db=db,
            )

        await db.commit()

        return AutoRefillResponse(
            enabled=result["enabled"],
            threshold=result.get("threshold"),
            refill_amount=result.get("refill_amount"),
        )

    except StripeNotConfiguredError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Payment processing is not configured",
        )
    except PaymentFailedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/transactions", response_model=list[TransactionResponse])
async def get_transactions(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get user's transaction history.
    """
    result = await db.execute(
        select(Transaction)
        .where(Transaction.user_id == current_user.id)
        .order_by(desc(Transaction.created_at))
        .offset(offset)
        .limit(limit)
    )
    transactions = result.scalars().all()

    return [
        TransactionResponse(
            id=t.id,
            type=t.type.value,
            amount=t.amount,
            balance_after=t.balance_after,
            description=t.description,
            reference_id=t.reference_id,
            created_at=t.created_at.isoformat() if t.created_at else "",
        )
        for t in transactions
    ]


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
    db: AsyncSession = Depends(get_db),
):
    """
    Handle Stripe webhook events.

    This endpoint is called by Stripe to notify of payment events.
    Configure webhook URL in Stripe Dashboard: https://yourdomain.com/api/v1/billing/webhook
    """
    service = get_stripe_billing_service()

    if not service.is_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stripe not configured",
        )

    if not stripe_signature:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Stripe-Signature header",
        )

    try:
        payload = await request.body()
        result = await service.process_webhook(
            payload=payload,
            signature=stripe_signature,
            db=db,
        )

        return {"status": "ok", "result": result}

    except PaymentFailedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook processing failed",
        )


# Manual deposit endpoint (for testing/admin)


class ManualDepositRequest(BaseModel):
    """Manual deposit (for testing/admin)."""

    amount: float = Field(..., gt=0, le=10000)
    description: str = Field("Manual deposit", max_length=255)


@router.post("/deposit", response_model=TransactionResponse)
async def manual_deposit(
    request: ManualDepositRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Add funds manually (for testing or admin credits).

    In production, this should be restricted to admins only.
    """
    # Update balance
    current_user.balance += request.amount

    # Create transaction
    transaction = Transaction(
        user_id=current_user.id,
        type="deposit",
        amount=request.amount,
        balance_after=current_user.balance,
        description=request.description,
    )
    db.add(transaction)

    await db.commit()
    await db.refresh(transaction)

    return TransactionResponse(
        id=transaction.id,
        type=transaction.type.value if hasattr(transaction.type, 'value') else transaction.type,
        amount=transaction.amount,
        balance_after=transaction.balance_after,
        description=transaction.description,
        reference_id=transaction.reference_id,
        created_at=transaction.created_at.isoformat() if transaction.created_at else "",
    )
