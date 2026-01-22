"""
Stripe Billing Service.

Handles all Stripe payment operations:
- Customer creation and management
- Payment method attachment
- One-time charges (add funds)
- Auto-refill subscriptions
- Webhook processing
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Optional

import stripe
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brain.config import get_settings
from brain.models.billing import Transaction, TransactionType
from brain.models.user import User

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize Stripe
if settings.stripe_secret_key:
    stripe.api_key = settings.stripe_secret_key


class StripeNotConfiguredError(Exception):
    """Raised when Stripe is not configured."""

    pass


class PaymentFailedError(Exception):
    """Raised when a payment fails."""

    pass


class StripeBillingService:
    """
    Service for managing Stripe payments.

    Handles:
    - Customer creation
    - Payment intents for adding funds
    - Checkout sessions for card payments
    - Webhook processing
    - Auto-refill configuration
    """

    def __init__(self):
        self._verify_stripe_configured()

    def _verify_stripe_configured(self) -> None:
        """Verify Stripe is configured."""
        if not settings.stripe_enabled or not settings.stripe_secret_key:
            logger.warning("Stripe is not configured. Payment features disabled.")

    @property
    def is_enabled(self) -> bool:
        """Check if Stripe is enabled and configured."""
        return bool(settings.stripe_enabled and settings.stripe_secret_key)

    async def get_or_create_customer(
        self,
        user: User,
        db: AsyncSession,
    ) -> str:
        """
        Get or create a Stripe customer for a user.

        Args:
            user: User model
            db: Database session

        Returns:
            Stripe customer ID
        """
        if not self.is_enabled:
            raise StripeNotConfiguredError("Stripe is not configured")

        # Return existing customer ID if present
        if user.stripe_customer_id:
            return user.stripe_customer_id

        # Create new Stripe customer
        try:
            customer = stripe.Customer.create(
                email=user.email,
                metadata={
                    "user_id": user.id,
                    "platform": "gpu-cloud",
                },
            )

            # Save customer ID to user
            user.stripe_customer_id = customer.id
            await db.flush()

            logger.info(f"Created Stripe customer {customer.id} for user {user.id}")
            return customer.id

        except stripe.StripeError as e:
            logger.error(f"Failed to create Stripe customer: {e}")
            raise PaymentFailedError(f"Failed to create customer: {str(e)}")

    async def create_checkout_session(
        self,
        user: User,
        amount: Decimal,
        success_url: str,
        cancel_url: str,
        db: AsyncSession,
    ) -> dict:
        """
        Create a Stripe Checkout session for adding funds.

        Args:
            user: User adding funds
            amount: Amount in USD (e.g., 50.00)
            success_url: URL to redirect on success
            cancel_url: URL to redirect on cancel
            db: Database session

        Returns:
            Dict with checkout session URL and ID
        """
        if not self.is_enabled:
            raise StripeNotConfiguredError("Stripe is not configured")

        customer_id = await self.get_or_create_customer(user, db)

        # Convert to cents
        amount_cents = int(amount * 100)

        try:
            session = stripe.checkout.Session.create(
                customer=customer_id,
                payment_method_types=["card"],
                line_items=[
                    {
                        "price_data": {
                            "currency": "usd",
                            "product_data": {
                                "name": "GPU Cloud Credits",
                                "description": f"Add ${amount:.2f} to your GPU Cloud balance",
                            },
                            "unit_amount": amount_cents,
                        },
                        "quantity": 1,
                    }
                ],
                mode="payment",
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={
                    "user_id": user.id,
                    "type": "add_funds",
                    "amount": str(amount),
                },
            )

            logger.info(
                f"Created checkout session {session.id} for user {user.id}, amount ${amount}"
            )

            return {
                "session_id": session.id,
                "url": session.url,
                "amount": float(amount),
            }

        except stripe.StripeError as e:
            logger.error(f"Failed to create checkout session: {e}")
            raise PaymentFailedError(f"Failed to create checkout: {str(e)}")

    async def create_payment_intent(
        self,
        user: User,
        amount: Decimal,
        db: AsyncSession,
    ) -> dict:
        """
        Create a Payment Intent for client-side payment.

        Use this for custom payment forms with Stripe Elements.

        Args:
            user: User adding funds
            amount: Amount in USD
            db: Database session

        Returns:
            Dict with client_secret for frontend
        """
        if not self.is_enabled:
            raise StripeNotConfiguredError("Stripe is not configured")

        customer_id = await self.get_or_create_customer(user, db)
        amount_cents = int(amount * 100)

        try:
            intent = stripe.PaymentIntent.create(
                amount=amount_cents,
                currency="usd",
                customer=customer_id,
                metadata={
                    "user_id": user.id,
                    "type": "add_funds",
                },
                automatic_payment_methods={"enabled": True},
            )

            return {
                "client_secret": intent.client_secret,
                "payment_intent_id": intent.id,
                "amount": float(amount),
            }

        except stripe.StripeError as e:
            logger.error(f"Failed to create payment intent: {e}")
            raise PaymentFailedError(f"Failed to create payment: {str(e)}")

    async def handle_successful_payment(
        self,
        user_id: str,
        amount: Decimal,
        stripe_payment_id: str,
        db: AsyncSession,
    ) -> Transaction:
        """
        Handle a successful payment - add funds to user balance.

        Args:
            user_id: User ID
            amount: Amount paid in USD
            stripe_payment_id: Stripe payment/session ID
            db: Database session

        Returns:
            Transaction record
        """
        # Get user
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user:
            raise ValueError(f"User {user_id} not found")

        # Update balance
        user.balance += float(amount)

        # Create transaction record
        transaction = Transaction(
            user_id=user_id,
            type=TransactionType.DEPOSIT,
            amount=float(amount),
            balance_after=user.balance,
            description=f"Added funds via Stripe",
            reference_id=stripe_payment_id,
        )
        db.add(transaction)

        await db.flush()

        logger.info(
            f"Added ${amount} to user {user_id} balance. New balance: ${user.balance}"
        )

        return transaction

    async def process_webhook(
        self,
        payload: bytes,
        signature: str,
        db: AsyncSession,
    ) -> dict:
        """
        Process a Stripe webhook event.

        Args:
            payload: Raw webhook payload
            signature: Stripe signature header
            db: Database session

        Returns:
            Dict with processing result
        """
        if not self.is_enabled:
            raise StripeNotConfiguredError("Stripe is not configured")

        if not settings.stripe_webhook_secret:
            raise StripeNotConfiguredError("Webhook secret not configured")

        try:
            event = stripe.Webhook.construct_event(
                payload,
                signature,
                settings.stripe_webhook_secret,
            )
        except stripe.SignatureVerificationError:
            logger.error("Invalid webhook signature")
            raise PaymentFailedError("Invalid signature")

        logger.info(f"Processing webhook event: {event.type}")

        # Handle different event types
        if event.type == "checkout.session.completed":
            session = event.data.object
            if session.metadata.get("type") == "add_funds":
                user_id = session.metadata.get("user_id")
                amount = Decimal(session.metadata.get("amount", "0"))

                if user_id and amount > 0:
                    await self.handle_successful_payment(
                        user_id=user_id,
                        amount=amount,
                        stripe_payment_id=session.id,
                        db=db,
                    )
                    await db.commit()

                    return {"status": "processed", "type": "add_funds", "amount": float(amount)}

        elif event.type == "payment_intent.succeeded":
            intent = event.data.object
            if intent.metadata.get("type") == "add_funds":
                user_id = intent.metadata.get("user_id")
                amount = Decimal(intent.amount / 100)  # Convert from cents

                if user_id and amount > 0:
                    await self.handle_successful_payment(
                        user_id=user_id,
                        amount=amount,
                        stripe_payment_id=intent.id,
                        db=db,
                    )
                    await db.commit()

                    return {"status": "processed", "type": "add_funds", "amount": float(amount)}

        elif event.type == "payment_intent.payment_failed":
            intent = event.data.object
            logger.warning(f"Payment failed: {intent.id} - {intent.last_payment_error}")
            return {"status": "failed", "error": str(intent.last_payment_error)}

        return {"status": "ignored", "type": event.type}

    async def setup_auto_refill(
        self,
        user: User,
        threshold: Decimal,
        refill_amount: Decimal,
        db: AsyncSession,
    ) -> dict:
        """
        Configure auto-refill for a user.

        When balance drops below threshold, automatically charge refill_amount.

        Args:
            user: User to configure
            threshold: Balance threshold to trigger refill
            refill_amount: Amount to add when triggered
            db: Database session

        Returns:
            Configuration details
        """
        if not self.is_enabled:
            raise StripeNotConfiguredError("Stripe is not configured")

        # Ensure customer exists
        await self.get_or_create_customer(user, db)

        # Store auto-refill settings on user
        user.auto_refill_enabled = True
        user.auto_refill_threshold = float(threshold)
        user.auto_refill_amount = float(refill_amount)

        await db.flush()

        logger.info(
            f"Configured auto-refill for user {user.id}: "
            f"threshold=${threshold}, amount=${refill_amount}"
        )

        return {
            "enabled": True,
            "threshold": float(threshold),
            "refill_amount": float(refill_amount),
        }

    async def disable_auto_refill(
        self,
        user: User,
        db: AsyncSession,
    ) -> dict:
        """Disable auto-refill for a user."""
        user.auto_refill_enabled = False
        await db.flush()

        return {"enabled": False}

    async def check_and_process_auto_refill(
        self,
        user: User,
        db: AsyncSession,
    ) -> Optional[Transaction]:
        """
        Check if auto-refill should trigger and process it.

        Called by billing tasks when balance is low.

        Args:
            user: User to check
            db: Database session

        Returns:
            Transaction if refill was processed, None otherwise
        """
        if not self.is_enabled:
            return None

        if not user.auto_refill_enabled:
            return None

        if not user.stripe_customer_id:
            return None

        if user.balance >= user.auto_refill_threshold:
            return None

        # Get default payment method
        try:
            customer = stripe.Customer.retrieve(
                user.stripe_customer_id,
                expand=["default_source", "invoice_settings.default_payment_method"],
            )

            payment_method = (
                customer.invoice_settings.default_payment_method
                or customer.default_source
            )

            if not payment_method:
                logger.warning(f"No payment method for auto-refill: user {user.id}")
                return None

            # Create and confirm payment intent
            amount_cents = int(user.auto_refill_amount * 100)

            intent = stripe.PaymentIntent.create(
                amount=amount_cents,
                currency="usd",
                customer=user.stripe_customer_id,
                payment_method=payment_method if isinstance(payment_method, str) else payment_method.id,
                off_session=True,
                confirm=True,
                metadata={
                    "user_id": user.id,
                    "type": "auto_refill",
                },
            )

            if intent.status == "succeeded":
                transaction = await self.handle_successful_payment(
                    user_id=user.id,
                    amount=Decimal(str(user.auto_refill_amount)),
                    stripe_payment_id=intent.id,
                    db=db,
                )

                logger.info(f"Auto-refill succeeded for user {user.id}: ${user.auto_refill_amount}")
                return transaction

        except stripe.CardError as e:
            logger.error(f"Auto-refill card error for user {user.id}: {e}")
        except stripe.StripeError as e:
            logger.error(f"Auto-refill failed for user {user.id}: {e}")

        return None

    def get_publishable_key(self) -> Optional[str]:
        """Get the Stripe publishable key for frontend."""
        return settings.stripe_publishable_key if self.is_enabled else None


# Global singleton
_stripe_billing_service: Optional[StripeBillingService] = None


def get_stripe_billing_service() -> StripeBillingService:
    """Get the global StripeBillingService instance."""
    global _stripe_billing_service
    if _stripe_billing_service is None:
        _stripe_billing_service = StripeBillingService()
    return _stripe_billing_service
