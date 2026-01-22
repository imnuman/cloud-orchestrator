"""
Payout Service for Provider Earnings.

Handles:
- Processing payout requests
- Crypto (USDC/USDT) payouts
- PayPal payouts
- Earnings settlement
- Payout status tracking
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from brain.config import get_settings
from brain.models.provider import (
    Provider,
    Payout,
    ProviderEarning,
    PayoutMethod,
    PayoutStatus,
)
from brain.models.billing import Transaction, TransactionType, PaymentMethod

logger = logging.getLogger(__name__)
settings = get_settings()


class PayoutService:
    """
    Service for processing provider payouts.

    Handles both crypto and PayPal payout methods.
    In production, this would integrate with:
    - Crypto: Circle (USDC), or direct blockchain transactions
    - PayPal: PayPal Payouts API
    """

    def __init__(self):
        """Initialize payout service."""
        # In production, initialize payment provider clients here
        # self.circle_client = CircleClient(api_key=settings.circle_api_key)
        # self.paypal_client = PayPalClient(client_id=..., secret=...)
        pass

    async def settle_provider_earnings(
        self,
        provider_id: str,
        db: AsyncSession,
    ) -> float:
        """
        Settle pending earnings for a provider.

        Moves earnings from pending to available balance after
        a settlement period (e.g., after the billing period closes).

        Args:
            provider_id: Provider ID to settle earnings for
            db: Database session

        Returns:
            Amount settled
        """
        # Get provider
        result = await db.execute(
            select(Provider).where(Provider.id == provider_id)
        )
        provider = result.scalar_one_or_none()

        if not provider:
            logger.warning(f"Provider {provider_id} not found for settlement")
            return 0.0

        # Get unsettled earnings
        earnings_result = await db.execute(
            select(ProviderEarning)
            .where(ProviderEarning.provider_id == provider_id)
            .where(ProviderEarning.is_settled == False)  # noqa: E712
        )
        unsettled = earnings_result.scalars().all()

        if not unsettled:
            return 0.0

        # Calculate total to settle
        total_earnings = sum(e.provider_earnings for e in unsettled)

        # Mark earnings as settled
        now = datetime.utcnow()
        for earning in unsettled:
            earning.is_settled = True
            earning.settled_at = now

        # Update provider balances
        provider.pending_earnings -= total_earnings
        provider.available_balance += total_earnings

        await db.flush()

        logger.info(
            f"Settled ${total_earnings:.2f} for provider {provider.display_name}"
        )

        return total_earnings

    async def settle_all_providers(self, db: AsyncSession) -> dict:
        """
        Settle earnings for all providers.

        Called periodically (e.g., daily) to move pending earnings
        to available balance.

        Args:
            db: Database session

        Returns:
            Dict with settlement summary
        """
        # Get all providers with pending earnings
        result = await db.execute(
            select(Provider).where(Provider.pending_earnings > 0)
        )
        providers = result.scalars().all()

        total_settled = 0.0
        providers_settled = 0

        for provider in providers:
            amount = await self.settle_provider_earnings(provider.id, db)
            if amount > 0:
                total_settled += amount
                providers_settled += 1

        return {
            "providers_settled": providers_settled,
            "total_settled": total_settled,
        }

    async def process_payout(
        self,
        payout_id: str,
        db: AsyncSession,
        admin_id: Optional[str] = None,
    ) -> bool:
        """
        Process a payout request.

        In production, this would initiate the actual transfer.
        Currently implements a manual processing flow.

        Args:
            payout_id: Payout ID to process
            db: Database session
            admin_id: Admin user processing the payout

        Returns:
            True if processing started successfully
        """
        # Get payout
        result = await db.execute(
            select(Payout).where(Payout.id == payout_id)
        )
        payout = result.scalar_one_or_none()

        if not payout:
            logger.error(f"Payout {payout_id} not found")
            return False

        if payout.status != PayoutStatus.PENDING:
            logger.warning(f"Payout {payout_id} is not pending: {payout.status}")
            return False

        # Get provider
        result = await db.execute(
            select(Provider).where(Provider.id == payout.provider_id)
        )
        provider = result.scalar_one_or_none()

        if not provider:
            payout.status = PayoutStatus.FAILED
            payout.status_message = "Provider not found"
            await db.flush()
            return False

        # Mark as processing
        payout.status = PayoutStatus.PROCESSING
        payout.processed_by = admin_id

        await db.flush()

        # Process based on payout method
        try:
            if payout.payout_method == PayoutMethod.CRYPTO:
                success = await self._process_crypto_payout(payout, provider, db)
            elif payout.payout_method == PayoutMethod.PAYPAL:
                success = await self._process_paypal_payout(payout, provider, db)
            else:
                payout.status = PayoutStatus.FAILED
                payout.status_message = f"Unsupported payout method: {payout.payout_method}"
                await db.flush()
                return False

            return success

        except Exception as e:
            logger.error(f"Error processing payout {payout_id}: {e}")
            payout.status = PayoutStatus.FAILED
            payout.status_message = str(e)
            # Refund the amount back to available balance
            provider.available_balance += payout.amount
            await db.flush()
            return False

    async def _process_crypto_payout(
        self,
        payout: Payout,
        provider: Provider,
        db: AsyncSession,
    ) -> bool:
        """
        Process a cryptocurrency payout.

        In production, this would:
        1. Call Circle API to initiate USDC transfer
        2. Or use web3.py to send transaction directly

        Args:
            payout: Payout record
            provider: Provider record
            db: Database session

        Returns:
            True if successful
        """
        logger.info(
            f"Processing crypto payout: ${payout.amount} to {payout.payout_address}"
        )

        # In production:
        # tx_hash = await self.circle_client.send_usdc(
        #     to_address=payout.payout_address,
        #     amount=payout.amount,
        # )

        # For now, simulate successful processing
        # In a real system, this would be async and completed via webhook
        payout.status = PayoutStatus.COMPLETED
        payout.processed_at = datetime.utcnow()
        payout.transaction_id = f"sim_tx_{payout.id[:8]}"  # Simulated tx ID
        payout.status_message = "Payout completed (simulation mode)"

        # Update provider totals
        provider.total_paid_out += payout.amount

        # Create transaction record
        # Note: This links to the User, not Provider, so we need to get the user
        transaction = Transaction(
            user_id=provider.user_id,
            type=TransactionType.PAYOUT,
            amount=-payout.amount,  # Negative because it's money out
            balance_after=provider.available_balance,  # Note: This is provider balance
            description=f"Payout to {payout.payout_address}",
            reference_id=payout.transaction_id,
            payment_method=PaymentMethod.CRYPTO,
            extra_data={
                "provider_id": provider.id,
                "payout_id": payout.id,
                "payout_method": payout.payout_method.value,
            },
        )
        db.add(transaction)

        await db.flush()

        logger.info(
            f"Crypto payout completed: ${payout.amount} to {provider.display_name}"
        )

        return True

    async def _process_paypal_payout(
        self,
        payout: Payout,
        provider: Provider,
        db: AsyncSession,
    ) -> bool:
        """
        Process a PayPal payout.

        In production, this would call PayPal Payouts API.

        Args:
            payout: Payout record
            provider: Provider record
            db: Database session

        Returns:
            True if successful
        """
        logger.info(
            f"Processing PayPal payout: ${payout.amount} to {payout.payout_address}"
        )

        # In production:
        # batch_id = await self.paypal_client.create_payout(
        #     email=payout.payout_address,
        #     amount=payout.amount,
        #     currency="USD",
        # )

        # Simulate successful processing
        payout.status = PayoutStatus.COMPLETED
        payout.processed_at = datetime.utcnow()
        payout.transaction_id = f"sim_paypal_{payout.id[:8]}"
        payout.status_message = "PayPal payout completed (simulation mode)"

        # Update provider totals
        provider.total_paid_out += payout.amount

        # Create transaction record
        transaction = Transaction(
            user_id=provider.user_id,
            type=TransactionType.PAYOUT,
            amount=-payout.amount,
            balance_after=provider.available_balance,
            description=f"PayPal payout to {payout.payout_address}",
            reference_id=payout.transaction_id,
            payment_method=PaymentMethod.MANUAL,  # PayPal via manual in this model
            extra_data={
                "provider_id": provider.id,
                "payout_id": payout.id,
                "payout_method": payout.payout_method.value,
            },
        )
        db.add(transaction)

        await db.flush()

        logger.info(
            f"PayPal payout completed: ${payout.amount} to {provider.display_name}"
        )

        return True

    async def get_pending_payouts(
        self,
        db: AsyncSession,
        limit: int = 50,
    ) -> list[Payout]:
        """
        Get list of pending payouts for admin processing.

        Args:
            db: Database session
            limit: Maximum number to return

        Returns:
            List of pending Payout records
        """
        result = await db.execute(
            select(Payout)
            .where(Payout.status == PayoutStatus.PENDING)
            .order_by(Payout.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_payout_stats(self, db: AsyncSession) -> dict:
        """
        Get payout statistics for admin dashboard.

        Args:
            db: Database session

        Returns:
            Dict with payout statistics
        """
        # Pending payouts
        pending_result = await db.execute(
            select(func.count(), func.sum(Payout.amount))
            .where(Payout.status == PayoutStatus.PENDING)
        )
        pending_row = pending_result.one()
        pending_count = pending_row[0] or 0
        pending_amount = pending_row[1] or 0.0

        # Processing payouts
        processing_result = await db.execute(
            select(func.count(), func.sum(Payout.amount))
            .where(Payout.status == PayoutStatus.PROCESSING)
        )
        processing_row = processing_result.one()
        processing_count = processing_row[0] or 0
        processing_amount = processing_row[1] or 0.0

        # Completed payouts (all time)
        completed_result = await db.execute(
            select(func.count(), func.sum(Payout.amount))
            .where(Payout.status == PayoutStatus.COMPLETED)
        )
        completed_row = completed_result.one()
        completed_count = completed_row[0] or 0
        completed_amount = completed_row[1] or 0.0

        # Total provider earnings
        earnings_result = await db.execute(
            select(func.sum(ProviderEarning.provider_earnings))
        )
        total_earnings = earnings_result.scalar() or 0.0

        # Total platform fees
        fees_result = await db.execute(
            select(func.sum(ProviderEarning.platform_fee))
        )
        total_platform_fees = fees_result.scalar() or 0.0

        return {
            "pending_payouts": {
                "count": pending_count,
                "amount": pending_amount,
            },
            "processing_payouts": {
                "count": processing_count,
                "amount": processing_amount,
            },
            "completed_payouts": {
                "count": completed_count,
                "amount": completed_amount,
            },
            "total_provider_earnings": total_earnings,
            "total_platform_fees": total_platform_fees,
        }

    async def record_provider_earning(
        self,
        provider_id: str,
        node_id: str,
        pod_id: Optional[str],
        gross_amount: float,
        gpu_hours: float,
        hourly_rate: float,
        period_start: datetime,
        period_end: datetime,
        db: AsyncSession,
        usage_record_id: Optional[str] = None,
    ) -> ProviderEarning:
        """
        Record an earning for a provider.

        Called by the billing system when a customer is charged.

        Args:
            provider_id: Provider ID
            node_id: Node ID that generated the earning
            pod_id: Pod ID (if applicable)
            gross_amount: Total amount charged to customer
            gpu_hours: Number of GPU hours used
            hourly_rate: Hourly rate charged
            period_start: Start of billing period
            period_end: End of billing period
            db: Database session
            usage_record_id: Optional link to UsageRecord

        Returns:
            Created ProviderEarning record
        """
        # Get provider for fee calculation
        result = await db.execute(
            select(Provider).where(Provider.id == provider_id)
        )
        provider = result.scalar_one_or_none()

        if not provider:
            raise ValueError(f"Provider {provider_id} not found")

        # Calculate earnings split
        provider_earnings, platform_fee = provider.calculate_provider_earnings(gross_amount)

        # Create earning record
        earning = ProviderEarning(
            provider_id=provider_id,
            node_id=node_id,
            pod_id=pod_id,
            usage_record_id=usage_record_id,
            gross_amount=gross_amount,
            platform_fee=platform_fee,
            provider_earnings=provider_earnings,
            gpu_hours=gpu_hours,
            hourly_rate=hourly_rate,
            period_start=period_start,
            period_end=period_end,
            is_settled=False,
        )
        db.add(earning)

        # Update provider totals
        provider.total_earnings += provider_earnings
        provider.pending_earnings += provider_earnings
        provider.total_gpu_hours += gpu_hours
        provider.total_jobs_completed += 1

        await db.flush()
        await db.refresh(earning)

        logger.debug(
            f"Recorded earning for provider {provider.display_name}: "
            f"${provider_earnings:.4f} (gross: ${gross_amount:.4f})"
        )

        return earning


# Global singleton instance
_payout_service: Optional[PayoutService] = None


def get_payout_service() -> PayoutService:
    """Get the global PayoutService instance."""
    global _payout_service
    if _payout_service is None:
        _payout_service = PayoutService()
    return _payout_service
