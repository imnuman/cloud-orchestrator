"""
Billing tasks - the "Meter" that charges users for usage.
"""

import asyncio
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brain.config import get_settings
from brain.models.base import async_session_maker
from brain.models.pod import Pod
from brain.models.user import User
from brain.models.billing import Transaction, TransactionType, UsageRecord
from brain.tasks.celery_app import celery_app
from shared.schemas import PodStatus

settings = get_settings()


async def _bill_running_pods() -> dict:
    """
    Core billing logic - charge users for running pods.
    Runs every billing interval (default: 60 seconds).
    """
    stats = {
        "pods_billed": 0,
        "total_charged": 0.0,
        "pods_stopped_for_balance": 0,
        "errors": [],
    }

    async with async_session_maker() as db:
        # Get all running pods
        result = await db.execute(
            select(Pod).where(Pod.status == PodStatus.RUNNING)
        )
        running_pods = result.scalars().all()

        for pod in running_pods:
            try:
                # Calculate charge for this billing period
                # hourly_price / 60 = per-minute rate
                # per-minute rate / 60 = per-second rate
                # per-second rate * billing_interval = charge
                seconds_in_period = settings.billing_interval_seconds
                hourly_rate = pod.hourly_price
                charge = (hourly_rate / 3600) * seconds_in_period

                # Get user
                user_result = await db.execute(
                    select(User).where(User.id == pod.user_id)
                )
                user = user_result.scalar_one_or_none()

                if not user:
                    stats["errors"].append(f"User not found for pod {pod.id}")
                    continue

                # Check if user can afford
                if user.balance < charge and user.balance <= 0:
                    # Stop pod due to insufficient balance
                    pod.status = PodStatus.STOPPING
                    pod.termination_reason = "Insufficient balance"
                    pod.stopped_at = datetime.utcnow()
                    stats["pods_stopped_for_balance"] += 1

                    # Create transaction for final charge
                    if user.balance > 0:
                        final_charge = user.balance
                        user.balance = 0
                        user.total_spent += final_charge

                        transaction = Transaction(
                            user_id=user.id,
                            type=TransactionType.CHARGE,
                            amount=-final_charge,
                            balance_after=user.balance,
                            description=f"Final charge for pod {pod.id[:8]}... (stopped: insufficient balance)",
                            pod_id=pod.id,
                        )
                        db.add(transaction)
                else:
                    # Normal billing
                    user.balance -= charge
                    user.total_spent += charge
                    pod.total_cost += charge
                    pod.last_billed_at = datetime.utcnow()

                    # Create transaction
                    transaction = Transaction(
                        user_id=user.id,
                        type=TransactionType.CHARGE,
                        amount=-charge,
                        balance_after=user.balance,
                        description=f"Usage charge for pod {pod.id[:8]}... ({pod.gpu_type})",
                        pod_id=pod.id,
                    )
                    db.add(transaction)

                    # Create usage record
                    usage = UsageRecord(
                        user_id=user.id,
                        pod_id=pod.id,
                        node_id=pod.node_id,
                        gpu_type=pod.gpu_type,
                        gpu_count=pod.gpu_count,
                        period_start=pod.last_billed_at or pod.started_at or datetime.utcnow(),
                        period_end=datetime.utcnow(),
                        duration_seconds=seconds_in_period,
                        hourly_rate=hourly_rate,
                        amount_charged=charge,
                        provider_cost=(pod.provider_cost / 3600) * seconds_in_period,
                        is_billed=True,
                    )
                    db.add(usage)

                    stats["pods_billed"] += 1
                    stats["total_charged"] += charge

            except Exception as e:
                stats["errors"].append(f"Error billing pod {pod.id}: {str(e)}")

        await db.commit()

    return stats


async def _check_low_balances() -> dict:
    """
    Check for users with low balances and send warnings.
    """
    stats = {
        "users_warned": 0,
        "users_with_running_pods": 0,
    }

    async with async_session_maker() as db:
        # Find users with low balance who have running pods
        result = await db.execute(
            select(User).where(
                User.balance <= settings.low_balance_warning_threshold,
                User.balance > 0,
                User.is_active == True,
            )
        )
        low_balance_users = result.scalars().all()

        for user in low_balance_users:
            # Check if user has running pods
            pod_result = await db.execute(
                select(Pod).where(
                    Pod.user_id == user.id,
                    Pod.status == PodStatus.RUNNING,
                )
            )
            running_pods = pod_result.scalars().all()

            if running_pods:
                stats["users_with_running_pods"] += 1
                # TODO: Send email/notification warning
                # For now, just log
                print(f"Low balance warning: User {user.email} has ${user.balance:.2f} with {len(running_pods)} running pods")
                stats["users_warned"] += 1

    return stats


@celery_app.task(name="brain.tasks.billing.bill_running_pods")
def bill_running_pods() -> dict:
    """Celery task wrapper for billing."""
    return asyncio.run(_bill_running_pods())


@celery_app.task(name="brain.tasks.billing.check_low_balances")
def check_low_balances() -> dict:
    """Celery task wrapper for low balance check."""
    return asyncio.run(_check_low_balances())
