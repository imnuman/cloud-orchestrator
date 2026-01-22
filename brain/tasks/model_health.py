"""
Model health monitoring Celery tasks.

Handles:
- Provisioning pending deployments
- Health checking active deployments
- Billing for model deployments
- Cleanup of stopped deployments
"""

import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brain.models.base import async_session_maker
from brain.models.model_catalog import Deployment, DeploymentStatus
from brain.services.model_deployment import get_model_deployment_service
from brain.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="brain.tasks.model_health.provision_pending_deployments")
def provision_pending_deployments() -> dict:
    """
    Provision pending model deployments.

    This task runs periodically to check for pending deployments
    and provision them on available nodes.
    """
    import asyncio
    return asyncio.run(_provision_pending_deployments_async())


async def _provision_pending_deployments_async() -> dict:
    """Async implementation of provision_pending_deployments."""
    service = get_model_deployment_service()
    provisioned = 0
    failed = 0

    async with async_session_maker() as db:
        try:
            # Get pending deployments
            deployments = await service.get_pending_deployments(db, limit=10)

            for deployment in deployments:
                try:
                    success = await service.provision_deployment(deployment, db)
                    if success:
                        provisioned += 1
                        logger.info(f"Provisioned deployment {deployment.id}")
                    else:
                        logger.warning(
                            f"Could not provision deployment {deployment.id}: "
                            f"{deployment.status_message}"
                        )
                except Exception as e:
                    failed += 1
                    logger.error(f"Error provisioning deployment {deployment.id}: {e}")
                    deployment.status = DeploymentStatus.FAILED
                    deployment.status_message = str(e)

            await db.commit()

        except Exception as e:
            logger.error(f"Error in provision_pending_deployments: {e}")
            await db.rollback()
            raise

    return {
        "provisioned": provisioned,
        "failed": failed,
    }


@celery_app.task(name="brain.tasks.model_health.check_deployment_health")
def check_deployment_health() -> dict:
    """
    Check health of active deployments.

    This task runs periodically to check the health of running deployments
    and update their status accordingly.
    """
    import asyncio
    return asyncio.run(_check_deployment_health_async())


async def _check_deployment_health_async() -> dict:
    """Async implementation of check_deployment_health."""
    service = get_model_deployment_service()
    checked = 0
    healthy = 0
    unhealthy = 0

    async with async_session_maker() as db:
        try:
            # Get deployments needing health check
            deployments = await service.get_deployments_needing_health_check(
                db, interval_seconds=30, limit=50
            )

            for deployment in deployments:
                try:
                    is_healthy = await service.check_deployment_health(deployment, db)
                    checked += 1
                    if is_healthy:
                        healthy += 1
                    else:
                        unhealthy += 1
                except Exception as e:
                    logger.error(f"Error checking deployment {deployment.id}: {e}")

            await db.commit()

        except Exception as e:
            logger.error(f"Error in check_deployment_health: {e}")
            await db.rollback()
            raise

    return {
        "checked": checked,
        "healthy": healthy,
        "unhealthy": unhealthy,
    }


@celery_app.task(name="brain.tasks.model_health.update_deployment_statuses")
def update_deployment_statuses() -> dict:
    """
    Update deployment statuses based on pod states.

    This task runs periodically to sync deployment status
    with the underlying pod status.
    """
    import asyncio
    return asyncio.run(_update_deployment_statuses_async())


async def _update_deployment_statuses_async() -> dict:
    """Async implementation of update_deployment_statuses."""
    service = get_model_deployment_service()
    updated = 0

    async with async_session_maker() as db:
        try:
            # Get active deployments
            result = await db.execute(
                select(Deployment).where(
                    Deployment.status.in_([
                        DeploymentStatus.PROVISIONING,
                        DeploymentStatus.STARTING,
                        DeploymentStatus.LOADING,
                        DeploymentStatus.STOPPING,
                    ])
                )
            )
            deployments = result.scalars().all()

            for deployment in deployments:
                old_status = deployment.status
                await service.update_deployment_status(deployment, db)
                if deployment.status != old_status:
                    updated += 1
                    logger.info(
                        f"Deployment {deployment.id} status changed: "
                        f"{old_status.value} -> {deployment.status.value}"
                    )

            await db.commit()

        except Exception as e:
            logger.error(f"Error in update_deployment_statuses: {e}")
            await db.rollback()
            raise

    return {"updated": updated}


@celery_app.task(name="brain.tasks.model_health.bill_model_deployments")
def bill_model_deployments() -> dict:
    """
    Bill active model deployments.

    This task runs periodically to update costs for running deployments.
    """
    import asyncio
    return asyncio.run(_bill_model_deployments_async())


async def _bill_model_deployments_async() -> dict:
    """Async implementation of bill_model_deployments."""
    billed = 0
    total_amount = 0.0

    async with async_session_maker() as db:
        try:
            # Get running deployments
            result = await db.execute(
                select(Deployment).where(
                    Deployment.status == DeploymentStatus.READY
                )
            )
            deployments = result.scalars().all()

            now = datetime.utcnow()

            for deployment in deployments:
                if not deployment.started_at:
                    continue

                # Calculate cost since last billing
                last_billed = deployment.last_billed_at or deployment.started_at
                elapsed_seconds = (now - last_billed).total_seconds()

                if elapsed_seconds < 60:  # Only bill every minute
                    continue

                elapsed_hours = elapsed_seconds / 3600.0
                cost = elapsed_hours * deployment.hourly_price

                # Update deployment
                deployment.total_cost += cost
                deployment.total_runtime_seconds += int(elapsed_seconds)
                deployment.last_billed_at = now

                billed += 1
                total_amount += cost

            await db.commit()

            if billed > 0:
                logger.info(
                    f"Billed {billed} deployments for ${total_amount:.4f}"
                )

        except Exception as e:
            logger.error(f"Error in bill_model_deployments: {e}")
            await db.rollback()
            raise

    return {
        "billed": billed,
        "total_amount": total_amount,
    }


@celery_app.task(name="brain.tasks.model_health.cleanup_stopped_deployments")
def cleanup_stopped_deployments() -> dict:
    """
    Cleanup stopped deployments.

    Mark deployments as fully stopped once their pods are terminated.
    """
    import asyncio
    return asyncio.run(_cleanup_stopped_deployments_async())


async def _cleanup_stopped_deployments_async() -> dict:
    """Async implementation of cleanup_stopped_deployments."""
    cleaned = 0

    async with async_session_maker() as db:
        try:
            # Get stopping deployments
            result = await db.execute(
                select(Deployment).where(
                    Deployment.status == DeploymentStatus.STOPPING
                )
            )
            deployments = result.scalars().all()

            for deployment in deployments:
                # Check if pod is terminated
                # In production, verify pod status
                # For now, just mark as stopped after a grace period

                if deployment.stopped_at:
                    elapsed = (datetime.utcnow() - deployment.stopped_at).total_seconds()
                    if elapsed > 60:  # 1 minute grace period
                        deployment.status = DeploymentStatus.STOPPED
                        deployment.total_cost = deployment.calculate_current_cost()
                        cleaned += 1

            await db.commit()

        except Exception as e:
            logger.error(f"Error in cleanup_stopped_deployments: {e}")
            await db.rollback()
            raise

    return {"cleaned": cleaned}
