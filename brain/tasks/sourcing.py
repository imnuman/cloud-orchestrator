"""
Celery tasks for GPU sourcing and provisioning.
"""

import asyncio
import logging
from datetime import datetime

from sqlalchemy import select

from brain.config import get_settings
from brain.models.base import async_session_maker
from brain.models.provisioned_instance import (
    ProvisionedInstance,
    ProvisioningStatus,
)
from brain.services.sourcing import get_sourcing_service
from brain.services.provisioning import get_provisioning_service
from brain.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)
settings = get_settings()


def run_async(coro):
    """Run async code in Celery tasks."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(name="brain.tasks.sourcing.search_and_provision")
def search_and_provision():
    """
    Search for GPU offers and provision instances if needed.

    Runs periodically to:
    1. Search providers for available GPUs matching criteria
    2. If auto-provisioning enabled and under limit, create instances
    3. Log available offers for monitoring
    """
    if not settings.sourcing_enabled:
        logger.debug("Sourcing disabled, skipping")
        return {"status": "disabled"}

    return run_async(_search_and_provision_async())


async def _search_and_provision_async():
    """Async implementation of search_and_provision."""
    sourcing_service = get_sourcing_service()
    provisioning_service = get_provisioning_service()

    try:
        # Get current instance count
        async with async_session_maker() as db:
            current_count = await provisioning_service.get_active_instance_count(db)

        if current_count >= settings.sourcing_max_instances:
            logger.info(
                f"At max instances ({current_count}/{settings.sourcing_max_instances}), "
                f"skipping provisioning"
            )
            return {
                "status": "at_limit",
                "current_instances": current_count,
                "max_instances": settings.sourcing_max_instances,
            }

        # Search for offers
        best_offers = await sourcing_service.get_cheapest_per_type()

        result = {
            "status": "success",
            "offers_found": {},
            "provisioned": [],
            "current_instances": current_count,
        }

        for gpu_type, offer in best_offers.items():
            if offer:
                result["offers_found"][gpu_type] = {
                    "offer_id": offer.id,
                    "price": offer.dph_total,
                    "reliability": offer.reliability2,
                    "location": offer.geolocation,
                }

                # Auto-provision if enabled and under limit
                if (
                    settings.auto_provisioning_enabled
                    and current_count < settings.sourcing_max_instances
                ):
                    logger.info(
                        f"Auto-provisioning {gpu_type} at ${offer.dph_total}/hr"
                    )
                    async with async_session_maker() as db:
                        instance = await provisioning_service.provision_from_offer(
                            offer, db
                        )
                        await db.commit()
                        result["provisioned"].append({
                            "instance_id": str(instance.id),
                            "gpu_type": gpu_type,
                            "price": offer.dph_total,
                        })
                        current_count += 1
            else:
                result["offers_found"][gpu_type] = None

        logger.info(
            f"Sourcing complete: {len(result['offers_found'])} GPU types searched, "
            f"{len(result['provisioned'])} instances provisioned"
        )

        return result

    except Exception as e:
        logger.error(f"Error in search_and_provision: {e}")
        return {"status": "error", "error": str(e)}

    finally:
        await sourcing_service.close()
        await provisioning_service.close()


@celery_app.task(name="brain.tasks.sourcing.check_provisioning_status")
def check_provisioning_status():
    """
    Check status of provisioning instances.

    Runs frequently to:
    1. Poll provider for instance status updates
    2. Check for agent registrations
    3. Handle timeouts and failures
    """
    if not settings.sourcing_enabled:
        return {"status": "disabled"}

    return run_async(_check_provisioning_status_async())


async def _check_provisioning_status_async():
    """Async implementation of check_provisioning_status."""
    provisioning_service = get_provisioning_service()

    try:
        async with async_session_maker() as db:
            # Get all provisioning instances
            result = await db.execute(
                select(ProvisionedInstance).where(
                    ProvisionedInstance.status.in_([
                        ProvisioningStatus.CREATING,
                        ProvisioningStatus.STARTING,
                        ProvisioningStatus.INSTALLING,
                        ProvisioningStatus.WAITING_REGISTRATION,
                    ])
                )
            )
            instances = result.scalars().all()

            if not instances:
                return {"status": "no_pending_instances"}

            updated = []
            for instance in instances:
                old_status = instance.status

                # Check provider status
                await provisioning_service.check_instance_status(instance, db)

                # Check for agent registration
                if instance.status in (
                    ProvisioningStatus.INSTALLING,
                    ProvisioningStatus.WAITING_REGISTRATION,
                ):
                    await provisioning_service.check_for_registration(instance, db)

                if instance.status != old_status:
                    updated.append({
                        "instance_id": str(instance.id),
                        "old_status": old_status.value,
                        "new_status": instance.status.value,
                    })

            await db.commit()

            logger.info(
                f"Checked {len(instances)} provisioning instances, "
                f"{len(updated)} status changes"
            )

            return {
                "status": "success",
                "checked": len(instances),
                "updated": updated,
            }

    except Exception as e:
        logger.error(f"Error in check_provisioning_status: {e}")
        return {"status": "error", "error": str(e)}

    finally:
        await provisioning_service.close()


@celery_app.task(name="brain.tasks.sourcing.update_instance_costs")
def update_instance_costs():
    """
    Update accumulated costs for active instances.

    Runs periodically to track spending on provisioned instances.
    """
    if not settings.sourcing_enabled:
        return {"status": "disabled"}

    return run_async(_update_instance_costs_async())


async def _update_instance_costs_async():
    """Async implementation of update_instance_costs."""
    provisioning_service = get_provisioning_service()

    try:
        async with async_session_maker() as db:
            count = await provisioning_service.update_costs(db)
            await db.commit()

            return {
                "status": "success",
                "updated": count,
            }

    except Exception as e:
        logger.error(f"Error in update_instance_costs: {e}")
        return {"status": "error", "error": str(e)}

    finally:
        await provisioning_service.close()


@celery_app.task(name="brain.tasks.sourcing.terminate_failed_instances")
def terminate_failed_instances():
    """
    Terminate instances that have failed or timed out.

    Cleanup task to ensure we don't keep paying for broken instances.
    """
    if not settings.sourcing_enabled:
        return {"status": "disabled"}

    return run_async(_terminate_failed_instances_async())


async def _terminate_failed_instances_async():
    """Async implementation of terminate_failed_instances."""
    provisioning_service = get_provisioning_service()

    try:
        async with async_session_maker() as db:
            # Get failed instances that haven't been terminated
            result = await db.execute(
                select(ProvisionedInstance).where(
                    ProvisionedInstance.status == ProvisioningStatus.FAILED,
                    ProvisionedInstance.provider_instance_id.isnot(None),
                )
            )
            failed_instances = result.scalars().all()

            terminated = []
            for instance in failed_instances:
                logger.info(f"Terminating failed instance {instance.id}")
                success = await provisioning_service.terminate_instance(instance, db)
                if success:
                    terminated.append(str(instance.id))

            await db.commit()

            return {
                "status": "success",
                "terminated": terminated,
            }

    except Exception as e:
        logger.error(f"Error in terminate_failed_instances: {e}")
        return {"status": "error", "error": str(e)}

    finally:
        await provisioning_service.close()
