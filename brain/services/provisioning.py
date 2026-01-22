"""
GPU Provisioning Service.

Handles creating and managing provisioned GPU instances from offers.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brain.adapters.base import InstanceConfig, InstanceStatus
from brain.adapters.vast_ai import VastClient, VastGpuOffer
from brain.config import get_settings
from brain.models.node import Node
from brain.models.provisioned_instance import (
    ProvisionedInstance,
    ProvisioningStatus,
)
from shared.schemas import ProviderType

logger = logging.getLogger(__name__)
settings = get_settings()


class ProvisioningService:
    """
    Service for provisioning and managing GPU instances.

    Handles:
    - Creating instances from offers
    - Tracking instance status through lifecycle
    - Matching agent registrations to provisioned instances
    - Cost tracking
    - Instance termination
    """

    def __init__(self, vast_client: Optional[VastClient] = None):
        """
        Initialize provisioning service.

        Args:
            vast_client: Optional VastClient instance
        """
        self._vast_client = vast_client

    @property
    def vast_client(self) -> VastClient:
        """Lazily initialize Vast.ai client."""
        if self._vast_client is None:
            self._vast_client = VastClient(
                api_key=settings.vast_ai_api_key,
                use_mock=settings.use_mock_providers,
            )
        return self._vast_client

    def _generate_install_script(self) -> str:
        """Generate the agent installation script for onstart."""
        brain_url = settings.brain_public_url.rstrip("/")

        return f"""#!/bin/bash
set -e

# GPU Agent Auto-Install Script
echo "Installing GPU Agent..."

# Set environment for agent
export BRAIN_URL="{brain_url}"
export PROVIDER_TYPE="vast_ai"
export PROVIDER_INSTANCE_ID="${{VAST_CONTAINERLABEL:-unknown}}"

# Download and run install script
curl -sSL "{brain_url}/api/v1/nodes/install.sh" | bash -s -- --auto

echo "GPU Agent installation complete"
"""

    async def provision_from_offer(
        self,
        offer: VastGpuOffer,
        db: AsyncSession,
        docker_image: str = "nvidia/cuda:12.2.0-base-ubuntu22.04",
    ) -> ProvisionedInstance:
        """
        Create a new instance from an offer.

        Args:
            offer: The Vast.ai offer to provision
            db: Database session
            docker_image: Docker image to use

        Returns:
            Created ProvisionedInstance record
        """
        logger.info(
            f"Provisioning offer {offer.id}: {offer.gpu_name} x{offer.num_gpus} "
            f"@ ${offer.dph_total}/hr"
        )

        # Create database record
        instance = ProvisionedInstance(
            provider_type=ProviderType.VAST_AI,
            provider_offer_id=str(offer.id),
            status=ProvisioningStatus.PENDING,
            gpu_type=offer.gpu_name,
            gpu_count=offer.num_gpus,
            gpu_vram_mb=offer.gpu_ram,
            hourly_cost=offer.dph_total,
            docker_image=docker_image,
            onstart_script=self._generate_install_script(),
        )
        db.add(instance)
        await db.flush()
        await db.refresh(instance)

        # Create instance with provider
        try:
            instance.status = ProvisioningStatus.CREATING

            config = InstanceConfig(
                docker_image=docker_image,
                disk_gb=max(20.0, offer.disk_space * 0.1),
                onstart_script=self._generate_install_script(),
                env_vars={
                    "PROVIDER_TYPE": "vast_ai",
                    "PROVIDER_INSTANCE_ID": str(instance.id),
                    "BRAIN_URL": settings.brain_public_url,
                },
                label=f"gpu-orch-{instance.id[:8]}",
            )

            provider_instance = await self.vast_client.create_instance(
                str(offer.id), config
            )

            instance.provider_instance_id = provider_instance.instance_id
            instance.status = ProvisioningStatus.STARTING
            instance.started_at = datetime.utcnow()

            logger.info(
                f"Instance created: {provider_instance.instance_id} "
                f"(status: {provider_instance.status})"
            )

        except Exception as e:
            logger.error(f"Failed to create instance: {e}")
            instance.status = ProvisioningStatus.FAILED
            instance.status_message = str(e)

        await db.flush()
        return instance

    async def check_instance_status(
        self,
        instance: ProvisionedInstance,
        db: AsyncSession,
    ) -> ProvisioningStatus:
        """
        Check and update instance status from provider.

        Args:
            instance: ProvisionedInstance to check
            db: Database session

        Returns:
            Updated status
        """
        if not instance.provider_instance_id:
            return instance.status

        try:
            provider_instance = await self.vast_client.get_instance(
                instance.provider_instance_id
            )

            if not provider_instance:
                logger.warning(
                    f"Instance {instance.provider_instance_id} not found at provider"
                )
                return instance.status

            instance.last_status_check_at = datetime.utcnow()

            # Update connection info
            if provider_instance.ssh_host:
                instance.ssh_host = provider_instance.ssh_host
                instance.ssh_port = provider_instance.ssh_port
                instance.public_ip = provider_instance.public_ip

            # Map provider status to our status
            if provider_instance.status == InstanceStatus.RUNNING:
                if instance.status == ProvisioningStatus.STARTING:
                    instance.status = ProvisioningStatus.INSTALLING
                    logger.info(
                        f"Instance {instance.id} is running, agent installing"
                    )
                elif instance.status == ProvisioningStatus.INSTALLING:
                    # Check if timeout exceeded
                    if instance.started_at:
                        elapsed = (
                            datetime.utcnow() - instance.started_at
                        ).total_seconds()
                        if elapsed > settings.provisioning_timeout_seconds:
                            instance.status = ProvisioningStatus.FAILED
                            instance.status_message = (
                                "Agent registration timeout exceeded"
                            )
                            logger.warning(
                                f"Instance {instance.id} timed out waiting "
                                f"for agent registration"
                            )
                        else:
                            instance.status = ProvisioningStatus.WAITING_REGISTRATION

            elif provider_instance.status == InstanceStatus.ERROR:
                instance.status = ProvisioningStatus.FAILED
                instance.status_message = "Provider reported error"

            elif provider_instance.status == InstanceStatus.STOPPED:
                if instance.status != ProvisioningStatus.TERMINATED:
                    instance.status = ProvisioningStatus.TERMINATED
                    instance.terminated_at = datetime.utcnow()

            await db.flush()
            return instance.status

        except Exception as e:
            logger.error(f"Error checking instance status: {e}")
            return instance.status

    async def check_for_registration(
        self,
        instance: ProvisionedInstance,
        db: AsyncSession,
    ) -> bool:
        """
        Check if an agent has registered and link it to the instance.

        Searches for nodes that registered with matching provider info.

        Args:
            instance: ProvisionedInstance waiting for registration
            db: Database session

        Returns:
            True if registration found and linked
        """
        if instance.node_id:
            return True  # Already linked

        if not instance.provider_instance_id:
            return False

        # Look for a node that registered with this provider instance ID
        result = await db.execute(
            select(Node).where(
                Node.provider_type == ProviderType.VAST_AI,
                Node.provider_id == instance.provider_instance_id,
            )
        )
        node = result.scalar_one_or_none()

        if node:
            logger.info(
                f"Found matching node {node.id} for instance {instance.id}"
            )
            instance.node_id = node.id
            instance.status = ProvisioningStatus.ACTIVE

            # Update node with cost info
            node.provider_cost = instance.hourly_cost
            node.hourly_price = instance.hourly_cost * (
                1 + settings.default_markup_percent / 100
            )

            await db.flush()
            return True

        return False

    async def terminate_instance(
        self,
        instance: ProvisionedInstance,
        db: AsyncSession,
    ) -> bool:
        """
        Terminate a provisioned instance.

        Args:
            instance: Instance to terminate
            db: Database session

        Returns:
            True if termination successful
        """
        logger.info(f"Terminating instance {instance.id}")

        if not instance.provider_instance_id:
            instance.status = ProvisioningStatus.TERMINATED
            await db.flush()
            return True

        try:
            instance.status = ProvisioningStatus.TERMINATING

            success = await self.vast_client.terminate_instance(
                instance.provider_instance_id
            )

            if success:
                instance.status = ProvisioningStatus.TERMINATED
                instance.terminated_at = datetime.utcnow()
                instance.total_cost = instance.calculate_current_cost()
                logger.info(
                    f"Instance {instance.id} terminated. "
                    f"Total cost: ${instance.total_cost:.2f}"
                )
            else:
                instance.status_message = "Termination request failed"
                logger.error(f"Failed to terminate instance {instance.id}")

            await db.flush()
            return success

        except Exception as e:
            logger.error(f"Error terminating instance: {e}")
            instance.status_message = str(e)
            await db.flush()
            return False

    async def update_costs(
        self,
        db: AsyncSession,
    ) -> int:
        """
        Update accumulated costs for all active instances.

        Args:
            db: Database session

        Returns:
            Number of instances updated
        """
        result = await db.execute(
            select(ProvisionedInstance).where(
                ProvisionedInstance.status.in_([
                    ProvisioningStatus.STARTING,
                    ProvisioningStatus.INSTALLING,
                    ProvisioningStatus.WAITING_REGISTRATION,
                    ProvisioningStatus.ACTIVE,
                ])
            )
        )
        instances = result.scalars().all()

        count = 0
        for instance in instances:
            old_cost = instance.total_cost
            instance.total_cost = instance.calculate_current_cost()
            if instance.total_cost != old_cost:
                count += 1

        if count > 0:
            await db.flush()
            logger.info(f"Updated costs for {count} active instances")

        return count

    async def get_active_instance_count(self, db: AsyncSession) -> int:
        """Get count of currently active/provisioning instances."""
        result = await db.execute(
            select(ProvisionedInstance).where(
                ProvisionedInstance.status.in_([
                    ProvisioningStatus.PENDING,
                    ProvisioningStatus.CREATING,
                    ProvisioningStatus.STARTING,
                    ProvisioningStatus.INSTALLING,
                    ProvisioningStatus.WAITING_REGISTRATION,
                    ProvisioningStatus.ACTIVE,
                ])
            )
        )
        return len(result.scalars().all())

    async def close(self) -> None:
        """Clean up resources."""
        if self._vast_client:
            await self._vast_client.close()


# Global singleton instance
_provisioning_service: Optional[ProvisioningService] = None


def get_provisioning_service() -> ProvisioningService:
    """Get the global ProvisioningService instance."""
    global _provisioning_service
    if _provisioning_service is None:
        _provisioning_service = ProvisioningService()
    return _provisioning_service
