"""
GPU Provisioning Service.

Handles creating and managing provisioned GPU instances from offers.
Supports multiple providers with automatic failover.
"""

import logging
from datetime import datetime
from typing import Optional, Union

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brain.adapters.base import GpuOffer, InstanceConfig, InstanceStatus
from brain.adapters.lambda_labs import LambdaClient, LambdaGpuOffer
from brain.adapters.runpod import RunPodClient, RunPodGpuOffer
from brain.adapters.vast_ai import VastClient, VastGpuOffer
from brain.config import get_settings
from brain.models.node import Node
from brain.models.provisioned_instance import (
    ProvisionedInstance,
    ProvisioningStatus,
)
from brain.services.multi_provider import (
    MultiProviderService,
    PricingStrategy,
    get_multi_provider_service,
)
from shared.schemas import ProviderType

logger = logging.getLogger(__name__)
settings = get_settings()

# Type alias for provider-specific offers
ProviderOffer = Union[VastGpuOffer, RunPodGpuOffer, LambdaGpuOffer, GpuOffer]


class ProvisioningService:
    """
    Service for provisioning and managing GPU instances.

    Handles:
    - Creating instances from offers (supporting multiple providers)
    - Automatic failover across providers
    - Tracking instance status through lifecycle
    - Matching agent registrations to provisioned instances
    - Cost tracking
    - Instance termination
    """

    def __init__(
        self,
        vast_client: Optional[VastClient] = None,
        multi_provider: Optional[MultiProviderService] = None,
    ):
        """
        Initialize provisioning service.

        Args:
            vast_client: Optional VastClient instance (for backward compatibility)
            multi_provider: Optional MultiProviderService instance
        """
        self._vast_client = vast_client
        self._multi_provider = multi_provider

    @property
    def vast_client(self) -> VastClient:
        """Lazily initialize Vast.ai client."""
        if self._vast_client is None:
            self._vast_client = VastClient(
                api_key=settings.vast_ai_api_key,
                use_mock=settings.use_mock_providers,
            )
        return self._vast_client

    @property
    def multi_provider(self) -> MultiProviderService:
        """Lazily initialize multi-provider service."""
        if self._multi_provider is None:
            self._multi_provider = get_multi_provider_service()
        return self._multi_provider

    def _provider_type_from_string(self, provider: str) -> ProviderType:
        """Convert provider string to ProviderType enum."""
        mapping = {
            "vast_ai": ProviderType.VAST_AI,
            "runpod": ProviderType.RUNPOD,
            "lambda_labs": ProviderType.LAMBDA_LABS,
        }
        return mapping.get(provider, ProviderType.VAST_AI)

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

    async def provision_with_failover(
        self,
        gpu_type: str,
        db: AsyncSession,
        max_price: Optional[float] = None,
        min_vram_mb: Optional[int] = None,
        docker_image: str = "nvidia/cuda:12.2.0-base-ubuntu22.04",
        preferred_providers: Optional[list[str]] = None,
    ) -> ProvisionedInstance:
        """
        Provision a GPU instance with automatic failover across providers.

        Args:
            gpu_type: GPU type to provision (e.g., "RTX 4090", "A100")
            db: Database session
            max_price: Maximum acceptable hourly price
            min_vram_mb: Minimum VRAM requirement
            docker_image: Docker image to use
            preferred_providers: Providers to try first (in order)

        Returns:
            Created ProvisionedInstance record
        """
        logger.info(
            f"Provisioning with failover: {gpu_type} "
            f"(max ${max_price}/hr, min {min_vram_mb}MB VRAM)"
        )

        # Get best offers from multi-provider service
        offers = await self.multi_provider.find_best_offers(
            gpu_type=gpu_type,
            max_price=max_price,
            min_vram_mb=min_vram_mb,
            strategy=PricingStrategy.BALANCED,
            limit=10,
        )

        if not offers:
            raise RuntimeError(f"No offers found for {gpu_type}")

        # Reorder by preferred providers if specified
        if preferred_providers:
            def sort_key(agg):
                try:
                    pref_idx = preferred_providers.index(agg.offer.provider)
                except ValueError:
                    pref_idx = len(preferred_providers)
                return (pref_idx, agg.value_score)

            offers.sort(key=sort_key)

        # Try each offer until one succeeds
        errors = []
        for agg in offers:
            offer = agg.offer
            provider = offer.provider
            provider_type = self._provider_type_from_string(provider)

            # Create database record
            instance = ProvisionedInstance(
                provider_type=provider_type,
                provider_offer_id=offer.offer_id,
                status=ProvisioningStatus.PENDING,
                gpu_type=offer.gpu_name,
                gpu_count=offer.gpu_count,
                gpu_vram_mb=offer.gpu_vram_mb,
                hourly_cost=offer.hourly_price,
                docker_image=docker_image,
                onstart_script=self._generate_install_script(provider),
            )
            db.add(instance)
            await db.flush()
            await db.refresh(instance)

            try:
                instance.status = ProvisioningStatus.CREATING

                config = InstanceConfig(
                    docker_image=docker_image,
                    disk_gb=max(20.0, offer.disk_gb * 0.1),
                    onstart_script=self._generate_install_script(provider),
                    env_vars={
                        "PROVIDER_TYPE": provider,
                        "PROVIDER_INSTANCE_ID": str(instance.id),
                        "BRAIN_URL": settings.brain_public_url,
                    },
                    label=f"gpu-orch-{instance.id[:8]}",
                )

                # Create instance using multi-provider service
                client = self.multi_provider._get_client_for_provider(provider)
                if not client:
                    raise RuntimeError(f"No client for provider: {provider}")

                provider_instance = await client.create_instance(offer.offer_id, config)

                instance.provider_instance_id = provider_instance.instance_id
                instance.status = ProvisioningStatus.STARTING
                instance.started_at = datetime.utcnow()

                logger.info(
                    f"Instance created on {provider}: {provider_instance.instance_id} "
                    f"({offer.gpu_name} @ ${offer.hourly_price}/hr)"
                )

                await db.flush()
                return instance

            except Exception as e:
                error_msg = f"{provider}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"Failed to provision on {provider}: {e}")

                # Mark this attempt as failed
                instance.status = ProvisioningStatus.FAILED
                instance.status_message = str(e)
                await db.flush()

        # All providers failed
        raise RuntimeError(
            f"Failed to provision {gpu_type} on any provider. Errors:\n"
            + "\n".join(errors)
        )

    def _generate_install_script(self, provider: str = "vast_ai") -> str:
        """Generate the agent installation script for onstart."""
        brain_url = settings.brain_public_url.rstrip("/")

        return f"""#!/bin/bash
set -e

# GPU Agent Auto-Install Script
echo "Installing GPU Agent..."

# Set environment for agent
export BRAIN_URL="{brain_url}"
export PROVIDER_TYPE="{provider}"

# Download and run install script
curl -sSL "{brain_url}/api/v1/nodes/install.sh" | bash -s -- --auto

echo "GPU Agent installation complete"
"""

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
            # Use appropriate client based on provider type
            provider = instance.provider_type.value
            provider_instance = await self.multi_provider.get_instance_status(
                instance.provider_instance_id,
                provider,
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
        # Support any provider type
        result = await db.execute(
            select(Node).where(
                Node.provider_type == instance.provider_type,
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

            # Use appropriate client based on provider type
            provider = instance.provider_type.value
            success = await self.multi_provider.terminate_instance(
                instance.provider_instance_id,
                provider,
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
        if self._multi_provider:
            await self._multi_provider.close()


# Global singleton instance
_provisioning_service: Optional[ProvisioningService] = None


def get_provisioning_service() -> ProvisioningService:
    """Get the global ProvisioningService instance."""
    global _provisioning_service
    if _provisioning_service is None:
        _provisioning_service = ProvisioningService()
    return _provisioning_service
