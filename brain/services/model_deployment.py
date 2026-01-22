"""
Model Deployment Service.

Handles the lifecycle of model deployments:
- Provisioning nodes for model deployments
- Creating pods with proper configuration
- Health checking and monitoring
- Scaling and resource management
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from brain.config import get_settings
from brain.models.node import Node
from brain.models.pod import Pod
from brain.models.model_catalog import (
    ModelTemplate,
    Deployment,
    DeploymentStatus,
)
from shared.schemas import NodeStatus, PodStatus

logger = logging.getLogger(__name__)
settings = get_settings()


def escape_like(value: str) -> str:
    """Escape special LIKE pattern characters to prevent SQL injection."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class ModelDeploymentService:
    """
    Service for managing model deployments.

    Handles:
    - Finding suitable nodes for deployments
    - Creating pods with model-specific configuration
    - Updating deployment status
    - Health monitoring
    - Graceful shutdown
    """

    async def provision_deployment(
        self,
        deployment: Deployment,
        db: AsyncSession,
    ) -> bool:
        """
        Provision a deployment by finding a node and creating a pod.

        Args:
            deployment: Deployment to provision
            db: Database session

        Returns:
            True if provisioning started successfully
        """
        logger.info(f"Provisioning deployment {deployment.id}: {deployment.name}")

        # Get model template
        result = await db.execute(
            select(ModelTemplate).where(ModelTemplate.id == deployment.model_template_id)
        )
        template = result.scalar_one_or_none()

        if not template:
            deployment.status = DeploymentStatus.FAILED
            deployment.status_message = "Model template not found"
            await db.flush()
            return False

        # Find a suitable node
        node = await self._find_suitable_node(
            min_vram_gb=template.min_vram_gb,
            gpu_type=deployment.gpu_type,
            gpu_count=deployment.gpu_count,
            db=db,
        )

        if not node:
            deployment.status = DeploymentStatus.PENDING
            deployment.status_message = "Waiting for available GPU resources"
            await db.flush()
            return False

        # Update deployment status
        deployment.status = DeploymentStatus.PROVISIONING
        deployment.node_id = node.id
        deployment.hourly_price = node.hourly_price * deployment.gpu_count

        # Create pod configuration
        env_vars = {**template.default_env, **deployment.custom_env}

        # Add model-specific environment variables
        if template.model_id:
            env_vars["MODEL"] = template.model_id
            env_vars["HF_MODEL_ID"] = template.model_id

        # Add deployment-specific env vars
        env_vars["DEPLOYMENT_ID"] = deployment.id
        env_vars["API_KEY"] = deployment.api_key or ""

        # Create the pod
        pod = Pod(
            user_id=deployment.user_id,
            node_id=node.id,
            name=f"model-{deployment.id[:8]}",
            docker_image=template.full_docker_image,
            gpu_type=node.gpu_model,
            gpu_count=deployment.gpu_count,
            status=PodStatus.PENDING,
            environment_variables=env_vars,
            startup_command=template.startup_command,
            hourly_price=deployment.hourly_price,
            provider_cost=node.provider_cost * deployment.gpu_count,
        )

        db.add(pod)
        await db.flush()
        await db.refresh(pod)

        # Link pod to deployment
        deployment.pod_id = pod.id
        deployment.status = DeploymentStatus.STARTING

        # Update node pod count
        node.current_pod_count += 1

        await db.flush()

        logger.info(
            f"Deployment {deployment.id} provisioned on node {node.id} "
            f"with pod {pod.id}"
        )

        return True

    async def _find_suitable_node(
        self,
        min_vram_gb: int,
        gpu_type: Optional[str],
        gpu_count: int,
        db: AsyncSession,
    ) -> Optional[Node]:
        """
        Find a suitable node for the deployment.

        Args:
            min_vram_gb: Minimum VRAM required in GB
            gpu_type: Preferred GPU type (optional)
            gpu_count: Number of GPUs needed
            db: Database session

        Returns:
            Suitable Node or None
        """
        min_vram_mb = min_vram_gb * 1024

        # Build query for available nodes
        query = (
            select(Node)
            .where(Node.status == NodeStatus.ONLINE)
            .where(Node.total_vram_mb >= min_vram_mb)
            .where(Node.gpu_count >= gpu_count)
        )

        # Filter by GPU type if specified
        if gpu_type:
            escaped_type = escape_like(gpu_type)
            query = query.where(Node.gpu_model.ilike(f"%{escaped_type}%"))

        # Order by price (cheapest first)
        query = query.order_by(Node.hourly_price)

        result = await db.execute(query)
        nodes = result.scalars().all()

        # Find first node with capacity
        for node in nodes:
            if node.is_available:
                return node

        return None

    async def update_deployment_status(
        self,
        deployment: Deployment,
        db: AsyncSession,
    ) -> DeploymentStatus:
        """
        Update deployment status based on pod status.

        Args:
            deployment: Deployment to update
            db: Database session

        Returns:
            Updated status
        """
        if not deployment.pod_id:
            return deployment.status

        # Get pod status
        result = await db.execute(
            select(Pod).where(Pod.id == deployment.pod_id)
        )
        pod = result.scalar_one_or_none()

        if not pod:
            deployment.status = DeploymentStatus.FAILED
            deployment.status_message = "Pod not found"
            await db.flush()
            return deployment.status

        # Map pod status to deployment status
        status_mapping = {
            PodStatus.PENDING: DeploymentStatus.PROVISIONING,
            PodStatus.PROVISIONING: DeploymentStatus.STARTING,
            PodStatus.RUNNING: DeploymentStatus.LOADING,  # Initially loading
            PodStatus.STOPPING: DeploymentStatus.STOPPING,
            PodStatus.STOPPED: DeploymentStatus.STOPPED,
            PodStatus.FAILED: DeploymentStatus.FAILED,
            PodStatus.TERMINATED: DeploymentStatus.STOPPED,
        }

        new_status = status_mapping.get(pod.status, deployment.status)

        # If pod is running and health check passes, mark as READY
        if pod.status == PodStatus.RUNNING and deployment.status == DeploymentStatus.LOADING:
            # Check startup timeout
            if deployment.started_at:
                result = await db.execute(
                    select(ModelTemplate).where(
                        ModelTemplate.id == deployment.model_template_id
                    )
                )
                template = result.scalar_one_or_none()

                elapsed = (datetime.utcnow() - deployment.started_at).total_seconds()
                timeout = template.startup_timeout_seconds if template else 600

                if elapsed > timeout:
                    deployment.status = DeploymentStatus.FAILED
                    deployment.status_message = "Model loading timeout"
                    await db.flush()
                    return deployment.status

        if new_status != deployment.status:
            deployment.status = new_status

            # Set timestamps
            if new_status == DeploymentStatus.READY and not deployment.started_at:
                deployment.started_at = datetime.utcnow()
            elif new_status == DeploymentStatus.STOPPED and not deployment.stopped_at:
                deployment.stopped_at = datetime.utcnow()
                deployment.total_cost = deployment.calculate_current_cost()

        await db.flush()
        return deployment.status

    async def check_deployment_health(
        self,
        deployment: Deployment,
        db: AsyncSession,
    ) -> bool:
        """
        Check deployment health.

        Args:
            deployment: Deployment to check
            db: Database session

        Returns:
            True if healthy
        """
        if deployment.status != DeploymentStatus.READY:
            return False

        # Get model template for health check URL
        result = await db.execute(
            select(ModelTemplate).where(ModelTemplate.id == deployment.model_template_id)
        )
        template = result.scalar_one_or_none()

        if not template or not template.health_check_url:
            return True  # No health check configured

        # In production, make HTTP request to health check endpoint
        # For now, simulate health check
        deployment.last_health_check_at = datetime.utcnow()

        # Simulate occasional failures for testing
        # In production, this would be actual HTTP call
        is_healthy = True  # await self._http_health_check(deployment, template)

        if is_healthy:
            deployment.health_check_failures = 0
        else:
            deployment.health_check_failures += 1

            # Mark unhealthy after 3 consecutive failures
            if deployment.health_check_failures >= 3:
                deployment.status = DeploymentStatus.UNHEALTHY
                deployment.status_message = "Health check failures exceeded threshold"

        await db.flush()
        return is_healthy

    async def mark_deployment_ready(
        self,
        deployment: Deployment,
        api_endpoint: str,
        ui_url: Optional[str],
        db: AsyncSession,
    ) -> None:
        """
        Mark deployment as ready with connection info.

        Args:
            deployment: Deployment to update
            api_endpoint: API endpoint URL
            ui_url: Optional UI URL
            db: Database session
        """
        deployment.status = DeploymentStatus.READY
        deployment.api_endpoint = api_endpoint
        deployment.ui_url = ui_url
        deployment.started_at = datetime.utcnow()
        deployment.status_message = None

        await db.flush()

        logger.info(
            f"Deployment {deployment.id} is ready at {api_endpoint}"
        )

    async def stop_deployment(
        self,
        deployment: Deployment,
        db: AsyncSession,
    ) -> bool:
        """
        Stop a deployment.

        Args:
            deployment: Deployment to stop
            db: Database session

        Returns:
            True if stop initiated
        """
        logger.info(f"Stopping deployment {deployment.id}")

        deployment.status = DeploymentStatus.STOPPING

        # Get and stop the pod
        if deployment.pod_id:
            result = await db.execute(
                select(Pod).where(Pod.id == deployment.pod_id)
            )
            pod = result.scalar_one_or_none()

            if pod and pod.status not in [PodStatus.STOPPED, PodStatus.TERMINATED]:
                pod.status = PodStatus.STOPPING

                # Update node pod count
                if deployment.node_id:
                    node_result = await db.execute(
                        select(Node).where(Node.id == deployment.node_id)
                    )
                    node = node_result.scalar_one_or_none()
                    if node and node.current_pod_count > 0:
                        node.current_pod_count -= 1

        # Update deployment
        deployment.stopped_at = datetime.utcnow()
        deployment.total_cost = deployment.calculate_current_cost()

        await db.flush()

        return True

    async def get_pending_deployments(
        self,
        db: AsyncSession,
        limit: int = 10,
    ) -> list[Deployment]:
        """
        Get pending deployments that need provisioning.

        Args:
            db: Database session
            limit: Maximum number to return

        Returns:
            List of pending Deployments
        """
        result = await db.execute(
            select(Deployment)
            .where(Deployment.status == DeploymentStatus.PENDING)
            .order_by(Deployment.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_deployments_needing_health_check(
        self,
        db: AsyncSession,
        interval_seconds: int = 30,
        limit: int = 50,
    ) -> list[Deployment]:
        """
        Get deployments that need health checking.

        Args:
            db: Database session
            interval_seconds: Minimum time since last check
            limit: Maximum number to return

        Returns:
            List of Deployments needing health check
        """
        cutoff = datetime.utcnow()

        result = await db.execute(
            select(Deployment)
            .where(Deployment.status.in_([
                DeploymentStatus.READY,
                DeploymentStatus.UNHEALTHY,
            ]))
            .where(
                (Deployment.last_health_check_at == None)  # noqa: E711
                | (Deployment.last_health_check_at < cutoff)
            )
            .limit(limit)
        )
        return list(result.scalars().all())


# Global singleton instance
_model_deployment_service: Optional[ModelDeploymentService] = None


def get_model_deployment_service() -> ModelDeploymentService:
    """Get the global ModelDeploymentService instance."""
    global _model_deployment_service
    if _model_deployment_service is None:
        _model_deployment_service = ModelDeploymentService()
    return _model_deployment_service
