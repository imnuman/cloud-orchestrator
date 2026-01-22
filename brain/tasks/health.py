"""
Health check tasks for nodes and pods.
"""

import asyncio
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brain.config import get_settings
from brain.models.base import async_session_maker
from brain.models.node import Node
from brain.models.pod import Pod
from brain.tasks.celery_app import celery_app
from shared.schemas import NodeStatus, PodStatus

settings = get_settings()


async def _check_node_health() -> dict:
    """
    Check node health based on heartbeats.
    Mark nodes as offline if they miss too many heartbeats.
    """
    stats = {
        "nodes_checked": 0,
        "nodes_marked_offline": 0,
        "nodes_online": 0,
    }

    timeout_threshold = datetime.utcnow() - timedelta(
        seconds=settings.heartbeat_timeout_seconds
    )

    async with async_session_maker() as db:
        # Get all nodes that should be online
        result = await db.execute(
            select(Node).where(Node.status != NodeStatus.MAINTENANCE)
        )
        nodes = result.scalars().all()

        for node in nodes:
            stats["nodes_checked"] += 1

            if node.status == NodeStatus.ONLINE:
                # Check if heartbeat is stale
                if node.last_heartbeat_at and node.last_heartbeat_at < timeout_threshold:
                    node.consecutive_missed_heartbeats += 1

                    # Mark offline after 3 missed heartbeats
                    if node.consecutive_missed_heartbeats >= 3:
                        node.status = NodeStatus.OFFLINE
                        stats["nodes_marked_offline"] += 1
                        print(f"Node {node.hostname} marked offline (missed heartbeats)")

                        # TODO: Handle running pods on this node
                        # - Notify users
                        # - Attempt migration
                        # - Stop billing
                else:
                    stats["nodes_online"] += 1
            elif node.status == NodeStatus.OFFLINE:
                # Check if node came back online recently
                if node.last_heartbeat_at and node.last_heartbeat_at > timeout_threshold:
                    node.status = NodeStatus.ONLINE
                    node.consecutive_missed_heartbeats = 0
                    stats["nodes_online"] += 1

        await db.commit()

    return stats


async def _cleanup_stale_pods() -> dict:
    """
    Clean up pods that are stuck in transitional states.
    """
    stats = {
        "pods_checked": 0,
        "pods_cleaned": 0,
    }

    # Pods stuck in PROVISIONING for more than 10 minutes
    provisioning_timeout = datetime.utcnow() - timedelta(minutes=10)

    # Pods stuck in STOPPING for more than 5 minutes
    stopping_timeout = datetime.utcnow() - timedelta(minutes=5)

    async with async_session_maker() as db:
        # Check stuck provisioning pods
        result = await db.execute(
            select(Pod).where(
                Pod.status == PodStatus.PROVISIONING,
                Pod.created_at < provisioning_timeout,
            )
        )
        stuck_provisioning = result.scalars().all()

        for pod in stuck_provisioning:
            stats["pods_checked"] += 1
            pod.status = PodStatus.FAILED
            pod.status_message = "Provisioning timeout - please try again"
            pod.termination_reason = "Provisioning timeout"
            stats["pods_cleaned"] += 1

        # Check stuck stopping pods
        result = await db.execute(
            select(Pod).where(
                Pod.status == PodStatus.STOPPING,
                Pod.stopped_at < stopping_timeout,
            )
        )
        stuck_stopping = result.scalars().all()

        for pod in stuck_stopping:
            stats["pods_checked"] += 1
            pod.status = PodStatus.STOPPED
            pod.status_message = "Force stopped after timeout"
            stats["pods_cleaned"] += 1

        await db.commit()

    return stats


@celery_app.task(name="brain.tasks.health.check_node_health")
def check_node_health() -> dict:
    """Celery task wrapper for node health check."""
    return asyncio.run(_check_node_health())


@celery_app.task(name="brain.tasks.health.cleanup_stale_pods")
def cleanup_stale_pods() -> dict:
    """Celery task wrapper for stale pod cleanup."""
    return asyncio.run(_cleanup_stale_pods())
