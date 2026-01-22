"""
Pod management routes for users.
"""

from datetime import datetime
from typing import Annotated, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brain.config import get_settings
from brain.models.base import get_db
from brain.models.node import Node
from brain.models.pod import Pod
from brain.models.user import User
from brain.routes.auth import get_current_active_user
from shared.schemas import (
    PodCreateRequest,
    PodResponse,
    PodStatus,
    NodeStatus,
    PortMapping,
)

router = APIRouter(prefix="/pods", tags=["Pods"])
settings = get_settings()


def pod_to_response(pod: Pod) -> dict:
    """Convert Pod model to response dict."""
    return {
        "id": pod.id,
        "user_id": pod.user_id,
        "node_id": pod.node_id,
        "status": pod.status.value,
        "gpu_type": pod.gpu_type,
        "gpu_count": pod.gpu_count,
        "docker_image": pod.docker_image,
        "port_mappings": [
            PortMapping(
                host_port=int(hp),
                container_port=int(cp),
                protocol="tcp",
            )
            for cp, hp in pod.port_mappings.items()
        ] if pod.port_mappings else [],
        "ssh_connection_string": pod.connection_string,
        "jupyter_url": f"http://{pod.ssh_host}:{pod.jupyter_port}" if pod.jupyter_port else None,
        "web_terminal_url": None,  # TODO: Implement web terminal
        "hourly_price": pod.hourly_price,
        "created_at": pod.created_at,
        "started_at": pod.started_at,
        "total_cost": pod.total_cost,
    }


async def find_best_node(
    db: AsyncSession,
    gpu_type: str,
    gpu_count: int,
    min_vram_mb: Optional[int],
    max_price: Optional[float],
) -> Optional[Node]:
    """Find the best available node for the requested configuration."""
    query = select(Node).where(
        Node.status == NodeStatus.ONLINE,
        Node.gpu_model.ilike(f"%{gpu_type}%"),
        Node.gpu_count >= gpu_count,
    )

    if min_vram_mb:
        query = query.where(Node.total_vram_mb >= min_vram_mb)
    if max_price:
        query = query.where(Node.hourly_price <= max_price)

    # Order by price (cheapest first)
    query = query.order_by(Node.hourly_price.asc())

    result = await db.execute(query)
    nodes = result.scalars().all()

    # Find first available node
    for node in nodes:
        if node.is_available:
            return node

    return None


@router.post("/", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_pod(
    request: PodCreateRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """
    Create a new pod (GPU container instance).
    """
    # Check user balance
    if current_user.balance < settings.minimum_balance_for_pod:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Insufficient balance. Minimum ${settings.minimum_balance_for_pod} required.",
        )

    # Find available node
    node = await find_best_node(
        db,
        request.gpu_type,
        request.gpu_count,
        request.min_vram_mb,
        request.max_hourly_price,
    )

    if not node:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No available nodes with {request.gpu_type} GPU. Please try again later.",
        )

    # Calculate price with markup
    base_price = node.hourly_price
    markup = base_price * (settings.default_markup_percent / 100)
    final_price = base_price + markup

    # Create pod record
    pod = Pod(
        user_id=current_user.id,
        node_id=node.id,
        status=PodStatus.PENDING,
        docker_image=request.docker_image,
        gpu_type=node.gpu_model,
        gpu_count=request.gpu_count,
        port_mappings={str(pm.container_port): pm.host_port for pm in request.port_mappings},
        environment_variables=request.environment_variables,
        startup_command=request.startup_command,
        volume_mounts=request.volume_mounts,
        hourly_price=final_price,
        provider_cost=base_price,
    )

    db.add(pod)
    await db.flush()
    await db.refresh(pod)

    # TODO: Send deploy command to agent via task queue
    # For now, we'll mark it as provisioning
    pod.status = PodStatus.PROVISIONING
    await db.flush()

    return pod_to_response(pod)


@router.get("/", response_model=list[dict])
async def list_pods(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    status_filter: Optional[PodStatus] = None,
) -> list[dict]:
    """List all pods for the current user."""
    query = select(Pod).where(Pod.user_id == current_user.id)

    if status_filter:
        query = query.where(Pod.status == status_filter)

    query = query.order_by(Pod.created_at.desc())

    result = await db.execute(query)
    pods = result.scalars().all()

    return [pod_to_response(pod) for pod in pods]


@router.get("/{pod_id}", response_model=dict)
async def get_pod(
    pod_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Get a specific pod by ID."""
    result = await db.execute(
        select(Pod).where(Pod.id == str(pod_id), Pod.user_id == current_user.id)
    )
    pod = result.scalar_one_or_none()

    if not pod:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pod not found",
        )

    return pod_to_response(pod)


@router.post("/{pod_id}/stop")
async def stop_pod(
    pod_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Stop a running pod."""
    result = await db.execute(
        select(Pod).where(Pod.id == str(pod_id), Pod.user_id == current_user.id)
    )
    pod = result.scalar_one_or_none()

    if not pod:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pod not found",
        )

    if pod.status not in [PodStatus.RUNNING, PodStatus.PROVISIONING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot stop pod in {pod.status.value} status",
        )

    # Update status
    pod.status = PodStatus.STOPPING
    pod.stopped_at = datetime.utcnow()

    # Calculate final cost
    if pod.started_at:
        runtime_seconds = (pod.stopped_at - pod.started_at).total_seconds()
        pod.total_runtime_seconds = int(runtime_seconds)
        pod.total_cost = pod.calculate_current_cost()

    await db.flush()

    # TODO: Send stop command to agent via task queue

    return {"message": "Pod stop initiated", "pod_id": str(pod_id)}


@router.delete("/{pod_id}")
async def terminate_pod(
    pod_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Terminate and delete a pod."""
    result = await db.execute(
        select(Pod).where(Pod.id == str(pod_id), Pod.user_id == current_user.id)
    )
    pod = result.scalar_one_or_none()

    if not pod:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pod not found",
        )

    # If running, stop first
    if pod.status in [PodStatus.RUNNING, PodStatus.PROVISIONING]:
        pod.stopped_at = datetime.utcnow()
        if pod.started_at:
            pod.total_runtime_seconds = int(
                (pod.stopped_at - pod.started_at).total_seconds()
            )
            pod.total_cost = pod.calculate_current_cost()

    pod.status = PodStatus.TERMINATED
    pod.termination_reason = "User requested termination"

    await db.flush()

    # TODO: Send terminate command to agent

    return {"message": "Pod terminated", "pod_id": str(pod_id)}


@router.get("/{pod_id}/logs")
async def get_pod_logs(
    pod_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    tail: int = 100,
) -> dict:
    """Get logs from a pod."""
    result = await db.execute(
        select(Pod).where(Pod.id == str(pod_id), Pod.user_id == current_user.id)
    )
    pod = result.scalar_one_or_none()

    if not pod:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pod not found",
        )

    # TODO: Fetch logs from agent
    return {
        "pod_id": str(pod_id),
        "logs": "Log fetching not yet implemented",
        "tail": tail,
    }
