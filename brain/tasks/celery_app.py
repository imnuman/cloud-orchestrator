"""
Celery application configuration.
"""

from celery import Celery
from celery.schedules import crontab

from brain.config import get_settings

settings = get_settings()

celery_app = Celery(
    "gpu_orchestrator",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "brain.tasks.billing",
        "brain.tasks.health",
        "brain.tasks.sourcing",
        "brain.tasks.model_health",
    ],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max per task
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# Periodic tasks (beat schedule)
celery_app.conf.beat_schedule = {
    # Billing: Run every minute
    "bill-running-pods": {
        "task": "brain.tasks.billing.bill_running_pods",
        "schedule": settings.billing_interval_seconds,
    },
    # Check low balances: Every 5 minutes
    "check-low-balances": {
        "task": "brain.tasks.billing.check_low_balances",
        "schedule": 300,
    },
    # Node health check: Every 30 seconds
    "check-node-health": {
        "task": "brain.tasks.health.check_node_health",
        "schedule": settings.heartbeat_interval_seconds,
    },
    # Cleanup stale pods: Every 5 minutes
    "cleanup-stale-pods": {
        "task": "brain.tasks.health.cleanup_stale_pods",
        "schedule": 300,
    },
    # GPU Sourcing: Search for offers (configured interval, default 5 min)
    "search-and-provision": {
        "task": "brain.tasks.sourcing.search_and_provision",
        "schedule": settings.sourcing_interval_seconds,
    },
    # Check provisioning status: Every 30 seconds
    "check-provisioning-status": {
        "task": "brain.tasks.sourcing.check_provisioning_status",
        "schedule": settings.provisioning_check_interval_seconds,
    },
    # Update instance costs: Every 5 minutes
    "update-instance-costs": {
        "task": "brain.tasks.sourcing.update_instance_costs",
        "schedule": 300,
    },
    # Cleanup failed instances: Every 10 minutes
    "terminate-failed-instances": {
        "task": "brain.tasks.sourcing.terminate_failed_instances",
        "schedule": 600,
    },
    # Model Deployments: Provision pending every 30 seconds
    "provision-pending-deployments": {
        "task": "brain.tasks.model_health.provision_pending_deployments",
        "schedule": 30,
    },
    # Model Deployments: Health check every 30 seconds
    "check-deployment-health": {
        "task": "brain.tasks.model_health.check_deployment_health",
        "schedule": 30,
    },
    # Model Deployments: Update statuses every 15 seconds
    "update-deployment-statuses": {
        "task": "brain.tasks.model_health.update_deployment_statuses",
        "schedule": 15,
    },
    # Model Deployments: Bill every minute
    "bill-model-deployments": {
        "task": "brain.tasks.model_health.bill_model_deployments",
        "schedule": 60,
    },
    # Model Deployments: Cleanup stopped every 5 minutes
    "cleanup-stopped-deployments": {
        "task": "brain.tasks.model_health.cleanup_stopped_deployments",
        "schedule": 300,
    },
}
