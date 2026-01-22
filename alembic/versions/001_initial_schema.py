"""Initial schema with all models.

Revision ID: 001_initial
Revises:
Create Date: 2026-01-22

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("is_verified", sa.Boolean(), default=False),
        sa.Column("is_admin", sa.Boolean(), default=False),
        sa.Column("is_provider", sa.Boolean(), default=False),
        sa.Column("balance", sa.Float(), default=0.0),
        sa.Column("total_spent", sa.Float(), default=0.0),
        sa.Column("credit_limit", sa.Float(), default=0.0),
        sa.Column("provider_earnings", sa.Float(), default=0.0),
        sa.Column("provider_payout_email", sa.String(255), nullable=True),
        sa.Column("api_key", sa.String(255), unique=True, nullable=True, index=True),
        sa.Column("last_login_at", sa.DateTime(), nullable=True),
        sa.Column("verification_token", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Create nodes table
    op.create_table(
        "nodes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("hostname", sa.String(255), nullable=False),
        sa.Column("ip_address", sa.String(45), nullable=False),
        sa.Column("public_ip", sa.String(45), nullable=True),
        sa.Column("status", sa.String(50), default="offline", index=True),
        sa.Column("last_heartbeat_at", sa.DateTime(), nullable=True),
        sa.Column("consecutive_missed_heartbeats", sa.Integer(), default=0),
        sa.Column(
            "provider_type",
            sa.Enum("RUNPOD", "LAMBDA_LABS", "VAST_AI", "COMMUNITY", "INTERNAL", name="providertype"),
            default="COMMUNITY",
            index=True,
        ),
        sa.Column("provider_id", sa.String(255), nullable=True, index=True),
        sa.Column("owner_user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("gpu_model", sa.String(255), nullable=True),
        sa.Column("gpu_count", sa.Integer(), default=1),
        sa.Column("total_vram_mb", sa.Integer(), nullable=True),
        sa.Column("gpu_details", postgresql.JSON(), nullable=True),
        sa.Column("cpu_model", sa.String(255), nullable=True),
        sa.Column("cpu_cores", sa.Integer(), nullable=True),
        sa.Column("ram_total_mb", sa.Integer(), nullable=True),
        sa.Column("disk_total_gb", sa.Float(), nullable=True),
        sa.Column("os_info", sa.String(255), nullable=True),
        sa.Column("hourly_price", sa.Float(), nullable=True),
        sa.Column("provider_cost", sa.Float(), nullable=True),
        sa.Column("current_gpu_utilization", sa.Float(), nullable=True),
        sa.Column("current_vram_used_mb", sa.Integer(), nullable=True),
        sa.Column("current_temperature_c", sa.Integer(), nullable=True),
        sa.Column("max_pods", sa.Integer(), default=1),
        sa.Column("current_pod_count", sa.Integer(), default=0),
        sa.Column("agent_version", sa.String(50), nullable=True),
        sa.Column("agent_api_key", sa.String(255), unique=True, nullable=True, index=True),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Create pods table
    op.create_table(
        "pods",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("node_id", sa.String(36), sa.ForeignKey("nodes.id"), nullable=True, index=True),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING", "PROVISIONING", "RUNNING", "STOPPING", "STOPPED", "FAILED", "TERMINATED",
                name="podstatus"
            ),
            default="PENDING",
            index=True,
        ),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("docker_image", sa.String(512), nullable=False),
        sa.Column("gpu_type", sa.String(255), nullable=False),
        sa.Column("gpu_count", sa.Integer(), default=1),
        sa.Column("gpu_indices", postgresql.JSON(), nullable=True),
        sa.Column("port_mappings", postgresql.JSON(), nullable=True),
        sa.Column("ssh_port", sa.Integer(), nullable=True),
        sa.Column("ssh_host", sa.String(255), nullable=True),
        sa.Column("jupyter_port", sa.Integer(), nullable=True),
        sa.Column("container_id", sa.String(255), nullable=True),
        sa.Column("environment_variables", postgresql.JSON(), nullable=True),
        sa.Column("startup_command", sa.String(1024), nullable=True),
        sa.Column("volume_mounts", postgresql.JSON(), nullable=True),
        sa.Column("hourly_price", sa.Float(), nullable=False),
        sa.Column("provider_cost", sa.Float(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("stopped_at", sa.DateTime(), nullable=True),
        sa.Column("last_billed_at", sa.DateTime(), nullable=True),
        sa.Column("total_runtime_seconds", sa.Integer(), default=0),
        sa.Column("total_cost", sa.Float(), default=0.0),
        sa.Column("termination_reason", sa.String(255), nullable=True),
        sa.Column("auto_stop_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Create transactions table
    op.create_table(
        "transactions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column(
            "type",
            sa.Enum("DEPOSIT", "CHARGE", "REFUND", "CREDIT", "PAYOUT", "ADJUSTMENT", name="transactiontype"),
            nullable=False,
        ),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("balance_after", sa.Float(), nullable=False),
        sa.Column("description", sa.String(512), nullable=True),
        sa.Column("reference_id", sa.String(255), nullable=True),
        sa.Column("pod_id", sa.String(36), sa.ForeignKey("pods.id"), nullable=True),
        sa.Column(
            "payment_method",
            sa.Enum("STRIPE", "CRYPTO", "MANUAL", "PROMOTIONAL", name="paymentmethod"),
            nullable=True,
        ),
        sa.Column("metadata", postgresql.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
    )

    # Create usage_records table
    op.create_table(
        "usage_records",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("pod_id", sa.String(36), sa.ForeignKey("pods.id"), nullable=False, index=True),
        sa.Column("node_id", sa.String(36), sa.ForeignKey("nodes.id"), nullable=False),
        sa.Column("gpu_type", sa.String(255), nullable=False),
        sa.Column("gpu_count", sa.Integer(), default=1),
        sa.Column("period_start", sa.DateTime(), nullable=False),
        sa.Column("period_end", sa.DateTime(), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), default=0),
        sa.Column("hourly_rate", sa.Float(), nullable=False),
        sa.Column("amount_charged", sa.Float(), default=0.0),
        sa.Column("provider_cost", sa.Float(), default=0.0),
        sa.Column("is_billed", sa.Boolean(), default=False),
        sa.Column("transaction_id", sa.String(36), sa.ForeignKey("transactions.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
    )

    # Create provisioned_instances table
    op.create_table(
        "provisioned_instances",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "provider_type",
            sa.Enum("RUNPOD", "LAMBDA_LABS", "VAST_AI", "COMMUNITY", "INTERNAL", name="providertype"),
            default="VAST_AI",
            index=True,
        ),
        sa.Column("provider_instance_id", sa.String(255), nullable=True, index=True),
        sa.Column("provider_offer_id", sa.String(255), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING", "CREATING", "STARTING", "INSTALLING", "WAITING_REGISTRATION",
                "ACTIVE", "FAILED", "TERMINATING", "TERMINATED",
                name="provisioningstatus"
            ),
            default="PENDING",
            index=True,
        ),
        sa.Column("status_message", sa.String(1024), nullable=True),
        sa.Column("last_status_check_at", sa.DateTime(), nullable=True),
        sa.Column("ssh_host", sa.String(255), nullable=True),
        sa.Column("ssh_port", sa.Integer(), nullable=True),
        sa.Column("public_ip", sa.String(45), nullable=True),
        sa.Column("gpu_type", sa.String(255), nullable=False, index=True),
        sa.Column("gpu_count", sa.Integer(), default=1),
        sa.Column("gpu_vram_mb", sa.Integer(), nullable=False),
        sa.Column("hourly_cost", sa.Float(), nullable=False),
        sa.Column("total_cost", sa.Float(), default=0.0),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("terminated_at", sa.DateTime(), nullable=True),
        sa.Column("node_id", sa.String(36), sa.ForeignKey("nodes.id"), nullable=True, index=True),
        sa.Column("docker_image", sa.String(512), nullable=True),
        sa.Column("onstart_script", sa.String(4096), nullable=True),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("provisioned_instances")
    op.drop_table("usage_records")
    op.drop_table("transactions")
    op.drop_table("pods")
    op.drop_table("nodes")
    op.drop_table("users")

    # Drop enums
    op.execute("DROP TYPE IF EXISTS provisioningstatus")
    op.execute("DROP TYPE IF EXISTS paymentmethod")
    op.execute("DROP TYPE IF EXISTS transactiontype")
    op.execute("DROP TYPE IF EXISTS podstatus")
    op.execute("DROP TYPE IF EXISTS providertype")
