"""
Agent configuration.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class AgentSettings(BaseSettings):
    """Agent settings loaded from environment or config file."""

    # Identity
    node_id: Optional[str] = None
    api_key: Optional[str] = None

    # Brain connection
    brain_url: str = "http://localhost:8000"
    brain_api_prefix: str = "/api/v1"

    # Heartbeat
    heartbeat_interval_seconds: int = 30

    # Docker
    docker_socket: str = "unix:///var/run/docker.sock"
    default_docker_network: str = "gpu-orchestrator"

    # Networking
    ssh_base_port: int = 10000  # Starting port for SSH mappings
    jupyter_base_port: int = 11000  # Starting port for Jupyter mappings

    # Storage
    data_dir: Path = Path("/var/lib/gpu-agent")
    config_file: Path = Path("/etc/gpu-agent/config.json")

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = Path("/var/log/gpu-agent/agent.log")

    # Provider info (for auto-provisioned nodes)
    provider_type: str = "community"  # vast_ai, runpod, lambda_labs, community, internal
    provider_instance_id: Optional[str] = None  # Provider's instance ID
    hourly_price: Optional[float] = None

    class Config:
        env_prefix = "GPU_AGENT_"
        env_file = ".env"


def get_agent_settings() -> AgentSettings:
    """Get agent settings instance."""
    return AgentSettings()
