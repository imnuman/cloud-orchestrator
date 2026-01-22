"""
Configuration management for the Brain service.
Uses Pydantic Settings for environment variable parsing.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "GPU Cloud Orchestrator"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"

    # API
    api_prefix: str = "/api/v1"
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/gpu_orchestrator"
    database_pool_size: int = 20
    database_max_overflow: int = 10

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # JWT Authentication
    jwt_secret_key: str = "change-me-in-production-use-strong-secret"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60 * 24  # 24 hours

    # Agent Authentication
    agent_api_key_prefix: str = "gpu_agent_"

    # Billing
    billing_interval_seconds: int = 60  # How often to run billing task
    minimum_balance_for_pod: float = 1.0  # Minimum balance to start a pod
    low_balance_warning_threshold: float = 5.0  # Warn user when balance below this

    # Node Management
    heartbeat_timeout_seconds: int = 90  # Mark node offline after missing heartbeats
    heartbeat_interval_seconds: int = 30  # Expected heartbeat interval

    # Provider API Keys (Phase 1: Arbitrage)
    runpod_api_key: Optional[str] = None
    lambda_labs_api_key: Optional[str] = None
    vast_ai_api_key: Optional[str] = None

    # Pricing Markup (Phase 1: Arbitrage)
    default_markup_percent: float = 25.0  # 25% markup on provider prices

    # Sourcing Configuration
    sourcing_enabled: bool = False  # Must explicitly enable
    sourcing_interval_seconds: int = 300  # 5 minutes
    sourcing_max_price_per_hour: float = 0.50
    sourcing_target_gpu_types: list[str] = ["RTX 4090", "RTX 3090"]
    sourcing_min_vram_mb: int = 24000
    sourcing_min_reliability: float = 0.95
    sourcing_max_instances: int = 5  # Hard limit on auto-provisioned nodes

    # Provisioning Configuration
    auto_provisioning_enabled: bool = False  # Search only by default
    brain_public_url: str = "http://localhost:8000"  # URL for agents to reach brain
    provisioning_timeout_seconds: int = 600  # 10 minutes
    provisioning_check_interval_seconds: int = 30

    # Mock Mode
    use_mock_providers: bool = True  # No real API calls without changing

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
