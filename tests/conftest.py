"""
Pytest configuration and fixtures for GPU Cloud Orchestrator tests.
"""

import asyncio
import os
from typing import AsyncGenerator, Generator
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# Set test environment variables before importing app
os.environ["USE_MOCK_PROVIDERS"] = "true"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres@localhost:5432/gpu_orchestrator_test"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only"

from brain.main import app
from brain.models.base import Base, get_async_session
from brain.config import get_settings


# Test database URL
TEST_DATABASE_URL = os.environ["DATABASE_URL"]

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    poolclass=NullPool,
    echo=False,
)

# Test session factory
TestSessionLocal = sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for session-scoped async fixtures."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a clean database session for each test.

    Creates all tables before the test and drops them after.
    """
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        yield session
        await session.rollback()

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    Provide an async HTTP client for testing API endpoints.
    """
    # Override the database session dependency
    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_async_session] = override_get_session

    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def test_user_data() -> dict:
    """Sample user registration data."""
    return {
        "email": f"test_{uuid4().hex[:8]}@example.com",
        "password": "TestPassword123!",
    }


@pytest.fixture
def test_gpu_offer() -> dict:
    """Sample GPU offer data."""
    return {
        "offer_id": "test-offer-123",
        "provider": "vast_ai",
        "gpu_name": "RTX 4090",
        "gpu_count": 1,
        "gpu_vram_mb": 24576,
        "total_vram_mb": 24576,
        "cpu_cores": 16,
        "ram_mb": 65536,
        "disk_gb": 100.0,
        "hourly_price": 0.45,
        "reliability_score": 0.98,
        "location": "US-TX",
    }


@pytest.fixture
def test_pod_request() -> dict:
    """Sample pod creation request."""
    return {
        "gpu_type": "RTX 4090",
        "gpu_count": 1,
        "docker_image": "nvidia/cuda:12.2.0-base-ubuntu22.04",
        "port_mappings": [],
        "environment_variables": {},
    }
