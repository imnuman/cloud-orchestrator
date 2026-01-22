"""
Tests for Lambda Labs adapter.
"""

import pytest

from brain.adapters.lambda_labs import LambdaClient, create_lambda_client
from brain.adapters.base import OfferFilters, InstanceConfig


class TestLambdaLabsClient:
    """Test suite for LambdaClient."""

    @pytest.fixture
    def mock_client(self) -> LambdaClient:
        """Create a mock Lambda Labs client."""
        return LambdaClient(api_key=None, use_mock=True)

    @pytest.mark.asyncio
    async def test_client_initializes_in_mock_mode(self, mock_client: LambdaClient):
        """Test that client initializes in mock mode without API key."""
        assert mock_client.is_mock is True
        assert mock_client.provider_name == "lambda_labs"

    @pytest.mark.asyncio
    async def test_list_offers_returns_offers(self, mock_client: LambdaClient):
        """Test that list_offers returns mock GPU offers."""
        offers = await mock_client.list_offers()

        assert len(offers) > 0
        assert all(o.provider == "lambda_labs" for o in offers)
        assert all(o.hourly_price > 0 for o in offers)

    @pytest.mark.asyncio
    async def test_list_offers_with_gpu_filter(self, mock_client: LambdaClient):
        """Test filtering offers by GPU type."""
        filters = OfferFilters(gpu_name="A100")
        offers = await mock_client.list_offers(filters)

        assert len(offers) > 0
        assert all("A100" in o.gpu_name.upper() for o in offers)

    @pytest.mark.asyncio
    async def test_list_offers_with_price_filter(self, mock_client: LambdaClient):
        """Test filtering offers by max price."""
        filters = OfferFilters(max_hourly_price=1.50)
        offers = await mock_client.list_offers(filters)

        # All offers should be under max price
        assert all(o.hourly_price <= 1.50 for o in offers)

    @pytest.mark.asyncio
    async def test_list_offers_raw_returns_lambda_offers(self, mock_client: LambdaClient):
        """Test that list_offers_raw returns Lambda Labs specific offers."""
        offers = await mock_client.list_offers_raw()

        assert len(offers) > 0
        # Check for Lambda Labs specific fields
        assert all(hasattr(o, "instance_type") for o in offers)
        assert all(hasattr(o, "region") for o in offers)
        assert all(hasattr(o, "price_cents_per_hour") for o in offers)

    @pytest.mark.asyncio
    async def test_create_instance_returns_instance(self, mock_client: LambdaClient):
        """Test creating a mock instance."""
        config = InstanceConfig(
            docker_image="nvidia/cuda:12.2.0-base-ubuntu22.04",
            disk_gb=20.0,
            label="test-instance",
        )

        # Get an offer first
        offers = await mock_client.list_offers_raw()
        assert len(offers) > 0

        # Lambda Labs offer_id format: instance_type:region
        offer_id = f"{offers[0].instance_type}:{offers[0].region}"
        instance = await mock_client.create_instance(offer_id, config)

        assert instance is not None
        assert instance.instance_id is not None
        assert instance.provider == "lambda_labs"

    @pytest.mark.asyncio
    async def test_get_instance_returns_instance(self, mock_client: LambdaClient):
        """Test getting instance details."""
        config = InstanceConfig(
            docker_image="nvidia/cuda:12.2.0-base-ubuntu22.04",
            disk_gb=20.0,
        )

        # Create instance first
        offers = await mock_client.list_offers_raw()
        offer_id = f"{offers[0].instance_type}:{offers[0].region}"
        instance = await mock_client.create_instance(offer_id, config)

        # Get instance details
        fetched = await mock_client.get_instance(instance.instance_id)

        assert fetched is not None
        assert fetched.instance_id == instance.instance_id

    @pytest.mark.asyncio
    async def test_terminate_instance_removes_instance(self, mock_client: LambdaClient):
        """Test terminating an instance."""
        config = InstanceConfig(
            docker_image="nvidia/cuda:12.2.0-base-ubuntu22.04",
            disk_gb=20.0,
        )

        # Create instance first
        offers = await mock_client.list_offers_raw()
        offer_id = f"{offers[0].instance_type}:{offers[0].region}"
        instance = await mock_client.create_instance(offer_id, config)

        # Terminate
        success = await mock_client.terminate_instance(instance.instance_id)
        assert success is True

        # Verify it's gone
        fetched = await mock_client.get_instance(instance.instance_id)
        assert fetched is None

    @pytest.mark.asyncio
    async def test_create_lambda_client_factory(self):
        """Test the factory function creates a client."""
        client = create_lambda_client(use_mock=True)

        assert client is not None
        assert client.is_mock is True

        await client.close()


class TestLambdaLabsMultiGpu:
    """Test suite for Lambda Labs multi-GPU offerings."""

    @pytest.fixture
    def mock_client(self) -> LambdaClient:
        """Create a mock Lambda Labs client."""
        return LambdaClient(api_key=None, use_mock=True)

    @pytest.mark.asyncio
    async def test_offers_include_multi_gpu_configs(self, mock_client: LambdaClient):
        """Test that offers include multi-GPU configurations."""
        offers = await mock_client.list_offers()

        # Should have some multi-GPU offers
        multi_gpu = [o for o in offers if o.gpu_count > 1]
        assert len(multi_gpu) > 0

    @pytest.mark.asyncio
    async def test_multi_gpu_filter(self, mock_client: LambdaClient):
        """Test filtering for multi-GPU offers."""
        filters = OfferFilters(min_gpu_count=4)
        offers = await mock_client.list_offers(filters)

        # All offers should have at least 4 GPUs
        assert all(o.gpu_count >= 4 for o in offers)

    @pytest.mark.asyncio
    async def test_offers_span_multiple_regions(self, mock_client: LambdaClient):
        """Test that offers are available in multiple regions."""
        offers = await mock_client.list_offers_raw()
        regions = set(o.region for o in offers)

        # Should have multiple regions
        assert len(regions) > 1
