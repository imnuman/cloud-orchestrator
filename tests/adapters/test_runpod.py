"""
Tests for RunPod adapter.
"""

import pytest

from brain.adapters.runpod import RunPodClient, create_runpod_client
from brain.adapters.base import OfferFilters, InstanceConfig


class TestRunPodClient:
    """Test suite for RunPodClient."""

    @pytest.fixture
    def mock_client(self) -> RunPodClient:
        """Create a mock RunPod client."""
        return RunPodClient(api_key=None, use_mock=True)

    @pytest.mark.asyncio
    async def test_client_initializes_in_mock_mode(self, mock_client: RunPodClient):
        """Test that client initializes in mock mode without API key."""
        assert mock_client.is_mock is True
        assert mock_client.provider_name == "runpod"

    @pytest.mark.asyncio
    async def test_list_offers_returns_offers(self, mock_client: RunPodClient):
        """Test that list_offers returns mock GPU offers."""
        offers = await mock_client.list_offers()

        assert len(offers) > 0
        assert all(o.provider == "runpod" for o in offers)
        assert all(o.hourly_price > 0 for o in offers)

    @pytest.mark.asyncio
    async def test_list_offers_with_gpu_filter(self, mock_client: RunPodClient):
        """Test filtering offers by GPU type."""
        filters = OfferFilters(gpu_name="RTX 4090")
        offers = await mock_client.list_offers(filters)

        assert len(offers) > 0
        assert all("4090" in o.gpu_name.upper() for o in offers)

    @pytest.mark.asyncio
    async def test_list_offers_with_price_filter(self, mock_client: RunPodClient):
        """Test filtering offers by max price."""
        filters = OfferFilters(max_hourly_price=0.50)
        offers = await mock_client.list_offers(filters)

        # All offers should be under max price
        assert all(o.hourly_price <= 0.50 for o in offers)

    @pytest.mark.asyncio
    async def test_list_offers_raw_returns_runpod_offers(self, mock_client: RunPodClient):
        """Test that list_offers_raw returns RunPod specific offers."""
        offers = await mock_client.list_offers_raw()

        assert len(offers) > 0
        # Check for RunPod specific fields
        assert all(hasattr(o, "secure_price") for o in offers)
        assert all(hasattr(o, "community_price") for o in offers)

    @pytest.mark.asyncio
    async def test_create_instance_returns_instance(self, mock_client: RunPodClient):
        """Test creating a mock instance."""
        config = InstanceConfig(
            docker_image="nvidia/cuda:12.2.0-base-ubuntu22.04",
            disk_gb=20.0,
            label="test-instance",
        )

        # Get an offer first
        offers = await mock_client.list_offers_raw()
        assert len(offers) > 0

        instance = await mock_client.create_instance(offers[0].id, config)

        assert instance is not None
        assert instance.instance_id is not None
        assert instance.provider == "runpod"

    @pytest.mark.asyncio
    async def test_get_instance_returns_instance(self, mock_client: RunPodClient):
        """Test getting instance details."""
        config = InstanceConfig(
            docker_image="nvidia/cuda:12.2.0-base-ubuntu22.04",
            disk_gb=20.0,
        )

        # Create instance first
        offers = await mock_client.list_offers_raw()
        instance = await mock_client.create_instance(offers[0].id, config)

        # Get instance details
        fetched = await mock_client.get_instance(instance.instance_id)

        assert fetched is not None
        assert fetched.instance_id == instance.instance_id

    @pytest.mark.asyncio
    async def test_terminate_instance_removes_instance(self, mock_client: RunPodClient):
        """Test terminating an instance."""
        config = InstanceConfig(
            docker_image="nvidia/cuda:12.2.0-base-ubuntu22.04",
            disk_gb=20.0,
        )

        # Create instance first
        offers = await mock_client.list_offers_raw()
        instance = await mock_client.create_instance(offers[0].id, config)

        # Terminate
        success = await mock_client.terminate_instance(instance.instance_id)
        assert success is True

        # Verify it's gone
        fetched = await mock_client.get_instance(instance.instance_id)
        assert fetched is None

    @pytest.mark.asyncio
    async def test_create_runpod_client_factory(self):
        """Test the factory function creates a client."""
        client = create_runpod_client(use_mock=True)

        assert client is not None
        assert client.is_mock is True

        await client.close()


class TestRunPodOfferVariety:
    """Test suite for RunPod offer variety."""

    @pytest.fixture
    def mock_client(self) -> RunPodClient:
        """Create a mock RunPod client."""
        return RunPodClient(api_key=None, use_mock=True)

    @pytest.mark.asyncio
    async def test_offers_include_consumer_gpus(self, mock_client: RunPodClient):
        """Test that offers include consumer GPUs like RTX 4090."""
        offers = await mock_client.list_offers()
        gpu_names = [o.gpu_name for o in offers]

        # Should have some consumer GPUs
        consumer_gpus = ["RTX 4090", "RTX 3090"]
        found_consumer = any(
            any(cg in name for cg in consumer_gpus)
            for name in gpu_names
        )
        assert found_consumer

    @pytest.mark.asyncio
    async def test_offers_include_datacenter_gpus(self, mock_client: RunPodClient):
        """Test that offers include datacenter GPUs like A100, H100."""
        offers = await mock_client.list_offers()
        gpu_names = [o.gpu_name for o in offers]

        # Should have some datacenter GPUs
        dc_gpus = ["A100", "H100", "L40"]
        found_dc = any(
            any(dg in name for dg in dc_gpus)
            for name in gpu_names
        )
        assert found_dc

    @pytest.mark.asyncio
    async def test_offers_have_varied_pricing(self, mock_client: RunPodClient):
        """Test that offers have varied pricing (not all the same)."""
        offers = await mock_client.list_offers()
        prices = [o.hourly_price for o in offers]

        # Should have price variety
        unique_prices = set(prices)
        assert len(unique_prices) > 3
