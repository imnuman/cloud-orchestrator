"""
Tests for Vast.ai adapter.
"""

import pytest

from brain.adapters.vast_ai import VastClient, create_vast_client
from brain.adapters.base import OfferFilters, InstanceConfig


class TestVastAiClient:
    """Test suite for VastClient."""

    @pytest.fixture
    def mock_client(self) -> VastClient:
        """Create a mock Vast.ai client."""
        return VastClient(api_key=None, use_mock=True)

    @pytest.mark.asyncio
    async def test_client_initializes_in_mock_mode(self, mock_client: VastClient):
        """Test that client initializes in mock mode without API key."""
        assert mock_client.is_mock is True
        assert mock_client.provider_name == "vast_ai"

    @pytest.mark.asyncio
    async def test_list_offers_returns_offers(self, mock_client: VastClient):
        """Test that list_offers returns mock GPU offers."""
        offers = await mock_client.list_offers()

        assert len(offers) > 0
        assert all(o.provider == "vast_ai" for o in offers)
        assert all(o.hourly_price > 0 for o in offers)

    @pytest.mark.asyncio
    async def test_list_offers_with_filters(self, mock_client: VastClient):
        """Test filtering offers by GPU type and price."""
        filters = OfferFilters(
            gpu_name="RTX 4090",
            max_hourly_price=1.0,
        )
        offers = await mock_client.list_offers(filters)

        assert len(offers) > 0
        assert all("4090" in o.gpu_name.upper() for o in offers)
        assert all(o.hourly_price <= 1.0 for o in offers)

    @pytest.mark.asyncio
    async def test_list_offers_raw_returns_vast_offers(self, mock_client: VastClient):
        """Test that list_offers_raw returns Vast.ai specific offers."""
        offers = await mock_client.list_offers_raw()

        assert len(offers) > 0
        # Check for Vast.ai specific fields
        assert all(hasattr(o, "dph_total") for o in offers)
        assert all(hasattr(o, "reliability2") for o in offers)

    @pytest.mark.asyncio
    async def test_create_instance_returns_instance(self, mock_client: VastClient):
        """Test creating a mock instance."""
        config = InstanceConfig(
            docker_image="nvidia/cuda:12.2.0-base-ubuntu22.04",
            disk_gb=20.0,
            label="test-instance",
        )

        # Get an offer first
        offers = await mock_client.list_offers_raw()
        assert len(offers) > 0

        instance = await mock_client.create_instance(str(offers[0].id), config)

        assert instance is not None
        assert instance.instance_id is not None
        assert instance.provider == "vast_ai"

    @pytest.mark.asyncio
    async def test_get_instance_returns_instance(self, mock_client: VastClient):
        """Test getting instance details."""
        config = InstanceConfig(
            docker_image="nvidia/cuda:12.2.0-base-ubuntu22.04",
            disk_gb=20.0,
        )

        # Create instance first
        offers = await mock_client.list_offers_raw()
        instance = await mock_client.create_instance(str(offers[0].id), config)

        # Get instance details
        fetched = await mock_client.get_instance(instance.instance_id)

        assert fetched is not None
        assert fetched.instance_id == instance.instance_id

    @pytest.mark.asyncio
    async def test_terminate_instance_removes_instance(self, mock_client: VastClient):
        """Test terminating an instance."""
        config = InstanceConfig(
            docker_image="nvidia/cuda:12.2.0-base-ubuntu22.04",
            disk_gb=20.0,
        )

        # Create instance first
        offers = await mock_client.list_offers_raw()
        instance = await mock_client.create_instance(str(offers[0].id), config)

        # Terminate
        success = await mock_client.terminate_instance(instance.instance_id)
        assert success is True

        # Verify it's gone
        fetched = await mock_client.get_instance(instance.instance_id)
        assert fetched is None

    @pytest.mark.asyncio
    async def test_create_vast_client_factory(self):
        """Test the factory function creates a client."""
        client = create_vast_client(use_mock=True)

        assert client is not None
        assert client.is_mock is True

        await client.close()


class TestVastAiOfferNormalization:
    """Test suite for Vast.ai offer normalization."""

    @pytest.fixture
    def mock_client(self) -> VastClient:
        """Create a mock Vast.ai client."""
        return VastClient(api_key=None, use_mock=True)

    @pytest.mark.asyncio
    async def test_normalized_offer_has_required_fields(self, mock_client: VastClient):
        """Test that normalized offers have all required fields."""
        offers = await mock_client.list_offers()

        for offer in offers:
            assert offer.offer_id is not None
            assert offer.provider == "vast_ai"
            assert offer.gpu_name is not None
            assert offer.gpu_count >= 1
            assert offer.gpu_vram_mb > 0
            assert offer.hourly_price > 0
            assert 0 <= offer.reliability_score <= 1

    @pytest.mark.asyncio
    async def test_offer_sorting_by_price(self, mock_client: VastClient):
        """Test that offers are sorted by price."""
        offers = await mock_client.list_offers()

        # Check offers are sorted by price ascending
        prices = [o.hourly_price for o in offers]
        assert prices == sorted(prices)
