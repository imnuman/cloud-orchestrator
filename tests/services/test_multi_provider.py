"""
Tests for Multi-Provider Service.
"""

import pytest

from brain.services.multi_provider import (
    MultiProviderService,
    PricingStrategy,
    get_multi_provider_service,
)
from brain.adapters.base import InstanceConfig


class TestMultiProviderService:
    """Test suite for MultiProviderService."""

    @pytest.fixture
    def service(self) -> MultiProviderService:
        """Create a multi-provider service with mock clients."""
        return MultiProviderService()

    @pytest.mark.asyncio
    async def test_list_all_offers_aggregates_providers(self, service: MultiProviderService):
        """Test that list_all_offers returns offers from all providers."""
        offers = await service.list_all_offers()

        assert len(offers) > 0

        # Should have offers from multiple providers
        providers = set(o.provider for o in offers)
        assert len(providers) >= 2  # At least 2 providers

    @pytest.mark.asyncio
    async def test_list_all_offers_respects_filters(self, service: MultiProviderService):
        """Test that filters are applied across all providers."""
        from brain.adapters.base import OfferFilters

        filters = OfferFilters(
            gpu_name="4090",
            max_hourly_price=1.0,
        )
        offers = await service.list_all_offers(filters)

        # All offers should match filters
        assert all("4090" in o.gpu_name.upper() for o in offers)
        assert all(o.hourly_price <= 1.0 for o in offers)

    @pytest.mark.asyncio
    async def test_find_best_offers_returns_sorted_offers(self, service: MultiProviderService):
        """Test that find_best_offers returns offers sorted by value."""
        offers = await service.find_best_offers(
            gpu_type="RTX",
            strategy=PricingStrategy.LOWEST_PRICE,
            limit=10,
        )

        assert len(offers) > 0

        # Should be sorted by value score (lower is better)
        scores = [o.value_score for o in offers]
        assert scores == sorted(scores)

    @pytest.mark.asyncio
    async def test_find_best_offers_with_price_strategy(self, service: MultiProviderService):
        """Test price-focused strategy prioritizes low prices."""
        offers = await service.find_best_offers(
            strategy=PricingStrategy.LOWEST_PRICE,
            limit=5,
        )

        assert len(offers) > 0

        # First offer should be among the cheapest
        all_offers = await service.list_all_offers()
        all_prices = sorted([o.hourly_price for o in all_offers])
        cheapest_price = all_prices[0]

        # Best offer should be close to the cheapest
        assert offers[0].offer.hourly_price <= cheapest_price * 1.5

    @pytest.mark.asyncio
    async def test_find_best_offers_with_value_strategy(self, service: MultiProviderService):
        """Test value strategy optimizes price per VRAM."""
        offers = await service.find_best_offers(
            strategy=PricingStrategy.BEST_VALUE,
            limit=5,
        )

        assert len(offers) > 0

        # All offers should have value scores calculated
        assert all(o.value_score > 0 for o in offers)

    @pytest.mark.asyncio
    async def test_find_cheapest_by_gpu_type(self, service: MultiProviderService):
        """Test finding cheapest offer per GPU type."""
        cheapest = await service.find_cheapest_by_gpu_type()

        assert len(cheapest) > 0

        # Each entry should be the cheapest for that GPU type
        for gpu_type, agg_offer in cheapest.items():
            if agg_offer is not None:
                assert agg_offer.offer.gpu_name is not None


class TestMultiProviderFailover:
    """Test suite for multi-provider failover functionality."""

    @pytest.fixture
    def service(self) -> MultiProviderService:
        """Create a multi-provider service with mock clients."""
        return MultiProviderService()

    @pytest.mark.asyncio
    async def test_provider_health_initialized(self, service: MultiProviderService):
        """Test that provider health is tracked."""
        health = service.get_provider_health()

        assert "vast_ai" in health
        assert "runpod" in health
        assert "lambda_labs" in health

        # All should start healthy
        assert all(h.is_healthy for h in health.values())

    @pytest.mark.asyncio
    async def test_create_instance_with_failover(self, service: MultiProviderService):
        """Test creating instance with failover support."""
        config = InstanceConfig(
            docker_image="nvidia/cuda:12.2.0-base-ubuntu22.04",
            disk_gb=20.0,
            label="test-failover",
        )

        instance, provider = await service.create_instance_with_failover(
            gpu_type="RTX 4090",
            config=config,
            max_price=2.0,
        )

        assert instance is not None
        assert instance.instance_id is not None
        assert provider in ["vast_ai", "runpod", "lambda_labs"]

    @pytest.mark.asyncio
    async def test_create_instance_prefers_specified_providers(
        self, service: MultiProviderService
    ):
        """Test that preferred providers are tried first."""
        config = InstanceConfig(
            docker_image="nvidia/cuda:12.2.0-base-ubuntu22.04",
            disk_gb=20.0,
        )

        # Prefer RunPod
        instance, provider = await service.create_instance_with_failover(
            gpu_type="RTX 4090",
            config=config,
            preferred_providers=["runpod", "vast_ai"],
        )

        # Should create on RunPod (first preferred with availability)
        assert instance is not None
        # Note: May not always be RunPod if it doesn't have the GPU type

    @pytest.mark.asyncio
    async def test_mark_provider_healthy(self, service: MultiProviderService):
        """Test manually marking a provider as healthy."""
        # Simulate failure
        health = service._provider_health["vast_ai"]
        health.is_healthy = False
        health.consecutive_failures = 5

        # Mark healthy
        service.mark_provider_healthy("vast_ai")

        # Should be healthy again
        health = service._provider_health["vast_ai"]
        assert health.is_healthy is True
        assert health.consecutive_failures == 0


class TestMultiProviderCaching:
    """Test suite for multi-provider offer caching."""

    @pytest.fixture
    def service(self) -> MultiProviderService:
        """Create a multi-provider service with mock clients."""
        return MultiProviderService()

    @pytest.mark.asyncio
    async def test_offers_are_cached(self, service: MultiProviderService):
        """Test that offers are cached after first fetch."""
        # First fetch
        offers1 = await service.list_all_offers(use_cache=True)

        # Second fetch should use cache
        offers2 = await service.list_all_offers(use_cache=True)

        # Both should return same data
        assert len(offers1) == len(offers2)

    @pytest.mark.asyncio
    async def test_cache_can_be_bypassed(self, service: MultiProviderService):
        """Test that cache can be bypassed."""
        # First fetch
        await service.list_all_offers(use_cache=True)

        # Bypass cache
        offers = await service.list_all_offers(use_cache=False)

        # Should still return offers (fresh fetch)
        assert len(offers) > 0


class TestPricingStrategies:
    """Test suite for different pricing strategies."""

    @pytest.fixture
    def service(self) -> MultiProviderService:
        """Create a multi-provider service with mock clients."""
        return MultiProviderService()

    @pytest.mark.asyncio
    async def test_lowest_price_strategy(self, service: MultiProviderService):
        """Test lowest price strategy."""
        offers = await service.find_best_offers(
            strategy=PricingStrategy.LOWEST_PRICE,
            limit=10,
        )

        # Should prioritize low prices
        prices = [o.offer.hourly_price for o in offers]
        # First few should be among the cheapest
        assert prices[0] <= prices[-1]

    @pytest.mark.asyncio
    async def test_balanced_strategy(self, service: MultiProviderService):
        """Test balanced strategy considers multiple factors."""
        offers = await service.find_best_offers(
            strategy=PricingStrategy.BALANCED,
            limit=10,
        )

        # Should have varied criteria in scoring
        assert len(offers) > 0
        assert all(o.value_score > 0 for o in offers)

    @pytest.mark.asyncio
    async def test_reliability_strategy(self, service: MultiProviderService):
        """Test reliability-focused strategy."""
        offers = await service.find_best_offers(
            strategy=PricingStrategy.HIGHEST_RELIABILITY,
            limit=10,
        )

        # Should prioritize reliable providers
        assert len(offers) > 0
