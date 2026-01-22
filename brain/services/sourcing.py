"""
GPU Sourcing Service.

Discovers available GPU offers from all providers based on configured criteria.
Uses the MultiProviderService for aggregated, cross-provider searching.
"""

import logging
from typing import Optional

from brain.adapters.base import GpuOffer, OfferFilters
from brain.adapters.vast_ai import VastClient, VastGpuOffer
from brain.config import get_settings
from brain.services.multi_provider import (
    AggregatedOffer,
    MultiProviderService,
    PricingStrategy,
    get_multi_provider_service,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class SourcingService:
    """
    Service for discovering GPU offers from all providers.

    Searches all configured providers for offers matching criteria
    (GPU type, price, reliability, etc.) and returns the best options.
    """

    def __init__(
        self,
        vast_client: Optional[VastClient] = None,
        multi_provider: Optional[MultiProviderService] = None,
    ):
        """
        Initialize sourcing service.

        Args:
            vast_client: Optional VastClient instance (for backward compatibility)
            multi_provider: Optional MultiProviderService for cross-provider search
        """
        self._vast_client = vast_client
        self._multi_provider = multi_provider

    @property
    def vast_client(self) -> VastClient:
        """Lazily initialize Vast.ai client."""
        if self._vast_client is None:
            self._vast_client = VastClient(
                api_key=settings.vast_ai_api_key,
                use_mock=settings.use_mock_providers,
            )
        return self._vast_client

    @property
    def multi_provider(self) -> MultiProviderService:
        """Lazily initialize multi-provider service."""
        if self._multi_provider is None:
            self._multi_provider = get_multi_provider_service()
        return self._multi_provider

    async def find_offers(
        self,
        gpu_type: str,
        max_price: Optional[float] = None,
        min_vram_mb: Optional[int] = None,
        min_reliability: Optional[float] = None,
        providers: Optional[list[str]] = None,
    ) -> list[AggregatedOffer]:
        """
        Find offers for a specific GPU type across all providers.

        Args:
            gpu_type: GPU model name (e.g., "RTX 4090")
            max_price: Maximum hourly price
            min_vram_mb: Minimum VRAM requirement
            min_reliability: Minimum reliability score
            providers: Specific providers to search (None = all)

        Returns:
            List of matching AggregatedOffer sorted by value
        """
        effective_max_price = max_price or settings.sourcing_max_price_per_hour
        effective_min_vram = min_vram_mb or settings.sourcing_min_vram_mb

        logger.info(
            f"Searching for {gpu_type} offers across all providers "
            f"(max ${effective_max_price}/hr, min {effective_min_vram}MB VRAM)"
        )

        offers = await self.multi_provider.find_best_offers(
            gpu_type=gpu_type,
            max_price=effective_max_price,
            min_vram_mb=effective_min_vram,
            strategy=PricingStrategy.BALANCED,
            limit=50,
        )

        logger.info(f"Found {len(offers)} matching offers for {gpu_type}")
        return offers

    async def find_offers_raw(
        self,
        gpu_type: str,
        max_price: Optional[float] = None,
        min_vram_mb: Optional[int] = None,
        min_reliability: Optional[float] = None,
    ) -> list[VastGpuOffer]:
        """
        Find raw Vast.ai offers for backward compatibility.

        Args:
            gpu_type: GPU model name (e.g., "RTX 4090")
            max_price: Maximum hourly price
            min_vram_mb: Minimum VRAM requirement
            min_reliability: Minimum reliability score

        Returns:
            List of matching VastGpuOffer sorted by price
        """
        filters = OfferFilters(
            gpu_name=gpu_type,
            max_hourly_price=max_price or settings.sourcing_max_price_per_hour,
            min_vram_mb=min_vram_mb or settings.sourcing_min_vram_mb,
            min_reliability=min_reliability or settings.sourcing_min_reliability,
        )

        return await self.vast_client.list_offers_raw(filters)

    async def find_best_offer(
        self,
        gpu_type: str,
        max_price: Optional[float] = None,
    ) -> Optional[AggregatedOffer]:
        """
        Find the best offer for a GPU type across all providers.

        Args:
            gpu_type: GPU model name
            max_price: Maximum hourly price

        Returns:
            Best matching offer or None
        """
        offers = await self.find_offers(gpu_type, max_price)

        if not offers:
            logger.info(f"No offers found for {gpu_type}")
            return None

        # Already sorted by value, return the best
        best = offers[0]
        logger.info(
            f"Best offer for {gpu_type}: "
            f"{best.offer.provider} @ ${best.offer.hourly_price}/hr, "
            f"{best.offer.gpu_vram_mb}MB VRAM, "
            f"value score: {best.value_score:.3f}"
        )
        return best

    async def find_offers_for_all_targets(
        self,
    ) -> dict[str, list[AggregatedOffer]]:
        """
        Find offers for all configured target GPU types across all providers.

        Returns:
            Dict mapping GPU type to list of offers
        """
        results = {}

        for gpu_type in settings.sourcing_target_gpu_types:
            try:
                offers = await self.find_offers(
                    gpu_type,
                    max_price=settings.sourcing_max_price_per_hour,
                    min_vram_mb=settings.sourcing_min_vram_mb,
                    min_reliability=settings.sourcing_min_reliability,
                )
                results[gpu_type] = offers
            except Exception as e:
                logger.error(f"Error searching for {gpu_type}: {e}")
                results[gpu_type] = []

        total_offers = sum(len(offers) for offers in results.values())
        logger.info(
            f"Sourcing complete: {total_offers} total offers across "
            f"{len(settings.sourcing_target_gpu_types)} GPU types from all providers"
        )

        return results

    async def get_cheapest_per_type(
        self,
    ) -> dict[str, Optional[AggregatedOffer]]:
        """
        Get the cheapest offer for each target GPU type across all providers.

        Returns:
            Dict mapping GPU type to best offer (or None)
        """
        all_offers = await self.find_offers_for_all_targets()

        return {
            gpu_type: offers[0] if offers else None
            for gpu_type, offers in all_offers.items()
        }

    async def get_provider_comparison(
        self,
        gpu_type: str,
    ) -> dict[str, list[GpuOffer]]:
        """
        Get offers for a GPU type grouped by provider for comparison.

        Args:
            gpu_type: GPU model name

        Returns:
            Dict mapping provider name to list of offers
        """
        filters = OfferFilters(gpu_name=gpu_type)
        all_offers = await self.multi_provider.list_all_offers(filters)

        by_provider: dict[str, list[GpuOffer]] = {}
        for offer in all_offers:
            if offer.provider not in by_provider:
                by_provider[offer.provider] = []
            by_provider[offer.provider].append(offer)

        # Sort each provider's offers by price
        for provider in by_provider:
            by_provider[provider].sort(key=lambda o: o.hourly_price)

        return by_provider

    def get_provider_health(self) -> dict:
        """Get health status for all providers."""
        return {
            name: {
                "is_healthy": health.is_healthy,
                "consecutive_failures": health.consecutive_failures,
                "avg_response_time_ms": health.avg_response_time_ms,
                "error_message": health.error_message,
            }
            for name, health in self.multi_provider.get_provider_health().items()
        }

    async def close(self) -> None:
        """Clean up resources."""
        if self._vast_client:
            await self._vast_client.close()
        if self._multi_provider:
            await self._multi_provider.close()


# Global singleton instance
_sourcing_service: Optional[SourcingService] = None


def get_sourcing_service() -> SourcingService:
    """Get the global SourcingService instance."""
    global _sourcing_service
    if _sourcing_service is None:
        _sourcing_service = SourcingService()
    return _sourcing_service
