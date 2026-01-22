"""
GPU Sourcing Service.

Discovers available GPU offers from providers based on configured criteria.
"""

import logging
from typing import Optional

from brain.adapters.base import OfferFilters
from brain.adapters.vast_ai import VastClient, VastGpuOffer
from brain.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SourcingService:
    """
    Service for discovering GPU offers from providers.

    Searches providers for offers matching configured criteria
    (GPU type, price, reliability, etc.).
    """

    def __init__(self, vast_client: Optional[VastClient] = None):
        """
        Initialize sourcing service.

        Args:
            vast_client: Optional VastClient instance. If not provided,
                        creates one based on settings.
        """
        self._vast_client = vast_client

    @property
    def vast_client(self) -> VastClient:
        """Lazily initialize Vast.ai client."""
        if self._vast_client is None:
            self._vast_client = VastClient(
                api_key=settings.vast_ai_api_key,
                use_mock=settings.use_mock_providers,
            )
        return self._vast_client

    async def find_offers(
        self,
        gpu_type: str,
        max_price: Optional[float] = None,
        min_vram_mb: Optional[int] = None,
        min_reliability: Optional[float] = None,
    ) -> list[VastGpuOffer]:
        """
        Find offers for a specific GPU type.

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

        logger.info(f"Searching for {gpu_type} offers (max ${max_price}/hr)")

        raw_offers = await self.vast_client.list_offers_raw(filters)

        logger.info(f"Found {len(raw_offers)} matching offers for {gpu_type}")
        return raw_offers

    async def find_best_offer(
        self,
        gpu_type: str,
        max_price: Optional[float] = None,
    ) -> Optional[VastGpuOffer]:
        """
        Find the best offer for a GPU type (lowest price meeting criteria).

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

        # Already sorted by price, return the cheapest
        best = offers[0]
        logger.info(
            f"Best offer for {gpu_type}: "
            f"${best.dph_total}/hr, {best.gpu_ram}MB VRAM, "
            f"{best.reliability2:.0%} reliability"
        )
        return best

    async def find_offers_for_all_targets(
        self,
    ) -> dict[str, list[VastGpuOffer]]:
        """
        Find offers for all configured target GPU types.

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
            f"{len(settings.sourcing_target_gpu_types)} GPU types"
        )

        return results

    async def get_cheapest_per_type(
        self,
    ) -> dict[str, Optional[VastGpuOffer]]:
        """
        Get the cheapest offer for each target GPU type.

        Returns:
            Dict mapping GPU type to best offer (or None)
        """
        all_offers = await self.find_offers_for_all_targets()

        return {
            gpu_type: offers[0] if offers else None
            for gpu_type, offers in all_offers.items()
        }

    async def close(self) -> None:
        """Clean up resources."""
        if self._vast_client:
            await self._vast_client.close()


# Global singleton instance
_sourcing_service: Optional[SourcingService] = None


def get_sourcing_service() -> SourcingService:
    """Get the global SourcingService instance."""
    global _sourcing_service
    if _sourcing_service is None:
        _sourcing_service = SourcingService()
    return _sourcing_service
