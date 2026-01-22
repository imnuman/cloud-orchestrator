"""
Multi-Provider GPU Service.

Aggregates GPU offers from multiple providers, implements price optimization,
and provides automatic failover capabilities.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from brain.adapters.base import (
    BaseProviderAdapter,
    GpuOffer,
    Instance,
    InstanceConfig,
    InstanceStatus,
    OfferFilters,
)
from brain.adapters.lambda_labs import LambdaClient, create_lambda_client
from brain.adapters.runpod import RunPodClient, create_runpod_client
from brain.adapters.vast_ai import VastClient
from brain.adapters.vast_ai.client import create_vast_client
from brain.config import get_settings
from shared.schemas import ProviderType

logger = logging.getLogger(__name__)
settings = get_settings()


class PricingStrategy(str, Enum):
    """Strategy for selecting GPU offers."""

    LOWEST_PRICE = "lowest_price"
    BEST_VALUE = "best_value"  # Price per GB VRAM
    HIGHEST_RELIABILITY = "highest_reliability"
    BALANCED = "balanced"  # Mix of price and reliability


@dataclass
class ProviderHealth:
    """Health status for a provider."""

    provider: str
    is_healthy: bool
    last_check: datetime
    consecutive_failures: int
    avg_response_time_ms: float
    error_message: Optional[str] = None


@dataclass
class AggregatedOffer:
    """GPU offer with additional metadata for comparison."""

    offer: GpuOffer
    provider_health: ProviderHealth
    value_score: float  # Lower is better
    normalized_price: float  # Price per GPU per hour

    def __lt__(self, other: "AggregatedOffer") -> bool:
        return self.value_score < other.value_score


class MultiProviderService:
    """
    Service that aggregates GPU capacity from multiple providers.

    Features:
    - Multi-provider offer aggregation
    - Price optimization with configurable strategies
    - Automatic failover on provider issues
    - Health monitoring for each provider
    - Caching for offer data
    """

    def __init__(
        self,
        vast_client: Optional[VastClient] = None,
        runpod_client: Optional[RunPodClient] = None,
        lambda_client: Optional[LambdaClient] = None,
    ):
        """
        Initialize multi-provider service.

        Args:
            vast_client: Optional VastClient instance
            runpod_client: Optional RunPodClient instance
            lambda_client: Optional LambdaClient instance
        """
        self._vast_client = vast_client
        self._runpod_client = runpod_client
        self._lambda_client = lambda_client

        # Provider health tracking
        self._provider_health: dict[str, ProviderHealth] = {}

        # Offer cache
        self._offer_cache: dict[str, list[GpuOffer]] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 60  # Cache offers for 60 seconds

        # Initialize health status for all providers
        for provider in ["vast_ai", "runpod", "lambda_labs"]:
            self._provider_health[provider] = ProviderHealth(
                provider=provider,
                is_healthy=True,
                last_check=datetime.utcnow(),
                consecutive_failures=0,
                avg_response_time_ms=0.0,
            )

    @property
    def vast_client(self) -> VastClient:
        """Lazily initialize Vast.ai client."""
        if self._vast_client is None:
            self._vast_client = create_vast_client(
                api_key=settings.vast_ai_api_key,
                use_mock=settings.use_mock_providers,
            )
        return self._vast_client

    @property
    def runpod_client(self) -> RunPodClient:
        """Lazily initialize RunPod client."""
        if self._runpod_client is None:
            self._runpod_client = create_runpod_client(
                api_key=settings.runpod_api_key,
                use_mock=settings.use_mock_providers,
            )
        return self._runpod_client

    @property
    def lambda_client(self) -> LambdaClient:
        """Lazily initialize Lambda Labs client."""
        if self._lambda_client is None:
            self._lambda_client = create_lambda_client(
                api_key=settings.lambda_labs_api_key,
                use_mock=settings.use_mock_providers,
            )
        return self._lambda_client

    def _get_all_clients(self) -> dict[str, BaseProviderAdapter]:
        """Get all provider clients."""
        return {
            "vast_ai": self.vast_client,
            "runpod": self.runpod_client,
            "lambda_labs": self.lambda_client,
        }

    def _get_client_for_provider(self, provider: str) -> Optional[BaseProviderAdapter]:
        """Get client for a specific provider."""
        clients = self._get_all_clients()
        return clients.get(provider)

    async def _fetch_offers_from_provider(
        self,
        provider: str,
        client: BaseProviderAdapter,
        filters: Optional[OfferFilters] = None,
    ) -> list[GpuOffer]:
        """
        Fetch offers from a single provider with health tracking.

        Args:
            provider: Provider name
            client: Provider client
            filters: Optional offer filters

        Returns:
            List of offers from the provider
        """
        health = self._provider_health[provider]
        start_time = datetime.utcnow()

        try:
            offers = await client.list_offers(filters)

            # Update health on success
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            health.is_healthy = True
            health.consecutive_failures = 0
            health.last_check = datetime.utcnow()
            health.avg_response_time_ms = (
                health.avg_response_time_ms * 0.7 + response_time * 0.3
            )
            health.error_message = None

            logger.debug(
                f"Fetched {len(offers)} offers from {provider} in {response_time:.0f}ms"
            )
            return offers

        except Exception as e:
            # Update health on failure
            health.consecutive_failures += 1
            health.last_check = datetime.utcnow()
            health.error_message = str(e)

            # Mark unhealthy after 3 consecutive failures
            if health.consecutive_failures >= 3:
                health.is_healthy = False

            logger.warning(
                f"Failed to fetch offers from {provider}: {e} "
                f"(failures: {health.consecutive_failures})"
            )
            return []

    async def list_all_offers(
        self,
        filters: Optional[OfferFilters] = None,
        providers: Optional[list[str]] = None,
        use_cache: bool = True,
    ) -> list[GpuOffer]:
        """
        List GPU offers from all (or specified) providers.

        Args:
            filters: Optional filters to apply
            providers: List of provider names to query (default: all healthy)
            use_cache: Whether to use cached offers

        Returns:
            Combined list of offers from all providers
        """
        # Check cache
        if use_cache and self._is_cache_valid():
            all_offers = []
            for provider_offers in self._offer_cache.values():
                all_offers.extend(provider_offers)
            return self._apply_filters(all_offers, filters)

        # Determine which providers to query
        if providers is None:
            providers = [
                p for p, h in self._provider_health.items()
                if h.is_healthy
            ]

        # Fetch from all providers in parallel
        clients = self._get_all_clients()
        tasks = []

        for provider in providers:
            if provider in clients:
                task = self._fetch_offers_from_provider(
                    provider, clients[provider], filters
                )
                tasks.append((provider, task))

        # Execute all fetches concurrently
        results = await asyncio.gather(
            *[t[1] for t in tasks],
            return_exceptions=True,
        )

        # Combine results and update cache
        all_offers = []
        for i, (provider, _) in enumerate(tasks):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"Error fetching from {provider}: {result}")
                self._offer_cache[provider] = []
            else:
                self._offer_cache[provider] = result
                all_offers.extend(result)

        self._cache_timestamp = datetime.utcnow()

        return all_offers

    def _is_cache_valid(self) -> bool:
        """Check if the offer cache is still valid."""
        if not self._cache_timestamp or not self._offer_cache:
            return False

        age = (datetime.utcnow() - self._cache_timestamp).total_seconds()
        return age < self._cache_ttl_seconds

    def _apply_filters(
        self, offers: list[GpuOffer], filters: Optional[OfferFilters]
    ) -> list[GpuOffer]:
        """Apply filters to a list of offers."""
        if not filters:
            return offers

        filtered = offers

        if filters.gpu_name:
            filtered = [
                o for o in filtered
                if filters.gpu_name.lower() in o.gpu_name.lower()
            ]
        if filters.min_gpu_count > 1:
            filtered = [o for o in filtered if o.gpu_count >= filters.min_gpu_count]
        if filters.min_vram_mb:
            filtered = [o for o in filtered if o.gpu_vram_mb >= filters.min_vram_mb]
        if filters.max_hourly_price:
            filtered = [o for o in filtered if o.hourly_price <= filters.max_hourly_price]
        if filters.min_reliability > 0:
            filtered = [o for o in filtered if o.reliability_score >= filters.min_reliability]

        return filtered

    def _calculate_value_score(
        self,
        offer: GpuOffer,
        health: ProviderHealth,
        strategy: PricingStrategy,
    ) -> float:
        """
        Calculate a value score for an offer (lower is better).

        Args:
            offer: The GPU offer
            health: Provider health status
            strategy: Pricing strategy to use

        Returns:
            Value score (lower = better value)
        """
        # Base score is price per GPU hour
        price_per_gpu = offer.hourly_price / offer.gpu_count

        # Reliability penalty (unhealthy providers get penalty)
        reliability_factor = 1.0
        if not health.is_healthy:
            reliability_factor = 2.0  # Double effective price for unhealthy
        elif health.consecutive_failures > 0:
            reliability_factor = 1.0 + (health.consecutive_failures * 0.1)

        # VRAM value (price per GB VRAM)
        vram_gb = offer.gpu_vram_mb / 1024
        price_per_vram = offer.hourly_price / (vram_gb * offer.gpu_count)

        if strategy == PricingStrategy.LOWEST_PRICE:
            return price_per_gpu * reliability_factor

        elif strategy == PricingStrategy.BEST_VALUE:
            # Optimize for price per GB VRAM
            return price_per_vram * reliability_factor

        elif strategy == PricingStrategy.HIGHEST_RELIABILITY:
            # Heavily weight reliability
            reliability_score = offer.reliability_score or 0.9
            return price_per_gpu / reliability_score * reliability_factor

        elif strategy == PricingStrategy.BALANCED:
            # Balance price, VRAM value, and reliability
            reliability_score = offer.reliability_score or 0.9
            return (
                price_per_gpu * 0.4 +
                price_per_vram * 0.3 +
                (1 - reliability_score) * 0.3
            ) * reliability_factor

        return price_per_gpu

    async def find_best_offers(
        self,
        gpu_type: Optional[str] = None,
        min_vram_mb: Optional[int] = None,
        max_price: Optional[float] = None,
        min_gpu_count: int = 1,
        strategy: PricingStrategy = PricingStrategy.BALANCED,
        limit: int = 10,
    ) -> list[AggregatedOffer]:
        """
        Find the best GPU offers across all providers.

        Args:
            gpu_type: GPU model name filter
            min_vram_mb: Minimum VRAM requirement
            max_price: Maximum hourly price
            min_gpu_count: Minimum number of GPUs
            strategy: Pricing strategy to use
            limit: Maximum number of offers to return

        Returns:
            List of aggregated offers sorted by value score
        """
        filters = OfferFilters(
            gpu_name=gpu_type,
            min_vram_mb=min_vram_mb,
            max_hourly_price=max_price,
            min_gpu_count=min_gpu_count,
        )

        offers = await self.list_all_offers(filters)

        # Create aggregated offers with scores
        aggregated = []
        for offer in offers:
            health = self._provider_health.get(
                offer.provider,
                ProviderHealth(
                    provider=offer.provider,
                    is_healthy=True,
                    last_check=datetime.utcnow(),
                    consecutive_failures=0,
                    avg_response_time_ms=0.0,
                ),
            )

            score = self._calculate_value_score(offer, health, strategy)
            normalized_price = offer.hourly_price / offer.gpu_count

            aggregated.append(
                AggregatedOffer(
                    offer=offer,
                    provider_health=health,
                    value_score=score,
                    normalized_price=normalized_price,
                )
            )

        # Sort by value score
        aggregated.sort()

        return aggregated[:limit]

    async def find_cheapest_by_gpu_type(
        self,
    ) -> dict[str, Optional[AggregatedOffer]]:
        """
        Find the cheapest offer for each GPU type across all providers.

        Returns:
            Dict mapping GPU type to best offer
        """
        offers = await self.list_all_offers()

        # Group by GPU type
        by_type: dict[str, list[AggregatedOffer]] = {}
        for offer in offers:
            gpu_type = offer.gpu_name

            health = self._provider_health.get(offer.provider)
            if not health:
                continue

            score = self._calculate_value_score(
                offer, health, PricingStrategy.LOWEST_PRICE
            )

            agg = AggregatedOffer(
                offer=offer,
                provider_health=health,
                value_score=score,
                normalized_price=offer.hourly_price / offer.gpu_count,
            )

            if gpu_type not in by_type:
                by_type[gpu_type] = []
            by_type[gpu_type].append(agg)

        # Get cheapest for each type
        result = {}
        for gpu_type, type_offers in by_type.items():
            type_offers.sort()
            result[gpu_type] = type_offers[0] if type_offers else None

        return result

    async def create_instance_with_failover(
        self,
        gpu_type: str,
        config: InstanceConfig,
        max_price: Optional[float] = None,
        preferred_providers: Optional[list[str]] = None,
    ) -> tuple[Instance, str]:
        """
        Create an instance with automatic failover across providers.

        Tries each provider in order of price until one succeeds.

        Args:
            gpu_type: GPU type to provision
            config: Instance configuration
            max_price: Maximum acceptable hourly price
            preferred_providers: Providers to try first (in order)

        Returns:
            Tuple of (created instance, provider name)

        Raises:
            RuntimeError: If all providers fail
        """
        # Get best offers
        offers = await self.find_best_offers(
            gpu_type=gpu_type,
            max_price=max_price,
            strategy=PricingStrategy.BALANCED,
            limit=20,
        )

        if not offers:
            raise RuntimeError(f"No offers found for {gpu_type}")

        # Reorder by preferred providers if specified
        if preferred_providers:
            def sort_key(agg: AggregatedOffer) -> tuple:
                try:
                    pref_idx = preferred_providers.index(agg.offer.provider)
                except ValueError:
                    pref_idx = len(preferred_providers)
                return (pref_idx, agg.value_score)

            offers.sort(key=sort_key)

        # Try each offer until one succeeds
        errors = []
        for agg in offers:
            offer = agg.offer
            provider = offer.provider
            client = self._get_client_for_provider(provider)

            if not client:
                continue

            try:
                logger.info(
                    f"Attempting to create instance on {provider} "
                    f"({offer.gpu_name} @ ${offer.hourly_price}/hr)"
                )

                instance = await client.create_instance(offer.offer_id, config)

                logger.info(f"Successfully created instance on {provider}: {instance.instance_id}")
                return instance, provider

            except Exception as e:
                error_msg = f"{provider}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"Failed to create instance on {provider}: {e}")

                # Update provider health
                health = self._provider_health.get(provider)
                if health:
                    health.consecutive_failures += 1
                    if health.consecutive_failures >= 3:
                        health.is_healthy = False

        # All providers failed
        raise RuntimeError(
            f"Failed to create instance on any provider. Errors:\n"
            + "\n".join(errors)
        )

    async def get_instance_status(
        self,
        instance_id: str,
        provider: str,
    ) -> Optional[Instance]:
        """
        Get instance status from a specific provider.

        Args:
            instance_id: The instance ID
            provider: Provider name

        Returns:
            Instance info or None if not found
        """
        client = self._get_client_for_provider(provider)
        if not client:
            return None

        return await client.get_instance(instance_id)

    async def terminate_instance(
        self,
        instance_id: str,
        provider: str,
    ) -> bool:
        """
        Terminate an instance on a specific provider.

        Args:
            instance_id: The instance ID
            provider: Provider name

        Returns:
            True if termination successful
        """
        client = self._get_client_for_provider(provider)
        if not client:
            return False

        return await client.terminate_instance(instance_id)

    def get_provider_health(self) -> dict[str, ProviderHealth]:
        """Get health status for all providers."""
        return self._provider_health.copy()

    def mark_provider_healthy(self, provider: str) -> None:
        """Manually mark a provider as healthy."""
        if provider in self._provider_health:
            health = self._provider_health[provider]
            health.is_healthy = True
            health.consecutive_failures = 0
            health.error_message = None
            health.last_check = datetime.utcnow()

    async def close(self) -> None:
        """Clean up all provider clients."""
        if self._vast_client:
            await self._vast_client.close()
        if self._runpod_client:
            await self._runpod_client.close()
        if self._lambda_client:
            await self._lambda_client.close()


# Global singleton instance
_multi_provider_service: Optional[MultiProviderService] = None


def get_multi_provider_service() -> MultiProviderService:
    """Get the global MultiProviderService instance."""
    global _multi_provider_service
    if _multi_provider_service is None:
        _multi_provider_service = MultiProviderService()
    return _multi_provider_service
