"""
Vast.ai API client with mock support.
Automatically uses mock mode when no API key is provided.
"""

import logging
from typing import Optional

import httpx

from brain.adapters.base import (
    BaseProviderAdapter,
    GpuOffer,
    Instance,
    InstanceConfig,
    InstanceStatus,
    OfferFilters,
)
from brain.adapters.vast_ai.mock import MockVastClient
from brain.adapters.vast_ai.schemas import (
    VastGpuOffer,
    VastInstance,
    VastInstanceConfig,
    VastInstanceStatus,
    VastOfferFilters,
)


logger = logging.getLogger(__name__)

VAST_API_BASE = "https://console.vast.ai/api/v0"


class VastClient(BaseProviderAdapter):
    """
    Vast.ai API client.

    Automatically falls back to mock mode when no API key is provided,
    making development and testing easier.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_mock: bool = False,
        base_url: str = VAST_API_BASE,
    ):
        """
        Initialize Vast.ai client.

        Args:
            api_key: Vast.ai API key. If None, mock mode is used.
            use_mock: Force mock mode even with API key.
            base_url: API base URL (mainly for testing).
        """
        self._api_key = api_key
        self._use_mock = use_mock or (api_key is None)
        self._base_url = base_url
        self._http_client: Optional[httpx.AsyncClient] = None
        self._mock_client: Optional[MockVastClient] = None

        if self._use_mock:
            logger.info("VastClient initialized in MOCK mode")
            self._mock_client = MockVastClient()
        else:
            logger.info("VastClient initialized with real API")
            self._http_client = httpx.AsyncClient(
                base_url=base_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )

    @property
    def provider_name(self) -> str:
        return "vast_ai"

    @property
    def is_mock(self) -> bool:
        """Check if client is in mock mode."""
        return self._use_mock

    async def list_offers(
        self, filters: Optional[OfferFilters] = None
    ) -> list[GpuOffer]:
        """List available GPU offers."""
        if self._use_mock:
            return await self._mock_client.list_offers(filters)

        return [o.to_normalized() for o in await self.list_offers_raw(filters)]

    async def list_offers_raw(
        self, filters: Optional[OfferFilters] = None
    ) -> list[VastGpuOffer]:
        """
        List offers with full Vast.ai-specific details.

        Args:
            filters: Optional filters to apply

        Returns:
            List of VastGpuOffer with full provider details
        """
        if self._use_mock:
            vast_filters = None
            if filters:
                vast_filters = VastOfferFilters(
                    gpu_name=filters.gpu_name,
                    num_gpus=filters.min_gpu_count,
                    min_gpu_ram=filters.min_vram_mb,
                    max_dph=filters.max_hourly_price,
                    min_reliability=filters.min_reliability,
                    min_disk=filters.min_disk_gb,
                )
            return self._mock_client.list_offers_raw(vast_filters)

        # Build query parameters
        params = {
            "q": self._build_search_query(filters),
            "order": "dph_total",
            "type": "on-demand",
        }

        response = await self._http_client.get("/bundles", params=params)
        response.raise_for_status()
        data = response.json()

        offers = []
        for offer_data in data.get("offers", []):
            try:
                offers.append(VastGpuOffer(**offer_data))
            except Exception as e:
                logger.warning(f"Failed to parse offer: {e}")

        return offers

    def _build_search_query(self, filters: Optional[OfferFilters]) -> str:
        """Build Vast.ai search query string."""
        conditions = ["rentable=true"]

        if filters:
            if filters.gpu_name:
                conditions.append(f"gpu_name={filters.gpu_name}")
            if filters.min_gpu_count > 1:
                conditions.append(f"num_gpus>={filters.min_gpu_count}")
            if filters.min_vram_mb:
                conditions.append(f"gpu_ram>={filters.min_vram_mb}")
            if filters.max_hourly_price:
                conditions.append(f"dph_total<={filters.max_hourly_price}")
            if filters.min_reliability:
                conditions.append(f"reliability2>={filters.min_reliability}")
            if filters.min_disk_gb:
                conditions.append(f"disk_space>={filters.min_disk_gb}")

        return " ".join(conditions)

    async def create_instance(
        self, offer_id: str, config: InstanceConfig
    ) -> Instance:
        """
        Create a new instance from an offer.

        Args:
            offer_id: The Vast.ai offer/machine ID
            config: Instance configuration

        Returns:
            The created instance
        """
        if self._use_mock:
            return await self._mock_client.create_instance(offer_id, config)

        vast_config = VastInstanceConfig(
            client_id="me",
            image=config.docker_image,
            disk=config.disk_gb,
            onstart=config.onstart_script,
            env=config.env_vars,
            label=config.label,
        )

        response = await self._http_client.put(
            f"/asks/{offer_id}/",
            json=vast_config.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise RuntimeError(f"Failed to create instance: {data}")

        # Get the new instance ID and fetch details
        instance_id = str(data.get("new_contract"))
        instance = await self.get_instance(instance_id)
        if not instance:
            raise RuntimeError(f"Instance created but not found: {instance_id}")

        return instance

    async def get_instance(self, instance_id: str) -> Optional[Instance]:
        """
        Get information about an instance.

        Args:
            instance_id: The Vast.ai instance ID

        Returns:
            Instance info or None if not found
        """
        if self._use_mock:
            return await self._mock_client.get_instance(instance_id)

        raw = await self.get_instance_raw(instance_id)
        return raw.to_normalized() if raw else None

    async def get_instance_raw(self, instance_id: str) -> Optional[VastInstance]:
        """
        Get raw Vast.ai instance details.

        Args:
            instance_id: The instance ID

        Returns:
            VastInstance with full details or None
        """
        if self._use_mock:
            return self._mock_client.get_instance_raw(instance_id)

        response = await self._http_client.get("/instances", params={"owner": "me"})
        response.raise_for_status()
        data = response.json()

        for inst_data in data.get("instances", []):
            if str(inst_data.get("id")) == instance_id:
                try:
                    return VastInstance(**inst_data)
                except Exception as e:
                    logger.warning(f"Failed to parse instance: {e}")
                    return None

        return None

    async def terminate_instance(self, instance_id: str) -> bool:
        """
        Terminate/destroy an instance.

        Args:
            instance_id: The instance ID to terminate

        Returns:
            True if termination was successful
        """
        if self._use_mock:
            return await self._mock_client.terminate_instance(instance_id)

        response = await self._http_client.delete(f"/instances/{instance_id}/")
        return response.status_code == 200

    async def close(self) -> None:
        """Clean up HTTP client."""
        if self._http_client:
            await self._http_client.aclose()


def create_vast_client(
    api_key: Optional[str] = None,
    use_mock: Optional[bool] = None,
) -> VastClient:
    """
    Factory function to create a VastClient.

    Args:
        api_key: Vast.ai API key (uses VAST_AI_API_KEY env var if not provided)
        use_mock: Force mock mode. If None, auto-detect based on api_key.

    Returns:
        Configured VastClient
    """
    import os

    if api_key is None:
        api_key = os.getenv("VAST_AI_API_KEY")

    if use_mock is None:
        use_mock = os.getenv("USE_MOCK_PROVIDERS", "true").lower() == "true"

    return VastClient(api_key=api_key, use_mock=use_mock)
