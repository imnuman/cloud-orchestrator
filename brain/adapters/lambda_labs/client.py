"""
Lambda Labs API client with mock support.
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
    OfferFilters,
)
from brain.adapters.lambda_labs.mock import MockLambdaClient
from brain.adapters.lambda_labs.schemas import (
    LambdaGpuOffer,
    LambdaInstance,
    LambdaInstanceConfig,
    LambdaOfferFilters,
)


logger = logging.getLogger(__name__)

LAMBDA_API_BASE = "https://cloud.lambdalabs.com/api/v1"


class LambdaClient(BaseProviderAdapter):
    """
    Lambda Labs API client.

    Automatically falls back to mock mode when no API key is provided,
    making development and testing easier.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_mock: bool = False,
        base_url: str = LAMBDA_API_BASE,
    ):
        """
        Initialize Lambda Labs client.

        Args:
            api_key: Lambda Labs API key. If None, mock mode is used.
            use_mock: Force mock mode even with API key.
            base_url: API base URL (mainly for testing).
        """
        self._api_key = api_key
        self._use_mock = use_mock or (api_key is None)
        self._base_url = base_url
        self._http_client: Optional[httpx.AsyncClient] = None
        self._mock_client: Optional[MockLambdaClient] = None

        if self._use_mock:
            logger.info("LambdaClient initialized in MOCK mode")
            self._mock_client = MockLambdaClient()
        else:
            logger.info("LambdaClient initialized with real API")
            self._http_client = httpx.AsyncClient(
                base_url=base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )

    @property
    def provider_name(self) -> str:
        return "lambda_labs"

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
    ) -> list[LambdaGpuOffer]:
        """
        List offers with full Lambda Labs-specific details.

        Uses Lambda Labs' instance-types API to fetch available configurations.
        """
        if self._use_mock:
            lambda_filters = None
            if filters:
                lambda_filters = LambdaOfferFilters(
                    gpu_name=filters.gpu_name,
                    min_gpus=filters.min_gpu_count,
                    max_price_cents=(
                        int(filters.max_hourly_price * 100) if filters.max_hourly_price else None
                    ),
                )
            return self._mock_client.list_offers_raw(lambda_filters)

        # Fetch instance types
        response = await self._http_client.get("/instance-types")
        response.raise_for_status()
        data = response.json()

        offers = []
        instance_types = data.get("data", {})

        for type_name, type_info in instance_types.items():
            regions_with_capacity = type_info.get("regions_with_capacity_available", [])

            # Parse GPU info from description
            description = type_info.get("instance_type", {}).get("description", "")
            specs = type_info.get("instance_type", {}).get("specs", {})

            # Extract GPU details
            gpu_count = specs.get("gpus", 1)
            vram_gb = specs.get("memory_gib", 24)
            vcpus = specs.get("vcpus", 8)
            memory_gb = specs.get("ram_gib", 32)
            storage_gb = specs.get("storage_gib", 100)
            price_cents = type_info.get("instance_type", {}).get("price_cents_per_hour", 0)

            # Parse GPU name from description
            gpu_name = "Unknown"
            if "A100" in description:
                gpu_name = "A100 80GB" if "80" in description else "A100 40GB"
            elif "H100" in description:
                gpu_name = "H100 SXM" if "SXM" in description else "H100 PCIe"
            elif "A10" in description:
                gpu_name = "A10"
            elif "A6000" in description:
                gpu_name = "RTX A6000"
            elif "GH200" in description:
                gpu_name = "GH200"
            elif "RTX" in description:
                gpu_name = description.split(",")[0].strip()

            # Create offer for each available region
            for region_info in regions_with_capacity:
                region_name = region_info.get("name", "unknown")
                region_desc = region_info.get("description", "")

                try:
                    offer = LambdaGpuOffer(
                        instance_type=type_name,
                        region=region_name,
                        price_cents_per_hour=price_cents,
                        description=description,
                        gpu_name=gpu_name,
                        gpu_count=gpu_count,
                        vram_gb=vram_gb,
                        vcpus=vcpus,
                        memory_gb=memory_gb,
                        storage_gb=storage_gb,
                        available=True,
                    )
                    offers.append(offer)
                except Exception as e:
                    logger.warning(f"Failed to parse offer {type_name}: {e}")

        # Apply filters
        if filters:
            if filters.gpu_name:
                offers = [
                    o for o in offers
                    if filters.gpu_name.lower() in o.gpu_name.lower()
                ]
            if filters.min_gpu_count > 1:
                offers = [o for o in offers if o.gpu_count >= filters.min_gpu_count]
            if filters.min_vram_mb:
                min_gb = filters.min_vram_mb // 1024
                offers = [o for o in offers if o.vram_gb >= min_gb]
            if filters.max_hourly_price:
                max_cents = int(filters.max_hourly_price * 100)
                offers = [o for o in offers if o.price_cents_per_hour <= max_cents]

        # Sort by price per GPU
        offers.sort(key=lambda o: o.price_cents_per_hour / o.gpu_count)
        return offers

    async def create_instance(
        self, offer_id: str, config: InstanceConfig
    ) -> Instance:
        """
        Create a new instance from an offer.

        Args:
            offer_id: The offer ID (format: "instance_type:region")
            config: Instance configuration

        Returns:
            The created instance
        """
        if self._use_mock:
            return await self._mock_client.create_instance(offer_id, config)

        # Parse offer_id
        parts = offer_id.split(":")
        instance_type = parts[0]
        region = parts[1] if len(parts) > 1 else "us-west-1"

        # Lambda Labs requires SSH keys to be pre-registered
        # For now, we'll use "default" as a placeholder
        ssh_key_names = ["default"]

        payload = {
            "region_name": region,
            "instance_type_name": instance_type,
            "ssh_key_names": ssh_key_names,
            "file_system_names": [],
            "quantity": 1,
        }

        if config.label:
            payload["name"] = config.label

        response = await self._http_client.post("/instance-operations/launch", json=payload)
        response.raise_for_status()
        data = response.json()

        instance_ids = data.get("data", {}).get("instance_ids", [])
        if not instance_ids:
            raise RuntimeError("Failed to create instance: no instance ID returned")

        # Fetch the created instance details
        instance_id = instance_ids[0]
        instance = await self.get_instance(instance_id)

        if not instance:
            raise RuntimeError(f"Instance created but not found: {instance_id}")

        logger.info(f"Lambda Labs instance created: {instance_id}")
        return instance

    async def get_instance(self, instance_id: str) -> Optional[Instance]:
        """
        Get information about an instance.

        Args:
            instance_id: The Lambda Labs instance ID

        Returns:
            Instance info or None if not found
        """
        if self._use_mock:
            return await self._mock_client.get_instance(instance_id)

        raw = await self.get_instance_raw(instance_id)
        return raw.to_normalized() if raw else None

    async def get_instance_raw(self, instance_id: str) -> Optional[LambdaInstance]:
        """
        Get raw Lambda Labs instance details.

        Args:
            instance_id: The instance ID

        Returns:
            LambdaInstance with full details or None
        """
        if self._use_mock:
            return self._mock_client.get_instance_raw(instance_id)

        response = await self._http_client.get(f"/instances/{instance_id}")

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()
        instance_data = data.get("data", {})

        if not instance_data:
            return None

        try:
            return LambdaInstance(
                id=instance_data.get("id", instance_id),
                name=instance_data.get("name"),
                ip=instance_data.get("ip"),
                status=instance_data.get("status", "unknown"),
                ssh_key_names=instance_data.get("ssh_key_names", []),
                file_system_names=instance_data.get("file_system_names", []),
                region=instance_data.get("region", {}),
                instance_type=instance_data.get("instance_type", {}),
                hostname=instance_data.get("hostname"),
                jupyter_token=instance_data.get("jupyter_token"),
                jupyter_url=instance_data.get("jupyter_url"),
            )
        except Exception as e:
            logger.warning(f"Failed to parse instance {instance_id}: {e}")
            return None

    async def list_instances(self) -> list[Instance]:
        """
        List all instances.

        Returns:
            List of all instances
        """
        if self._use_mock:
            return [
                inst.to_normalized()
                for inst in self._mock_client._instances.values()
            ]

        response = await self._http_client.get("/instances")
        response.raise_for_status()
        data = response.json()

        instances = []
        for inst_data in data.get("data", []):
            try:
                lambda_inst = LambdaInstance(
                    id=inst_data.get("id"),
                    name=inst_data.get("name"),
                    ip=inst_data.get("ip"),
                    status=inst_data.get("status", "unknown"),
                    ssh_key_names=inst_data.get("ssh_key_names", []),
                    file_system_names=inst_data.get("file_system_names", []),
                    region=inst_data.get("region", {}),
                    instance_type=inst_data.get("instance_type", {}),
                    hostname=inst_data.get("hostname"),
                )
                instances.append(lambda_inst.to_normalized())
            except Exception as e:
                logger.warning(f"Failed to parse instance: {e}")

        return instances

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

        payload = {"instance_ids": [instance_id]}

        try:
            response = await self._http_client.post(
                "/instance-operations/terminate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            terminated = data.get("data", {}).get("terminated_instances", [])
            success = instance_id in [t.get("id") for t in terminated]

            if success:
                logger.info(f"Lambda Labs instance terminated: {instance_id}")

            return success
        except Exception as e:
            logger.error(f"Failed to terminate instance {instance_id}: {e}")
            return False

    async def close(self) -> None:
        """Clean up HTTP client."""
        if self._http_client:
            await self._http_client.aclose()


def create_lambda_client(
    api_key: Optional[str] = None,
    use_mock: Optional[bool] = None,
) -> LambdaClient:
    """
    Factory function to create a LambdaClient.

    Args:
        api_key: Lambda Labs API key (uses LAMBDA_LABS_API_KEY env var if not provided)
        use_mock: Force mock mode. If None, auto-detect based on api_key.

    Returns:
        Configured LambdaClient
    """
    import os

    if api_key is None:
        api_key = os.getenv("LAMBDA_LABS_API_KEY")

    if use_mock is None:
        use_mock = os.getenv("USE_MOCK_PROVIDERS", "true").lower() == "true"

    return LambdaClient(api_key=api_key, use_mock=use_mock)
