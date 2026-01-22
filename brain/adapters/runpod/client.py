"""
RunPod API client with mock support.
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
from brain.adapters.runpod.mock import MockRunPodClient
from brain.adapters.runpod.schemas import (
    RunPodGpuOffer,
    RunPodInstance,
    RunPodInstanceConfig,
    RunPodOfferFilters,
)


logger = logging.getLogger(__name__)

RUNPOD_API_BASE = "https://api.runpod.io"
RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"


class RunPodClient(BaseProviderAdapter):
    """
    RunPod API client.

    Automatically falls back to mock mode when no API key is provided,
    making development and testing easier.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_mock: bool = False,
        base_url: str = RUNPOD_API_BASE,
    ):
        """
        Initialize RunPod client.

        Args:
            api_key: RunPod API key. If None, mock mode is used.
            use_mock: Force mock mode even with API key.
            base_url: API base URL (mainly for testing).
        """
        self._api_key = api_key
        self._use_mock = use_mock or (api_key is None)
        self._base_url = base_url
        self._graphql_url = RUNPOD_GRAPHQL_URL
        self._http_client: Optional[httpx.AsyncClient] = None
        self._mock_client: Optional[MockRunPodClient] = None

        if self._use_mock:
            logger.info("RunPodClient initialized in MOCK mode")
            self._mock_client = MockRunPodClient()
        else:
            logger.info("RunPodClient initialized with real API")
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
        return "runpod"

    @property
    def is_mock(self) -> bool:
        """Check if client is in mock mode."""
        return self._use_mock

    async def _graphql_query(self, query: str, variables: Optional[dict] = None) -> dict:
        """Execute a GraphQL query."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        response = await self._http_client.post(
            self._graphql_url,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        if "errors" in data:
            raise RuntimeError(f"GraphQL error: {data['errors']}")

        return data.get("data", {})

    async def list_offers(
        self, filters: Optional[OfferFilters] = None
    ) -> list[GpuOffer]:
        """List available GPU offers."""
        if self._use_mock:
            return await self._mock_client.list_offers(filters)

        return [o.to_normalized() for o in await self.list_offers_raw(filters)]

    async def list_offers_raw(
        self, filters: Optional[OfferFilters] = None
    ) -> list[RunPodGpuOffer]:
        """
        List offers with full RunPod-specific details.

        Uses RunPod's GraphQL API to fetch GPU types and pricing.
        """
        if self._use_mock:
            runpod_filters = None
            if filters:
                runpod_filters = RunPodOfferFilters(
                    gpu_type_id=filters.gpu_name,
                    min_memory_in_gb=(
                        filters.min_vram_mb // 1024 if filters.min_vram_mb else None
                    ),
                    max_price_per_hour=filters.max_hourly_price,
                )
            return self._mock_client.list_offers_raw(runpod_filters)

        # GraphQL query for GPU types
        query = """
        query GpuTypes {
            gpuTypes {
                id
                displayName
                memoryInGb
                secureCloud
                communityCloud
                lowestPrice(input: { gpuCount: 1 }) {
                    minimumBidPrice
                    uninterruptablePrice
                }
            }
        }
        """

        data = await self._graphql_query(query)
        gpu_types = data.get("gpuTypes", [])

        offers = []
        for gpu_data in gpu_types:
            try:
                lowest_price_data = gpu_data.get("lowestPrice", {})
                secure_price = None
                community_price = None
                lowest_price = None

                if gpu_data.get("secureCloud"):
                    secure_price = lowest_price_data.get("uninterruptablePrice")
                if gpu_data.get("communityCloud"):
                    community_price = lowest_price_data.get("minimumBidPrice")

                # Calculate lowest available price
                prices = [p for p in [secure_price, community_price] if p]
                lowest_price = min(prices) if prices else None

                offer = RunPodGpuOffer(
                    id=gpu_data["id"],
                    display_name=gpu_data["displayName"],
                    memory_in_gb=gpu_data["memoryInGb"],
                    secure_price=secure_price,
                    community_price=community_price,
                    lowest_price=lowest_price,
                    stock_status="available" if lowest_price else "unavailable",
                )
                offers.append(offer)
            except Exception as e:
                logger.warning(f"Failed to parse GPU type: {e}")

        # Apply filters
        if filters:
            if filters.gpu_name:
                offers = [
                    o for o in offers
                    if filters.gpu_name.lower() in o.display_name.lower()
                ]
            if filters.min_vram_mb:
                min_gb = filters.min_vram_mb // 1024
                offers = [o for o in offers if o.memory_in_gb >= min_gb]
            if filters.max_hourly_price:
                offers = [
                    o for o in offers
                    if o.lowest_price and o.lowest_price <= filters.max_hourly_price
                ]

        # Sort by price
        offers.sort(key=lambda o: o.lowest_price or float("inf"))
        return offers

    async def create_instance(
        self, offer_id: str, config: InstanceConfig
    ) -> Instance:
        """
        Create a new pod from an offer.

        Args:
            offer_id: The GPU type ID (e.g., "NVIDIA RTX 4090")
            config: Instance configuration

        Returns:
            The created instance
        """
        if self._use_mock:
            return await self._mock_client.create_instance(offer_id, config)

        # Convert environment variables to RunPod format
        env_vars = {**config.env_vars}

        # Build ports string
        ports = "22/tcp,8888/http"

        # GraphQL mutation for creating a pod
        mutation = """
        mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                desiredStatus
                imageName
                gpuCount
                volumeInGb
                containerDiskInGb
                costPerHr
                machine {
                    gpuDisplayName
                }
            }
        }
        """

        variables = {
            "input": {
                "cloudType": "ALL",
                "gpuCount": 1,
                "volumeInGb": int(config.disk_gb),
                "containerDiskInGb": int(config.disk_gb),
                "minVcpuCount": 2,
                "minMemoryInGb": 8,
                "gpuTypeId": offer_id,
                "name": config.label or f"gpu-orch-{offer_id[:8]}",
                "imageName": config.docker_image,
                "ports": ports,
                "volumeMountPath": "/workspace",
                "env": [{"key": k, "value": v} for k, v in env_vars.items()],
            }
        }

        if config.onstart_script:
            variables["input"]["dockerArgs"] = f"bash -c '{config.onstart_script}'"

        data = await self._graphql_query(mutation, variables)
        pod_data = data.get("podFindAndDeployOnDemand", {})

        if not pod_data:
            raise RuntimeError("Failed to create pod: no data returned")

        instance = RunPodInstance(
            id=pod_data["id"],
            name=pod_data.get("name"),
            desiredStatus=pod_data.get("desiredStatus", "CREATED"),
            imageName=pod_data.get("imageName", config.docker_image),
            gpuCount=pod_data.get("gpuCount", 1),
            volumeInGb=pod_data.get("volumeInGb", config.disk_gb),
            containerDiskInGb=pod_data.get("containerDiskInGb", config.disk_gb),
            costPerHr=pod_data.get("costPerHr", 0),
            machine=pod_data.get("machine"),
        )

        logger.info(f"RunPod instance created: {instance.id}")
        return instance.to_normalized()

    async def get_instance(self, instance_id: str) -> Optional[Instance]:
        """
        Get information about a pod.

        Args:
            instance_id: The RunPod pod ID

        Returns:
            Instance info or None if not found
        """
        if self._use_mock:
            return await self._mock_client.get_instance(instance_id)

        raw = await self.get_instance_raw(instance_id)
        return raw.to_normalized() if raw else None

    async def get_instance_raw(self, instance_id: str) -> Optional[RunPodInstance]:
        """
        Get raw RunPod pod details.

        Args:
            instance_id: The pod ID

        Returns:
            RunPodInstance with full details or None
        """
        if self._use_mock:
            return self._mock_client.get_instance_raw(instance_id)

        query = """
        query Pod($podId: String!) {
            pod(input: { podId: $podId }) {
                id
                name
                desiredStatus
                imageName
                gpuCount
                volumeInGb
                containerDiskInGb
                costPerHr
                uptimeSeconds
                runtime {
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                }
                machine {
                    gpuDisplayName
                    podHostId
                }
            }
        }
        """

        try:
            data = await self._graphql_query(query, {"podId": instance_id})
            pod_data = data.get("pod")

            if not pod_data:
                return None

            return RunPodInstance(
                id=pod_data["id"],
                name=pod_data.get("name"),
                desiredStatus=pod_data.get("desiredStatus", "UNKNOWN"),
                runtime=pod_data.get("runtime"),
                machine=pod_data.get("machine"),
                imageName=pod_data.get("imageName", ""),
                gpuCount=pod_data.get("gpuCount", 1),
                volumeInGb=pod_data.get("volumeInGb", 0),
                containerDiskInGb=pod_data.get("containerDiskInGb", 0),
                costPerHr=pod_data.get("costPerHr", 0),
                uptimeSeconds=pod_data.get("uptimeSeconds"),
            )
        except Exception as e:
            logger.warning(f"Failed to get pod {instance_id}: {e}")
            return None

    async def terminate_instance(self, instance_id: str) -> bool:
        """
        Terminate/destroy a pod.

        Args:
            instance_id: The pod ID to terminate

        Returns:
            True if termination was successful
        """
        if self._use_mock:
            return await self._mock_client.terminate_instance(instance_id)

        mutation = """
        mutation TerminatePod($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """

        try:
            data = await self._graphql_query(mutation, {"input": {"podId": instance_id}})
            success = data.get("podTerminate") is not None
            if success:
                logger.info(f"RunPod instance terminated: {instance_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to terminate pod {instance_id}: {e}")
            return False

    async def stop_instance(self, instance_id: str) -> bool:
        """
        Stop a pod (without terminating).

        Args:
            instance_id: The pod ID to stop

        Returns:
            True if stop was successful
        """
        if self._use_mock:
            return True

        mutation = """
        mutation StopPod($input: PodStopInput!) {
            podStop(input: $input) {
                id
                desiredStatus
            }
        }
        """

        try:
            data = await self._graphql_query(mutation, {"input": {"podId": instance_id}})
            return data.get("podStop") is not None
        except Exception as e:
            logger.error(f"Failed to stop pod {instance_id}: {e}")
            return False

    async def resume_instance(self, instance_id: str) -> bool:
        """
        Resume a stopped pod.

        Args:
            instance_id: The pod ID to resume

        Returns:
            True if resume was successful
        """
        if self._use_mock:
            return True

        mutation = """
        mutation ResumePod($input: PodResumeInput!) {
            podResume(input: $input) {
                id
                desiredStatus
            }
        }
        """

        try:
            data = await self._graphql_query(mutation, {"input": {"podId": instance_id}})
            return data.get("podResume") is not None
        except Exception as e:
            logger.error(f"Failed to resume pod {instance_id}: {e}")
            return False

    async def close(self) -> None:
        """Clean up HTTP client."""
        if self._http_client:
            await self._http_client.aclose()


def create_runpod_client(
    api_key: Optional[str] = None,
    use_mock: Optional[bool] = None,
) -> RunPodClient:
    """
    Factory function to create a RunPodClient.

    Args:
        api_key: RunPod API key (uses RUNPOD_API_KEY env var if not provided)
        use_mock: Force mock mode. If None, auto-detect based on api_key.

    Returns:
        Configured RunPodClient
    """
    import os

    if api_key is None:
        api_key = os.getenv("RUNPOD_API_KEY")

    if use_mock is None:
        use_mock = os.getenv("USE_MOCK_PROVIDERS", "true").lower() == "true"

    return RunPodClient(api_key=api_key, use_mock=use_mock)
