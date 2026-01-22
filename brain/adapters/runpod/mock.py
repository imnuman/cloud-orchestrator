"""
Mock RunPod client for development and testing.
Returns realistic fake data without making actual API calls.
"""

import logging
import random
import uuid
from datetime import datetime
from typing import Optional

from brain.adapters.base import GpuOffer, Instance, InstanceConfig, InstanceStatus, OfferFilters
from brain.adapters.runpod.schemas import (
    RunPodGpuOffer,
    RunPodInstance,
    RunPodOfferFilters,
)

logger = logging.getLogger(__name__)


# Mock GPU offers that simulate RunPod's inventory
MOCK_GPU_OFFERS = [
    RunPodGpuOffer(
        id="NVIDIA RTX 4090",
        display_name="RTX 4090",
        memory_in_gb=24,
        secure_price=0.74,
        community_price=0.44,
        community_spot_price=0.34,
        secure_spot_price=0.59,
        lowest_price=0.34,
        stock_status="available",
        max_gpu_count=8,
        cuda_version="12.2",
        cpu_cores=16,
        ram_mb=62464,
        disk_gb=200.0,
        reliability_score=0.99,
        location="US-TX",
    ),
    RunPodGpuOffer(
        id="NVIDIA RTX 3090",
        display_name="RTX 3090",
        memory_in_gb=24,
        secure_price=0.44,
        community_price=0.31,
        community_spot_price=0.22,
        secure_spot_price=0.35,
        lowest_price=0.22,
        stock_status="available",
        max_gpu_count=4,
        cuda_version="12.2",
        cpu_cores=12,
        ram_mb=48128,
        disk_gb=150.0,
        reliability_score=0.98,
        location="US-TX",
    ),
    RunPodGpuOffer(
        id="NVIDIA A100 80GB PCIe",
        display_name="A100 80GB",
        memory_in_gb=80,
        secure_price=1.99,
        community_price=1.64,
        community_spot_price=1.44,
        secure_spot_price=1.79,
        lowest_price=1.44,
        stock_status="available",
        max_gpu_count=8,
        cuda_version="12.2",
        cpu_cores=24,
        ram_mb=128000,
        disk_gb=500.0,
        reliability_score=0.99,
        location="US-TX",
    ),
    RunPodGpuOffer(
        id="NVIDIA A100 SXM 80GB",
        display_name="A100 SXM 80GB",
        memory_in_gb=80,
        secure_price=2.49,
        community_price=None,
        community_spot_price=None,
        secure_spot_price=2.09,
        lowest_price=2.09,
        stock_status="limited",
        max_gpu_count=8,
        cuda_version="12.2",
        cpu_cores=32,
        ram_mb=256000,
        disk_gb=1000.0,
        reliability_score=0.99,
        location="US-TX",
    ),
    RunPodGpuOffer(
        id="NVIDIA H100 PCIe",
        display_name="H100 PCIe",
        memory_in_gb=80,
        secure_price=4.49,
        community_price=3.89,
        community_spot_price=3.49,
        secure_spot_price=3.99,
        lowest_price=3.49,
        stock_status="limited",
        max_gpu_count=8,
        cuda_version="12.3",
        cpu_cores=32,
        ram_mb=256000,
        disk_gb=1000.0,
        reliability_score=0.99,
        location="US-TX",
    ),
    RunPodGpuOffer(
        id="NVIDIA H100 SXM",
        display_name="H100 SXM",
        memory_in_gb=80,
        secure_price=5.49,
        community_price=None,
        community_spot_price=None,
        secure_spot_price=4.89,
        lowest_price=4.89,
        stock_status="limited",
        max_gpu_count=8,
        cuda_version="12.3",
        cpu_cores=48,
        ram_mb=512000,
        disk_gb=2000.0,
        reliability_score=0.99,
        location="US-TX",
    ),
    RunPodGpuOffer(
        id="NVIDIA L40S",
        display_name="L40S",
        memory_in_gb=48,
        secure_price=1.14,
        community_price=0.94,
        community_spot_price=0.79,
        secure_spot_price=0.99,
        lowest_price=0.79,
        stock_status="available",
        max_gpu_count=8,
        cuda_version="12.2",
        cpu_cores=24,
        ram_mb=128000,
        disk_gb=500.0,
        reliability_score=0.99,
        location="US-TX",
    ),
    RunPodGpuOffer(
        id="NVIDIA RTX A6000",
        display_name="RTX A6000",
        memory_in_gb=48,
        secure_price=0.79,
        community_price=0.59,
        community_spot_price=0.49,
        secure_spot_price=0.69,
        lowest_price=0.49,
        stock_status="available",
        max_gpu_count=4,
        cuda_version="12.2",
        cpu_cores=16,
        ram_mb=62464,
        disk_gb=200.0,
        reliability_score=0.98,
        location="EU-RO",
    ),
]


class MockRunPodClient:
    """Mock RunPod client for development/testing."""

    def __init__(self):
        """Initialize mock client with simulated state."""
        self._instances: dict[str, RunPodInstance] = {}
        logger.info("MockRunPodClient initialized")

    def list_offers_raw(
        self, filters: Optional[RunPodOfferFilters] = None
    ) -> list[RunPodGpuOffer]:
        """List mock GPU offers with optional filtering."""
        offers = MOCK_GPU_OFFERS.copy()

        if filters:
            if filters.gpu_type_id:
                offers = [
                    o for o in offers
                    if filters.gpu_type_id.lower() in o.display_name.lower()
                ]
            if filters.min_memory_in_gb:
                offers = [o for o in offers if o.memory_in_gb >= filters.min_memory_in_gb]
            if filters.max_price_per_hour:
                offers = [
                    o for o in offers
                    if o.lowest_price and o.lowest_price <= filters.max_price_per_hour
                ]

        # Sort by price
        offers.sort(key=lambda o: o.lowest_price or float("inf"))
        return offers

    async def list_offers(
        self, filters: Optional[OfferFilters] = None
    ) -> list[GpuOffer]:
        """List normalized GPU offers."""
        runpod_filters = None
        if filters:
            runpod_filters = RunPodOfferFilters(
                gpu_type_id=filters.gpu_name,
                min_memory_in_gb=(
                    filters.min_vram_mb // 1024 if filters.min_vram_mb else None
                ),
                max_price_per_hour=filters.max_hourly_price,
            )

        raw_offers = self.list_offers_raw(runpod_filters)
        return [o.to_normalized() for o in raw_offers]

    async def create_instance(
        self, offer_id: str, config: InstanceConfig
    ) -> Instance:
        """Create a mock instance."""
        instance_id = f"pod_{uuid.uuid4().hex[:12]}"

        # Find the offer to get pricing
        matching_offers = [o for o in MOCK_GPU_OFFERS if o.id == offer_id]
        if not matching_offers:
            # Use first available if not found
            matching_offers = MOCK_GPU_OFFERS[:1]

        offer = matching_offers[0]

        mock_instance = RunPodInstance(
            id=instance_id,
            name=config.label or f"mock-{instance_id[:8]}",
            desiredStatus="RUNNING",
            runtime={
                "ports": [
                    {"privatePort": 22, "publicPort": random.randint(20000, 30000), "ip": "mock.runpod.net"},
                    {"privatePort": 8888, "publicPort": random.randint(30000, 40000), "ip": "mock.runpod.net"},
                ],
            },
            machine={
                "gpuDisplayName": offer.display_name,
                "gpuId": offer.id,
            },
            imageName=config.docker_image,
            gpuCount=1,
            volumeInGb=config.disk_gb,
            containerDiskInGb=config.disk_gb,
            costPerHr=offer.lowest_price or 0.0,
        )

        self._instances[instance_id] = mock_instance
        logger.info(f"Mock RunPod instance created: {instance_id}")

        return mock_instance.to_normalized()

    async def get_instance(self, instance_id: str) -> Optional[Instance]:
        """Get a mock instance by ID."""
        mock_instance = self._instances.get(instance_id)
        if mock_instance:
            return mock_instance.to_normalized()
        return None

    def get_instance_raw(self, instance_id: str) -> Optional[RunPodInstance]:
        """Get raw mock instance."""
        return self._instances.get(instance_id)

    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate a mock instance."""
        if instance_id in self._instances:
            del self._instances[instance_id]
            logger.info(f"Mock RunPod instance terminated: {instance_id}")
            return True
        return False
