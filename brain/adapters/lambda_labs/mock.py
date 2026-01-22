"""
Mock Lambda Labs client for development and testing.
Returns realistic fake data without making actual API calls.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from brain.adapters.base import GpuOffer, Instance, InstanceConfig, InstanceStatus, OfferFilters
from brain.adapters.lambda_labs.schemas import (
    LambdaGpuOffer,
    LambdaInstance,
    LambdaOfferFilters,
)

logger = logging.getLogger(__name__)


# Mock GPU offers that simulate Lambda Labs' inventory
MOCK_GPU_OFFERS = [
    LambdaGpuOffer(
        instance_type="gpu_1x_a10",
        region="us-west-1",
        price_cents_per_hour=60,
        description="1x A10 (24 GB PCIe)",
        gpu_name="A10",
        gpu_count=1,
        vram_gb=24,
        vcpus=30,
        memory_gb=200,
        storage_gb=1400,
        available=True,
    ),
    LambdaGpuOffer(
        instance_type="gpu_1x_a100",
        region="us-west-1",
        price_cents_per_hour=129,
        description="1x A100 (40 GB PCIe)",
        gpu_name="A100 40GB",
        gpu_count=1,
        vram_gb=40,
        vcpus=30,
        memory_gb=200,
        storage_gb=1400,
        available=True,
    ),
    LambdaGpuOffer(
        instance_type="gpu_1x_a100_sxm4",
        region="us-west-1",
        price_cents_per_hour=179,
        description="1x A100 (80 GB SXM4)",
        gpu_name="A100 SXM 80GB",
        gpu_count=1,
        vram_gb=80,
        vcpus=30,
        memory_gb=200,
        storage_gb=1400,
        available=True,
    ),
    LambdaGpuOffer(
        instance_type="gpu_4x_a100",
        region="us-west-1",
        price_cents_per_hour=516,
        description="4x A100 (40 GB PCIe)",
        gpu_name="A100 40GB",
        gpu_count=4,
        vram_gb=40,
        vcpus=120,
        memory_gb=800,
        storage_gb=5600,
        available=True,
    ),
    LambdaGpuOffer(
        instance_type="gpu_8x_a100",
        region="us-west-1",
        price_cents_per_hour=1032,
        description="8x A100 (40 GB PCIe)",
        gpu_name="A100 40GB",
        gpu_count=8,
        vram_gb=40,
        vcpus=240,
        memory_gb=1600,
        storage_gb=11200,
        available=True,
    ),
    LambdaGpuOffer(
        instance_type="gpu_8x_a100_sxm4",
        region="us-west-1",
        price_cents_per_hour=1432,
        description="8x A100 (80 GB SXM4)",
        gpu_name="A100 SXM 80GB",
        gpu_count=8,
        vram_gb=80,
        vcpus=240,
        memory_gb=1800,
        storage_gb=20000,
        available=True,
    ),
    LambdaGpuOffer(
        instance_type="gpu_1x_a6000",
        region="us-west-1",
        price_cents_per_hour=80,
        description="1x RTX A6000 (48 GB)",
        gpu_name="RTX A6000",
        gpu_count=1,
        vram_gb=48,
        vcpus=14,
        memory_gb=100,
        storage_gb=200,
        available=True,
    ),
    LambdaGpuOffer(
        instance_type="gpu_1x_h100_pcie",
        region="us-west-1",
        price_cents_per_hour=249,
        description="1x H100 (80 GB PCIe)",
        gpu_name="H100 PCIe",
        gpu_count=1,
        vram_gb=80,
        vcpus=26,
        memory_gb=200,
        storage_gb=1000,
        available=True,
    ),
    LambdaGpuOffer(
        instance_type="gpu_8x_h100_sxm",
        region="us-west-1",
        price_cents_per_hour=2632,
        description="8x H100 (80 GB SXM5)",
        gpu_name="H100 SXM",
        gpu_count=8,
        vram_gb=80,
        vcpus=208,
        memory_gb=1800,
        storage_gb=25000,
        available=False,  # Typically hard to get
    ),
    LambdaGpuOffer(
        instance_type="gpu_1x_gh200",
        region="us-west-1",
        price_cents_per_hour=149,
        description="1x GH200 (96 GB)",
        gpu_name="GH200",
        gpu_count=1,
        vram_gb=96,
        vcpus=72,
        memory_gb=480,
        storage_gb=2000,
        available=True,
    ),
]

# Add offers in multiple regions
REGIONS = ["us-west-1", "us-east-1", "us-south-1", "europe-central-1"]


def _generate_regional_offers() -> list[LambdaGpuOffer]:
    """Generate offers across multiple regions."""
    offers = []
    for base_offer in MOCK_GPU_OFFERS:
        for region in REGIONS:
            # Copy offer with different region
            offer = base_offer.model_copy(update={"region": region})
            # Slight price variation by region
            if region == "europe-central-1":
                offer.price_cents_per_hour = int(offer.price_cents_per_hour * 1.1)
            offers.append(offer)
    return offers


class MockLambdaClient:
    """Mock Lambda Labs client for development/testing."""

    def __init__(self):
        """Initialize mock client with simulated state."""
        self._instances: dict[str, LambdaInstance] = {}
        self._offers = _generate_regional_offers()
        logger.info("MockLambdaClient initialized")

    def list_offers_raw(
        self, filters: Optional[LambdaOfferFilters] = None
    ) -> list[LambdaGpuOffer]:
        """List mock GPU offers with optional filtering."""
        offers = [o for o in self._offers if o.available]

        if filters:
            if filters.instance_type:
                offers = [
                    o for o in offers
                    if filters.instance_type.lower() in o.instance_type.lower()
                ]
            if filters.region:
                offers = [
                    o for o in offers
                    if filters.region.lower() in o.region.lower()
                ]
            if filters.min_gpus > 1:
                offers = [o for o in offers if o.gpu_count >= filters.min_gpus]
            if filters.max_price_cents:
                offers = [
                    o for o in offers
                    if o.price_cents_per_hour <= filters.max_price_cents
                ]
            if filters.gpu_name:
                offers = [
                    o for o in offers
                    if filters.gpu_name.lower() in o.gpu_name.lower()
                ]

        # Sort by price per GPU
        offers.sort(key=lambda o: o.price_cents_per_hour / o.gpu_count)
        return offers

    async def list_offers(
        self, filters: Optional[OfferFilters] = None
    ) -> list[GpuOffer]:
        """List normalized GPU offers."""
        lambda_filters = None
        if filters:
            lambda_filters = LambdaOfferFilters(
                gpu_name=filters.gpu_name,
                min_gpus=filters.min_gpu_count,
                max_price_cents=(
                    int(filters.max_hourly_price * 100) if filters.max_hourly_price else None
                ),
            )

        raw_offers = self.list_offers_raw(lambda_filters)
        return [o.to_normalized() for o in raw_offers]

    async def create_instance(
        self, offer_id: str, config: InstanceConfig
    ) -> Instance:
        """Create a mock instance."""
        instance_id = f"i-{uuid.uuid4().hex[:12]}"

        # Parse offer_id (format: instance_type:region)
        parts = offer_id.split(":")
        instance_type = parts[0] if parts else "gpu_1x_a100"
        region = parts[1] if len(parts) > 1 else "us-west-1"

        # Find matching offer
        matching_offers = [
            o for o in self._offers
            if o.instance_type == instance_type and o.region == region
        ]
        if not matching_offers:
            matching_offers = self._offers[:1]

        offer = matching_offers[0]

        mock_instance = LambdaInstance(
            id=instance_id,
            name=config.label or f"mock-{instance_id[:8]}",
            ip=f"10.0.{hash(instance_id) % 256}.{hash(instance_id + '1') % 256}",
            status="active",
            ssh_key_names=["default"],
            region={"name": region, "description": f"Mock {region}"},
            instance_type={
                "name": instance_type,
                "description": f"{offer.gpu_count}x {offer.gpu_name}",
                "price_cents_per_hour": offer.price_cents_per_hour,
            },
            hostname=f"{instance_id}.cloud.lambdalabs.com",
        )

        self._instances[instance_id] = mock_instance
        logger.info(f"Mock Lambda Labs instance created: {instance_id}")

        return mock_instance.to_normalized()

    async def get_instance(self, instance_id: str) -> Optional[Instance]:
        """Get a mock instance by ID."""
        mock_instance = self._instances.get(instance_id)
        if mock_instance:
            return mock_instance.to_normalized()
        return None

    def get_instance_raw(self, instance_id: str) -> Optional[LambdaInstance]:
        """Get raw mock instance."""
        return self._instances.get(instance_id)

    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate a mock instance."""
        if instance_id in self._instances:
            del self._instances[instance_id]
            logger.info(f"Mock Lambda Labs instance terminated: {instance_id}")
            return True
        return False
