"""
Mock Vast.ai client for development and testing.
Provides realistic responses without making actual API calls.
"""

import asyncio
import random
from datetime import datetime
from typing import Optional

from brain.adapters.base import (
    BaseProviderAdapter,
    GpuOffer,
    Instance,
    InstanceConfig,
    InstanceStatus,
    OfferFilters,
)
from brain.adapters.vast_ai.schemas import (
    VastGpuOffer,
    VastInstance,
    VastInstanceStatus,
    VastOfferFilters,
)


# Mock data: Realistic GPU offers
MOCK_OFFERS: list[dict] = [
    {
        "id": 10001,
        "machine_id": 5001,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "total_flops": 82.58,
        "cuda_max_good": 12.2,
        "driver_version": "535.154.05",
        "cpu_cores": 16,
        "cpu_cores_effective": 16.0,
        "cpu_name": "AMD Ryzen 9 5950X",
        "cpu_ram": 65536,
        "disk_space": 500.0,
        "disk_bw": 3500.0,
        "inet_up": 1000.0,
        "inet_down": 1000.0,
        "dph_total": 0.35,
        "min_bid": 0.30,
        "reliability2": 0.98,
        "dlperf": 42.5,
        "dlperf_per_dphtotal": 121.43,
        "duration": 168.0,
        "verification": "verified",
        "geolocation": "US-West",
        "country": "US",
        "rentable": True,
        "rented": False,
    },
    {
        "id": 10002,
        "machine_id": 5002,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "total_flops": 82.58,
        "cuda_max_good": 12.1,
        "driver_version": "535.129.03",
        "cpu_cores": 12,
        "cpu_cores_effective": 12.0,
        "cpu_name": "Intel Core i9-12900K",
        "cpu_ram": 32768,
        "disk_space": 250.0,
        "disk_bw": 2500.0,
        "inet_up": 500.0,
        "inet_down": 1000.0,
        "dph_total": 0.42,
        "min_bid": 0.35,
        "reliability2": 0.96,
        "dlperf": 41.2,
        "dlperf_per_dphtotal": 98.10,
        "duration": 72.0,
        "verification": "verified",
        "geolocation": "US-East",
        "country": "US",
        "rentable": True,
        "rented": False,
    },
    {
        "id": 10003,
        "machine_id": 5003,
        "gpu_name": "RTX 3090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "total_flops": 35.58,
        "cuda_max_good": 12.0,
        "driver_version": "530.30.02",
        "cpu_cores": 8,
        "cpu_cores_effective": 8.0,
        "cpu_name": "AMD Ryzen 7 5800X",
        "cpu_ram": 32768,
        "disk_space": 200.0,
        "disk_bw": 1500.0,
        "inet_up": 200.0,
        "inet_down": 500.0,
        "dph_total": 0.25,
        "min_bid": 0.20,
        "reliability2": 0.95,
        "dlperf": 28.5,
        "dlperf_per_dphtotal": 114.0,
        "duration": 240.0,
        "verification": "verified",
        "geolocation": "EU-West",
        "country": "DE",
        "rentable": True,
        "rented": False,
    },
    {
        "id": 10004,
        "machine_id": 5004,
        "gpu_name": "RTX 4090",
        "num_gpus": 2,
        "gpu_ram": 24576,
        "total_flops": 165.16,
        "cuda_max_good": 12.2,
        "driver_version": "535.154.05",
        "cpu_cores": 32,
        "cpu_cores_effective": 32.0,
        "cpu_name": "AMD EPYC 7542",
        "cpu_ram": 131072,
        "disk_space": 1000.0,
        "disk_bw": 5000.0,
        "inet_up": 2000.0,
        "inet_down": 2000.0,
        "dph_total": 0.75,
        "min_bid": 0.65,
        "reliability2": 0.99,
        "dlperf": 85.0,
        "dlperf_per_dphtotal": 113.33,
        "duration": 336.0,
        "verification": "verified",
        "geolocation": "US-Central",
        "country": "US",
        "rentable": True,
        "rented": False,
    },
    {
        "id": 10005,
        "machine_id": 5005,
        "gpu_name": "A100",
        "num_gpus": 1,
        "gpu_ram": 81920,
        "total_flops": 312.0,
        "cuda_max_good": 12.2,
        "driver_version": "535.154.05",
        "cpu_cores": 64,
        "cpu_cores_effective": 16.0,
        "cpu_name": "AMD EPYC 7763",
        "cpu_ram": 262144,
        "disk_space": 500.0,
        "disk_bw": 7000.0,
        "inet_up": 10000.0,
        "inet_down": 10000.0,
        "dph_total": 1.50,
        "min_bid": 1.20,
        "reliability2": 0.99,
        "dlperf": 150.0,
        "dlperf_per_dphtotal": 100.0,
        "duration": 720.0,
        "verification": "verified",
        "geolocation": "US-West",
        "country": "US",
        "rentable": True,
        "rented": False,
    },
]


class MockVastClient(BaseProviderAdapter):
    """
    Mock Vast.ai client for development and testing.
    Simulates the Vast.ai API without making real calls.
    """

    def __init__(self):
        """Initialize mock client."""
        self._offers = [VastGpuOffer(**offer) for offer in MOCK_OFFERS]
        self._instances: dict[str, VastInstance] = {}
        self._instance_counter = 20000
        self._startup_delays: dict[str, int] = {}

    @property
    def provider_name(self) -> str:
        return "vast_ai"

    def list_offers_raw(
        self, filters: Optional[VastOfferFilters] = None
    ) -> list[VastGpuOffer]:
        """List raw Vast.ai offers with full details."""
        offers = [o for o in self._offers if o.rentable and not o.rented]

        if filters:
            if filters.gpu_name:
                gpu_filter = filters.gpu_name.lower()
                offers = [
                    o for o in offers if gpu_filter in o.gpu_name.lower()
                ]

            if filters.num_gpus:
                offers = [o for o in offers if o.num_gpus >= filters.num_gpus]

            if filters.min_gpu_ram:
                offers = [o for o in offers if o.gpu_ram >= filters.min_gpu_ram]

            if filters.max_dph:
                offers = [o for o in offers if o.dph_total <= filters.max_dph]

            if filters.min_reliability:
                offers = [
                    o for o in offers if o.reliability2 >= filters.min_reliability
                ]

            if filters.min_disk:
                offers = [o for o in offers if o.disk_space >= filters.min_disk]

            if filters.cuda_vers:
                offers = [
                    o for o in offers if o.cuda_max_good >= filters.cuda_vers
                ]

        # Sort by price
        offers.sort(key=lambda o: o.dph_total)

        return offers[:filters.limit] if filters else offers

    async def list_offers(
        self, filters: Optional[OfferFilters] = None
    ) -> list[GpuOffer]:
        """List available GPU offers."""
        # Simulate network delay
        await asyncio.sleep(0.1)

        vast_filters = None
        if filters:
            vast_filters = VastOfferFilters(
                gpu_name=filters.gpu_name,
                num_gpus=filters.min_gpu_count,
                min_gpu_ram=filters.min_vram_mb,
                max_dph=filters.max_hourly_price,
                min_reliability=filters.min_reliability,
                min_disk=filters.min_disk_gb,
                cuda_vers=float(filters.cuda_version) if filters.cuda_version else None,
            )

        raw_offers = self.list_offers_raw(vast_filters)
        return [o.to_normalized() for o in raw_offers]

    async def create_instance(
        self, offer_id: str, config: InstanceConfig
    ) -> Instance:
        """Create a mock instance."""
        await asyncio.sleep(0.2)

        # Find the offer
        offer = next(
            (o for o in self._offers if str(o.id) == offer_id),
            None
        )
        if not offer:
            raise ValueError(f"Offer {offer_id} not found")

        # Create instance
        self._instance_counter += 1
        instance_id = str(self._instance_counter)

        vast_instance = VastInstance(
            id=self._instance_counter,
            machine_id=offer.machine_id,
            actual_status=VastInstanceStatus.LOADING,
            intended_status="running",
            gpu_name=offer.gpu_name,
            num_gpus=offer.num_gpus,
            gpu_ram=offer.gpu_ram,
            dph_total=offer.dph_total,
            docker_image=config.docker_image,
            onstart_cmd=config.onstart_script,
            disk_space=config.disk_gb,
            label=config.label,
            start_date=datetime.utcnow().timestamp(),
        )

        self._instances[instance_id] = vast_instance
        self._startup_delays[instance_id] = random.randint(2, 5)

        # Mark offer as rented
        for o in self._offers:
            if str(o.id) == offer_id:
                o.rented = True
                break

        return vast_instance.to_normalized()

    async def get_instance(self, instance_id: str) -> Optional[Instance]:
        """Get instance status with simulated startup progression."""
        await asyncio.sleep(0.1)

        vast_instance = self._instances.get(instance_id)
        if not vast_instance:
            return None

        # Simulate startup progression
        if instance_id in self._startup_delays:
            self._startup_delays[instance_id] -= 1
            if self._startup_delays[instance_id] <= 0:
                del self._startup_delays[instance_id]
                vast_instance.actual_status = VastInstanceStatus.RUNNING
                vast_instance.ssh_host = f"ssh{instance_id}.vast.ai"
                vast_instance.ssh_port = 22 + random.randint(1000, 9000)
                vast_instance.public_ipaddr = f"203.0.113.{random.randint(1, 254)}"

        return vast_instance.to_normalized()

    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate a mock instance."""
        await asyncio.sleep(0.1)

        vast_instance = self._instances.get(instance_id)
        if not vast_instance:
            return False

        # Mark instance as stopped
        vast_instance.actual_status = VastInstanceStatus.EXITED
        vast_instance.end_date = datetime.utcnow().timestamp()

        # Free up the offer
        for offer in self._offers:
            if offer.machine_id == vast_instance.machine_id:
                offer.rented = False
                break

        return True

    def get_instance_raw(self, instance_id: str) -> Optional[VastInstance]:
        """Get raw Vast.ai instance."""
        return self._instances.get(instance_id)

    def reset(self) -> None:
        """Reset mock state (useful for testing)."""
        self._instances.clear()
        self._startup_delays.clear()
        for offer in self._offers:
            offer.rented = False
