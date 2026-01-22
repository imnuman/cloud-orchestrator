"""
GPU Agent - Main daemon that runs on GPU worker nodes.
Handles:
- Registration with Brain
- Periodic heartbeats
- Pod deployment/management
- GPU telemetry
"""

import asyncio
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

import httpx

from agent.config import AgentSettings, get_agent_settings
from agent.gpu_detector import detect_gpus
from agent.system_info import detect_system_info
from agent.docker_manager import DockerManager
from shared.schemas import (
    NodeRegistrationRequest,
    NodeRegistrationResponse,
    NodeHeartbeatRequest,
    NodeHeartbeatResponse,
    ProviderType,
    PodDeployCommand,
    PodStopCommand,
)


class GpuAgent:
    """GPU Agent daemon."""

    def __init__(self, settings: Optional[AgentSettings] = None):
        """Initialize the agent."""
        self.settings = settings or get_agent_settings()
        self.node_id: Optional[str] = None
        self.api_key: Optional[str] = None
        self.heartbeat_interval: int = self.settings.heartbeat_interval_seconds
        self.running = False
        self.docker_manager: Optional[DockerManager] = None

        # HTTP client
        self.http_client = httpx.AsyncClient(
            base_url=f"{self.settings.brain_url}{self.settings.brain_api_prefix}",
            timeout=30.0,
        )

        # Load saved config
        self._load_config()

    def _load_config(self) -> None:
        """Load saved configuration from file."""
        config_file = self.settings.config_file
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    self.node_id = config.get("node_id")
                    self.api_key = config.get("api_key")
                    print(f"Loaded config: node_id={self.node_id}")
            except Exception as e:
                print(f"Error loading config: {e}")

    def _save_config(self) -> None:
        """Save configuration to file."""
        config_file = self.settings.config_file
        config_file.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "node_id": self.node_id,
            "api_key": self.api_key,
        }

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved config to {config_file}")

    async def register(self) -> bool:
        """
        Register this node with the Brain.

        Returns:
            True if registration successful
        """
        print("Detecting hardware...")
        gpus = detect_gpus()
        system_info = detect_system_info()

        if not gpus:
            print("WARNING: No GPUs detected! Continuing anyway for testing.")

        # Get IP address (in production, this would be VPN IP)
        import socket
        ip_address = socket.gethostbyname(socket.gethostname())

        # Map provider type string to enum
        provider_type_map = {
            "vast_ai": ProviderType.VAST_AI,
            "runpod": ProviderType.RUNPOD,
            "lambda_labs": ProviderType.LAMBDA_LABS,
            "community": ProviderType.COMMUNITY,
            "internal": ProviderType.INTERNAL,
        }
        provider_type = provider_type_map.get(
            self.settings.provider_type.lower(),
            ProviderType.COMMUNITY
        )

        request = NodeRegistrationRequest(
            hostname=system_info.hostname,
            ip_address=ip_address,
            gpus=gpus,
            system_info=system_info,
            agent_version="0.1.0",
            provider_type=provider_type,
            provider_id=self.settings.provider_instance_id,
            hourly_price=self.settings.hourly_price,
        )

        print(f"Registering with Brain at {self.settings.brain_url}...")
        print(f"  Hostname: {request.hostname}")
        print(f"  IP: {request.ip_address}")
        print(f"  GPUs: {len(gpus)}")
        if gpus:
            for gpu in gpus:
                print(f"    - {gpu.name} ({gpu.memory_total_mb} MB)")

        try:
            response = await self.http_client.post(
                "/nodes/register",
                json=request.model_dump(),
            )
            response.raise_for_status()

            data = response.json()
            self.node_id = data["node_id"]
            self.api_key = data["api_key"]
            self.heartbeat_interval = data.get(
                "heartbeat_interval_seconds",
                self.settings.heartbeat_interval_seconds,
            )

            print(f"Registration successful!")
            print(f"  Node ID: {self.node_id}")
            print(f"  Heartbeat interval: {self.heartbeat_interval}s")

            # Save config
            self._save_config()

            return True

        except httpx.HTTPStatusError as e:
            print(f"Registration failed: {e.response.status_code} - {e.response.text}")
            return False
        except Exception as e:
            print(f"Registration error: {e}")
            return False

    async def send_heartbeat(self) -> bool:
        """
        Send heartbeat to Brain.

        Returns:
            True if heartbeat acknowledged
        """
        if not self.node_id or not self.api_key:
            print("Not registered, cannot send heartbeat")
            return False

        # Get current status
        gpus = detect_gpus()
        system_info = detect_system_info()

        # Get running pods
        running_pods = []
        if self.docker_manager:
            running_pods = [
                UUID(pod_id) for pod_id in self.docker_manager.get_running_containers()
            ]

        request = NodeHeartbeatRequest(
            node_id=UUID(self.node_id),
            gpus=gpus,
            running_pods=running_pods,
            system_info=system_info,
            timestamp=datetime.utcnow(),
        )

        try:
            response = await self.http_client.post(
                "/nodes/heartbeat",
                json=request.model_dump(mode="json"),
                headers={"X-Agent-API-Key": self.api_key},
            )
            response.raise_for_status()

            data = response.json()

            # Process any commands from Brain
            if data.get("commands"):
                await self._process_commands(data["commands"])

            return data.get("acknowledged", False)

        except httpx.HTTPStatusError as e:
            print(f"Heartbeat failed: {e.response.status_code}")
            return False
        except Exception as e:
            print(f"Heartbeat error: {e}")
            return False

    async def _process_commands(self, commands: list[dict]) -> None:
        """Process commands received from Brain."""
        for cmd in commands:
            cmd_type = cmd.get("type")
            print(f"Processing command: {cmd_type}")

            if cmd_type == "deploy_pod":
                command = PodDeployCommand(**cmd["data"])
                if self.docker_manager:
                    result = await self.docker_manager.deploy_pod(command)
                    # TODO: Report result back to Brain
                    print(f"Pod deployment: {'success' if result.success else 'failed'}")

            elif cmd_type == "stop_pod":
                command = PodStopCommand(**cmd["data"])
                if self.docker_manager:
                    result = await self.docker_manager.stop_pod(command)
                    print(f"Pod stop: {'success' if result.success else 'failed'}")

    async def heartbeat_loop(self) -> None:
        """Main heartbeat loop."""
        while self.running:
            success = await self.send_heartbeat()
            if success:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Heartbeat OK")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Heartbeat FAILED")

            await asyncio.sleep(self.heartbeat_interval)

    async def start(self) -> None:
        """Start the agent."""
        print("=" * 60)
        print("GPU Agent Starting...")
        print("=" * 60)

        # Initialize Docker manager
        try:
            self.docker_manager = DockerManager()
            print("Docker manager initialized")
        except Exception as e:
            print(f"WARNING: Docker not available: {e}")
            self.docker_manager = None

        # Register if not already registered
        if not self.node_id or not self.api_key:
            if not await self.register():
                print("Failed to register, exiting")
                return

        self.running = True

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        print("Starting heartbeat loop...")
        await self.heartbeat_loop()

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        print("\nShutting down...")
        self.running = False
        await self.http_client.aclose()
        print("Agent stopped")


async def main():
    """Main entry point."""
    agent = GpuAgent()
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
