"""
Docker container management for GPU workloads.
"""

import random
from typing import Optional
from uuid import UUID

import docker
from docker.errors import ContainerError, ImageNotFound, APIError
from docker.models.containers import Container

from agent.config import get_agent_settings
from shared.schemas import PodDeployCommand, PodDeployResponse, PodStopCommand, PodStopResponse

settings = get_agent_settings()


class DockerManager:
    """Manages Docker containers for GPU workloads."""

    def __init__(self):
        """Initialize Docker client."""
        self.client = docker.from_env()
        self._ensure_network()

    def _ensure_network(self) -> None:
        """Ensure the orchestrator network exists."""
        try:
            self.client.networks.get(settings.default_docker_network)
        except docker.errors.NotFound:
            self.client.networks.create(
                settings.default_docker_network,
                driver="bridge",
            )

    def _allocate_port(self, base_port: int) -> int:
        """
        Allocate an available port starting from base_port.
        In production, this should check actual port availability.
        """
        # Simple implementation: random offset from base
        # In production: track allocated ports in a database
        return base_port + random.randint(0, 999)

    def _build_gpu_device_request(self, gpu_indices: list[int]) -> list[dict]:
        """Build Docker device request for NVIDIA GPUs."""
        if not gpu_indices:
            # Request all GPUs
            return [
                docker.types.DeviceRequest(
                    count=-1,  # All GPUs
                    capabilities=[["gpu"]],
                )
            ]

        # Request specific GPUs
        device_ids = [str(i) for i in gpu_indices]
        return [
            docker.types.DeviceRequest(
                device_ids=device_ids,
                capabilities=[["gpu"]],
            )
        ]

    async def deploy_pod(self, command: PodDeployCommand) -> PodDeployResponse:
        """
        Deploy a new pod (container) with GPU access.

        Args:
            command: Pod deployment configuration

        Returns:
            Deployment result with container info
        """
        try:
            # Pull image if not present
            try:
                self.client.images.get(command.docker_image)
            except ImageNotFound:
                print(f"Pulling image: {command.docker_image}")
                self.client.images.pull(command.docker_image)

            # Allocate ports
            ssh_port = self._allocate_port(settings.ssh_base_port)
            jupyter_port = self._allocate_port(settings.jupyter_base_port)

            # Build port mappings
            port_bindings = {
                "22/tcp": ssh_port,  # SSH
                "8888/tcp": jupyter_port,  # Jupyter
            }

            # Add custom port mappings
            mapped_ports = {22: ssh_port, 8888: jupyter_port}
            for pm in command.port_mappings:
                host_port = self._allocate_port(12000)
                port_bindings[f"{pm.container_port}/{pm.protocol}"] = host_port
                mapped_ports[pm.container_port] = host_port

            # Environment variables
            env = {
                "POD_ID": str(command.pod_id),
                **command.environment_variables,
            }

            # GPU device request
            device_requests = self._build_gpu_device_request(command.gpu_indices)

            # Volume mounts
            volumes = {}
            for mount in command.volume_mounts:
                # Format: "/host/path:/container/path"
                parts = mount.split(":")
                if len(parts) == 2:
                    volumes[parts[0]] = {"bind": parts[1], "mode": "rw"}

            # Create container
            container = self.client.containers.run(
                command.docker_image,
                command=command.startup_command,
                name=f"pod-{command.pod_id}",
                detach=True,
                environment=env,
                ports=port_bindings,
                device_requests=device_requests,
                volumes=volumes,
                network=settings.default_docker_network,
                restart_policy={"Name": "unless-stopped"},
                shm_size="16g",  # Shared memory for ML workloads
                **command.resource_limits,
            )

            # Get host IP (for SSH connection string)
            # In production, this would be the VPN/public IP
            import socket
            host_ip = socket.gethostbyname(socket.gethostname())

            return PodDeployResponse(
                pod_id=command.pod_id,
                success=True,
                container_id=container.id,
                ssh_host=host_ip,
                ssh_port=ssh_port,
                mapped_ports=mapped_ports,
            )

        except ImageNotFound as e:
            return PodDeployResponse(
                pod_id=command.pod_id,
                success=False,
                error_message=f"Image not found: {command.docker_image}",
            )
        except APIError as e:
            return PodDeployResponse(
                pod_id=command.pod_id,
                success=False,
                error_message=f"Docker API error: {str(e)}",
            )
        except Exception as e:
            return PodDeployResponse(
                pod_id=command.pod_id,
                success=False,
                error_message=f"Deployment failed: {str(e)}",
            )

    async def stop_pod(self, command: PodStopCommand) -> PodStopResponse:
        """
        Stop a running pod.

        Args:
            command: Stop command with pod ID

        Returns:
            Stop result
        """
        container_name = f"pod-{command.pod_id}"

        try:
            container = self.client.containers.get(container_name)

            if command.force:
                container.kill()
            else:
                container.stop(timeout=command.timeout_seconds)

            # Remove container
            container.remove()

            return PodStopResponse(
                pod_id=command.pod_id,
                success=True,
            )

        except docker.errors.NotFound:
            return PodStopResponse(
                pod_id=command.pod_id,
                success=True,  # Already gone
                error_message="Container not found (may already be stopped)",
            )
        except Exception as e:
            return PodStopResponse(
                pod_id=command.pod_id,
                success=False,
                error_message=f"Failed to stop: {str(e)}",
            )

    def get_running_containers(self) -> list[str]:
        """Get list of running pod container IDs."""
        containers = self.client.containers.list(
            filters={"name": "pod-"},
        )
        return [c.name.replace("pod-", "") for c in containers]

    def get_container_logs(self, pod_id: str, tail: int = 100) -> str:
        """Get logs from a container."""
        container_name = f"pod-{pod_id}"
        try:
            container = self.client.containers.get(container_name)
            return container.logs(tail=tail).decode("utf-8")
        except docker.errors.NotFound:
            return "Container not found"
        except Exception as e:
            return f"Error getting logs: {e}"

    def get_container_stats(self, pod_id: str) -> Optional[dict]:
        """Get resource usage stats for a container."""
        container_name = f"pod-{pod_id}"
        try:
            container = self.client.containers.get(container_name)
            stats = container.stats(stream=False)
            return {
                "cpu_percent": self._calculate_cpu_percent(stats),
                "memory_usage_mb": stats["memory_stats"].get("usage", 0) // (1024 * 1024),
                "memory_limit_mb": stats["memory_stats"].get("limit", 0) // (1024 * 1024),
            }
        except Exception:
            return None

    def _calculate_cpu_percent(self, stats: dict) -> float:
        """Calculate CPU usage percentage from Docker stats."""
        try:
            cpu_delta = (
                stats["cpu_stats"]["cpu_usage"]["total_usage"]
                - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            system_delta = (
                stats["cpu_stats"]["system_cpu_usage"]
                - stats["precpu_stats"]["system_cpu_usage"]
            )
            if system_delta > 0:
                num_cpus = len(stats["cpu_stats"]["cpu_usage"].get("percpu_usage", [1]))
                return (cpu_delta / system_delta) * num_cpus * 100.0
            return 0.0
        except (KeyError, ZeroDivisionError):
            return 0.0
