"""
System information detection.
"""

import os
import platform
import socket
import subprocess
from typing import Optional

from shared.schemas import SystemInfo


def get_hostname() -> str:
    """Get system hostname."""
    return socket.gethostname()


def get_os_info() -> str:
    """Get OS name and version."""
    try:
        # Try to get pretty name from /etc/os-release
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=")[1].strip().strip('"')
    except FileNotFoundError:
        pass
    return f"{platform.system()} {platform.release()}"


def get_kernel_version() -> str:
    """Get kernel version."""
    return platform.release()


def get_cpu_info() -> tuple[str, int]:
    """Get CPU model and core count."""
    cpu_model = "Unknown"
    cpu_cores = os.cpu_count() or 1

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":")[1].strip()
                    break
    except FileNotFoundError:
        cpu_model = platform.processor() or "Unknown"

    return cpu_model, cpu_cores


def get_memory_info() -> tuple[int, int]:
    """Get total and available RAM in MB."""
    try:
        with open("/proc/meminfo") as f:
            mem_total = 0
            mem_available = 0
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) // 1024  # KB to MB
                elif line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1]) // 1024
            return mem_total, mem_available
    except FileNotFoundError:
        return 0, 0


def get_disk_info() -> tuple[float, float]:
    """Get total and available disk space in GB."""
    try:
        stat = os.statvfs("/")
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
        available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        return round(total_gb, 2), round(available_gb, 2)
    except Exception:
        return 0.0, 0.0


def get_docker_version() -> Optional[str]:
    """Get Docker version if installed."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Parse "Docker version X.Y.Z, build abc123"
            parts = result.stdout.split()
            if len(parts) >= 3:
                return parts[2].rstrip(",")
        return None
    except Exception:
        return None


def get_nvidia_driver_version() -> Optional[str]:
    """Get NVIDIA driver version."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
        return None
    except Exception:
        return None


def detect_system_info() -> SystemInfo:
    """Detect and return system information."""
    cpu_model, cpu_cores = get_cpu_info()
    ram_total, ram_available = get_memory_info()
    disk_total, disk_available = get_disk_info()

    return SystemInfo(
        hostname=get_hostname(),
        os_name=get_os_info(),
        kernel_version=get_kernel_version(),
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        ram_total_mb=ram_total,
        ram_available_mb=ram_available,
        disk_total_gb=disk_total,
        disk_available_gb=disk_available,
        docker_version=get_docker_version(),
        nvidia_driver_version=get_nvidia_driver_version(),
    )


if __name__ == "__main__":
    # Test system info detection
    import json
    info = detect_system_info()
    print(json.dumps(info.model_dump(), indent=2))
