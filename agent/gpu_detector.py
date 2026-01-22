"""
GPU detection using nvidia-smi.
"""

import json
import subprocess
from typing import Optional

from shared.schemas import GpuInfo


def run_nvidia_smi(query: str, format_type: str = "csv,noheader,nounits") -> Optional[str]:
    """
    Run nvidia-smi with specified query.

    Args:
        query: Comma-separated list of query fields
        format_type: Output format

    Returns:
        Command output or None if failed
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                f"--format={format_type}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"Error running nvidia-smi: {e}")
        return None


def detect_gpus() -> list[GpuInfo]:
    """
    Detect all NVIDIA GPUs on the system.

    Returns:
        List of GpuInfo objects for each detected GPU
    """
    gpus = []

    # Query GPU information
    query_fields = [
        "index",
        "name",
        "memory.total",
        "memory.used",
        "memory.free",
        "temperature.gpu",
        "utilization.gpu",
        "power.draw",
        "driver_version",
    ]

    output = run_nvidia_smi(",".join(query_fields))

    if not output:
        return gpus

    # Parse output (one line per GPU)
    for line in output.split("\n"):
        if not line.strip():
            continue

        values = [v.strip() for v in line.split(",")]

        if len(values) < 9:
            continue

        try:
            gpu = GpuInfo(
                index=int(values[0]),
                name=values[1],
                memory_total_mb=int(values[2]) if values[2] else 0,
                memory_used_mb=int(values[3]) if values[3] else 0,
                memory_free_mb=int(values[4]) if values[4] else 0,
                temperature_c=int(values[5]) if values[5] and values[5] != "[N/A]" else None,
                utilization_percent=int(values[6]) if values[6] and values[6] != "[N/A]" else None,
                power_draw_w=float(values[7]) if values[7] and values[7] != "[N/A]" else None,
                driver_version=values[8] if values[8] else None,
            )
            gpus.append(gpu)
        except (ValueError, IndexError) as e:
            print(f"Error parsing GPU info: {e}")
            continue

    return gpus


def get_cuda_version() -> Optional[str]:
    """Get CUDA version from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # CUDA version is shown in the header, not easily queryable
        # Alternative: check nvcc
        nvcc_result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if nvcc_result.returncode == 0:
            # Parse "Cuda compilation tools, release X.Y" line
            for line in nvcc_result.stdout.split("\n"):
                if "release" in line.lower():
                    parts = line.split("release")
                    if len(parts) > 1:
                        version = parts[1].strip().split(",")[0]
                        return version
        return None
    except Exception:
        return None


def get_gpu_stats_json() -> dict:
    """
    Get GPU statistics as a JSON-serializable dictionary.
    Useful for debugging and API responses.
    """
    gpus = detect_gpus()
    cuda_version = get_cuda_version()

    return {
        "gpu_count": len(gpus),
        "cuda_version": cuda_version,
        "gpus": [gpu.model_dump() for gpu in gpus],
    }


if __name__ == "__main__":
    # Test GPU detection
    stats = get_gpu_stats_json()
    print(json.dumps(stats, indent=2))
