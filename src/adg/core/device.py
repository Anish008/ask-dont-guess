"""Hardware detection and device selection.

Per spec §15.1-15.2: Detect CUDA, MPS, CPU; choose appropriate dtype and device_map.
"""

import os
from typing import Literal

import torch

from adg.core.logging import get_logger

log = get_logger(__name__)

DeviceType = Literal["cuda", "mps", "cpu"]
DType = Literal["auto", "fp16", "fp32", "bf16"]

# Constants
AMPERE_COMPUTE_CAPABILITY = 8  # Ampere architecture starts at compute capability 8.0


class HardwareInfo:
    """Hardware capabilities and recommended settings."""

    def __init__(
        self,
        device: DeviceType,
        dtype: DType,
        *,  # Keyword-only arguments after this
        cuda_available: bool = False,
        mps_available: bool = False,
        gpu_name: str | None = None,
        gpu_memory_gb: float | None = None,
        cpu_count: int = 0,
        ram_gb: float | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.cuda_available = cuda_available
        self.mps_available = mps_available
        self.gpu_name = gpu_name
        self.gpu_memory_gb = gpu_memory_gb
        self.cpu_count = cpu_count
        self.ram_gb = ram_gb

    def __repr__(self) -> str:
        if self.device == "cuda":
            return f"HardwareInfo(cuda, {self.gpu_name}, {self.gpu_memory_gb:.1f}GB VRAM)"
        elif self.device == "mps":
            return f"HardwareInfo(mps, {self.ram_gb:.1f}GB RAM)"
        else:
            return f"HardwareInfo(cpu, {self.cpu_count} cores, {self.ram_gb:.1f}GB RAM)"


def detect_hardware() -> HardwareInfo:
    """Detect available hardware and return recommended settings.

    Returns:
        HardwareInfo with device, dtype, and capabilities

    Per spec §15.2:
    - CUDA GPU → device=cuda, dtype=auto (bf16 on Ampere+, fp16 otherwise)
    - Apple MPS → device=mps, dtype=fp16
    - CPU → device=cpu, dtype=fp32
    """
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    # Detect CPU info
    cpu_count = os.cpu_count() or 0

    # Rough RAM estimation (platform-dependent)
    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        ram_gb = None

    if cuda_available:
        # CUDA: Use GPU 0, detect VRAM and name
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Choose dtype: bf16 for Ampere+ (compute capability >= 8.0), else fp16
        capability = torch.cuda.get_device_capability(0)
        if capability[0] >= AMPERE_COMPUTE_CAPABILITY:
            dtype: DType = "bf16"
        else:
            dtype = "fp16"

        log.info(
            "hardware_detected",
            device="cuda",
            gpu=gpu_name,
            vram_gb=round(gpu_memory_gb, 1),
            dtype=dtype,
            compute_capability=f"{capability[0]}.{capability[1]}",
        )

        return HardwareInfo(
            device="cuda",
            dtype=dtype,
            cuda_available=True,
            mps_available=False,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            cpu_count=cpu_count,
            ram_gb=ram_gb,
        )

    elif mps_available:
        # Apple Silicon MPS
        log.info(
            "hardware_detected",
            device="mps",
            dtype="fp16",
            ram_gb=ram_gb,
            note="Apple Silicon",
        )

        return HardwareInfo(
            device="mps",
            dtype="fp16",
            cuda_available=False,
            mps_available=True,
            cpu_count=cpu_count,
            ram_gb=ram_gb,
        )

    else:
        # CPU fallback
        log.info(
            "hardware_detected",
            device="cpu",
            dtype="fp32",
            cpu_count=cpu_count,
            ram_gb=ram_gb,
            note="No GPU detected",
        )

        return HardwareInfo(
            device="cpu",
            dtype="fp32",
            cuda_available=False,
            mps_available=False,
            cpu_count=cpu_count,
            ram_gb=ram_gb,
        )


def get_torch_dtype(dtype: DType) -> torch.dtype:
    """Convert dtype string to torch.dtype.

    Args:
        dtype: One of "auto", "fp16", "fp32", "bf16"

    Returns:
        Corresponding torch.dtype

    Note:
        "auto" resolves to fp16 for CUDA/MPS, fp32 for CPU
    """
    if dtype == "auto":
        # Auto: fp16 on GPU, fp32 on CPU
        if torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            return torch.float16
        else:
            return torch.float32
    elif dtype == "fp16":
        return torch.float16
    elif dtype == "bf16":
        return torch.bfloat16
    elif dtype == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def get_device_map(device: DeviceType) -> str | dict[str, str] | None:
    """Get device_map argument for HF model loading.

    Args:
        device: Device type

    Returns:
        - "auto" for CUDA (uses accelerate's balanced placement)
        - "mps" for Apple Silicon
        - "cpu" for CPU
        - None for mock or when device_map not needed
    """
    device_map_lookup = {
        "cuda": "auto",  # Let accelerate handle multi-GPU
        "mps": "mps",
        "cpu": "cpu",
    }
    return device_map_lookup.get(device)


def report_memory_usage(device: DeviceType | None = None) -> dict[str, float]:
    """Report current GPU memory usage (if available).

    Args:
        device: Device to check (None = auto-detect)

    Returns:
        Dict with allocated_gb, reserved_gb, free_gb (empty dict if not CUDA)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free = total - allocated

        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(free, 2),
            "total_gb": round(total, 2),
        }
    else:
        return {}
