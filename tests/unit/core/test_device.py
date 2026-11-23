"""Tests for device detection."""

import torch
from adg.core.device import (
    detect_hardware,
    get_device_map,
    get_torch_dtype,
    report_memory_usage,
)


def test_detect_hardware():
    """Test hardware detection returns valid info."""
    hw = detect_hardware()

    assert hw.device in ("cuda", "mps", "cpu")
    assert hw.dtype in ("auto", "fp16", "fp32", "bf16")
    assert isinstance(hw.cuda_available, bool)
    assert isinstance(hw.mps_available, bool)
    assert hw.cpu_count >= 0

    # Verify device matches availability
    if hw.cuda_available:
        assert hw.device == "cuda"
        assert hw.gpu_name is not None
        assert hw.gpu_memory_gb and hw.gpu_memory_gb > 0
    elif hw.mps_available:
        assert hw.device == "mps"
    else:
        assert hw.device == "cpu"


def test_get_torch_dtype():
    """Test dtype conversion."""
    assert get_torch_dtype("fp16") == torch.float16
    assert get_torch_dtype("fp32") == torch.float32
    assert get_torch_dtype("bf16") == torch.bfloat16

    # Auto resolves to fp16/fp32
    auto_dtype = get_torch_dtype("auto")
    assert auto_dtype in (torch.float16, torch.float32)


def test_get_device_map():
    """Test device_map generation."""
    assert get_device_map("cuda") == "auto"
    assert get_device_map("mps") == "mps"
    assert get_device_map("cpu") == "cpu"


def test_report_memory_usage_cpu():
    """Test memory reporting on CPU (should be empty)."""
    mem = report_memory_usage("cpu")
    assert mem == {}


def test_report_memory_usage_cuda():
    """Test memory reporting on CUDA (if available)."""
    if torch.cuda.is_available():
        mem = report_memory_usage("cuda")
        assert "allocated_gb" in mem
        assert "total_gb" in mem
        assert mem["total_gb"] > 0
    else:
        mem = report_memory_usage(None)  # Auto-detect should use CPU
        assert mem == {}
