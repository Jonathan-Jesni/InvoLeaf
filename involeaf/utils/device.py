"""Single point of contact with the accelerator.

Nothing else in the codebase may call ``.cuda()``, check ``torch.cuda.is_available()``
or hardcode a dtype. This module exists because the project runs on an NVIDIA RTX 5050
(CUDA) through September and moves to an AMD Radeon PRO W7800 (ROCm) in October; every
result row records which backend produced it so the two are never silently compared.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, asdict
from typing import Any

import torch


@dataclass
class DeviceInfo:
    """Everything needed to reproduce, or correctly discount, a timing number."""

    device: str            # "cuda" | "directml" | "cpu"
    backend: str           # "cuda" | "rocm" | "directml" | "cpu"
    name: str              # marketing name of the accelerator
    total_memory_gb: float | None
    amp_dtype: str         # "bfloat16" | "float16" | "none"
    torch_version: str
    platform: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_rocm() -> bool:
    # A ROCm build of torch still reports device type "cuda"; torch.version.hip is the
    # only reliable discriminator.
    return getattr(torch.version, "hip", None) is not None


def _directml_device():
    try:
        import torch_directml  # type: ignore
    except ImportError:
        return None
    try:
        if torch_directml.device_count() > 0:
            return torch_directml.device(0)
    except Exception:
        return None
    return None


def resolve_device(prefer: str = "auto") -> torch.device:
    """Return the best available device.

    ``prefer`` is one of ``auto``, ``cuda``, ``directml`` or ``cpu``. ``cuda`` covers
    ROCm as well, since a ROCm torch build exposes AMD cards under the CUDA API.
    """
    if prefer == "cpu":
        return torch.device("cpu")

    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")

    if prefer in ("auto", "directml"):
        dml = _directml_device()
        if dml is not None:
            return dml

    if prefer not in ("auto", "cpu"):
        raise RuntimeError(f"requested device {prefer!r} is not available")
    return torch.device("cpu")


def supports_bf16(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
        return False


def amp_dtype_for(device: torch.device, prefer: str = "auto") -> torch.dtype | None:
    """Pick the autocast dtype.

    bfloat16 is preferred wherever it is supported: it needs no GradScaler and is the
    safer choice on RDNA3, where fp16 dynamic range has bitten people. ``None`` means
    run in full fp32 (autocast disabled).
    """
    if prefer == "none" or device.type == "cpu":
        return None
    if prefer == "bfloat16":
        return torch.bfloat16
    if prefer == "float16":
        return torch.float16
    # auto
    if supports_bf16(device):
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return None


def describe(device: torch.device, amp_dtype: torch.dtype | None = None) -> DeviceInfo:
    """Build the provenance record that gets stamped onto every results row."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        backend = "rocm" if _is_rocm() else "cuda"
        name, total = props.name, props.total_memory / 1024 ** 3
    elif device.type == "cpu":
        backend, name, total = "cpu", platform.processor() or "cpu", None
    else:
        backend, name, total = "directml", str(device), None

    return DeviceInfo(
        device=str(device),
        backend=backend,
        name=name,
        total_memory_gb=round(total, 1) if total else None,
        amp_dtype=str(amp_dtype).replace("torch.", "") if amp_dtype else "none",
        torch_version=torch.__version__,
        platform=f"{platform.system()} {platform.release()}",
    )


def autocast(device: torch.device, dtype: torch.dtype | None):
    """Context manager for mixed precision; a no-op when ``dtype`` is None."""
    if dtype is None:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=dtype)


def needs_grad_scaler(dtype: torch.dtype | None) -> bool:
    """Only fp16 needs loss scaling; bf16 has fp32's exponent range."""
    return dtype == torch.float16


def peak_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024 ** 3


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def synchronize(device: torch.device) -> None:
    """Required before any wall-clock measurement - GPU work is asynchronous."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
