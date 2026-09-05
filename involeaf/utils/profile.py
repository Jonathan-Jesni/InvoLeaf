"""Cost measurement: parameters, FLOPs, latency, peak memory.

Every measurement is stamped with the device record from ``involeaf.utils.device`` so
that September RTX 5050 numbers and October W7800 numbers are never merged into one
table. Latency in particular is not portable between backends.

A standing caveat for the write-up: involution's FLOP savings do not fully translate
into wall-clock time, because there is no fused kernel for it. The original paper says
so itself (Table 2: RedNet-50 14.3 ms vs ResNet-50 11.4 ms on GPU despite fewer FLOPs),
and our shift-based implementation trades speed for memory even further, running K*K
small kernels per block. Report the gap; do not hide it.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from involeaf.utils import device as dev


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "total_m": round(total / 1e6, 3)}


def count_flops(model: nn.Module, input_size: tuple[int, ...] = (1, 3, 224, 224)) -> dict:
    """FLOPs via fvcore, cross-checked against thop when both are installed.

    Both tools miss operations they have no handler for; the involution kernel
    generation is standard 1x1 convolution and is counted, but the elementwise
    multiply-accumulate over K*K offsets is not. The returned dict therefore includes an
    analytic correction term rather than pretending the tool output is complete.
    """
    result: dict = {"input_size": list(input_size)}
    model = model.eval()
    x = torch.randn(*input_size)

    try:
        from fvcore.nn import FlopCountAnalysis

        fca = FlopCountAnalysis(model, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        result["fvcore_gflops"] = round(fca.total() / 1e9, 4)
        result["fvcore_uncounted"] = sorted(set(fca.unsupported_ops().keys()))
    except Exception as exc:                        # noqa: BLE001
        result["fvcore_error"] = str(exc)

    try:
        from thop import profile as thop_profile

        macs, _ = thop_profile(model, inputs=(x,), verbose=False)
        result["thop_gmacs"] = round(macs / 1e9, 4)
    except Exception as exc:                        # noqa: BLE001
        result["thop_error"] = str(exc)

    result["involution_analytic_gflops"] = round(
        _involution_multiply_accumulate_flops(model, input_size) / 1e9, 4
    )
    return result


def _involution_multiply_accumulate_flops(
    model: nn.Module, input_size: tuple[int, ...]
) -> float:
    """The multiply-accumulate term that FLOP counters miss.

    Per involution: C * K*K * H * W multiplies plus the same number of adds.
    """
    from involeaf.ops.involution import Involution2d

    patch = getattr(getattr(model, "patch_embed", None), "patch_size", (16, 16))
    stride = patch[0] if isinstance(patch, (tuple, list)) else patch
    h = w = input_size[-1] // stride

    total = 0.0
    for m in model.modules():
        if isinstance(m, Involution2d):
            total += 2.0 * m.channels * (m.kernel_size ** 2) * h * w
    return total


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    device: torch.device,
    input_size: tuple[int, ...] = (1, 3, 224, 224),
    warmup: int = 20,
    iters: int = 100,
    amp_dtype: torch.dtype | None = None,
) -> dict:
    """Wall-clock latency, batch 1 by default.

    Synchronises around every timed region -- GPU work is asynchronous, and timing
    without a sync measures queueing, not compute.
    """
    model = model.eval().to(device)
    x = torch.randn(*input_size, device=device)

    for _ in range(warmup):
        with dev.autocast(device, amp_dtype):
            model(x)
    dev.synchronize(device)

    samples = []
    for _ in range(iters):
        dev.synchronize(device)
        t0 = time.perf_counter()
        with dev.autocast(device, amp_dtype):
            model(x)
        dev.synchronize(device)
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    n = len(samples)
    return {
        "mean_ms": round(sum(samples) / n, 3),
        "median_ms": round(samples[n // 2], 3),
        "p90_ms": round(samples[int(n * 0.9)], 3),
        "min_ms": round(samples[0], 3),
        "iters": iters,
        "batch_size": input_size[0],
    }


def measure_train_memory(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 16,
    input_size: int = 224,
    amp_dtype: torch.dtype | None = None,
) -> dict:
    """Peak allocated memory for one forward+backward step."""
    if device.type != "cuda":
        return {"peak_gb": None, "note": "peak memory is only tracked on CUDA/ROCm"}

    model = model.train().to(device)
    dev.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    x = torch.randn(batch_size, 3, input_size, input_size, device=device)
    y = torch.randint(0, max(1, model.num_classes if hasattr(model, "num_classes") else 10),
                      (batch_size,), device=device)
    try:
        with dev.autocast(device, amp_dtype):
            loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        peak = dev.peak_memory_gb(device)
        status = "ok"
    except torch.OutOfMemoryError:
        peak, status = None, "OOM"

    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return {"peak_gb": round(peak, 3) if peak else None,
            "batch_size": batch_size, "status": status}


def full_profile(
    model: nn.Module,
    device: torch.device,
    input_size: int = 224,
    train_batch: int = 16,
    amp_dtype: torch.dtype | None = None,
) -> dict:
    """Everything the efficiency table needs, for one model on one device."""
    return {
        "params": count_parameters(model),
        "flops": count_flops(model, (1, 3, input_size, input_size)),
        "latency_gpu": measure_latency(
            model, device, (1, 3, input_size, input_size), amp_dtype=amp_dtype
        ),
        "latency_cpu": measure_latency(
            model, torch.device("cpu"), (1, 3, input_size, input_size),
            warmup=3, iters=20, amp_dtype=None,
        ),
        "train_memory": measure_train_memory(
            model, device, train_batch, input_size, amp_dtype
        ),
        "device": dev.describe(device, amp_dtype).as_dict(),
    }
