"""Cost measurement: parameters, MACs ("FLOPs"), latency, peak memory.

Every measurement is stamped with the device record from ``involeaf.utils.device`` so
that September RTX 5050 numbers and October W7900D numbers are never merged into one
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
    """Multiply-accumulates (MACs) via fvcore, completed analytically.

    fvcore counts one fused multiply-add as one "flop", so its total is MACs, the unit
    timm and most ViT papers quote as "FLOPs" (ViT-B/16: ~17.6 G). Two terms it has no
    handler for are added analytically, in the same unit:

    - the involution multiply-accumulate over K*K offsets (its kernel generation is
      ordinary 1x1 convolution and is counted);
    - the two attention matmuls inside ``scaled_dot_product_attention`` (Q K^T and
      A V), which fvcore skips for every ViT, and twice per block for AgriTL-ViT.

    ``total_gmacs`` is the figure to report. Before 2026-10-19 the involution term was in
    FLOPs (2x) and attention was missing, as the older ``*_gflops`` keys show.
    """
    result: dict = {"input_size": list(input_size), "unit": "GMACs"}
    model = model.eval()
    x = torch.randn(*input_size)

    try:
        from fvcore.nn import FlopCountAnalysis

        fca = FlopCountAnalysis(model, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        result["fvcore_gmacs"] = round(fca.total() / 1e9, 4)
        result["fvcore_uncounted"] = sorted(set(fca.unsupported_ops().keys()))
    except Exception as exc:                        # noqa: BLE001
        result["fvcore_error"] = str(exc)

    try:
        from thop import profile as thop_profile

        macs, _ = thop_profile(model, inputs=(x,), verbose=False)
        result["thop_gmacs"] = round(macs / 1e9, 4)
    except Exception as exc:                        # noqa: BLE001
        result["thop_error"] = str(exc)

    result["involution_analytic_gmacs"] = round(
        _involution_macs(model, input_size) / 1e9, 4)
    result["attention_analytic_gmacs"] = round(
        _attention_macs(model, input_size) / 1e9, 4)
    if "fvcore_gmacs" in result:
        result["total_gmacs"] = round(
            result["fvcore_gmacs"] + result["involution_analytic_gmacs"]
            + result["attention_analytic_gmacs"], 4)
    return result


def _token_grid(model: nn.Module, input_size: tuple[int, ...]) -> int:
    patch = getattr(getattr(model, "patch_embed", None), "patch_size", (16, 16))
    stride = patch[0] if isinstance(patch, (tuple, list)) else patch
    return input_size[-1] // stride


def _involution_macs(model: nn.Module, input_size: tuple[int, ...]) -> float:
    """Per involution: one multiply-accumulate per channel, kernel tap and location."""
    from involeaf.ops.involution import Involution2d

    side = _token_grid(model, input_size)
    return float(sum(m.channels * m.kernel_size ** 2 * side * side
                     for m in model.modules() if isinstance(m, Involution2d)))


def _attention_macs(model: nn.Module, input_size: tuple[int, ...]) -> float:
    """Per attention module: Q K^T and A V, each N*N*D multiply-accumulates.

    Counted per Attention module, so AgriTL-ViT's second attention in every block is
    included. The qkv and output projections are ordinary linears that fvcore counts.
    """
    try:
        from timm.models.vision_transformer import Attention
    except ImportError:                              # pragma: no cover
        return 0.0
    side = _token_grid(model, input_size)
    n = side * side + getattr(model, "num_prefix_tokens", 0)
    return float(sum(2 * n * n * m.qkv.in_features
                     for m in model.modules() if isinstance(m, Attention)))


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
    dev.reset_peak_memory(device)
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
