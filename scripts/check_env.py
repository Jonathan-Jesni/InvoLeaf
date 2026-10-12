#!/usr/bin/env python
"""Verify a fresh machine can train this project, and recommend a batch size.

    python scripts/check_env.py

Run this first on any new machine. It exists because the project moves between an
NVIDIA CUDA laptop and an AMD ROCm cloud instance, and the failure modes there are
quiet: a ROCm build that reports no device, bfloat16 silently unsupported, or a batch
size that fits in theory and OOMs on the first backward pass. Each check prints PASS or
FAIL with the reason, and the script exits non-zero if anything is broken.

It does not train anything and writes no files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OK, BAD = "PASS", "FAIL"
failures: list[str] = []


def report(name: str, passed: bool, detail: str = "") -> bool:
    print(f"  [{OK if passed else BAD}] {name}" + (f" - {detail}" if detail else ""))
    if not passed:
        failures.append(name)
    return passed


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def check_torch() -> "object | None":
    section("Framework")
    try:
        import torch
    except ImportError as exc:
        report("torch importable", False, str(exc))
        return None

    report("torch importable", True, f"version {torch.__version__}")

    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    if hip:
        report("build flavour", True, f"ROCm/HIP {hip}")
    elif cuda:
        report("build flavour", True, f"CUDA {cuda}")
    else:
        report("build flavour", False, "CPU-only build - training will be unusably slow")
    return torch


def check_device(torch):
    section("Accelerator")
    from involeaf.utils import device as dev

    d = dev.resolve_device()
    if d.type == "cpu":
        report(
            "accelerator visible", False,
            "resolve_device() returned CPU. On ROCm check that the container sees "
            "/dev/kfd and /dev/dri, and that HSA_OVERRIDE_GFX_VERSION is unset unless "
            "your card actually needs it.",
        )
        return d, None

    amp = dev.amp_dtype_for(d)
    info = dev.describe(d, amp)
    report("accelerator visible", True, f"{info.name} ({info.backend})")
    report(
        "device memory reported", info.total_memory_gb is not None,
        f"{info.total_memory_gb} GB" if info.total_memory_gb else "unknown",
    )
    report(
        "bfloat16 available", amp is not None and "bfloat16" in str(amp),
        f"autocast dtype = {info.amp_dtype}. bf16 needs no GradScaler and is the safer "
        f"choice on RDNA3; fp16 would also work but is more prone to overflow.",
    )
    return d, amp


def check_data() -> None:
    section("Data")
    import json

    splits = sorted((REPO / "splits").glob("*.json"))
    report("split files present", bool(splits), f"{len(splits)} found")

    for path in splits:
        meta = json.loads(path.read_text(encoding="utf-8"))
        root = REPO / meta["root"]
        total = sum(len(v) for v in meta["splits"].values())
        if not root.is_dir():
            report(
                f"{path.name} images", False,
                f"{meta['root']} is missing - run scripts/prepare_data.py --clone",
            )
            continue
        # Spot-check rather than stat 18,000 files.
        probe = [rel for rel, _ in next(iter(meta["splits"].values()))[:5]]
        missing = [p for p in probe if not (root / p).is_file()]
        report(
            f"{path.name} images", not missing,
            f"{total} samples, root OK" if not missing else f"missing e.g. {missing[0]}",
        )


def check_model_and_batch(torch, device, amp, batch_sizes: list[int], model_cfg: str) -> None:
    """Build the real model and run one forward+backward at each candidate batch size.

    This is the only check that matters for picking a batch size. A VRAM calculation
    cannot predict allocator fragmentation or the peak inside the backward pass.
    """
    section("Model and batch size")
    from involeaf.models.registry import build_model
    from involeaf.utils import device as dev
    from involeaf.utils.config import load_config

    cfg = load_config(REPO / model_cfg)
    cfg["pretrained"] = False          # weights are irrelevant to a memory probe
    try:
        model = build_model(cfg, 10).to(device)
    except Exception as exc:          # noqa: BLE001
        report("model builds", False, str(exc))
        return
    params = sum(p.numel() for p in model.parameters()) / 1e6
    report("model builds", True, f"{cfg['name']} at {params:.2f}M parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    largest_ok = None
    for bs in batch_sizes:
        dev.reset_peak_memory(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            x = torch.randn(bs, 3, 224, 224, device=device)
            y = torch.randint(0, 10, (bs,), device=device)
            with dev.autocast(device, amp):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            peak = dev.peak_memory_gb(device)
            print(f"  [{OK}] batch {bs:>3} - peak {peak:.2f} GB" if peak
                  else f"  [{OK}] batch {bs:>3}")
            largest_ok = bs
            del x, y, loss
        except torch.OutOfMemoryError:
            print(f"  [ -- ] batch {bs:>3} - out of memory")
            opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report("at least one batch size fits", largest_ok is not None)
    if largest_ok:
        effective = 64
        accum = max(1, effective // largest_ok)
        print(
            f"\n  Recommended: batch_size={largest_ok}, which gives accum_steps="
            f"{accum} against the pinned effective batch of {effective}."
        )
        if effective % largest_ok != 0:
            print(
                f"  WARNING: {largest_ok} does not divide {effective} exactly. Use the "
                f"next power of two down so the effective batch stays comparable with "
                f"runs from the other machine."
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/involeaf_b16.yaml",
                    help="model to probe; the heaviest model you plan to train")
    ap.add_argument("--batch-sizes", type=int, nargs="*",
                    default=[8, 16, 32, 64, 128],
                    help="candidate batch sizes, tried in order")
    ap.add_argument("--skip-memory", action="store_true",
                    help="skip the model build and batch-size probe")
    args = ap.parse_args()

    print("InvoLeaf environment check")
    print("=" * 26)

    torch = check_torch()
    if torch is None:
        sys.exit(1)

    device, amp = check_device(torch)
    check_data()
    if not args.skip_memory and device.type != "cpu":
        check_model_and_batch(torch, device, amp, args.batch_sizes, args.config)

    section("Summary")
    if failures:
        print(f"  {len(failures)} check(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("  All checks passed. Safe to start training.")


if __name__ == "__main__":
    main()
