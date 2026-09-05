"""Training loop.

Effective batch size is pinned by gradient accumulation
--------------------------------------------------------
``batch_size * accum_steps`` is held at a fixed ``effective_batch`` regardless of what
fits in VRAM. This project trains on an 8 GB RTX 5050 through September and a 32 GB
Radeon PRO W7800 from October; without pinning, the two halves of the experiment would
use different effective batch sizes and could not be compared. The physical batch size
is a memory detail, the effective batch size is a hyperparameter.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from involeaf.data.datasets import SplitDataset
from involeaf.data.transforms import eval_transform, train_transform
from involeaf.engine.evaluate import evaluate
from involeaf.models.registry import build_model, param_groups
from involeaf.utils import device as dev
from involeaf.utils.logging import RunLogger, get_logger
from involeaf.utils.metrics import AverageMeter
from involeaf.utils.seed import seed_everything, worker_init_fn

log = get_logger()


def build_loaders(cfg: dict) -> tuple[DataLoader, DataLoader, SplitDataset]:
    split_file = cfg["split_file"]
    size = int(cfg.get("image_size", 224))
    batch_size = int(cfg["batch_size"])
    workers = int(cfg.get("workers", 4))

    train_ds = SplitDataset(split_file, "train", transform=train_transform(size))
    val_ds = SplitDataset(split_file, "val", transform=eval_transform(size))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg.get("eval_batch_size", batch_size * 2)),
        shuffle=False, num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
    )
    return train_loader, val_loader, train_ds


def accumulation_steps(cfg: dict) -> int:
    """How many micro-batches make up one optimiser step."""
    effective = int(cfg.get("effective_batch", 64))
    batch_size = int(cfg["batch_size"])
    if effective % batch_size != 0:
        raise ValueError(
            f"effective_batch ({effective}) must be a multiple of batch_size "
            f"({batch_size}) so the effective batch is exact"
        )
    return effective // batch_size


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """Cosine decay with linear warmup, stepped per optimiser step."""
    epochs = int(cfg.get("epochs", 20))
    warmup_epochs = float(cfg.get("warmup_epochs", 2))
    total = max(1, epochs * steps_per_epoch)
    warmup = int(warmup_epochs * steps_per_epoch)
    min_factor = float(cfg.get("min_lr_factor", 0.01))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model, loader, optimizer, scheduler, scaler, criterion,
    device, amp_dtype, accum, grad_clip, limit_batches=None,
) -> dict:
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(loader):
        if limit_batches is not None and step >= limit_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with dev.autocast(device, amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)

        # Scale so the gradient equals the mean over the full effective batch.
        scaled = loss / accum
        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        if (step + 1) % accum == 0:
            if scaler is not None:
                if grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        batch_n = labels.size(0)
        loss_meter.update(loss.item(), batch_n)
        acc_meter.update(
            (logits.float().argmax(1) == labels).float().mean().item(), batch_n
        )

    return {"train_loss": loss_meter.avg, "train_acc": acc_meter.avg}


def train(cfg: dict, run_name: str | None = None, limit_batches: int | None = None) -> dict:
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    device = dev.resolve_device(cfg.get("device", "auto"))
    amp_dtype = dev.amp_dtype_for(device, cfg.get("amp", "auto"))
    device_info = dev.describe(device, amp_dtype).as_dict()
    log.info(f"device: {device_info['name']} ({device_info['backend']}, amp={device_info['amp_dtype']})")

    train_loader, val_loader, train_ds = build_loaders(cfg)
    num_classes = train_ds.num_classes

    model = build_model(cfg, num_classes).to(device)
    accum = accumulation_steps(cfg)
    steps_per_epoch = max(1, len(train_loader) // accum)

    optimizer = torch.optim.AdamW(param_groups(model, cfg))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type) if dev.needs_grad_scaler(amp_dtype) else None
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.1)))

    run_name = run_name or f"{cfg.get('name', 'run')}_seed{seed}"
    logger = RunLogger(run_name, cfg, device_info)
    logger.set("num_classes", num_classes)
    logger.set("class_names", train_ds.class_names)
    logger.set("effective_batch", int(cfg.get("effective_batch", 64)))
    logger.set("accum_steps", accum)
    logger.set("params_total", sum(p.numel() for p in model.parameters()))
    logger.set("model_meta", getattr(model, "involeaf_meta", {}))

    ckpt_dir = Path(cfg.get("checkpoint_dir", "results/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{run_name}.pth"

    best_metric, best_epoch = -1.0, -1
    epochs = int(cfg.get("epochs", 20))
    dev.reset_peak_memory(device)

    for epoch in range(epochs):
        stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion,
            device, amp_dtype, accum, float(cfg.get("grad_clip", 1.0)), limit_batches,
        )
        val = evaluate(model, val_loader, device, train_ds.class_names, amp_dtype)
        logger.log_epoch(
            epoch, **stats, val_acc=val["accuracy"], val_macro_f1=val["macro_f1"],
            lr=optimizer.param_groups[0]["lr"],
        )

        # Checkpoint selection uses macro-F1 on the *validation* split. The test split
        # is not read until the final evaluation.
        if val["macro_f1"] > best_metric:
            best_metric, best_epoch = val["macro_f1"], epoch
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                 "val_macro_f1": best_metric, "class_names": train_ds.class_names},
                ckpt_path,
            )

    logger.set("best_val_macro_f1", best_metric)
    logger.set("best_epoch", best_epoch)
    logger.set("checkpoint", str(ckpt_path))
    logger.set("peak_train_memory_gb", dev.peak_memory_gb(device))
    logger.save()

    return {"checkpoint": str(ckpt_path), "best_val_macro_f1": best_metric,
            "best_epoch": best_epoch, "run_name": run_name}
