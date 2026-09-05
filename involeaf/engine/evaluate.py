"""Evaluation loop and the robustness / resolution / cross-domain sweeps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from involeaf.data.datasets import SplitDataset
from involeaf.data.transforms import CORRUPTIONS, build_eval_transform
from involeaf.utils import device as dev
from involeaf.utils.metrics import compute_metrics


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with dev.autocast(device, amp_dtype):
            logits = model(images)
        preds.append(logits.float().argmax(dim=1).cpu())
        targets.append(labels)
    return torch.cat(targets).numpy(), torch.cat(preds).numpy()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    amp_dtype: torch.dtype | None = None,
) -> dict:
    y_true, y_pred = predict(model, loader, device, amp_dtype)
    return compute_metrics(y_true, y_pred, class_names)


def _loader(dataset: SplitDataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def robustness_sweep(
    model: nn.Module,
    split_file: str | Path,
    device: torch.device,
    split: str = "test",
    batch_size: int = 32,
    workers: int = 4,
    amp_dtype: torch.dtype | None = None,
    seed: int = 0,
) -> dict:
    """Evaluate one set of trained weights under each corruption.

    No retraining: the same checkpoint is tested against progressively harsher inputs.
    This is the AgriTL-ViT Table 17 protocol, where their reported accuracy falls from
    98.5% to 74.9% on tomato under combined corruption.
    """
    results = {}
    for corruption in CORRUPTIONS:
        transform = build_eval_transform(corruption=corruption, seed=seed)
        ds = SplitDataset(split_file, split, transform=transform)
        results[corruption] = evaluate(
            model, _loader(ds, batch_size, workers), device, ds.class_names, amp_dtype
        )
    return results


def resolution_sweep(
    model: nn.Module,
    split_file: str | Path,
    device: torch.device,
    resolutions: tuple[int, ...] = (128, 200, 224),
    split: str = "test",
    batch_size: int = 32,
    workers: int = 4,
    amp_dtype: torch.dtype | None = None,
) -> dict:
    """Information-loss protocol: downscale to the target, restore to 224.

    See ``involeaf/data/transforms.py`` for why a native-resolution 200x200 evaluation
    is impossible on a patch-16 ViT.
    """
    results = {}
    for res in resolutions:
        low = None if res == 224 else res
        transform = build_eval_transform(corruption="clean", low_res=low)
        ds = SplitDataset(split_file, split, transform=transform)
        results[str(res)] = evaluate(
            model, _loader(ds, batch_size, workers), device, ds.class_names, amp_dtype
        )
        results[str(res)]["protocol"] = (
            "native" if res == 224 else f"downscale to {res}, restore to 224"
        )
    return results


def cross_domain_eval(
    model: nn.Module,
    split_file: str | Path,
    device: torch.device,
    batch_size: int = 32,
    workers: int = 4,
    amp_dtype: torch.dtype | None = None,
) -> dict:
    """Train on PlantVillage (lab), test on PlantDoc (field).

    The split file must have been built against the training dataset's class list, so
    that label indices refer to the same classes.
    """
    transform = build_eval_transform(corruption="clean")
    ds = SplitDataset(split_file, "test", transform=transform)
    out = evaluate(model, _loader(ds, batch_size, workers), device, ds.class_names, amp_dtype)
    out["n_images"] = len(ds)
    return out
