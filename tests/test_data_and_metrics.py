"""Tests for splits, deterministic corruption, config loading and the metrics guard."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from involeaf.data.datasets import SplitDataset
from involeaf.data.prepare import stratified_split, write_split_file
from involeaf.data.transforms import (
    DeterministicCorruption, build_eval_transform, low_resolution_transform,
    native_resolution_transform,
)
from involeaf.utils.config import load_config
from involeaf.utils.metrics import MetricConsistencyError, compute_metrics


def textured_image(seed: int = 0) -> Image.Image:
    """A noisy image, not a flat colour.

    GaussianBlur of a uniform image is that same image, so a flat-colour fixture cannot
    detect whether blur was applied at all. Every corruption test needs real texture.
    """
    rng = np.random.default_rng(seed)
    return Image.fromarray(
        rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8), mode="RGB"
    )


@pytest.fixture
def fake_dataset(tmp_path):
    """Three classes with deliberately unequal counts, to test stratification."""
    root = tmp_path / "raw"
    counts = {"Tomato___healthy": 20, "Tomato___Early_blight": 13, "Tomato___Late_blight": 7}
    classes = {}
    for name, n in counts.items():
        folder = root / name
        folder.mkdir(parents=True)
        paths = []
        for i in range(n):
            p = folder / f"img_{i:03d}.jpg"
            Image.new("RGB", (32, 32), (i * 8 % 256, 100, 150)).save(p)
            paths.append(p)
        classes[name] = paths
    return root, classes, counts


# --------------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------------- #

def test_split_is_deterministic(fake_dataset):
    """Two runs of the same seed must give byte-identical splits.

    Filesystem iteration order is not stable across machines, so the split sorts before
    shuffling. Without that, the September and October machines would train on different
    partitions while both claiming seed 42.
    """
    root, classes, _ = fake_dataset
    a = stratified_split(classes, root, seed=42)
    b = stratified_split(classes, root, seed=42)
    assert a == b


def test_split_partitions_without_overlap(fake_dataset):
    """No image may appear in more than one split - the classic silent leak."""
    root, classes, counts = fake_dataset
    _, splits = stratified_split(classes, root, seed=42)

    seen = [rel for split in splits.values() for rel, _ in split]
    assert len(seen) == len(set(seen)), "an image appears in more than one split"
    assert len(seen) == sum(counts.values()), "images were lost or duplicated"


def test_split_is_stratified(fake_dataset):
    """Every class must appear in every split, roughly in proportion."""
    root, classes, _ = fake_dataset
    class_names, splits = stratified_split(classes, root, seed=42)
    for split, items in splits.items():
        present = {label for _, label in items}
        assert present == set(range(len(class_names))), (
            f"split {split!r} is missing classes {set(range(len(class_names))) - present}"
        )


def test_different_seeds_give_different_splits(fake_dataset):
    root, classes, _ = fake_dataset
    assert stratified_split(classes, root, seed=42) != stratified_split(classes, root, seed=7)


def test_dataset_reads_from_a_split_file(tmp_path, fake_dataset):
    root, classes, _ = fake_dataset
    class_names, splits = stratified_split(classes, root, seed=42)
    split_file = tmp_path / "split.json"
    write_split_file(split_file, "fake", root, class_names, splits)

    ds = SplitDataset(split_file, "train", transform=build_eval_transform())
    assert ds.num_classes == 3
    image, label = ds[0]
    assert image.shape == (3, 224, 224)
    assert 0 <= label < 3
    assert sum(ds.class_counts().values()) == len(ds)


def test_dataset_rejects_an_unknown_split(tmp_path, fake_dataset):
    root, classes, _ = fake_dataset
    class_names, splits = stratified_split(classes, root, seed=42)
    split_file = tmp_path / "split.json"
    write_split_file(split_file, "fake", root, class_names, splits)
    with pytest.raises(KeyError):
        SplitDataset(split_file, "nonexistent")


def test_committed_splits_use_portable_roots():
    """Split files are committed, so an absolute path in one is a portability bug.

    A root like "D:/Projects/.../data/..." loads fine here and fails on the October
    ROCm machine, on WSL, and on any collaborator's checkout.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    files = sorted(repo.glob("splits/*.json"))
    for path in files:
        root = json.loads(path.read_text(encoding="utf-8"))["root"]
        assert not Path(root).is_absolute(), (
            f"{path.name} stores an absolute root ({root!r}); regenerate it with "
            f"scripts/prepare_data.py"
        )
        assert ":" not in root, f"{path.name} stores a drive-lettered root ({root!r})"


def test_split_roots_resolve_against_the_repository(tmp_path, fake_dataset):
    from involeaf.utils.paths import REPO_ROOT, resolve_root, to_portable

    inside = REPO_ROOT / "data" / "whatever"
    assert to_portable(inside) == "data/whatever"
    assert resolve_root("data/whatever") == inside


# --------------------------------------------------------------------------------- #
# Corruptions
# --------------------------------------------------------------------------------- #

@pytest.mark.parametrize("corruption", ["colorjitter", "blur", "erasing", "combined"])
def test_corruption_is_reproducible_for_a_given_index(corruption):
    """A robustness number that moves between runs is not a measurement."""
    img = textured_image()
    op = DeterministicCorruption(corruption, seed=0)
    assert torch.equal(op(img, index=5), op(img, index=5))


@pytest.mark.parametrize("corruption", ["colorjitter", "erasing", "combined"])
def test_corruption_differs_between_indices(corruption):
    """Otherwise every image would receive an identical perturbation."""
    img = textured_image()
    op = DeterministicCorruption(corruption, seed=0)
    assert not torch.equal(op(img, index=1), op(img, index=2))


def test_corruption_does_not_disturb_the_global_rng():
    """It forks the RNG state; leaking a seed would silently change data ordering."""
    img = textured_image()
    op = DeterministicCorruption("combined", seed=0)
    torch.manual_seed(1234)
    before = torch.randn(4)
    torch.manual_seed(1234)
    op(img, index=99)
    assert torch.equal(before, torch.randn(4))


def test_corruption_actually_changes_the_image():
    img = textured_image()
    clean = build_eval_transform("clean")(img)
    for name in ("colorjitter", "blur", "erasing", "combined"):
        assert not torch.allclose(DeterministicCorruption(name)(img, index=0), clean)


def test_native_resolution_rejects_sizes_not_divisible_by_the_patch_size():
    """200x200 cannot be tokenised by a patch-16 ViT; the error must say so."""
    with pytest.raises(ValueError, match="divisible"):
        native_resolution_transform(200, patch_size=16)
    assert native_resolution_transform(192, patch_size=16) is not None


def test_low_resolution_protocol_restores_the_input_size():
    """Information loss at a fixed input size, so the ViT still accepts the tensor."""
    img = textured_image()
    assert low_resolution_transform(128)(img).shape == (3, 224, 224)
    assert low_resolution_transform(200)(img).shape == (3, 224, 224)


# --------------------------------------------------------------------------------- #
# Metrics guard
# --------------------------------------------------------------------------------- #

def test_metrics_are_internally_consistent():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    m = compute_metrics(y_true, y_pred, ["a", "b", "c"])
    for stats in m["per_class"].values():
        p, r, f = stats["precision"], stats["recall"], stats["f1"]
        assert min(p, r) - 1e-6 <= f <= max(p, r) + 1e-6


def test_metrics_guard_rejects_an_impossible_f1():
    """The exact triple published for Tomato Early_blight in AgriTL-ViT Table 5."""
    from involeaf.utils.metrics import _check_consistency

    with pytest.raises(MetricConsistencyError, match="harmonic mean"):
        _check_consistency([0.977], [0.958], [0.857], ["Tomato___Early_blight"])


def test_support_is_derived_from_the_labels():
    """Support cannot be copied between tables if it is computed from y_true."""
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 2])
    m = compute_metrics(y_true, y_pred, ["a", "b", "c"])
    assert [s["support"] for s in m["per_class"].values()] == [3, 2, 1]
    assert sum(s["support"] for s in m["per_class"].values()) == len(y_true)


# --------------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------------- #

def test_config_inherits_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("epochs: 20\nlr: 1.0e-4\nbatch_size: 16\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\nname: child\nbatch_size: 8\n")

    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["epochs"] == 20        # inherited
    assert cfg["batch_size"] == 8     # overridden
    assert cfg["lr"] == 1e-4          # parsed as a float, not a string

    cfg = load_config(tmp_path / "child.yaml", ["kernel_size=3", "delta_init=false"])
    assert cfg["kernel_size"] == 3 and cfg["delta_init"] is False


def test_real_configs_all_load():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    for path in sorted(repo.glob("configs/**/*.yaml")):
        if path.name == "base.yaml":
            continue
        cfg = load_config(path)
        assert "name" in cfg and "backbone" in cfg, f"{path} is incomplete"
        assert cfg["effective_batch"] % cfg["batch_size"] == 0, (
            f"{path}: effective_batch must be a multiple of batch_size"
        )
