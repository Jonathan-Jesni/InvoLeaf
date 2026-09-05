"""Build fixed, committed train/val/test splits.

Sources
-------
PlantVillage  https://github.com/spMohanty/PlantVillage-Dataset  (git clone, ~1.6 GB,
              no credentials). Class folders live under ``raw/color/``.
PlantDoc      https://github.com/pratikkayal/PlantDoc-Dataset     (git clone). Real
              field photographs, used only as an unseen cross-domain test set.

Split policy
------------
Stratified 70/15/15 with seed 42. AgriTL-ViT uses 70/30 with no validation split, which
leaves no honest way to select a checkpoint -- selecting on the test set inflates the
reported number. We split their 30% into 15% validation and 15% test; the test partition
is untouched until final evaluation. The difference is documented rather than hidden.

Note on counts: raw PlantVillage tomato holds ~18,160 images across 10 classes, while
AgriTL-ViT reports 14,529. We use the full set and report the discrepancy rather than
silently sampling down to match a number we cannot reconstruct.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from involeaf.utils.paths import to_portable

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

# PlantVillage crops. Note which of AgriTL-ViT's datasets are reachable from here:
# tomato and maize are their two main crops; apple and grape are their cross-dataset
# tests; potato and strawberry are their region-shift tests. Their third main crop,
# Rice (11,810 images / 9 classes), is NOT in PlantVillage -- PlantVillage covers 14
# crops and rice is not one of them, so it would need a separate Kaggle source.
CROP_PREFIXES = {
    "tomato": "Tomato___",
    "maize": "Corn_(maize)___",
    "potato": "Potato___",
    "apple": "Apple___",
    "grape": "Grape___",
    "strawberry": "Strawberry___",
    "peach": "Peach___",
    "cherry": "Cherry_(including_sour)___",
    "pepper": "Pepper,_bell___",
}

# PlantDoc folder names -> PlantVillage class names, for the cross-domain test. Only
# classes present in both are usable; everything else is dropped and reported.
PLANTDOC_TO_PLANTVILLAGE = {
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": (
        "Tomato___Spider_mites Two-spotted_spider_mite"
    ),
    "Tomato leaf": "Tomato___healthy",
}


def _list_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix in IMAGE_SUFFIXES
    )


def scan_class_folders(root: Path, prefix: str | None = None) -> dict[str, list[Path]]:
    """Map class-folder name -> image paths, optionally filtered to one crop."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} does not exist. Clone the dataset first, e.g.\n"
            f"  git clone https://github.com/spMohanty/PlantVillage-Dataset data/PlantVillage-Dataset"
        )
    classes = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if prefix and not folder.name.startswith(prefix):
            continue
        images = _list_images(folder)
        if images:
            classes[folder.name] = images
    if not classes:
        raise RuntimeError(f"no class folders found under {root} (prefix={prefix!r})")
    return classes


def stratified_split(
    classes: dict[str, list[Path]],
    root: Path,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[list[str], dict[str, list[tuple[str, int]]]]:
    """Per-class shuffle then proportional cut, so every split keeps the class balance."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    class_names = sorted(classes)
    splits: dict[str, list[tuple[str, int]]] = defaultdict(list)
    rng = random.Random(seed)

    for label, name in enumerate(class_names):
        paths = sorted(classes[name])           # sort first: filesystem order is not stable
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        chunks = {
            "train": paths[:n_train],
            "val": paths[n_train : n_train + n_val],
            "test": paths[n_train + n_val :],
        }
        for split, items in chunks.items():
            for p in items:
                splits[split].append((p.relative_to(root).as_posix(), label))

    for split in splits:
        splits[split].sort()
    return class_names, dict(splits)


def write_split_file(
    out_path: Path,
    name: str,
    root: Path,
    class_names: list[str],
    splits: dict[str, list[tuple[str, int]]],
    extra: dict | None = None,
) -> Path:
    payload = {
        "name": name,
        # Repo-relative where possible: this file is committed and must load on the
        # October ROCm machine as well as here.
        "root": to_portable(root),
        "class_names": class_names,
        "counts": {s: len(v) for s, v in splits.items()},
        "splits": splits,
    }
    if extra:
        payload.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return out_path


def prepare_plantvillage(
    data_root: Path,
    crop: str,
    out_path: Path,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict:
    """Build the split file for one PlantVillage crop."""
    if crop not in CROP_PREFIXES:
        raise ValueError(f"unknown crop {crop!r}; expected one of {sorted(CROP_PREFIXES)}")

    color_root = data_root / "raw" / "color"
    if not color_root.is_dir():          # tolerate a flattened copy
        color_root = data_root

    classes = scan_class_folders(color_root, prefix=CROP_PREFIXES[crop])
    class_names, splits = stratified_split(classes, color_root, ratios, seed)
    write_split_file(
        out_path, f"plantvillage_{crop}", color_root, class_names, splits,
        extra={"source": "PlantVillage", "crop": crop, "seed": seed, "ratios": list(ratios)},
    )
    return {
        "crop": crop,
        "classes": len(class_names),
        "total": sum(len(v) for v in classes.values()),
        "counts": {s: len(v) for s, v in splits.items()},
        "per_class": {k: len(v) for k, v in classes.items()},
        "out": str(out_path),
    }


def prepare_plantdoc(
    data_root: Path,
    out_path: Path,
    reference_split: Path,
) -> dict:
    """Build a cross-domain *test-only* split, relabelled into PlantVillage classes.

    Labels must come from the training dataset's class list, or the predicted indices
    would refer to a different class ordering and every cross-domain number would be
    silently wrong.
    """
    reference = json.loads(Path(reference_split).read_text(encoding="utf-8"))
    class_names: list[str] = reference["class_names"]
    index = {name: i for i, name in enumerate(class_names)}

    samples: list[tuple[str, int]] = []
    matched, skipped = {}, []
    for sub in ("train", "test"):          # we use every PlantDoc image as test data
        sub_root = data_root / sub
        if not sub_root.is_dir():
            continue
        for folder in sorted(p for p in sub_root.iterdir() if p.is_dir()):
            target = PLANTDOC_TO_PLANTVILLAGE.get(folder.name)
            if target is None or target not in index:
                skipped.append(folder.name)
                continue
            images = _list_images(folder)
            matched[folder.name] = matched.get(folder.name, 0) + len(images)
            for p in images:
                samples.append((p.relative_to(data_root).as_posix(), index[target]))

    if not samples:
        raise RuntimeError(
            f"no PlantDoc folders mapped into {reference_split}. Check the clone layout "
            f"and PLANTDOC_TO_PLANTVILLAGE."
        )

    samples.sort()
    write_split_file(
        out_path, "plantdoc_tomato", data_root, class_names, {"test": samples},
        extra={
            "source": "PlantDoc",
            "role": "cross-domain test only",
            "label_space_from": str(reference_split),
            "unmapped_folders": sorted(set(skipped)),
        },
    )
    return {
        "total": len(samples),
        "matched_folders": matched,
        "unmapped": sorted(set(skipped)),
        "out": str(out_path),
    }
