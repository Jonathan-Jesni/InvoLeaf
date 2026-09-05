"""Dataset built from a committed split file.

Splits live in ``splits/*.json`` and are version-controlled, so every run -- across
machines, across the September RTX 5050 and the October W7800 -- sees exactly the same
train/val/test partition. Nothing regenerates a split at training time.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from involeaf.data.transforms import DeterministicCorruption
from involeaf.utils.paths import resolve_root


class SplitDataset(Dataset):
    """Reads (relative_path, label) pairs for one split of one dataset.

    ``DeterministicCorruption`` transforms receive the sample index so the corruption
    applied to a given image is fixed; ordinary transforms are called normally.
    """

    def __init__(
        self,
        split_file: str | Path,
        split: str,
        transform=None,
        root: str | Path | None = None,
    ) -> None:
        self.split_file = Path(split_file)
        meta = json.loads(self.split_file.read_text(encoding="utf-8"))

        if split not in meta["splits"]:
            raise KeyError(
                f"split {split!r} not in {self.split_file}; "
                f"available: {sorted(meta['splits'])}"
            )

        self.root = Path(root) if root is not None else resolve_root(meta["root"])
        self.class_names: list[str] = meta["class_names"]
        self.samples: list[tuple[str, int]] = [
            (rel, int(label)) for rel, label in meta["splits"][split]
        ]
        self.split = split
        self.transform = transform
        self.dataset_name = meta.get("name", self.split_file.stem)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def class_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(self.class_names, 0)
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return counts

    def __getitem__(self, index: int):
        rel, label = self.samples[index]
        with Image.open(self.root / rel) as img:
            img = img.convert("RGB")

        if self.transform is None:
            return img, label
        if isinstance(self.transform, DeterministicCorruption):
            return self.transform(img, index=index), label
        return self.transform(img), label
