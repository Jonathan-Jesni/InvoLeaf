"""Train/eval transforms and the test-time corruption suite.

The corruption suite reproduces the protocol of AgriTL-ViT (ESWA 2025) Table 17 --
ColorJitter, GaussianBlur, RandomErasing, individually and combined -- so that our
numbers sit directly alongside theirs. Corruptions are applied at *test* time only, to
weights trained on clean data; no retraining is involved, which makes the whole
robustness stage nearly free.

Determinism
-----------
RandomErasing and ColorJitter are stochastic. A robustness number that changes between
runs is not a measurement, so ``DeterministicCorruption`` derives a per-sample seed from
the sample index. The same image therefore receives the same corruption on every run and
for every model, which is what makes cross-model comparison legitimate.

Resolution protocol
-------------------
AgriTL-ViT reports a 200x200 low-resolution test. 200 is not divisible by the patch size
16, so it cannot be a native-resolution evaluation of a patch-16 ViT: the patch embedding
rejects the input outright. We therefore read "low resolution" as information loss at a
fixed input size -- downscale to the target, then resize back to 224 -- which is both the
only coherent reading and the more meaningful test. ``native_resolution_transform`` is
provided separately for genuine native-resolution evaluation, and it accepts only
multiples of the patch size.
"""

from __future__ import annotations

import torch
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_SIZE = 224


def train_transform(size: int = DEFAULT_SIZE) -> T.Compose:
    """Mild augmentation: leaf disease is a fine-grained, colour-sensitive task.

    Colour distortion is deliberately omitted here -- lesion colour is the signal, and
    jittering it during training would both hurt accuracy and contaminate the
    ColorJitter robustness test, which is only meaningful if the model never trained
    on colour-shifted data.
    """
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size: int = DEFAULT_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def native_resolution_transform(size: int, patch_size: int = 16) -> T.Compose:
    """Genuine native-resolution evaluation. Requires ``size % patch_size == 0``."""
    if size % patch_size != 0:
        raise ValueError(
            f"native-resolution evaluation needs a size divisible by the patch size "
            f"({patch_size}); {size} is not. Use low_resolution_transform({size}) for "
            f"the information-loss protocol instead."
        )
    return eval_transform(size)


def low_resolution_transform(
    low: int, size: int = DEFAULT_SIZE, patch_size: int = 16
) -> T.Compose:
    """Downscale to ``low`` then restore to ``size`` -- information loss, fixed input."""
    del patch_size  # accepted for symmetry with native_resolution_transform
    return T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.Resize(low),
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------------- #
# Corruptions
# --------------------------------------------------------------------------------- #

def _color_jitter() -> T.ColorJitter:
    """Lighting and white-balance drift, as when photographing in a field."""
    return T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)


def _gaussian_blur() -> T.GaussianBlur:
    """Focus error from a hand-held phone camera."""
    return T.GaussianBlur(kernel_size=5, sigma=(0.6, 2.0))


def _random_erasing() -> T.RandomErasing:
    """Occlusion by another leaf, a hand, or debris. Operates on tensors."""
    return T.RandomErasing(p=1.0, scale=(0.05, 0.20), ratio=(0.3, 3.3), value="random")


CORRUPTIONS = ("clean", "colorjitter", "blur", "erasing", "combined")


class DeterministicCorruption:
    """Apply one named corruption with a seed derived from the sample index.

    Wrapping the transform rather than seeding globally keeps the corruption identical
    across models and runs while leaving the rest of the pipeline untouched.
    """

    def __init__(self, name: str, size: int = DEFAULT_SIZE, seed: int = 0) -> None:
        if name not in CORRUPTIONS:
            raise ValueError(f"unknown corruption {name!r}; expected one of {CORRUPTIONS}")
        self.name = name
        self.seed = seed

        self.resize = T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size)])
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

        self.pil_ops: list = []
        self.tensor_ops: list = []
        if name in ("colorjitter", "combined"):
            self.pil_ops.append(_color_jitter())
        if name in ("blur", "combined"):
            self.pil_ops.append(_gaussian_blur())
        if name in ("erasing", "combined"):
            self.tensor_ops.append(_random_erasing())

    def __call__(self, img, index: int | None = None):
        if index is not None:
            # Fork the RNG so the corruption is reproducible without disturbing the
            # global stream used by shuffling and augmentation elsewhere.
            gen_state = torch.get_rng_state()
            torch.manual_seed(self.seed * 1_000_003 + index)
        try:
            out = self.resize(img)
            for op in self.pil_ops:
                out = op(out)
            out = self.to_tensor(out)
            for op in self.tensor_ops:
                out = op(out)
            return self.normalize(out)
        finally:
            if index is not None:
                torch.set_rng_state(gen_state)


def build_eval_transform(
    corruption: str = "clean",
    low_res: int | None = None,
    size: int = DEFAULT_SIZE,
    seed: int = 0,
):
    """Single entry point used by the evaluation scripts."""
    if corruption == "clean" and low_res is None:
        return eval_transform(size)
    if corruption == "clean":
        return low_resolution_transform(low_res, size)
    if low_res is not None:
        raise ValueError("combining a corruption with a low-resolution test is not part "
                         "of the protocol; run them separately")
    return DeterministicCorruption(corruption, size=size, seed=seed)
