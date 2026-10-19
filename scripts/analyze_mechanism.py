#!/usr/bin/env python
"""Why does involution trade colour robustness for occlusion robustness?

    python scripts/analyze_mechanism.py                 # seed-42 checkpoints
    python scripts/analyze_mechanism.py --smoke         # random weights, 8 images

Inference only, on the same test images and the same seeded corruptions as
eval_robustness.py, for ResNet-50, ViT-B/16 and InvoLeaf. Writes two files:

results/mechanism_seed<N>.json
    Representation drift: for every stage of every model and every corruption, the
    relative change ||f(corrupted) - f(clean)|| / ||f(clean)|| of the patch-token
    features, averaged over images. Under occlusion the drift is also split into tokens
    inside and outside the erased box, which measures how far the damage spreads. For
    InvoLeaf it also records the drift of the generated involution kernels and of the
    involution's input, so kernel drift / input drift says whether generating kernels
    from the image's own content amplifies a perturbation.

results/interp_seed<N>.npz
    Grad-CAM maps (for the predicted class) and InvoLeaf kernel maps on one test image
    per class, clean, colour-shifted and occluded, plus the images themselves. Small
    enough to commit; scripts/plot_mechanism.py draws the figures from it.

The hypothesis being tested, stated before the numbers exist: involution kernels are
generated from each token's own colour features, so a global colour shift changes every
kernel, while occlusion changes only the kernels near the erased box. If so, InvoLeaf's
kernel drift should be large under colour shift, and small outside the box under
occlusion, with ViT-B's attention spreading occlusion damage further than involution.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from involeaf.data.datasets import SplitDataset  # noqa: E402
from involeaf.data.transforms import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, DeterministicCorruption, eval_transform,
)
from involeaf.models.registry import build_model  # noqa: E402
from involeaf.ops.involution import Involution2d  # noqa: E402
from involeaf.utils import device as dev  # noqa: E402
from involeaf.utils.config import load_config  # noqa: E402
from involeaf.utils.logging import get_logger  # noqa: E402

log = get_logger()

MODELS = ["resnet50", "vit_b16_baseline", "involeaf_b16"]
CONDS = ["colorjitter", "blur", "erasing", "combined"]
VIZ_CONDS = ["clean", "colorjitter", "erasing"]
SPLIT_FILE = "splits/plantvillage_tomato.json"


# --- models ---------------------------------------------------------------------------

def load_model(name: str, args, device: torch.device, num_classes: int):
    """A trained checkpoint, or (with --smoke) the same architecture untrained."""
    if args.smoke:
        cfg = load_config(REPO / "configs" / f"{name}.yaml")
        cfg["pretrained"] = False
        source = "random init (smoke)"
        state = None
    else:
        path = Path(args.ckpt_dir) / f"{name}_seed{args.seed}.pth"
        if not path.is_absolute():
            path = REPO / path
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = dict(ckpt["cfg"], pretrained=False)
        source = f"{path} (epoch {ckpt['epoch']}, val macro-F1 {ckpt['val_macro_f1']:.4f})"
        state = ckpt["model"]
    if cfg.get("arch") == "involeaf":
        # shift and unfold compute the same function (tests/test_involution.py); seed 42
        # was trained with shift. unfold is faster on the 48 GB card, shift fits 8 GB.
        cfg["impl"] = "shift" if args.smoke else "unfold"
    model = build_model(cfg, num_classes)
    if state is not None:
        model.load_state_dict(state)
    log.info(f"{name}: {source}")
    return model.to(device).eval()


def stages(model) -> list[tuple[str, torch.nn.Module]]:
    if hasattr(model, "blocks"):
        return [(f"block{i}", b) for i, b in enumerate(model.blocks)]
    return [(n, getattr(model, n)) for n in ("layer1", "layer2", "layer3", "layer4")]


def gradcam_layer(model) -> torch.nn.Module:
    return model.blocks[-1].norm1 if hasattr(model, "blocks") else model.layer4


def to_tokens(t: torch.Tensor, n_prefix: int) -> torch.Tensor:
    """(B, N, C) token sequence or (B, C, H, W) map -> (B, P, C) spatial tokens."""
    if t.dim() == 3:
        return t[:, n_prefix:]
    return t.flatten(2).transpose(1, 2)


def to_grid(t: torch.Tensor, n_prefix: int) -> torch.Tensor:
    """(B, N, C) or (B, C, H, W) -> (B, C, H, W)."""
    if t.dim() == 4:
        return t
    tok = t[:, n_prefix:]
    side = math.isqrt(tok.shape[1])
    return tok.transpose(1, 2).reshape(tok.shape[0], tok.shape[2], side, side)


class Capture:
    """Forward hooks that record every stage output and, for InvoLeaf, every
    involution's input and generated kernel (the raw ``span`` output)."""

    def __init__(self, model) -> None:
        self.n_prefix = getattr(model, "num_prefix_tokens", 0)
        self.feats: dict[str, torch.Tensor] = {}
        self.handles = []
        for name, m in stages(model):
            self.handles.append(m.register_forward_hook(self._store(name)))
        self.involutions = [m for m in model.modules() if isinstance(m, Involution2d)]
        for i, m in enumerate(self.involutions):
            self.handles.append(m.register_forward_hook(self._store_input(f"inv_in{i}")))
            self.handles.append(m.span.register_forward_hook(self._store(f"kern{i}")))

    def _store(self, key):
        def hook(_m, _inp, out):
            self.feats[key] = out.detach()
        return hook

    def _store_input(self, key):
        def hook(_m, inp, _out):
            self.feats[key] = inp[0].detach()
        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run(model, cap: Capture, x: torch.Tensor):
    cap.feats = {}
    feats = model.forward_features(x)
    pre = model.forward_head(feats, pre_logits=True)
    logits = model.forward_head(feats)
    return pre.float(), logits.argmax(1), dict(cap.feats)


# --- drift ----------------------------------------------------------------------------

def rel_drift(clean: torch.Tensor, corr: torch.Tensor,
              mask: torch.Tensor | None = None) -> torch.Tensor:
    """Per-image ||corr - clean|| / ||clean|| over (tokens, channels).

    ``mask`` (B, P) restricts both norms to the selected tokens; an image with no
    selected token gets NaN, so it drops out of the nan-mean.
    """
    d = (corr - clean).float()
    a = clean.float()
    if mask is not None:
        m = mask.unsqueeze(-1).float()
        d, a = d * m, a * m
        empty = mask.sum(1) == 0
    num = d.pow(2).sum((1, 2)).sqrt()
    den = a.pow(2).sum((1, 2)).sqrt().clamp_min(1e-12)
    out = num / den
    if mask is not None:
        out[empty] = float("nan")
    return out


def token_mask(pixel_mask: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """(B, 224, 224) erased-pixel mask -> (B, P): a token is 'inside' if any of its
    receptive-field pixels was erased."""
    side = math.isqrt(n_tokens)
    pooled = F.adaptive_max_pool2d(pixel_mask.float().unsqueeze(1), (side, side))
    return pooled.flatten(1) > 0


def kernel_tokens(raw: torch.Tensor) -> torch.Tensor:
    """span output (B, G*K*K, H, W) -> (B, HW, G*K*K)."""
    return raw.flatten(2).transpose(1, 2)


# --- interpretability -----------------------------------------------------------------

def gradcam(model, x: torch.Tensor, n_prefix: int, chunk: int = 5):
    """Grad-CAM for the predicted class. Returns (cams (B, h, w) in [0, 1], preds)."""
    layer = gradcam_layer(model)
    cams, preds = [], []
    for start in range(0, x.shape[0], chunk):
        xb = x[start:start + chunk]
        store = {}
        handle = layer.register_forward_hook(lambda _m, _i, o: store.__setitem__("a", o))
        try:
            with torch.enable_grad():
                logits = model(xb)
                pred = logits.argmax(1)
                score = logits.gather(1, pred[:, None]).sum()
                grad, = torch.autograd.grad(score, store["a"])
        finally:
            handle.remove()
        act = to_grid(store["a"].detach(), n_prefix).float()
        grad = to_grid(grad.detach(), n_prefix).float()
        weights = grad.mean((2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(1))
        cam = cam / cam.flatten(1).max(1).values.clamp_min(1e-12)[:, None, None]
        cams.append(cam.cpu())
        preds.append(pred.cpu())
    return torch.cat(cams).numpy(), torch.cat(preds).numpy()


def kernel_maps(raw: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Share of each location's kernel energy that sits off the centre tap, averaged
    over groups: 0 = the kernel only passes the token through (its delta
    initialisation), 1 = it mixes neighbours only. Shape (B, H, W)."""
    b, _, h, w = raw.shape
    k = raw.float().view(b, -1, kernel_size * kernel_size, h, w)
    total = k.pow(2).sum(2).clamp_min(1e-12)
    centre = k[:, :, (kernel_size * kernel_size) // 2].pow(2)
    return ((total - centre) / total).mean(1)


def to_uint8(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    img = (x.cpu() * std + mean).clamp(0, 1)
    return (img.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)


# --- main -----------------------------------------------------------------------------

def load_images(ds: SplitDataset, indices, clean_tf, corr_tf):
    """Every condition for these test indices, with the evaluation's exact seeds."""
    pil = [ds[i] for i in indices]
    labels = torch.tensor([lab for _, lab in pil])
    x = {"clean": torch.stack([clean_tf(img) for img, _ in pil])}
    for c, tf in corr_tf.items():
        x[c] = torch.stack([tf(img, index=int(i)) for (img, _), i in zip(pil, indices)])
    erased = (x["erasing"] - x["clean"]).abs().sum(1) > 1e-5 if "erasing" in x else None
    return x, labels, erased


def summarise(values: dict[str, list[float]]) -> dict:
    out = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        out[key] = {"mean": float(arr.mean()) if arr.size else None,
                    "std": float(arr.std(ddof=1)) if arr.size > 1 else None,
                    "n": int(arr.size)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42, help="which checkpoints to load")
    ap.add_argument("--ckpt-dir", default="results/checkpoints")
    ap.add_argument("--n-images", type=int, default=300,
                    help="test images for the drift measurement, evenly spaced")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--smoke", action="store_true",
                    help="untrained models and 8 images: checks the code, not the science")
    args = ap.parse_args()
    if args.smoke:
        args.n_images, args.batch_size = 8, 4

    torch.manual_seed(0)
    device = dev.resolve_device()
    t0 = time.time()

    ds = SplitDataset(REPO / SPLIT_FILE, "test", transform=None)
    class_names = ds.class_names
    indices = np.unique(np.linspace(0, len(ds) - 1, args.n_images).round().astype(int))
    clean_tf = eval_transform()
    corr_tf = {c: DeterministicCorruption(c, seed=0) for c in CONDS}

    models = {name: load_model(name, args, device, len(class_names)) for name in MODELS}
    caps = {name: Capture(m) for name, m in models.items()}
    inv_k = caps["involeaf_b16"].involutions[0].kernel_size

    # ---------------- drift over the evenly spaced test images ----------------
    vals: dict[str, list[float]] = defaultdict(list)
    for start in range(0, len(indices), args.batch_size):
        idx = indices[start:start + args.batch_size]
        x, labels, erased = load_images(ds, idx, clean_tf, corr_tf)
        for name, model in models.items():
            cap = caps[name]
            pre0, pred0, f0 = run(model, cap, x["clean"].to(device))
            vals[f"{name}/clean/acc"] += (pred0.cpu() == labels).float().tolist()
            for c in CONDS:
                pre1, pred1, f1 = run(model, cap, x[c].to(device))
                base = f"{name}/{c}"
                vals[f"{base}/acc"] += (pred1.cpu() == labels).float().tolist()
                vals[f"{base}/flip"] += (pred1 != pred0).float().cpu().tolist()
                vals[f"{base}/prelogit"] += rel_drift(
                    pre0.unsqueeze(1), pre1.unsqueeze(1)).cpu().tolist()
                for key in f0:
                    if key.startswith("kern"):
                        a, b = kernel_tokens(f0[key]), kernel_tokens(f1[key])
                    else:
                        a, b = to_tokens(f0[key], cap.n_prefix), to_tokens(f1[key], cap.n_prefix)
                    vals[f"{base}/{key}/all"] += rel_drift(a, b).cpu().tolist()
                    if c == "erasing":
                        inside = token_mask(erased, a.shape[1]).to(device)
                        vals[f"{base}/{key}/inside"] += rel_drift(a, b, inside).cpu().tolist()
                        vals[f"{base}/{key}/outside"] += rel_drift(a, b, ~inside).cpu().tolist()
        log.info(f"drift: {min(start + args.batch_size, len(indices))}/{len(indices)} images")

    stats = summarise(vals)
    drift: dict = {}
    for name, model in models.items():
        cap = caps[name]
        stage_names = [s for s, _ in stages(model)]
        drift[name] = {"stages": stage_names,
                       "clean_acc": stats[f"{name}/clean/acc"]["mean"]}
        for c in CONDS:
            base = f"{name}/{c}"
            entry = {
                "acc": stats[f"{base}/acc"]["mean"],
                "flip_rate": stats[f"{base}/flip"]["mean"],
                "prelogit_drift": stats[f"{base}/prelogit"],
                "stage_drift": [stats[f"{base}/{s}/all"] for s in stage_names],
            }
            if c == "erasing":
                entry["stage_drift_inside"] = [stats[f"{base}/{s}/inside"] for s in stage_names]
                entry["stage_drift_outside"] = [stats[f"{base}/{s}/outside"] for s in stage_names]
            n_inv = len(cap.involutions)
            if n_inv:
                entry["kernel_drift"] = [stats[f"{base}/kern{i}/all"] for i in range(n_inv)]
                entry["inv_input_drift"] = [stats[f"{base}/inv_in{i}/all"] for i in range(n_inv)]
                entry["kernel_amplification"] = [
                    (k["mean"] / i["mean"]) if k["mean"] and i["mean"] else None
                    for k, i in zip(entry["kernel_drift"], entry["inv_input_drift"])
                ]
                if c == "erasing":
                    entry["kernel_drift_inside"] = [
                        stats[f"{base}/kern{i}/inside"] for i in range(n_inv)]
                    entry["kernel_drift_outside"] = [
                        stats[f"{base}/kern{i}/outside"] for i in range(n_inv)]
            drift[name][c] = entry

    # ---------------- Grad-CAM and kernel maps, one test image per class ----------------
    labels_all = [lab for _, lab in ds.samples]
    viz_idx = np.array([labels_all.index(k) for k in range(len(class_names))
                        if k in labels_all])
    x, labels, _ = load_images(ds, viz_idx, clean_tf,
                               {c: corr_tf[c] for c in VIZ_CONDS if c != "clean"})
    npz: dict[str, np.ndarray] = {"viz_indices": viz_idx, "labels": labels.numpy()}
    for c in VIZ_CONDS:
        npz[f"image_{c}"] = to_uint8(x[c])
        for name, model in models.items():
            cam, pred = gradcam(model, x[c].to(device), caps[name].n_prefix)
            npz[f"cam_{name}_{c}"] = cam.astype(np.float32)
            npz[f"pred_{name}_{c}"] = pred
        _, _, f = run(models["involeaf_b16"], caps["involeaf_b16"], x[c].to(device))
        kern_keys = sorted((k for k in f if k.startswith("kern")), key=lambda k: int(k[4:]))
        npz[f"kmap_{c}"] = torch.stack(
            [kernel_maps(f[k], inv_k) for k in kern_keys], dim=0).cpu().numpy()
        if c == "clean":
            npz["kprofile"] = torch.stack([
                f[k].float().view(f[k].shape[0], -1, inv_k * inv_k, *f[k].shape[2:])
                .abs().mean((0, 1, 3, 4)) for k in kern_keys
            ]).cpu().numpy()

    for cap in caps.values():
        cap.remove()

    tag = "smoke" if args.smoke else f"seed{args.seed}"
    out = {
        "seed": args.seed,
        "smoke": args.smoke,
        "n_images": int(len(indices)),
        "split_file": SPLIT_FILE,
        "corruption_seed": 0,
        "kernel_size": inv_k,
        "class_names": class_names,
        "device": dev.describe(device, None).as_dict(),
        "drift": drift,
        "minutes": round((time.time() - t0) / 60, 2),
    }
    dest = REPO / "results" / f"mechanism_{tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savez_compressed(REPO / "results" / f"interp_{tag}.npz", **npz)
    log.info(f"wrote {dest} and results/interp_{tag}.npz ({out['minutes']} min)")

    # ---------------- headline numbers ----------------
    log.info("relative drift of the final pre-logit feature / prediction flip rate")
    for name in MODELS:
        row = "  ".join(
            f"{c} {drift[name][c]['prelogit_drift']['mean']:.3f}/"
            f"{drift[name][c]['flip_rate']:.2f}" for c in CONDS)
        log.info(f"  {name:<17} {row}")
    log.info("occlusion: mean stage drift outside / inside the erased box")
    for name in MODELS:
        e = drift[name]["erasing"]
        o = np.nanmean([s["mean"] for s in e["stage_drift_outside"] if s["mean"] is not None])
        i = np.nanmean([s["mean"] for s in e["stage_drift_inside"] if s["mean"] is not None])
        log.info(f"  {name:<17} outside {o:.3f}  inside {i:.3f}")
    inv = drift["involeaf_b16"]
    # Kernels that never left their initialisation (the last block's, which no gradient
    # reaches) are constant, so their drift is exactly zero under every corruption.
    # Averaging them in would understate kernel drift; the per-block JSON keeps them.
    trained = [b for b in range(len(inv["colorjitter"]["kernel_drift"]))
               if any((inv[c]["kernel_drift"][b]["mean"] or 0) > 1e-6 for c in CONDS)]
    skipped = sorted(set(range(len(inv["colorjitter"]["kernel_drift"]))) - set(trained))
    if skipped:
        log.info(f"kernel averages exclude block(s) {[b + 1 for b in skipped]}: "
                 f"constant kernels, never trained")

    def kmean(entries):
        vals = [entries[b]["mean"] for b in trained if entries[b]["mean"] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    for c in CONDS:
        amp = [inv[c]["kernel_amplification"][b] for b in trained]
        amp = np.mean([a for a in amp if a is not None]) if amp else float("nan")
        log.info(f"InvoLeaf kernels, {c:<11}: drift {kmean(inv[c]['kernel_drift']):.3f}, "
                 f"kernel/input drift {amp:.2f}")
    e = inv["erasing"]
    log.info(f"InvoLeaf kernels under occlusion: outside "
             f"{kmean(e['kernel_drift_outside']):.3f}, inside "
             f"{kmean(e['kernel_drift_inside']):.3f}")


if __name__ == "__main__":
    main()
