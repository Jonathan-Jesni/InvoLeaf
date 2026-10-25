#!/usr/bin/env python
"""Draw the mechanism and interpretability figures from analyze_mechanism.py's output.

    python scripts/plot_mechanism.py                 # results/*_seed42.*
    python scripts/plot_mechanism.py --tag smoke

Reads results/mechanism_<tag>.json and results/interp_<tag>.npz and writes, next to the
make_plots.py figures in results/figures/:

fig8_gradcam        Grad-CAM for the predicted class, clean vs occluded, ResNet-50 /
                    ViT-B / InvoLeaf on the same leaves.
fig9_kernel_maps    How much each InvoLeaf kernel mixes its neighbours, per location and
                    depth, on the same leaf clean / colour-shifted / occluded.
fig10_drift         How far a corruption moves each model's features, block by block:
                    colour shift everywhere, occlusion outside the erased box.

The leaves shown are the first test image of each class in --classes, fixed in advance
rather than picked for how the maps look.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_plots import (  # noqa: E402
    BLUE, INK, INK_2, K_RAMP, MUTED, ORANGE, REPO, SHORT_CLASS, SURFACE,
    _save, _style, plt,
)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RESULTS = REPO / "results"
COLUMNS = [("resnet50", "ResNet-50"), ("vit_b16_baseline", "ViT-B/16"),
           ("involeaf_b16", "InvoLeaf")]
COND_LABEL = {"clean": "clean", "colorjitter": "colour-shifted", "erasing": "occluded"}

# Single-hue ramps from the reference palette: transparent-to-blue for overlays on the
# leaf photo, light-to-dark blue for the standalone kernel maps.
CAM_CMAP = LinearSegmentedColormap.from_list(
    "cam", [(0.11, 0.36, 0.67, 0.0), (0.11, 0.36, 0.67, 0.8)])
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def _patch(meta: dict) -> str:
    """Which occlusion fill the analysis used (files before 2026-10-25 used the noise patch)."""
    return ("black patch" if meta.get("occlusion") == "erasing_black"
            else "out-of-range noise patch")


def _short(name: str) -> str:
    return SHORT_CLASS.get(name.split("___", 1)[-1], name)


def _upsample(cam: np.ndarray, size: int = 224) -> np.ndarray:
    """Min-max scale, then nearest-neighbour upsample to the image.

    Min-max (as pytorch-grad-cam does) rather than divide-by-max: InvoLeaf's raw maps
    carry a large constant offset (minimum 0.5-0.9 of the maximum), which would tint the
    whole image and hide where the map actually varies.
    """
    lo, hi = cam.min(), cam.max()
    cam = (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam)
    reps = size // cam.shape[0]
    return np.kron(cam, np.ones((reps, reps)))[:size, :size]


def _trained_blocks(npz) -> list[int]:
    """Blocks whose involution left its identity initialisation.

    The last block's involution never trains: only the CLS token reaches the classifier,
    the CLS token bypasses involution, and nothing after the last block mixes patch
    tokens back into it, so no gradient reaches those kernels. They stay at the delta
    (centre tap ~0.99 of the weight). Showing them would read as a finding about the
    operator when it is a fact about where it sits.
    """
    prof = npz["kprofile"]
    centre = prof.shape[1] // 2
    return [b for b in range(prof.shape[0]) if prof[b, centre] / prof[b].sum() < 0.95]


def _pick(classes: list[str], class_names: list[str], labels: np.ndarray) -> list[int]:
    """Positions in the npz arrays of the requested classes, in the requested order."""
    rows = []
    for want in classes:
        matches = [k for k, n in enumerate(class_names) if want.lower() in n.lower()]
        if matches and matches[0] in labels.tolist():
            rows.append(labels.tolist().index(matches[0]))
    return rows or list(range(min(3, len(labels))))


def _blank(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_gradcam(meta: dict, npz, rows: list[int], out: Path) -> None:
    names = meta["class_names"]
    conds = ["clean", "erasing"]
    n_rows = len(rows) * len(conds)
    fig, axes = plt.subplots(n_rows, 1 + len(COLUMNS),
                             figsize=(2.1 * (1 + len(COLUMNS)), 2.25 * n_rows))
    axes = np.atleast_2d(axes)
    for r_i, pos in enumerate(rows):
        true = int(npz["labels"][pos])
        for c_i, cond in enumerate(conds):
            row = r_i * len(conds) + c_i
            img = npz[f"image_{cond}"][pos]
            ax = axes[row, 0]
            ax.imshow(img)
            _blank(ax)
            ax.set_ylabel(f"{_short(names[true])}\n{COND_LABEL[cond]}", color=INK_2,
                          fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
            for m_i, (model, label) in enumerate(COLUMNS, start=1):
                ax = axes[row, m_i]
                ax.imshow(img)
                ax.imshow(_upsample(npz[f"cam_{model}_{cond}"][pos]), cmap=CAM_CMAP,
                          vmin=0, vmax=1)
                _blank(ax)
                pred = int(npz[f"pred_{model}_{cond}"][pos])
                ok = pred == true
                ax.set_xlabel(("" if ok else "wrong: ") + _short(names[pred]),
                              fontsize=8, color=INK_2 if ok else INK,
                              fontweight="normal" if ok else "semibold")
            if row == 0:
                axes[row, 0].set_title("Input", fontsize=10, loc="center")
                for m_i, (_, label) in enumerate(COLUMNS, start=1):
                    axes[row, m_i].set_title(label, fontsize=10, loc="center")
    fig.suptitle(f"Grad-CAM for the predicted class: where each model looks "
                 f"(occlusion: {_patch(meta)})",
                 x=0.02, ha="left", fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    _save(fig, out, "fig8_gradcam")


def fig_kernel_maps(meta: dict, npz, pos: int, out: Path) -> None:
    names = meta["class_names"]
    conds = ["clean", "colorjitter", "erasing"]
    trained = _trained_blocks(npz)
    n = len(trained)
    blocks = sorted({trained[0], trained[n // 3], trained[(2 * n) // 3], trained[-1]})
    fig, axes = plt.subplots(len(conds), 1 + len(blocks),
                             figsize=(2.0 * (1 + len(blocks)) + 0.6, 2.1 * len(conds)))
    axes = np.atleast_2d(axes)
    im = None
    for r, cond in enumerate(conds):
        ax = axes[r, 0]
        ax.imshow(npz[f"image_{cond}"][pos])
        _blank(ax)
        ax.set_ylabel(COND_LABEL[cond], color=INK_2, fontsize=9, rotation=0,
                      ha="right", va="center", labelpad=8)
        for c, b in enumerate(blocks, start=1):
            ax = axes[r, c]
            im = ax.imshow(npz[f"kmap_{cond}"][b, pos], cmap=SEQ_CMAP, vmin=0, vmax=1)
            _blank(ax)
            if r == 0:
                ax.set_title(f"Block {b + 1}", fontsize=10, loc="center")
    axes[0, 0].set_title(_short(names[int(npz["labels"][pos])]), fontsize=10, loc="center")
    cbar = fig.colorbar(im, ax=axes[:, 1:], fraction=0.025, pad=0.02)
    cbar.set_label("Share of kernel weight on neighbours", color=INK_2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_2, length=0)
    fig.suptitle(f"InvoLeaf's generated kernels: how much each location mixes its "
                 f"neighbours (occlusion: {_patch(meta)})",
                 x=0.02, ha="left", fontsize=12, fontweight="semibold", color=INK)
    _save(fig, out, "fig9_kernel_maps")


def _series(entries: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    m = np.array([e["mean"] if e["mean"] is not None else np.nan for e in entries])
    s = np.array([e["std"] if e.get("std") is not None else 0.0 for e in entries])
    return m, s


def fig_drift(meta: dict, npz, out: Path) -> None:
    d = meta["drift"]
    trained = set(_trained_blocks(npz))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    panels = [
        ("colorjitter", "stage_drift", "kernel_drift", "Colour shift: drift of all tokens"),
        ("erasing", "stage_drift_outside", "kernel_drift_outside",
         f"Occlusion ({_patch(meta)}): drift outside the box"),
    ]
    for ax, (cond, key, kkey, title) in zip(axes, panels):
        for model, color, label in [("vit_b16_baseline", ORANGE, "ViT-B/16 features"),
                                    ("involeaf_b16", BLUE, "InvoLeaf features")]:
            m, _ = _series(d[model][cond][key])
            xs = np.arange(1, len(m) + 1)
            ax.plot(xs, m, color=color, linewidth=2, marker="o", markersize=4,
                    markeredgecolor=SURFACE, label=label, zorder=3)
            ax.annotate(label.split()[0], (xs[-1], m[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=color)
        km, _ = _series(d["involeaf_b16"][cond][kkey])
        xs = np.arange(1, len(km) + 1)
        km[[b for b in range(len(km)) if b not in trained]] = np.nan
        ax.plot(xs, km, color=K_RAMP[3], linewidth=1.6, marker="s", markersize=4,
                markeredgecolor=SURFACE, label="InvoLeaf generated kernels", zorder=2)
        ax.set_xticks(xs)
        ax.set_xlabel("Encoder block")
        ax.set_ylabel("Relative change vs clean image")
        ax.set_ylim(bottom=0)
        ax.set_title(title)
    axes[0].legend(loc="center right")
    ax_note = (f"{meta['n_images']} test images, corruptions seeded exactly as in the "
               f"robustness evaluation" + ("  (SMOKE: untrained models)" if meta["smoke"]
                                            else f", seed-{meta['seed']} checkpoints"))
    untrained = [str(b + 1) for b in range(len(km)) if b not in trained]
    if untrained:
        ax_note += (f". Kernels not drawn for block {', '.join(untrained)}: no gradient "
                    f"reaches them, so they never leave their identity initialisation")
    fig.text(0.01, -0.02, ax_note, fontsize=8, color=MUTED)
    fig.tight_layout()
    _save(fig, out, "fig10_drift")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None,
                    help="default: seed42_black if present, else seed42 (noise patch)")
    ap.add_argument("--classes", default="Early_blight,Late_blight,Septoria",
                    help="comma-separated class-name fragments, first test image of each")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    if args.tag is None:
        args.tag = ("seed42_black" if (RESULTS / "mechanism_seed42_black.json").exists()
                    else "seed42")
    meta = json.loads((RESULTS / f"mechanism_{args.tag}.json").read_text(encoding="utf-8"))
    npz = np.load(RESULTS / f"interp_{args.tag}.npz")
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    _style()

    rows = _pick(args.classes.split(","), meta["class_names"], npz["labels"])
    print(f"drawing mechanism figures into {out}")
    fig_gradcam(meta, npz, rows, out)
    fig_kernel_maps(meta, npz, rows[0], out)
    fig_drift(meta, npz, out)


if __name__ == "__main__":
    main()
