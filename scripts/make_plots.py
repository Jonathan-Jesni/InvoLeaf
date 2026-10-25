#!/usr/bin/env python
"""Draw the report's figures from results/*.json.

    python scripts/make_plots.py

Like make_tables.py, every figure is derived from the result files, so a figure can
never drift from the runs behind it. Rerun whenever new results are pushed. Figures go
to results/figures/ as PNG (300 dpi, for Word/slides) and PDF (vector, for LaTeX).

Values are means across the seeds available. Error bars are the sample standard
deviation across seeds and appear only once a model has two or more seeds; the legend
shows how many seeds each model has. The tables from make_tables.py are the exact-value
companion to every figure.

Colour follows the model in every figure: blue is involution (ours), orange is the plain
ViT it modifies, aqua is AgriTL-ViT, yellow is ResNet-50. Where both backbone sizes
appear together, marker shape carries the size (circle ViT-B, square ViT-S). Kernel
size, being ordered, is a light-to-dark blue ramp rather than three unrelated hues. The
palette is the dataviz reference order, validated for colour-vision deficiency.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

# --- chart chrome (light surface) ---------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
K_RAMP = {3: "#86b6ef", 5: "#3987e5", 7: "#1c5cab"}

MODELS = {
    "involeaf_b16":     dict(label="InvoLeaf (ViT-B)", color=BLUE,      marker="o"),
    "vit_b16_baseline": dict(label="ViT-B/16",         color=ORANGE,    marker="o"),
    "agritl_vit":       dict(label="AgriTL-ViT",       color=AQUA,      marker="o"),
    "resnet50":         dict(label="ResNet-50",        color=YELLOW,    marker="o"),
    "vit_s16_baseline": dict(label="ViT-S/16",         color=ORANGE,    marker="s"),
    "abl_s16_k3":       dict(label="InvoLeaf-S, K=3",  color=K_RAMP[3], marker="s"),
    "abl_s16_k5":       dict(label="InvoLeaf-S, K=5",  color=K_RAMP[5], marker="s"),
    "abl_s16_k7":       dict(label="InvoLeaf-S, K=7",  color=K_RAMP[7], marker="s"),
}
B16_SET = ["involeaf_b16", "vit_b16_baseline", "agritl_vit", "resnet50"]
S16_SET = ["vit_s16_baseline", "abl_s16_k3", "abl_s16_k5", "abl_s16_k7"]

CONDITIONS = [
    ("Clean", lambda r: r["robustness"]["clean"]["accuracy"]),
    ("Colour shift", lambda r: r["robustness"]["colorjitter"]["accuracy"]),
    ("Blur", lambda r: r["robustness"]["blur"]["accuracy"]),
    # Occlusion and the combined test use the in-range black patch. The original
    # "erasing" patch is out-of-range noise (see involeaf/data/transforms.py) and only
    # appears in fig7, which shows what it does to ResNet.
    ("Occlusion\n(black patch)", lambda r: r["robustness"]["erasing_black"]["accuracy"]),
    ("Combined\n(black patch)", lambda r: r["robustness"]["combined_black"]["accuracy"]),
    ("Field photos\n(PlantDoc)", lambda r: r["cross_domain"]["accuracy"]),
]

SHORT_CLASS = {
    "Bacterial_spot": "Bacterial spot",
    "Early_blight": "Early blight",
    "Late_blight": "Late blight",
    "Leaf_Mold": "Leaf mould",
    "Septoria_leaf_spot": "Septoria",
    "Spider_mites Two-spotted_spider_mite": "Spider mites",
    "Target_Spot": "Target spot",
    "Tomato_Yellow_Leaf_Curl_Virus": "Yellow leaf curl",
    "Tomato_mosaic_virus": "Mosaic virus",
    "healthy": "Healthy",
}


# --- data -----------------------------------------------------------------------------

def _load(pattern: str) -> list[dict]:
    records = []
    for path in sorted(RESULTS.glob(pattern)):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"skipping malformed {path.name}")
    return records


def _by_model(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        grouped[r["config"].get("name")].append(r)
    return grouped


def _stats(records: list[dict], getter) -> tuple[float, float, int] | None:
    """(mean, std across seeds, number of seeds), or None when nothing is available."""
    values = []
    for r in records:
        try:
            values.append(getter(r))
        except (KeyError, TypeError):
            pass
    if not values:
        return None
    return mean(values), (stdev(values) if len(values) > 1 else 0.0), len(values)


def _seed_note(n: int) -> str:
    return f"{n} seed" if n == 1 else f"{n} seeds"


# --- style ----------------------------------------------------------------------------

def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def _point(ax, x, y, xerr, yerr, model: str, size: float = 70) -> None:
    """A labelled-entity dot with a surface ring, plus error bars when multi-seed."""
    spec = MODELS[model]
    if xerr or yerr:
        ax.errorbar(x, y, xerr=xerr or None, yerr=yerr or None, fmt="none",
                    ecolor=MUTED, elinewidth=0.9, capsize=2, zorder=2)
    ax.scatter([x], [y], s=size, color=spec["color"], marker=spec["marker"],
               edgecolors=SURFACE, linewidths=1.5, zorder=3)


K_MODELS = ("abl_s16_k3", "abl_s16_k5", "abl_s16_k7")


def _label_points(ax, pts: dict[str, tuple[float, float]], offset=(8, 6),
                  k_above: bool = False) -> None:
    """Direct labels, one per model, except the three kernel sizes.

    The K=3/5/7 runs land almost on top of each other, so separate labels collide.
    They get one shared label beside the cluster; their shades are told apart in the
    kernel-ablation figure.
    """
    for m, (x, y) in pts.items():
        if m not in K_MODELS:
            ax.annotate(MODELS[m]["label"], (x, y), xytext=offset,
                        textcoords="offset points", fontsize=8.5, color=INK_2)
    ks = [pts[m] for m in K_MODELS if m in pts]
    if not ks:
        return
    text = "InvoLeaf-S (K=3, 5, 7;\nlight to dark)"
    if k_above:     # centred above the cluster, clear of neighbours to its right
        ax.annotate(text, (mean(p[0] for p in ks), max(p[1] for p in ks)),
                    xytext=(0, 10), textcoords="offset points", fontsize=8.5,
                    color=INK_2, ha="center", va="bottom")
    else:           # below and to the right of the cluster
        ax.annotate(text, (max(p[0] for p in ks), min(p[1] for p in ks)),
                    xytext=(8, -16), textcoords="offset points", fontsize=8.5,
                    color=INK_2, va="top")


def _shape_legend(ax, loc: str = "lower left") -> None:
    handles = [
        Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=7,
               label="ViT-B-size models"),
        Line2D([], [], marker="s", linestyle="", color=MUTED, markersize=7,
               label="ViT-S-size models"),
    ]
    ax.legend(handles=handles, loc=loc, labelcolor=INK_2)


# --- figures --------------------------------------------------------------------------

def fig_robustness(evals, models: list[str], title: str, name: str, out: Path) -> None:
    """Accuracy under each condition, one bar per model."""
    present = [m for m in models if evals.get(m)]
    if not present:
        print(f"  skipped {name}: no evaluation results")
        return
    fig, ax = plt.subplots(figsize=(10, 4.6))
    width = 0.8 / len(present)
    xs = range(len(CONDITIONS))
    for i, m in enumerate(present):
        means, errs, counts = [], [], []
        for _, get in CONDITIONS:
            s = _stats(evals[m], get)
            # A run evaluated before a test existed has no bar there, rather than a
            # zero-height bar that would read as 0% accuracy.
            means.append(s[0] * 100 if s else float("nan"))
            errs.append(s[1] * 100 if s else float("nan"))
            counts.append(s[2] if s else 0)
        offsets = [x - 0.4 + width * (i + 0.5) for x in xs]
        n = len(evals[m])
        # The legend gives the model's seed count; a bar that averages fewer seeds (a
        # test added after some runs were evaluated) says so on the bar itself.
        for x, y, e, k in zip(offsets, means, errs, counts):
            if 0 < k < n:
                ax.annotate(f"n={k}", (x, y + (e if e == e else 0)), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=6.5,
                            color=INK_2)
        # A surface-coloured edge gives the 2px gap between adjacent bars.
        ax.bar(offsets, means, width, color=MODELS[m]["color"], edgecolor=SURFACE,
               linewidth=1.2, label=f"{MODELS[m]['label']}  ({_seed_note(n)})", zorder=2)
        if n > 1:
            ax.errorbar(offsets, means, yerr=errs, fmt="none", ecolor=INK_2,
                        elinewidth=0.9, capsize=2, zorder=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([c[0] for c in CONDITIONS])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test accuracy (%)")
    ax.xaxis.grid(False)
    ax.set_title(title)
    ax.legend(ncol=len(present), loc="upper center", bbox_to_anchor=(0.5, -0.14))
    _save(fig, out, name)


def fig_tradeoff(evals, out: Path) -> None:
    """Occlusion accuracy against colour-shift accuracy, with baseline-to-involution arrows."""
    occ = CONDITIONS[3][1]
    col = CONDITIONS[1][1]
    pts = {}
    fewer = []
    for m in MODELS:
        if evals.get(m):
            sx, sy = _stats(evals[m], occ), _stats(evals[m], col)
            if sx and sy:
                pts[m] = (sx[0] * 100, sy[0] * 100, sx[1] * 100, sy[1] * 100)
                if sx[2] < len(evals[m]):
                    fewer.append((MODELS[m]["label"], sx[2]))
    if not pts:
        print("  skipped fig3_tradeoff: no evaluation results")
        return

    fig, ax = plt.subplots(figsize=(8, 5.6))
    # Arrows: what adding involution does to each backbone. For ViT-S, one arrow to
    # the centre of the three kernel sizes rather than three near-identical arrows.
    targets = {"vit_b16_baseline": pts.get("involeaf_b16")}
    ks = [pts[k] for k in K_MODELS if k in pts]
    if ks:
        targets["vit_s16_baseline"] = (mean(p[0] for p in ks), mean(p[1] for p in ks))
    for src, dst in targets.items():
        if src in pts and dst:
            ax.annotate("", xy=dst[:2], xytext=pts[src][:2], zorder=1,
                        arrowprops=dict(arrowstyle="-|>", color=AXIS, lw=1.2,
                                        shrinkA=7, shrinkB=9))
    for m, (x, y, xe, ye) in pts.items():
        _point(ax, x, y, xe, ye, m)
    _label_points(ax, {m: p[:2] for m, p in pts.items()})
    ax.set_xlabel("Accuracy under occlusion, black patch (%)")
    ax.set_ylabel("Accuracy under colour shift (%)")
    ax.set_title("Occlusion vs colour-shift robustness")
    ax.margins(x=0.12, y=0.12)
    _shape_legend(ax)
    ax.text(0.99, 0.02, "Arrows: plain ViT → same ViT with involution", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=MUTED)
    if fewer:
        counts = sorted({k for _, k in fewer})
        note = "; ".join(
            f"occlusion averages {k} seeds for " + ", ".join(lab for lab, kk in fewer if kk == k)
            for k in counts)
        fig.text(0.01, -0.01, note[0].upper() + note[1:], fontsize=7.5, color=MUTED)
    _save(fig, out, "fig3_tradeoff")


def fig_lab_vs_field(evals, out: Path) -> None:
    """Lab test accuracy against field (PlantDoc) accuracy, one row per model."""
    lab_get, field_get = CONDITIONS[0][1], CONDITIONS[5][1]
    rows = []
    for m in MODELS:
        if evals.get(m):
            lab, field = _stats(evals[m], lab_get), _stats(evals[m], field_get)
            if lab and field:
                rows.append((m, lab, field))
    if not rows:
        print("  skipped fig4_lab_vs_field: no cross-domain results")
        return
    rows.sort(key=lambda r: r[2][0])        # weakest in the field at the bottom

    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(rows) + 1.6))
    for i, (m, lab, field) in enumerate(rows):
        ax.plot([field[0] * 100, lab[0] * 100], [i, i], color=AXIS, lw=2, zorder=1,
                solid_capstyle="round")
        ax.scatter([lab[0] * 100], [i], s=55, color=MUTED, edgecolors=SURFACE,
                   linewidths=1.5, zorder=3)
        _point(ax, field[0] * 100, i, field[1] * 100, 0, m, size=75)
        # Anchor the value left of the error bar, not the dot, so the two never collide.
        ax.annotate(f"{field[0] * 100:.0f}%", ((field[0] - field[1]) * 100, i),
                    xytext=(-10, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([MODELS[m]["label"] for m, _, _ in rows])
    ax.set_xlim(0, 102)
    ax.set_xlabel("Accuracy (%)")
    ax.yaxis.grid(False)
    ax.set_title("Lab accuracy vs real field photographs")
    handles = [
        Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=7,
               label="PlantVillage test (lab)"),
        Line2D([], [], marker="o", linestyle="", color=INK_2, markersize=7,
               label="PlantDoc (field), coloured by model"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2,
              labelcolor=INK_2)
    _save(fig, out, "fig4_lab_vs_field")


def fig_params_vs_accuracy(evals, train, out: Path) -> None:
    """Parameter count against clean test accuracy."""
    pts = []
    for m in MODELS:
        if evals.get(m) and train.get(m):
            acc = _stats(evals[m], CONDITIONS[0][1])
            params = train[m][0].get("params_total")
            if acc and params:
                pts.append((m, params / 1e6, acc))
    if not pts:
        print("  skipped fig5_params_vs_accuracy: missing results")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, params, (acc, sd, _) in pts:
        _point(ax, params, acc * 100, 0, sd * 100, m)
    # In this figure ViT-S sits just right of the kernel cluster, so its label goes above.
    _label_points(ax, {m: (params, acc * 100) for m, params, (acc, _, _) in pts},
                  offset=(8, -3), k_above=True)
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Clean test accuracy (%)")
    ax.set_title("Model size vs clean accuracy")
    ax.margins(x=0.15, y=0.2)
    ax.text(0.99, 0.02, "Note: the y-axis is zoomed; every model is above 99%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
    _shape_legend(ax, loc="lower left")
    _save(fig, out, "fig5_params_vs_accuracy")


def fig_training_curves(train, out: Path) -> None:
    """Validation macro-F1 per epoch, mean across seeds with a ±1 std band."""
    present = [m for m in B16_SET if train.get(m)]
    if not present:
        print("  skipped fig6_training_curves: no training results")
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for m in present:
        curves = [[e["val_macro_f1"] for e in r["epochs"]] for r in train[m]]
        length = min(len(c) for c in curves)
        epochs = list(range(1, length + 1))
        means = [mean(c[i] for c in curves) for i in range(length)]
        ax.plot(epochs, means, color=MODELS[m]["color"], lw=2,
                label=f"{MODELS[m]['label']}  ({_seed_note(len(curves))})")
        if len(curves) > 1:
            sds = [stdev(c[i] for c in curves) for i in range(length)]
            ax.fill_between(epochs, [a - b for a, b in zip(means, sds)],
                            [a + b for a, b in zip(means, sds)],
                            color=MODELS[m]["color"], alpha=0.15, linewidth=0)
    ax.set_ylim(0.90, 1.001)
    ax.set_xlim(1, None)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Training progress (ViT-B-size models and ResNet-50)")
    ax.text(0.99, 0.02, "y-axis starts at 0.90; AgriTL-ViT's first epoch is below it",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    _save(fig, out, "fig6_training_curves")


def fig_resnet_collapse(evals, out: Path) -> None:
    """Where ResNet-50's predictions go: clean, out-of-range noise patch, black patch.

    The round-1 "collapse to Yellow Leaf Curl" happens only with the out-of-range noise
    patch; a black patch in the same box barely moves the predictions. The figure is
    the evidence for that methodological finding.
    """
    records = evals.get("resnet50")
    if not records:
        print("  skipped fig7_resnet_occlusion: no ResNet results")
        return

    def share(condition: str) -> tuple[list[float], int] | None:
        per_seed = []
        for r in records:
            if condition not in r["robustness"]:
                continue
            cm = r["robustness"][condition]["confusion_matrix"]
            cols = [sum(row[j] for row in cm) for j in range(len(cm))]
            total = sum(cols)
            per_seed.append([c / total * 100 for c in cols])
        if not per_seed:
            return None
        return [mean(s[j] for s in per_seed) for j in range(len(per_seed[0]))], len(per_seed)

    names = list(records[0]["robustness"]["clean"]["per_class"].keys())
    labels = [SHORT_CLASS.get(n.split("___", 1)[-1], n) for n in names]
    series = [("clean", "Clean test images", MUTED),
              ("erasing", "Out-of-range noise patch", YELLOW),
              ("erasing_black", "Black patch, same boxes", INK_2)]
    series = [(label, color, share(cond)) for cond, label, color in series]
    series = [(label, color, sh) for label, color, sh in series if sh]

    fig, ax = plt.subplots(figsize=(10, 4.4))
    xs = range(len(labels))
    w = 0.8 / len(series)
    for i, (label, color, (values, n)) in enumerate(series):
        offs = [x - 0.4 + w * (i + 0.5) for x in xs]
        ax.bar(offs, values, w, color=color, edgecolor=SURFACE, linewidth=1.2,
               label=f"{label}  ({_seed_note(n)})", zorder=2)
        if label.startswith("Out-of-range"):
            top = max(range(len(values)), key=lambda j: values[j])
            ax.annotate(f"{values[top]:.1f}%", (offs[top], values[top]), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Share of all predictions (%)")
    ax.xaxis.grid(False)
    ax.set_title("ResNet-50's occlusion collapse comes from the out-of-range patch")
    ax.legend(loc="upper left")
    _save(fig, out, "fig7_resnet_occlusion")


# --- main -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    _style()

    evals = _by_model(_load("eval_*.json"))
    train = _by_model([r for r in _load("*_seed*.json") if "epochs" in r])

    print(f"drawing figures into {out}")
    fig_robustness(evals, B16_SET, "Accuracy under corruption: ViT-B-size models and ResNet-50",
                   "fig1_robustness_b16", out)
    fig_robustness(evals, S16_SET, "Kernel-size ablation on ViT-S/16",
                   "fig2_kernel_ablation_s16", out)
    fig_tradeoff(evals, out)
    fig_lab_vs_field(evals, out)
    fig_params_vs_accuracy(evals, train, out)
    fig_training_curves(train, out)
    fig_resnet_collapse(evals, out)


if __name__ == "__main__":
    main()
