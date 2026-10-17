#!/usr/bin/env python
"""Regenerate every report table from results/*.json.

    python scripts/make_tables.py

No number in the report is ever hand-copied. Each table is derived from the JSON written
by scripts/train.py, scripts/benchmark.py and scripts/eval_robustness.py, so a table can
never drift from the run that produced it, and a stale table is impossible to publish by
accident. Markdown goes to results/tables.md and a flat CSV to results/summary.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from involeaf.utils.logging import get_logger  # noqa: E402

log = get_logger()


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_no results yet_\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _load(pattern: str) -> list[dict]:
    records = []
    for path in sorted((REPO / "results").glob(pattern)):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            log.info(f"skipping malformed {path}")
    return records


def training_table() -> str:
    """Validation macro-F1 per config, aggregated over seeds."""
    runs = [r for r in _load("*.json")
            if "epochs" in r and "best_val_macro_f1" in r]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        grouped[r["config"].get("name", r["run_name"])].append(r)

    rows = []
    for name, group in sorted(grouped.items()):
        scores = [g["best_val_macro_f1"] for g in group]
        params = group[0].get("params_total", 0) / 1e6
        spread = f"{stdev(scores):.4f}" if len(scores) > 1 else "-"
        flag = "" if len(scores) >= 3 else "  (needs 3 seeds)"
        rows.append([name, f"{params:.2f}M", len(scores),
                     f"{mean(scores):.4f}", spread + flag])
    return _md_table(
        ["config", "params", "seeds", "val macro-F1 (mean)", "std"], rows
    )


def efficiency_table() -> str:
    """Cost metrics. Grouped by device, because latency is not portable."""
    bench = _load("benchmark*.json")
    if not bench:
        return "_no benchmark results yet_\n"

    chunks = []
    for record in bench:
        d = record["device"]
        rows = []
        for name, m in record["models"].items():
            flops = m["flops"]
            counted = flops.get("fvcore_gflops")
            extra = flops.get("involution_analytic_gflops", 0.0)
            total = f"{counted + extra:.2f}" if counted is not None else "n/a"
            rows.append([
                name,
                f"{m['params']['total_m']}M",
                total,
                m["latency_gpu"]["median_ms"],
                m["latency_cpu"]["median_ms"],
                m["train_memory"]["peak_gb"] or m["train_memory"].get("status", "-"),
            ])
        chunks.append(
            f"**{d['name']}** ({d['backend']}, torch {d['torch_version']}, "
            f"amp {d['amp_dtype']})\n\n"
            + _md_table(
                ["model", "params", "GFLOPs", "GPU ms", "CPU ms", "peak train GB"], rows
            )
            + "\nGFLOPs include an analytic term for the involution multiply-accumulate, "
              "which FLOP counters do not handle. Latency is measured at batch 1 and is "
              "not comparable across devices.\n"
        )
    return "\n".join(chunks)


SEED_NOTE = (
    "\nEach cell is the mean over the seeds listed, as `mean ± std` (sample standard "
    "deviation) once there is more than one seed. A single number is one seed and "
    "carries no error estimate.\n"
)


def _cell(values: list[float], scale: float = 1.0, digits: int = 4) -> str:
    """Mean, or mean ± std across seeds when there is more than one."""
    values = [v * scale for v in values if v is not None]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{mean(values):.{digits}f} ± {stdev(values):.{digits}f}"


def _evals_by_model() -> dict[str, list[dict]]:
    """Evaluation records grouped by config name, so seeds of one model share a row."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in _load("eval_*.json"):
        name = record["config"].get("name", Path(record["checkpoint"]).stem)
        grouped[name].append(record)
    return dict(sorted(grouped.items()))


def _seeds(group: list[dict]) -> str:
    return ", ".join(str(s) for s in sorted(r["config"].get("seed", "?") for r in group))


def robustness_table() -> str:
    """Accuracy under each corruption, one row per model, averaged over seeds."""
    grouped = _evals_by_model()
    if not grouped:
        return "_no evaluation results yet_\n"

    corruptions = ["clean", "colorjitter", "blur", "erasing", "combined"]
    rows = []
    for name, group in grouped.items():
        robs = [r.get("robustness", {}) for r in group]
        row = [name, _seeds(group)]
        for c in corruptions:
            row.append(_cell([rob[c]["accuracy"] for rob in robs if c in rob]))
        # The drop is computed per seed, then averaged, so its spread is the spread of
        # the drop itself rather than a combination of two separately averaged columns.
        drops = [rob["clean"]["accuracy"] - rob["combined"]["accuracy"]
                 for rob in robs if "clean" in rob and "combined" in rob]
        row.append(_cell(drops, scale=100, digits=1))
        rows.append(row)
    return _md_table(
        ["model", "seeds"] + corruptions + ["clean-combined drop (pp)"], rows
    ) + SEED_NOTE


def resolution_table() -> str:
    grouped = _evals_by_model()
    if not grouped:
        return "_no evaluation results yet_\n"
    resolutions = ["128", "200", "224"]
    rows = []
    for name, group in grouped.items():
        res = [r.get("resolution", {}) for r in group]
        rows.append([name, _seeds(group)] + [
            _cell([x[r]["accuracy"] for x in res if r in x]) for r in resolutions
        ])
    return _md_table(["model", "seeds"] + [f"{r}px" for r in resolutions], rows) + (
        SEED_NOTE
        + "\n128 and 200 use the information-loss protocol (downscale, then restore to "
        "224). A native 200x200 evaluation is impossible on a patch-16 ViT, since 200 "
        "is not divisible by 16.\n"
    )


def cross_domain_table() -> str:
    grouped = {name: [r for r in group if "cross_domain" in r]
               for name, group in _evals_by_model().items()}
    grouped = {name: group for name, group in grouped.items() if group}
    if not grouped:
        return "_no cross-domain results yet_\n"
    rows = []
    for name, group in grouped.items():
        cds = [r["cross_domain"] for r in group]
        cleans = [r.get("robustness", {}).get("clean", {}).get("accuracy") for r in group]
        # PlantDoc has no Target_Spot folder, so plain macro-F1 averages in a class with
        # zero support and understates the result. Quote F1 over the classes present.
        present_f1 = [cd.get("macro_f1_present", cd["macro_f1"]) for cd in cds]
        first = cds[0]
        classes = (f"{first['classes_present']}/{first['classes_total']}"
                   if "classes_present" in first else "-")
        rows.append([
            name,
            _seeds(group),
            _cell(cleans),
            _cell([cd["accuracy"] for cd in cds]),
            _cell(present_f1),
            classes,
            first["n_images"],
        ])
    return _md_table(
        ["model", "seeds", "PlantVillage test", "PlantDoc (field)",
         "PlantDoc macro-F1 (classes present)", "classes", "images"],
        rows,
    ) + SEED_NOTE


def write_csv(dest: Path) -> None:
    import csv

    runs = [r for r in _load("*.json") if "best_val_macro_f1" in r]
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run_name", "config", "seed", "params_m",
                         "best_val_macro_f1", "best_epoch", "device", "backend"])
        for r in runs:
            writer.writerow([
                r["run_name"], r["config"].get("name"), r["config"].get("seed"),
                round(r.get("params_total", 0) / 1e6, 3),
                round(r["best_val_macro_f1"], 5), r.get("best_epoch"),
                r["device"]["name"], r["device"]["backend"],
            ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/tables.md")
    args = ap.parse_args()

    sections = [
        ("Training results", training_table()),
        ("Efficiency", efficiency_table()),
        ("Robustness (AgriTL-ViT Table 17 protocol)", robustness_table()),
        ("Resolution", resolution_table()),
        ("Cross-domain generalisation", cross_domain_table()),
    ]
    body = "# InvoLeaf results\n\nGenerated by `scripts/make_tables.py` from `results/*.json`. Do not edit by hand.\n\n"
    body += "\n".join(f"## {title}\n\n{content}\n" for title, content in sections)

    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(body, encoding="utf-8")
    write_csv(dest.with_name("summary.csv"))
    log.info(f"wrote {dest} and {dest.with_name('summary.csv')}")
    print(body)


if __name__ == "__main__":
    main()
