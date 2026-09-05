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


def robustness_table() -> str:
    """Accuracy under each corruption, one row per model."""
    evals = _load("eval_*.json")
    if not evals:
        return "_no evaluation results yet_\n"

    corruptions = ["clean", "colorjitter", "blur", "erasing", "combined"]
    rows = []
    for record in evals:
        name = record["config"].get("name", Path(record["checkpoint"]).stem)
        rob = record.get("robustness", {})
        clean = rob.get("clean", {}).get("accuracy")
        row = [name] + [
            f"{rob[c]['accuracy']:.4f}" if c in rob else "-" for c in corruptions
        ]
        if clean and "combined" in rob:
            row.append(f"{(clean - rob['combined']['accuracy']) * 100:.1f}")
        else:
            row.append("-")
        rows.append(row)
    return _md_table(["model"] + corruptions + ["clean-combined drop (pp)"], rows)


def resolution_table() -> str:
    evals = _load("eval_*.json")
    if not evals:
        return "_no evaluation results yet_\n"
    resolutions = ["128", "200", "224"]
    rows = []
    for record in evals:
        name = record["config"].get("name", Path(record["checkpoint"]).stem)
        res = record.get("resolution", {})
        rows.append([name] + [
            f"{res[r]['accuracy']:.4f}" if r in res else "-" for r in resolutions
        ])
    return _md_table(["model"] + [f"{r}px" for r in resolutions], rows) + (
        "\n128 and 200 use the information-loss protocol (downscale, then restore to "
        "224). A native 200x200 evaluation is impossible on a patch-16 ViT, since 200 "
        "is not divisible by 16.\n"
    )


def cross_domain_table() -> str:
    evals = [r for r in _load("eval_*.json") if "cross_domain" in r]
    if not evals:
        return "_no cross-domain results yet_\n"
    rows = []
    for record in evals:
        name = record["config"].get("name", Path(record["checkpoint"]).stem)
        cd = record["cross_domain"]
        clean = record.get("robustness", {}).get("clean", {}).get("accuracy")
        rows.append([
            name,
            f"{clean:.4f}" if clean else "-",
            f"{cd['accuracy']:.4f}",
            f"{cd['macro_f1']:.4f}",
            cd["n_images"],
        ])
    return _md_table(
        ["model", "PlantVillage test", "PlantDoc (field)", "PlantDoc macro-F1", "images"],
        rows,
    )


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
