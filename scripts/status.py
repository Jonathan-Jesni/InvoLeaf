#!/usr/bin/env python
"""Show what has run, what is running now, and what is left, for every seed.

    python scripts/status.py

Reads only result files, logs and the process list, so it is safe to run while
training. It reflects the files on THIS machine: results pushed from another account
only appear after a `git pull`.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
LOGS = RESULTS / "logs"

PHASES = {
    "S": ["resnet50", "vit_b16_baseline", "involeaf_b16", "agritl_vit"],
    "C": ["vit_s16_baseline", "abl_s16_k3", "abl_s16_k5", "abl_s16_k7"],
    "E": ["abl_variant_pure", "abl_variant_parallel"],
    "F": ["benchmark"],
    "P": ["abl_placement_every3", "abl_placement_last4", "abl_placement_last6",
          "abl_placement_first11"],
    "D": ["abl_ffn_lr_backbone", "abl_no_delta_init", "abl_kernel_k3", "abl_norm_bn",
          "abl_ffninit_random", "abl_groups_gc64", "abl_groups_gc768"],
    "M": ["maize_resnet50", "maize_vit_b16_baseline", "maize_involeaf_b16",
          "maize_agritl_vit"],
}
# The split agreed for the project: every seed runs S, plus one ablation each. The
# second round (Oct 19) repeats the kernel and variant ablations on all three seeds,
# adds the planned ViT-B ablations (P, D) on seed 42 and maize (M) on every seed.
PLAN = {42: "SCEPDM", 43: "SCEM", 44: "SCEFM"}


def active_runs() -> dict[str, str]:
    """Run name -> what is happening to it, read from the process list."""
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True,
                             text=True, check=False).stdout
    except OSError:          # e.g. on Windows, where there is no ps
        return {}
    active = {}
    for line in out.splitlines():
        if "scripts/train.py" in line:
            name = re.search(r"name=(\S+)", line)
            seed = re.search(r"seed=(\d+)", line)
            if name and seed:
                active[f"{name.group(1)}_seed{seed.group(1)}"] = "training"
        elif "scripts/eval_robustness.py" in line:
            run = re.search(r"(\w+_seed\d+)\.pth", line)
            if run:
                active[run.group(1)] = "evaluating"
        elif "scripts/benchmark.py" in line:
            active["benchmark"] = "benchmarking"
    return active


def progress(run: str) -> str:
    """The latest completed epoch of a run, from the end of its log."""
    log = LOGS / f"{run}.log"
    if not log.exists():
        return "starting"
    with log.open("rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 16384))
        lines = fh.read().decode("utf-8", "replace").splitlines()
    for line in reversed(lines):
        m = re.search(r"(\d\d:\d\d:\d\d).*\] epoch\s+(\d+).*val_acc=([\d.]+)", line)
        if m:
            return (f"epoch {int(m.group(2)) + 1}/20 finished at {m.group(1)}, "
                    f"val acc {float(m.group(3)):.4f}")
    return "starting (no epoch finished yet)"


def state(run: str, active: dict[str, str]) -> tuple[str, str]:
    if run in active:
        what = active[run]
        detail = progress(run) if what == "training" else ""
        return "RUNNING", f"{what}{': ' + detail if detail else ''}"
    if run == "benchmark":
        return ("done", "") if (RESULTS / "benchmark.json").exists() else ("left", "")
    trained = (RESULTS / f"{run}.json").exists()
    evaluated = (RESULTS / f"eval_{run}.json").exists()
    if trained and evaluated:
        return "done", ""
    if trained:
        return "trained", "not evaluated yet"
    return "left", ""


def main() -> None:
    active = active_runs()
    counts = {"done": 0, "RUNNING": 0, "trained": 0, "left": 0}

    for seed, phases in PLAN.items():
        print(f"\nseed {seed}  (phases {', '.join(phases)})")
        for phase in phases:
            for base in PHASES[phase]:
                run = base if base == "benchmark" else f"{base}_seed{seed}"
                status, detail = state(run, active)
                counts[status] += 1
                print(f"  {status:<8}  {base:<22} {detail}")

    total = sum(counts.values())
    print(f"\n{counts['done']}/{total} done, {counts['RUNNING']} running, "
          f"{counts['trained']} awaiting evaluation, {counts['left']} left")
    if not active:
        print("Nothing is running on this machine right now.")


if __name__ == "__main__":
    main()
