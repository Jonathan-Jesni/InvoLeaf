#!/usr/bin/env python
"""Clone the datasets if needed and write the fixed split files.

    python scripts/prepare_data.py --dataset tomato
    python scripts/prepare_data.py --dataset plantdoc

Splits are deterministic given the seed, and are committed to the repository so every
machine trains on the same partition.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from involeaf.data.prepare import prepare_plantdoc, prepare_plantvillage  # noqa: E402
from involeaf.utils.logging import get_logger  # noqa: E402

log = get_logger()

SOURCES = {
    "plantvillage": (
        "https://github.com/spMohanty/PlantVillage-Dataset",
        "data/PlantVillage-Dataset",
    ),
    "plantdoc": (
        "https://github.com/pratikkayal/PlantDoc-Dataset",
        "data/PlantDoc-Dataset",
    ),
}


def ensure_clone(key: str, clone: bool) -> Path:
    url, rel = SOURCES[key]
    dest = REPO / rel
    if dest.is_dir():
        log.info(f"{key}: found at {dest}")
        return dest
    if not clone:
        raise SystemExit(
            f"{key} is not present at {dest}.\n"
            f"Run with --clone, or clone it yourself:\n"
            f"    git clone --depth 1 {url} {rel}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"{key}: cloning {url} -> {dest} (this is large; --depth 1 is used)")
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)], check=True, cwd=REPO
    )
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="tomato",
                    choices=["tomato", "maize", "potato", "apple", "grape", "plantdoc"])
    ap.add_argument("--clone", action="store_true",
                    help="git clone the source dataset if it is not already present")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--reference-split", default="splits/plantvillage_tomato.json",
                    help="for plantdoc: the split whose class list defines the labels")
    args = ap.parse_args()

    splits_dir = REPO / "splits"
    splits_dir.mkdir(exist_ok=True)

    if args.dataset == "plantdoc":
        root = ensure_clone("plantdoc", args.clone)
        ref = REPO / args.reference_split
        if not ref.exists():
            raise SystemExit(
                f"{ref} does not exist. Build the PlantVillage tomato split first:\n"
                f"    python scripts/prepare_data.py --dataset tomato --clone"
            )
        summary = prepare_plantdoc(root, splits_dir / "plantdoc_tomato.json", ref)
        log.info(f"PlantDoc cross-domain test set: {summary['total']} images")
        for folder, n in sorted(summary["matched_folders"].items()):
            log.info(f"  mapped {folder!r}: {n}")
        if summary["unmapped"]:
            log.info(f"  unmapped (dropped): {summary['unmapped']}")
    else:
        root = ensure_clone("plantvillage", args.clone)
        out = splits_dir / f"plantvillage_{args.dataset}.json"
        summary = prepare_plantvillage(
            root, args.dataset, out, seed=args.seed, ratios=tuple(args.ratios)
        )
        log.info(f"{args.dataset}: {summary['classes']} classes, {summary['total']} images")
        for name, n in sorted(summary["per_class"].items()):
            log.info(f"  {name}: {n}")
        log.info(f"split sizes: {summary['counts']}")
        if args.dataset == "tomato":
            log.info(
                f"note: AgriTL-ViT reports 14,529 tomato images; PlantVillage holds "
                f"{summary['total']}. The difference is reported, not matched."
            )

    log.info(f"wrote {summary['out']}")


if __name__ == "__main__":
    main()
