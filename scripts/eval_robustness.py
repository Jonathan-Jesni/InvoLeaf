#!/usr/bin/env python
"""Evaluate a trained checkpoint: clean, corrupted, low-resolution, cross-domain.

    python scripts/eval_robustness.py --ckpt results/checkpoints/involeaf_b16_seed42.pth

No retraining happens here. One set of weights is tested against progressively harsher
inputs, which is what makes the whole robustness stage cheap. The corruption protocol
matches AgriTL-ViT Table 17 so the numbers sit directly alongside theirs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from involeaf.engine.evaluate import (  # noqa: E402
    cross_domain_eval, resolution_sweep, robustness_sweep,
)
from involeaf.models.registry import build_model  # noqa: E402
from involeaf.utils import device as dev  # noqa: E402
from involeaf.utils.logging import get_logger  # noqa: E402

log = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split-file", default=None,
                    help="defaults to the split recorded in the checkpoint config")
    ap.add_argument("--cross-domain-split", default="splits/plantdoc_tomato.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resolutions", type=int, nargs="*", default=[128, 200, 224])
    ap.add_argument("--skip-cross-domain", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = torch.load(REPO / args.ckpt if not Path(args.ckpt).is_absolute() else args.ckpt,
                      map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    class_names = ckpt["class_names"]

    device = dev.resolve_device(cfg.get("device", "auto"))
    amp_dtype = dev.amp_dtype_for(device, cfg.get("amp", "auto"))

    model = build_model(dict(cfg, pretrained=False), len(class_names))
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    log.info(f"loaded {args.ckpt} (epoch {ckpt['epoch']}, "
             f"val macro-F1 {ckpt['val_macro_f1']:.4f})")

    split_file = REPO / (args.split_file or cfg["split_file"])
    out = {
        "checkpoint": str(args.ckpt),
        "config": cfg,
        "device": dev.describe(device, amp_dtype).as_dict(),
    }

    log.info("robustness sweep (clean / colorjitter / blur / erasing / combined)")
    out["robustness"] = robustness_sweep(
        model, split_file, device, batch_size=args.batch_size,
        workers=args.workers, amp_dtype=amp_dtype,
    )
    for name, res in out["robustness"].items():
        log.info(f"  {name:<12} acc={res['accuracy']:.4f}  macroF1={res['macro_f1']:.4f}")

    log.info("resolution sweep")
    out["resolution"] = resolution_sweep(
        model, split_file, device, tuple(args.resolutions),
        batch_size=args.batch_size, workers=args.workers, amp_dtype=amp_dtype,
    )
    for name, res in out["resolution"].items():
        log.info(f"  {name:<12} acc={res['accuracy']:.4f}  ({res['protocol']})")

    cross_path = REPO / args.cross_domain_split
    if not args.skip_cross_domain and cross_path.exists():
        log.info("cross-domain: PlantVillage (lab) -> PlantDoc (field)")
        out["cross_domain"] = cross_domain_eval(
            model, cross_path, device, batch_size=args.batch_size,
            workers=args.workers, amp_dtype=amp_dtype,
        )
        log.info(f"  acc={out['cross_domain']['accuracy']:.4f} "
                 f"on {out['cross_domain']['n_images']} field images")
    elif not args.skip_cross_domain:
        log.info(f"cross-domain skipped: {cross_path} not found "
                 f"(run scripts/prepare_data.py --dataset plantdoc --clone)")

    name = Path(args.ckpt).stem
    dest = REPO / (args.out or f"results/eval_{name}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info(f"wrote {dest}")


if __name__ == "__main__":
    main()
