#!/usr/bin/env python
"""Measure parameters, FLOPs, latency and peak training memory.

    python scripts/benchmark.py --all
    python scripts/benchmark.py --config configs/involeaf_b16.yaml

Latency is device-specific and the results file records which backend produced it.
Numbers from the RTX 5050 and the W7800 must never be placed in the same table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from involeaf.models.registry import build_model  # noqa: E402
from involeaf.utils import device as dev  # noqa: E402
from involeaf.utils.config import load_config  # noqa: E402
from involeaf.utils.logging import get_logger  # noqa: E402
from involeaf.utils.profile import full_profile  # noqa: E402

log = get_logger()

DEFAULT_SET = [
    "configs/resnet50.yaml",
    "configs/vit_b16_baseline.yaml",
    "configs/agritl_vit.yaml",
    "configs/involeaf_b16.yaml",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="benchmark the four headline models")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--train-batch", type=int, default=16)
    ap.add_argument("--out", default="results/benchmark.json")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="skip weight download; cost metrics do not depend on weights")
    args = ap.parse_args()

    configs = args.config or (DEFAULT_SET if args.all else None)
    if not configs:
        ap.error("pass --all or one or more --config paths")

    device = dev.resolve_device()
    amp_dtype = dev.amp_dtype_for(device)
    log.info(f"benchmarking on {dev.describe(device, amp_dtype).name}")

    out = {"device": dev.describe(device, amp_dtype).as_dict(), "models": {}}
    for path in configs:
        cfg = load_config(REPO / path if not Path(path).is_absolute() else path)
        if args.no_pretrained:
            cfg["pretrained"] = False
        log.info(f"--- {cfg['name']} ---")
        model = build_model(cfg, args.num_classes)
        out["models"][cfg["name"]] = full_profile(
            model, device, train_batch=args.train_batch, amp_dtype=amp_dtype
        )
        p = out["models"][cfg["name"]]["params"]["total_m"]
        lat = out["models"][cfg["name"]]["latency_gpu"]["median_ms"]
        mem = out["models"][cfg["name"]]["train_memory"]["peak_gb"]
        log.info(f"{cfg['name']}: {p}M params, {lat} ms/img, peak train mem {mem} GB")
        del model

    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info(f"wrote {dest}")


if __name__ == "__main__":
    main()
