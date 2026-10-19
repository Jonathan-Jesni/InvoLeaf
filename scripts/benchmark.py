#!/usr/bin/env python
"""Measure parameters, FLOPs, latency and peak training memory.

    python scripts/benchmark.py --all
    python scripts/benchmark.py --config configs/involeaf_b16.yaml
    python scripts/benchmark.py --flops-only        # CPU, no weights: results/flops.json

Latency is device-specific and the results file records which backend produced it.
Numbers from the RTX 5050 and the W7900D must never be placed in the same table.
MACs and parameters are not device-specific, so --flops-only runs anywhere (the laptop
is fine) and make_tables.py prefers its numbers over the MACs inside benchmark.json.
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
from involeaf.utils.profile import count_flops, count_parameters, full_profile  # noqa: E402

log = get_logger()

DEFAULT_SET = [
    "configs/resnet50.yaml",
    "configs/vit_b16_baseline.yaml",
    "configs/agritl_vit.yaml",
    "configs/involeaf_b16.yaml",
]

# MACs are cheap to count, so the --flops-only table covers every architecture the
# report discusses, not just the four headline models.
FLOPS_SET = DEFAULT_SET + [
    "configs/vit_s16_baseline.yaml",
    "configs/involeaf_s16.yaml",
    "configs/ablations/variant_pure.yaml",
    "configs/ablations/variant_parallel.yaml",
    "configs/ablations/placement_first11.yaml",
    "configs/ablations/kernel_k3.yaml",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="benchmark the four headline models")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--train-batch", type=int, default=16)
    ap.add_argument("--out", default=None,
                    help="default results/benchmark.json, or results/flops.json")
    ap.add_argument("--flops-only", action="store_true",
                    help="parameters and MACs only, on CPU without weights")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="skip weight download; cost metrics do not depend on weights")
    args = ap.parse_args()

    if args.flops_only:
        flops_only(args.config or FLOPS_SET, args.num_classes,
                   REPO / (args.out or "results/flops.json"))
        return

    configs = args.config or (DEFAULT_SET if args.all else None)
    if not configs:
        ap.error("pass --all, --flops-only or one or more --config paths")

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

    dest = REPO / (args.out or "results/benchmark.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info(f"wrote {dest}")


def flops_only(configs: list[str], num_classes: int, dest: Path) -> None:
    out = {"unit": "GMACs", "input_size": [1, 3, 224, 224], "models": {}}
    for path in configs:
        cfg = load_config(REPO / path if not Path(path).is_absolute() else path)
        cfg["pretrained"] = False
        model = build_model(cfg, num_classes)
        out["models"][cfg["name"]] = {"params": count_parameters(model),
                                      "flops": count_flops(model)}
        f = out["models"][cfg["name"]]["flops"]
        log.info(f"{cfg['name']:<24} {out['models'][cfg['name']]['params']['total_m']:>8}M  "
                 f"{f.get('total_gmacs', 'n/a')} GMACs (attention "
                 f"{f['attention_analytic_gmacs']}, involution "
                 f"{f['involution_analytic_gmacs']})")
        del model
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info(f"wrote {dest}")


if __name__ == "__main__":
    main()
