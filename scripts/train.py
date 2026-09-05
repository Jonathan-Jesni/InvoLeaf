#!/usr/bin/env python
"""Train one model.

    python scripts/train.py --config configs/involeaf_b16.yaml
    python scripts/train.py --config configs/vit_b16_baseline.yaml --epochs 1 --limit-batches 20
    python scripts/train.py --config configs/involeaf_b16.yaml --seeds 42 43 44

Any config key can be overridden inline:
    python scripts/train.py --config configs/involeaf_b16.yaml --set kernel_size=3 batch_size=8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from involeaf.engine.train import train  # noqa: E402
from involeaf.utils.config import load_config  # noqa: E402
from involeaf.utils.logging import get_logger  # noqa: E402

log = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override any config key")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="train once per seed; final configs should use three")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="smoke-test mode: stop each epoch after N batches")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    seeds = args.seeds if args.seeds else [cfg.get("seed", 42)]
    results = []
    for seed in seeds:
        cfg = dict(cfg, seed=seed)
        run_name = f"{cfg['name']}_seed{seed}"
        log.info(f"=== {run_name} ===")
        results.append(train(cfg, run_name=run_name, limit_batches=args.limit_batches))

    if len(results) > 1:
        scores = [r["best_val_macro_f1"] for r in results]
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / max(1, len(scores) - 1)
        log.info(f"{cfg['name']}: val macro-F1 {mean:.4f} +/- {var ** 0.5:.4f} "
                 f"over {len(scores)} seeds")


if __name__ == "__main__":
    main()
