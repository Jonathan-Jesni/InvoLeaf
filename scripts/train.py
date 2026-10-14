#!/usr/bin/env python
"""Train one model.

    python scripts/train.py --config configs/involeaf_b16.yaml
    python scripts/train.py --config configs/vit_b16_baseline.yaml --epochs 1 --limit-batches 20
    python scripts/train.py --config configs/involeaf_b16.yaml --seeds 42 43 44

Any config key can be overridden inline:
    python scripts/train.py --config configs/involeaf_b16.yaml --set kernel_size=3 batch_size=8

For time-limited sessions, pause cleanly before the budget runs out and continue later:
    python scripts/train.py --config configs/agritl_vit.yaml --resume --max-minutes 170

Exit codes: 0 finished, 3 paused (resume later with --resume), anything else failed.
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

PAUSED_EXIT = 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    # action="extend" so a repeated --set adds to the list instead of replacing it;
    # otherwise "--set batch_size=64 --set kernel_size=3" silently drops batch_size.
    ap.add_argument("--set", nargs="*", default=[], action="extend", metavar="KEY=VALUE",
                    help="override any config key; may be repeated")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="train once per seed; final configs should use three")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="smoke-test mode: stop each epoch after N batches")
    ap.add_argument("--resume", action="store_true",
                    help="continue from results/checkpoints/<run>.last.pth if it exists")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="pause cleanly after the epoch that would overrun this budget")
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
        result = train(cfg, run_name=run_name, limit_batches=args.limit_batches,
                       resume=args.resume, max_minutes=args.max_minutes)
        if result["status"] == "paused":
            log.info(f"{run_name} paused before epoch {result['next_epoch']} of "
                     f"{result['epochs']}; rerun the same command with --resume")
            sys.exit(PAUSED_EXIT)
        results.append(result)

    if len(results) > 1:
        scores = [r["best_val_macro_f1"] for r in results]
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / max(1, len(scores) - 1)
        log.info(f"{cfg['name']}: val macro-F1 {mean:.4f} +/- {var ** 0.5:.4f} "
                 f"over {len(scores)} seeds")


if __name__ == "__main__":
    main()
