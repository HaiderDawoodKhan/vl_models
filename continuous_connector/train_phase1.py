from __future__ import annotations

import argparse

from .config import get_config
from .training import train_phase1


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 connector initialization.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--representation", choices=["patches", "mean_pool"], default="patches")
    args = parser.parse_args()
    cfg = get_config(smoke=args.smoke)
    out = cfg.checkpoint_dir / "phase1" if args.output is None else cfg.checkpoint_dir / args.output
    metrics = train_phase1(
        cfg,
        cfg.train_cache if args.cache is None else cfg.cache_dir / args.cache,
        out,
        max_steps=args.max_steps,
        max_train_examples=args.max_train_examples,
        representation=args.representation,
    )
    print(metrics)


if __name__ == "__main__":
    main()
