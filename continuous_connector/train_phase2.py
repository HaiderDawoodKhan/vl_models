from __future__ import annotations

import argparse

from .config import get_config
from .training import train_phase2


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 VQA SFT with Alpaca replay.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--phase1-dir", default="phase1")
    parser.add_argument("--output", default="phase2")
    parser.add_argument("--lambda-replay", type=float, default=None)
    parser.add_argument("--norm-weight", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--representation", choices=["patches", "mean_pool"], default="patches")
    args = parser.parse_args()
    cfg = get_config(smoke=args.smoke)
    metrics = train_phase2(
        cfg,
        cfg.train_cache if args.cache is None else cfg.cache_dir / args.cache,
        cfg.checkpoint_dir / args.phase1_dir,
        cfg.checkpoint_dir / args.output,
        lambda_replay=args.lambda_replay,
        max_steps=args.max_steps,
        max_train_examples=args.max_train_examples,
        representation=args.representation,
        norm_weight=args.norm_weight,
    )
    print(metrics)


if __name__ == "__main__":
    main()
