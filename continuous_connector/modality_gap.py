from __future__ import annotations

import argparse

from .config import get_config
from .modality import compute_modality_gap


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute modality-gap metrics and UMAP plots.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint subdir, e.g. phase1/phase2/phase3. Omit for initial connector.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--representation", choices=["patches", "cls", "mean_pool"], default="patches")
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    cfg = get_config(smoke=args.smoke)
    cfg.ensure_dirs()
    ckpt = cfg.checkpoint_dir / args.checkpoint if args.checkpoint else None
    name = args.name or (args.checkpoint if args.checkpoint else "initial")
    metrics = compute_modality_gap(
        cfg,
        cfg.test_cache if args.cache is None else cfg.cache_dir / args.cache,
        ckpt,
        name,
        representation=args.representation,
        count=args.count,
    )
    print(metrics)


if __name__ == "__main__":
    main()
