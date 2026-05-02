from __future__ import annotations

import argparse

from .config import get_config
from .training import train_phase2
from .utils import save_json


def run_lambda_ablation(cfg, cache, phase1_dir, max_steps, max_train_examples):
    results = []
    for lam in [0.0, 0.05, 0.2, 0.5]:
        output = cfg.checkpoint_dir / f"phase2_lambda_{lam}"
        metrics = train_phase2(
            cfg,
            cache,
            phase1_dir,
            output,
            lambda_replay=lam,
            max_steps=max_steps,
            max_train_examples=max_train_examples,
        )
        results.append(metrics)
    save_json(cfg.metrics_dir / "lambda_ablation_summary.json", results)
    return results


def run_representation_ablation(cfg, cache, phase1_dir, max_steps, max_train_examples):
    results = []
    for representation in ["patches", "cls", "mean_pool"]:
        output = cfg.checkpoint_dir / f"phase2_repr_{representation}"
        metrics = train_phase2(
            cfg,
            cache,
            phase1_dir,
            output,
            lambda_replay=cfg.lambda_replay,
            max_steps=max_steps,
            max_train_examples=max_train_examples,
            representation=representation,
        )
        metrics["representation"] = representation
        metrics["visual_tokens"] = 49 if representation == "patches" else 1
        results.append(metrics)
    save_json(cfg.metrics_dir / "representation_ablation_summary.json", results)
    return results


def run_norm_alignment(cfg, cache, phase1_dir, max_steps, max_train_examples):
    metrics = train_phase2(
        cfg,
        cache,
        phase1_dir,
        cfg.checkpoint_dir / "phase2_norm_alignment",
        lambda_replay=cfg.lambda_replay,
        max_steps=max_steps,
        max_train_examples=max_train_examples,
        norm_weight=1e-2,
    )
    save_json(cfg.metrics_dir / "norm_alignment_summary.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part A ablations.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--kind", choices=["lambda", "representation", "norm", "all"], default="all")
    parser.add_argument("--cache", default=None, help="Use *_with_cls.pt for CLS representation ablation.")
    parser.add_argument("--phase1-dir", default="phase1")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    args = parser.parse_args()
    cfg = get_config(smoke=args.smoke)
    cfg.ensure_dirs()
    cache = cfg.train_cache if args.cache is None else cfg.cache_dir / args.cache
    phase1 = cfg.checkpoint_dir / args.phase1_dir
    outputs = {}
    if args.kind in {"lambda", "all"}:
        outputs["lambda"] = run_lambda_ablation(cfg, cache, phase1, args.max_steps, args.max_train_examples)
    if args.kind in {"representation", "all"}:
        outputs["representation"] = run_representation_ablation(cfg, cache, phase1, args.max_steps, args.max_train_examples)
    if args.kind in {"norm", "all"}:
        outputs["norm"] = run_norm_alignment(cfg, cache, phase1, args.max_steps, args.max_train_examples)
    print(outputs)


if __name__ == "__main__":
    main()
