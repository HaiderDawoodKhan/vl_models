from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from .config import get_config
from .data import load_alpaca_replay
from .evaluation import evaluate_majority_baseline, evaluate_vqa, save_qualitative_examples
from .model import load_lm_and_tokenizer
from .training import compute_lm_perplexity
from .utils import get_device, load_json, save_json


def plot_phase_curves(cfg, metrics_files: List[Path]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = []
    vqa = []
    ratios = []
    for path in metrics_files:
        if not path.exists():
            continue
        data = load_json(path)
        labels.append(data.get("phase", path.stem))
        if "vqa_accuracy" in data:
            vqa.append(data["vqa_accuracy"])
        if "forgetting_ratio" in data:
            ratios.append(data["forgetting_ratio"])
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    if labels and vqa:
        plt.figure(figsize=(6, 4))
        plt.plot(labels[: len(vqa)], vqa, marker="o")
        plt.ylabel("VQA accuracy")
        plt.tight_layout()
        plt.savefig(cfg.plots_dir / "vqa_accuracy_by_phase.png", dpi=160)
        plt.close()
    if labels and ratios:
        plt.figure(figsize=(6, 4))
        plt.plot(labels[: len(ratios)], ratios, marker="o")
        plt.ylabel("Forgetting ratio R")
        plt.tight_layout()
        plt.savefig(cfg.plots_dir / "forgetting_ratio_by_phase.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VQA and baselines.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint", default="phase3")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--train-cache", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--majority", action="store_true")
    parser.add_argument("--alpaca-ppl", action="store_true", help="Compute base-LM Alpaca perplexity for PPL0.")
    parser.add_argument("--qualitative", action="store_true", help="Save six qualitative examples with first-token top-5 logits.")
    parser.add_argument("--representation", choices=["patches", "cls", "mean_pool"], default="patches")
    args = parser.parse_args()
    cfg = get_config(smoke=args.smoke)
    cfg.ensure_dirs()
    val_cache = cfg.test_cache if args.cache is None else cfg.cache_dir / args.cache
    ckpt = None if args.text_only else cfg.checkpoint_dir / args.checkpoint
    if args.alpaca_ppl:
        device = get_device()
        lm, tokenizer = load_lm_and_tokenizer(cfg, device, freeze=True)
        texts = load_alpaca_replay(100 if args.smoke else cfg.alpaca_replay_examples)
        ppl = compute_lm_perplexity(lm, tokenizer, texts, device, batch_size=cfg.eval_batch_size)
        metrics = {"phase": "phase0", "alpaca_ppl": ppl}
        save_json(cfg.metrics_dir / "alpaca_ppl0.json", metrics)
    elif args.qualitative:
        if ckpt is None:
            raise SystemExit("--qualitative requires a visual checkpoint; omit --text-only.")
        metrics = save_qualitative_examples(cfg, val_cache, ckpt, representation=args.representation)
    elif args.majority:
        train_cache = cfg.train_cache if args.train_cache is None else cfg.cache_dir / args.train_cache
        metrics = evaluate_majority_baseline(cfg, train_cache, val_cache)
    else:
        metrics = evaluate_vqa(
            cfg,
            val_cache,
            ckpt,
            max_examples=args.max_examples,
            text_only=args.text_only,
            representation=args.representation,
        )
    print(metrics)


if __name__ == "__main__":
    main()
