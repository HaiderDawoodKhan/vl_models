from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

try:
    from .train_lm import train_lm
    from .train_vqvae import train_vqvae
    from .utils import save_json
    from .config import get_config
except ImportError:
    from train_lm import train_lm
    from train_vqvae import train_vqvae
    from utils import save_json
    from config import get_config


def run_vqvae_ablations(smoke: bool = False) -> List[Dict[str, Any]]:
    configs = [
        {"name": "vqvae_baseline_k256_beta025_grad", "k": 256, "beta": 0.25, "ema": False},
        {"name": "vqvae_k128_beta025_grad", "k": 128, "beta": 0.25, "ema": False},
        {"name": "vqvae_k256_beta100_grad", "k": 256, "beta": 1.0, "ema": False},
        {"name": "vqvae_k256_beta025_ema", "k": 256, "beta": 0.25, "ema": True},
    ]
    results = []
    for row in configs:
        results.append(train_vqvae(smoke=smoke, use_ema=row["ema"], k=row["k"], beta=row["beta"], output_name=row["name"]))
    cfg = get_config(smoke)
    save_json(cfg.outputs_dir / "vqvae_ablation_metrics.json", results)
    return results


def run_lm_weight_ablation(smoke: bool = False) -> List[Dict[str, Any]]:
    cfg = get_config(smoke)
    combos = [
        ("no_replay", 0.0, 0.0),
        ("weak", 0.05, 0.05),
        ("baseline", 0.2, 0.5),
        ("strong", 0.5, 0.5),
    ]
    results = []
    for name, lambda_lm, gamma_img in combos:
        result = train_lm(smoke=smoke, max_steps=4 if smoke else None, lambda_lm=lambda_lm, gamma_img=gamma_img)
        result["ablation"] = name
        result["lambda_lm"] = lambda_lm
        result["gamma_img"] = gamma_img
        results.append(result)
    save_json(cfg.outputs_dir / "lm_weight_ablation_metrics.json", results)
    return results


def run_no_projector_ablation(smoke: bool = False) -> Dict[str, Any]:
    result = train_lm(smoke=smoke, max_steps=4 if smoke else None, no_projector=True)
    result["ablation"] = "no_projector"
    cfg = get_config(smoke)
    save_json(cfg.outputs_dir / "no_projector_ablation_metrics.json", result)
    return result


def run_break_protection(smoke: bool = False) -> Dict[str, Any]:
    cfg = get_config(smoke)
    result = train_lm(smoke=smoke, max_steps=4 if smoke else None, lambda_lm=0.0, gamma_img=0.0, lora_r=64)
    result["ablation"] = "break_protection_lora64_no_replay"
    save_json(cfg.outputs_dir / "break_protection_metrics.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--kind", choices=["vqvae", "lm-weights", "no-projector", "break-protection"], required=True)
    args = parser.parse_args()
    if args.kind == "vqvae":
        print(run_vqvae_ablations(args.smoke))
    elif args.kind == "lm-weights":
        print(run_lm_weight_ablation(args.smoke))
    elif args.kind == "no-projector":
        print(run_no_projector_ablation(args.smoke))
    else:
        print(run_break_protection(args.smoke))
