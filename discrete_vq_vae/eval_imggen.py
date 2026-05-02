from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from .config import CLASS_NAMES, get_config
    from .eval_vqa import load_finetuned
    from .synthetic_data import build_imggen_examples
    from .train_lm import load_encoded
    from .train_vqvae import load_vqvae_checkpoint
    from .utils import get_device, require_torch, save_json, set_seed
except ImportError:
    from config import CLASS_NAMES, get_config
    from eval_vqa import load_finetuned
    from synthetic_data import build_imggen_examples
    from train_lm import load_encoded
    from train_vqvae import load_vqvae_checkpoint
    from utils import get_device, require_torch, save_json, set_seed


torch = require_torch()


def masked_visual_step(logits: Any, cfg: Any, temperature: float = 1.0, sample: bool = False) -> int:
    masked = torch.full_like(logits, -1e9)
    start = cfg.vtxt + 2
    end = start + cfg.k
    masked[:, start:end] = logits[:, start:end]
    if sample:
        probs = torch.softmax(masked / temperature, dim=-1)
        return int(torch.multinomial(probs[0], num_samples=1).item())
    return int(torch.argmax(masked[0]).item())


def generate_visual_tokens(lm: Any, overlay: Any, tokenizer: Any, cfg: Any, prompt: str, image_id: int, device: str, temperature: float = 1.0, sample: bool = False) -> tuple[List[int], Any, Any]:
    ids = [tokenizer.bos_token_id or tokenizer.eos_token_id] + tokenizer(prompt, add_special_tokens=False).input_ids + [image_id]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    first_raw = None
    first_masked = None
    for step in range(16):
        with torch.no_grad():
            logits = lm(inputs_embeds=overlay(input_ids), attention_mask=torch.ones_like(input_ids)).logits[:, -1, :]
        if step == 0:
            first_raw = logits.detach().cpu()
            first_masked = torch.full_like(first_raw, -1e9)
            first_masked[:, cfg.vtxt + 2 : cfg.vtxt + 2 + cfg.k] = first_raw[:, cfg.vtxt + 2 : cfg.vtxt + 2 + cfg.k]
        next_id = masked_visual_step(logits, cfg, temperature, sample)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    generated = input_ids[0, -16:].detach().cpu().tolist()
    return generated, first_raw, first_masked


def decode_visual_ids(vqvae: Any, token_ids: List[int], cfg: Any, device: str) -> Any:
    indices = torch.tensor([tid - (cfg.vtxt + 2) for tid in token_ids], dtype=torch.long, device=device).view(1, 4, 4)
    with torch.no_grad():
        return vqvae.decode_indices(indices).detach().cpu()[0], indices.detach().cpu()[0]


def save_generated_grid(cfg: Any, rows: List[Dict[str, Any]], name: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 2, figsize=(5, 2 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])
    for row_idx, row in enumerate(rows):
        axes[row_idx, 0].imshow(row["image"].permute(1, 2, 0).numpy().clip(0, 1))
        axes[row_idx, 0].set_title(row["class_name"])
        axes[row_idx, 1].imshow(row["token_map"].numpy(), cmap="tab20")
        axes[row_idx, 1].set_title("tokens")
        for col in range(2):
            axes[row_idx, col].axis("off")
    fig.tight_layout()
    fig.savefig(cfg.gen_dir / name, dpi=160)
    plt.close(fig)


def save_logit_histogram(cfg: Any, raw: Any, masked: Any) -> None:
    import matplotlib.pyplot as plt

    if raw is None or masked is None:
        return
    allowed = masked[masked > -1e8].flatten().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(raw.flatten().numpy(), bins=50)
    axes[0].set_title("raw logits")
    axes[1].hist(allowed, bins=50)
    axes[1].set_title("allowed visual logits")
    fig.tight_layout()
    fig.savefig(cfg.plots_dir / "imggen_logit_mask_histogram.png", dpi=160)
    plt.close(fig)


def evaluate_imggen(smoke: bool = False) -> Dict[str, Any]:
    cfg = get_config(smoke)
    set_seed(cfg.seed)
    device = get_device()
    lm, tokenizer, overlay, token_ids = load_finetuned(cfg, device)
    vqvae = load_vqvae_checkpoint(cfg.vqvae_path, cfg, device)
    vqvae.eval()
    enc = load_encoded(cfg.encoded_val_path)
    examples = build_imggen_examples(enc["labels"].tolist())
    selected = []
    for class_id in range(len(CLASS_NAMES)):
        class_examples = [ex for ex in examples if ex.label == class_id][:2]
        selected.extend(class_examples)
    if smoke:
        selected = selected[:3]
    rows = []
    first_raw = None
    first_masked = None
    for ex in selected:
        ids, raw, masked = generate_visual_tokens(lm, overlay, tokenizer, cfg, ex.prompt, token_ids["image_id"], device)
        if first_raw is None:
            first_raw, first_masked = raw, masked
        image, token_map = decode_visual_ids(vqvae, ids, cfg, device)
        rows.append({"class_name": ex.class_name, "prompt": ex.prompt, "token_ids": ids, "image": image, "token_map": token_map})
    save_generated_grid(cfg, rows, "generated_images.png")
    save_logit_histogram(cfg, first_raw, first_masked)

    temp_rows = []
    prompt = "Generate an image of a circle."
    for temp in [0.5, 1.0, 1.5]:
        ids, _raw, _masked = generate_visual_tokens(lm, overlay, tokenizer, cfg, prompt, token_ids["image_id"], device, temperature=temp, sample=True)
        image, token_map = decode_visual_ids(vqvae, ids, cfg, device)
        temp_rows.append({"class_name": f"T={temp}", "prompt": prompt, "token_ids": ids, "image": image, "token_map": token_map})
    save_generated_grid(cfg, temp_rows, "temperature_sweep.png")
    report = {
        "generated": [{k: v for k, v in row.items() if k not in {"image", "token_map"}} for row in rows],
        "temperature_sweep": [{k: v for k, v in row.items() if k not in {"image", "token_map"}} for row in temp_rows],
        "notes": "LM generates a flattened 1D raster of 16 visual tokens; decoded coherence can break vertical or 2D spatial continuity.",
    }
    save_json(cfg.outputs_dir / "imggen_metrics.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(evaluate_imggen(args.smoke))
