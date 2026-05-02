from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .config import VLMConfig
from .data import build_vqa_examples, load_clip_cache
from .model import build_connector, load_connector_checkpoint, load_lm_and_tokenizer, select_visual_tokens
from .utils import require_torch, save_json, set_seed


def fixed_indices(n: int, seed: int = 42, count: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(n, min(count, n), replace=False)


def compute_modality_gap(
    cfg: VLMConfig,
    cache_path: Path,
    checkpoint_dir: Path | None,
    output_name: str,
    representation: str = "patches",
    count: int = 200,
) -> Dict[str, Any]:
    torch = require_torch()
    import torch.nn.functional as F

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = load_clip_cache(cache_path)
    features = select_visual_tokens(cache["features"], representation)
    labels = cache["labels"]
    indices = fixed_indices(len(features), cfg.seed, count)
    examples = build_vqa_examples(labels[indices].tolist(), indices=indices)
    texts = [ex.question for ex in examples if ex.template == "object_recognition"][: len(indices)]
    lm, tokenizer = load_lm_and_tokenizer(cfg, device, freeze=True)
    connector = build_connector(cfg).to(device)
    if checkpoint_dir and (checkpoint_dir / "connector.pt").exists():
        load_connector_checkpoint(connector, checkpoint_dir, device)
    connector.eval()
    lm.eval()
    with torch.no_grad():
        z = features[indices].to(device)
        v = connector(z)
        v_repr = F.normalize(v.mean(dim=1), dim=-1)
        encoded = tokenizer(texts, padding=True, return_tensors="pt").to(device)
        embs = lm.get_input_embeddings()(encoded.input_ids)
        mask = encoded.attention_mask.unsqueeze(-1)
        t_repr = (embs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        t_repr = F.normalize(t_repr, dim=-1)
        v_bar = F.normalize(v_repr.mean(dim=0), dim=0)
        t_bar = F.normalize(t_repr.mean(dim=0), dim=0)
        visual_sim = v_repr @ v_repr.T
        text_sim = t_repr @ t_repr.T
        cross_sim = v_repr @ t_repr.T
        eye = ~torch.eye(v_repr.shape[0], dtype=torch.bool, device=device)
        metrics = {
            "output_name": output_name,
            "mg": float(torch.norm(v_bar - t_bar).detach().cpu()),
            "within_visual": float(visual_sim[eye].mean().detach().cpu()),
            "within_text": float(text_sim[eye].mean().detach().cpu()),
            "cross_modal": float(cross_sim.mean().detach().cpu()),
            "matched_cross": float(cross_sim.diag().mean().detach().cpu()),
            "rnorm": float((v.norm(dim=-1).mean() / lm.get_input_embeddings().weight.norm(dim=-1).mean()).detach().cpu()),
        }
    save_json(cfg.metrics_dir / f"modality_gap_{output_name}.json", metrics)
    save_umap_plot(cfg, output_name, v_repr.detach().cpu().numpy(), t_repr.detach().cpu().numpy())
    return metrics


def save_umap_plot(cfg: VLMConfig, output_name: str, visual: np.ndarray, text: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
        import umap
    except ImportError:
        return
    reducer = umap.UMAP(random_state=cfg.seed)
    coords = reducer.fit_transform(np.concatenate([visual, text], axis=0))
    n = len(visual)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter(coords[:n, 0], coords[:n, 1], s=14, label="visual", alpha=0.8)
    plt.scatter(coords[n:, 0], coords[n:, 1], s=14, label="text", alpha=0.8)
    plt.title(f"UMAP modality gap: {output_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.plots_dir / f"umap_{output_name}.png", dpi=160)
    plt.close()
