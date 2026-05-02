from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from .config import CLASS_NAMES, get_config
    from .synthetic_data import class_counts, load_synthetic, save_synthetic_cache
    from .utils import count_params, get_device, require_torch, save_json, set_seed, timer
    from .vqvae import build_vqvae_from_cfg
except ImportError:
    from config import CLASS_NAMES, get_config
    from synthetic_data import class_counts, load_synthetic, save_synthetic_cache
    from utils import count_params, get_device, require_torch, save_json, set_seed, timer
    from vqvae import build_vqvae_from_cfg


torch = require_torch()
F = torch.nn.functional


def make_loader(images: Any, labels: Any, batch_size: int, shuffle: bool = True) -> Any:
    dataset = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _mean(items: List[float]) -> float:
    return float(np.mean(items)) if items else float("nan")


def evaluate_vqvae(model: Any, images: Any, batch_size: int, device: str) -> Dict[str, float]:
    model.eval()
    logs: Dict[str, List[float]] = {"recon_loss": [], "perplexity": [], "dead_codes": []}
    counts = torch.zeros(model.quantizer.k, dtype=torch.long)
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = images[start : start + batch_size].to(device)
            recon, info = model(x)
            logs["recon_loss"].append(float(F.mse_loss(recon, x).cpu()))
            logs["perplexity"].append(float(info["perplexity"].cpu()))
            logs["dead_codes"].append(float(info["dead_codes"]))
            counts += torch.bincount(info["indices"].cpu().flatten(), minlength=model.quantizer.k)
    return {
        "recon_loss": _mean(logs["recon_loss"]),
        "perplexity": _mean(logs["perplexity"]),
        "dead_codes": int((counts == 0).sum()),
    }


def quantization_gap(model: Any, images: Any, batch_size: int, device: str) -> Dict[str, float]:
    model.eval()
    pre_losses: List[float] = []
    post_losses: List[float] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = images[start : start + batch_size].to(device)
            ze = model.encoder(x)
            x_pre = model.decoder(ze)
            _, info = model.quantizer(ze)
            x_post = model.decoder(info["zq"])
            pre_losses.append(float(F.mse_loss(x_pre, x).cpu()))
            post_losses.append(float(F.mse_loss(x_post, x).cpu()))
    pre = _mean(pre_losses)
    post = _mean(post_losses)
    return {"l_pre": pre, "l_post": post, "delta": post - pre}


def save_training_curves(cfg: Any, history: List[Dict[str, float]], name: str = "vqvae_training") -> Path:
    import matplotlib.pyplot as plt

    cfg.ensure_dirs()
    path = cfg.plots_dir / f"{name}.png"
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key in zip(axes, ["recon_loss", "perplexity", "dead_codes"]):
        ax.plot([row.get(key, float("nan")) for row in history])
        ax.set_title(key)
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_usage_histogram(cfg: Any, model: Any, images: Any, device: str, name: str = "vqvae_usage") -> Path:
    import matplotlib.pyplot as plt

    counts = torch.zeros(model.quantizer.k, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), cfg.vqvae_batch):
            idx = model.encode_to_indices(images[start : start + cfg.vqvae_batch].to(device))
            counts += torch.bincount(idx.cpu().flatten(), minlength=model.quantizer.k)
    path = cfg.plots_dir / f"{name}.png"
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(np.arange(model.quantizer.k), counts.numpy())
    ax.set_title("Code usage")
    ax.set_xlabel("code")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_codebook_heatmap(cfg: Any, model: Any, name: str = "vqvae_codebook_cosine") -> Path:
    import matplotlib.pyplot as plt

    with torch.no_grad():
        emb = F.normalize(model.quantizer.embedding.weight.detach().cpu(), dim=-1)
        sim = emb @ emb.t()
    path = cfg.plots_dir / f"{name}.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(sim.numpy(), cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_title("Codebook cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_token_maps(cfg: Any, model: Any, images: Any, labels: Any, device: str, name: str = "vqvae_token_maps") -> Path:
    import matplotlib.pyplot as plt

    selected = []
    label_np = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
    for class_id in range(len(CLASS_NAMES)):
        hits = np.flatnonzero(label_np == class_id)
        if len(hits):
            selected.append(int(hits[0]))
    x = images[selected].to(device)
    with torch.no_grad():
        recon, info = model(x)
    path = cfg.token_maps_dir / f"{name}.png"
    fig, axes = plt.subplots(len(selected), 3, figsize=(7, 2.2 * len(selected)))
    for row, idx in enumerate(selected):
        axes[row, 0].imshow(images[idx].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[row, 0].set_ylabel(CLASS_NAMES[int(labels[idx])])
        axes[row, 1].imshow(recon[row].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1))
        im = axes[row, 2].imshow(info["indices"][row].detach().cpu().numpy(), cmap="tab20")
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    axes[0, 0].set_title("original")
    axes[0, 1].set_title("reconstruction")
    axes[0, 2].set_title("4x4 codes")
    fig.colorbar(im, ax=axes[:, 2].ravel().tolist(), shrink=0.6)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def train_vqvae(
    smoke: bool = False,
    use_ema: bool = False,
    k: int | None = None,
    beta: float | None = None,
    max_epochs: int | None = None,
    output_name: str = "vqvae_best",
) -> Dict[str, Any]:
    cfg = get_config(smoke)
    if k is not None:
        cfg.k = k
    if beta is not None:
        cfg.beta = beta
    set_seed(cfg.seed)
    cfg.ensure_dirs()
    save_synthetic_cache(cfg)
    train = load_synthetic(cfg.synthetic_train_path)
    val = load_synthetic(cfg.synthetic_val_path)
    device = get_device()
    model = build_vqvae_from_cfg(cfg, use_ema=use_ema).to(device)
    params = list(model.encoder.parameters()) + list(model.decoder.parameters()) if use_ema else list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.vqvae_lr)
    loader = make_loader(train["images"], train["labels"], cfg.vqvae_batch, shuffle=True)
    history: List[Dict[str, float]] = []

    with timer() as elapsed:
        for epoch in range(1, (max_epochs or cfg.vqvae_epochs) + 1):
            model.train()
            epoch_logs: Dict[str, List[float]] = {"loss": [], "recon_loss": [], "codebook_loss": [], "commit_loss": [], "perplexity": [], "dead_codes": []}
            for images, _labels in loader:
                images = images.to(device)
                recon, info = model(images)
                loss, logs = model.loss(recon, images, info)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                for key, value in logs.items():
                    epoch_logs.setdefault(key, []).append(value)
            val_metrics = evaluate_vqvae(model, val["images"], cfg.vqvae_batch, device)
            row = {key: _mean(values) for key, values in epoch_logs.items()}
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            row["epoch"] = epoch
            history.append(row)
            print(row)

    gap = quantization_gap(model, val["images"], cfg.vqvae_batch, device)
    ckpt_path = cfg.weights_dir / f"{output_name}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "use_ema": use_ema,
            "k": cfg.k,
            "beta": cfg.beta,
        },
        ckpt_path,
    )
    plots = {
        "curves": str(save_training_curves(cfg, history, output_name)),
        "usage": str(save_usage_histogram(cfg, model, val["images"], device, output_name + "_usage")),
        "heatmap": str(save_codebook_heatmap(cfg, model, output_name + "_cosine")),
        "token_maps": str(save_token_maps(cfg, model, val["images"], val["labels"], device, output_name + "_token_maps")),
    }
    metrics = {
        "checkpoint": str(ckpt_path),
        "use_ema": use_ema,
        "k": cfg.k,
        "beta": cfg.beta,
        "params": count_params(model),
        "train_class_counts": class_counts(train["labels"].tolist()),
        "val_class_counts": class_counts(val["labels"].tolist()),
        "history": history,
        "quantization_gap": gap,
        "plots": plots,
        "seconds": elapsed["seconds"],
    }
    save_json(cfg.outputs_dir / f"{output_name}_metrics.json", metrics)
    return metrics


def load_vqvae_checkpoint(path: Path, cfg: Any, device: str, use_ema: bool | None = None) -> Any:
    ckpt = torch.load(path, map_location=device)
    model = build_vqvae_from_cfg(cfg, use_ema=bool(ckpt.get("use_ema", False) if use_ema is None else use_ema), k=int(ckpt.get("k", cfg.k)), beta=float(ckpt.get("beta", cfg.beta)))
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--name", default="vqvae_best")
    args = parser.parse_args()
    print(train_vqvae(args.smoke, args.ema, args.k, args.beta, args.epochs, args.name))
