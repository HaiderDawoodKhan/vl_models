from __future__ import annotations

import argparse
from pathlib import Path

from .config import get_config
from .data import load_cifar10_subset, save_clip_cache
from .model import load_clip_processor, load_clip_vision
from .utils import get_device, require_torch, set_seed


def cache_split(train: bool, smoke: bool = False, include_cls: bool = False, batch_size: int = 64) -> Path:
    torch = require_torch()
    cfg = get_config(smoke=smoke)
    cfg.ensure_dirs()
    set_seed(cfg.seed)
    device = get_device()
    dataset, indices = load_cifar10_subset(cfg, train=train, smoke=smoke)
    processor = load_clip_processor(cfg)
    clip = load_clip_vision(cfg, device)
    all_features = []
    labels = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            images = [dataset[i][0] for i in batch_indices]
            batch_labels = [dataset[i][1] for i in batch_indices]
            pixels = processor(images=images, return_tensors="pt").pixel_values.to(device)
            hidden = clip(pixel_values=pixels).last_hidden_state
            tokens = hidden if include_cls else hidden[:, 1:, :]
            if not include_cls and tokens.shape[1] != cfg.num_patches:
                raise RuntimeError(f"Expected {cfg.num_patches} patches, got {tokens.shape[1]}.")
            all_features.append(tokens.detach().cpu().float())
            labels.extend(batch_labels)
    features = torch.cat(all_features, dim=0)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    path = cfg.train_cache if train else cfg.test_cache
    if include_cls:
        path = path.with_name(path.stem + "_with_cls.pt")
    save_clip_cache(path, features, label_tensor, indices)
    print(f"saved {path} features={tuple(features.shape)} labels={tuple(label_tensor.shape)}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache CIFAR-10 CLIP patch features.")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--include-cls", action="store_true", help="Cache all 50 CLIP tokens for CLS ablations.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.split in {"train", "both"}:
        cache_split(True, smoke=args.smoke, include_cls=args.include_cls, batch_size=args.batch_size)
    if args.split in {"test", "both"}:
        cache_split(False, smoke=args.smoke, include_cls=args.include_cls, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
