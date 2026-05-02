from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    from .config import CLASS_NAMES, PartBConfig
    from .utils import require_torch, set_seed
except ImportError:
    from config import CLASS_NAMES, PartBConfig
    from utils import require_torch, set_seed


GEOMETRIC = {"triangle", "circle", "cross", "checkerboard"}
NON_GEOMETRIC = {"spiral", "gradient"}
SYMMETRY_AXES = {
    "spiral": "0",
    "triangle": "3",
    "circle": "infinite",
    "cross": "4",
    "checkerboard": "4",
    "gradient": "1",
}


@dataclass
class VQAExample:
    index: int
    label: int
    class_name: str
    question: str
    answer: str
    template: str


@dataclass
class ImageGenExample:
    index: int
    label: int
    class_name: str
    prompt: str


def _rgb(mask: np.ndarray, color: np.ndarray, bg: float = 0.0) -> np.ndarray:
    img = np.full((3, mask.shape[0], mask.shape[1]), bg, dtype=np.float32)
    for c in range(3):
        img[c] = np.where(mask, color[c], img[c])
    return img


def make_checkerboard(size: int = 16, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    block = int(rng.choice([1, 2, 4]))
    y, x = np.indices((size, size))
    mask = ((y // block + x // block) % 2).astype(np.float32)
    tint = rng.uniform(0.6, 1.0, size=3).astype(np.float32)
    return np.stack([mask * tint[c] for c in range(3)], axis=0)


def make_gradient(size: int = 16, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    x = np.linspace(0, 1, size, dtype=np.float32)
    base = np.tile(x[None, :], (size, 1))
    if rng.random() < 0.5:
        base = base.T
    tint = rng.uniform(0.5, 1.0, size=3).astype(np.float32)
    return np.stack([base * tint[c] for c in range(3)], axis=0)


def make_circle(size: int = 16, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    y, x = np.indices((size, size))
    cx, cy = size / 2 + rng.uniform(-1.2, 1.2, size=2)
    radius = rng.uniform(4.0, 6.0)
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    return _rgb(mask, rng.uniform(0.45, 1.0, size=3).astype(np.float32))


def make_cross(size: int = 16, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    y, x = np.indices((size, size))
    cx, cy = int(size / 2 + rng.integers(-1, 2)), int(size / 2 + rng.integers(-1, 2))
    width = int(rng.choice([1, 2]))
    mask = (np.abs(x - cx) <= width) | (np.abs(y - cy) <= width)
    return _rgb(mask, rng.uniform(0.45, 1.0, size=3).astype(np.float32))


def make_triangle(size: int = 16, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    top = np.array([size / 2 + rng.uniform(-1, 1), 2 + rng.uniform(-0.5, 1.0)])
    left = np.array([3 + rng.uniform(-1, 1), size - 3 + rng.uniform(-1, 1)])
    right = np.array([size - 4 + rng.uniform(-1, 1), size - 3 + rng.uniform(-1, 1)])
    y, x = np.indices((size, size))
    pts = np.stack([x, y], axis=-1).astype(np.float32)

    def sign(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        return (p1[..., 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[..., 1] - p3[1])

    b1 = sign(pts, top, left) < 0
    b2 = sign(pts, left, right) < 0
    b3 = sign(pts, right, top) < 0
    mask = (b1 == b2) & (b2 == b3)
    return _rgb(mask, rng.uniform(0.45, 1.0, size=3).astype(np.float32))


def make_spiral(size: int = 16, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    y, x = np.indices((size, size))
    cx, cy = size / 2 + rng.uniform(-0.8, 0.8, size=2)
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    turns = rng.uniform(1.8, 2.5)
    target = (theta + np.pi) / (2 * np.pi) * turns * size / 2
    mask = np.abs(r - target) < 0.9
    mask &= r < size / 2 - 1
    return _rgb(mask, rng.uniform(0.45, 1.0, size=3).astype(np.float32))


GENERATORS = {
    "spiral": make_spiral,
    "triangle": make_triangle,
    "circle": make_circle,
    "cross": make_cross,
    "checkerboard": make_checkerboard,
    "gradient": make_gradient,
}


def generate_synthetic_dataset(cfg: PartBConfig) -> Tuple[Any, Any]:
    torch = require_torch()
    rng = np.random.default_rng(cfg.seed)
    images: List[np.ndarray] = []
    labels: List[int] = []
    for label, class_name in enumerate(CLASS_NAMES):
        for _ in range(cfg.n_per_class):
            img = GENERATORS[class_name](cfg.img_size, rng)
            noise = rng.normal(0, 0.03, img.shape).astype(np.float32)
            images.append(np.clip(img + noise, 0, 1).astype(np.float32))
            labels.append(label)
    return torch.tensor(np.stack(images), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def stratified_split(labels: Sequence[int], cfg: PartBConfig) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(cfg.seed)
    labels_np = np.asarray(labels)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for class_id in range(cfg.num_classes):
        idx = np.flatnonzero(labels_np == class_id)
        rng.shuffle(idx)
        n_train = int(len(idx) * cfg.train_frac)
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def save_synthetic_cache(cfg: PartBConfig, force: bool = False) -> Dict[str, Any]:
    torch = require_torch()
    set_seed(cfg.seed)
    cfg.ensure_dirs()
    if cfg.synthetic_train_path.exists() and cfg.synthetic_val_path.exists() and not force:
        train = torch.load(cfg.synthetic_train_path, map_location="cpu")
        val = torch.load(cfg.synthetic_val_path, map_location="cpu")
        expected_train = cfg.num_classes * int(cfg.n_per_class * cfg.train_frac)
        expected_val = cfg.num_classes * cfg.n_per_class - expected_train
        if len(train.get("images", [])) == expected_train and len(val.get("images", [])) == expected_val:
            return {"train": str(cfg.synthetic_train_path), "val": str(cfg.synthetic_val_path), "cached": True}
    images, labels = generate_synthetic_dataset(cfg)
    train_idx, val_idx = stratified_split(labels.tolist(), cfg)
    train = {"images": images[train_idx], "labels": labels[train_idx], "indices": train_idx, "class_names": CLASS_NAMES}
    val = {"images": images[val_idx], "labels": labels[val_idx], "indices": val_idx, "class_names": CLASS_NAMES}
    torch.save(train, cfg.synthetic_train_path)
    torch.save(val, cfg.synthetic_val_path)
    save_synthetic_grid(cfg, images, labels)
    return {"train": str(cfg.synthetic_train_path), "val": str(cfg.synthetic_val_path), "cached": False}


def load_synthetic(path: Path) -> Dict[str, Any]:
    torch = require_torch()
    data = torch.load(path, map_location="cpu")
    if "images" not in data or "labels" not in data:
        raise ValueError(f"{path} is not a synthetic cache.")
    return data


def save_synthetic_grid(cfg: PartBConfig, images: Any, labels: Any) -> Path:
    import matplotlib.pyplot as plt

    cfg.ensure_dirs()
    fig, axes = plt.subplots(cfg.num_classes, 5, figsize=(8, 9))
    labels_np = labels.detach().cpu().numpy()
    for class_id, class_name in enumerate(CLASS_NAMES):
        idx = np.flatnonzero(labels_np == class_id)[:5]
        for col, image_idx in enumerate(idx):
            ax = axes[class_id, col]
            img = images[image_idx].detach().cpu().permute(1, 2, 0).numpy()
            ax.imshow(np.clip(img, 0, 1))
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(class_name)
    fig.tight_layout()
    path = cfg.outputs_dir / "synthetic_grid.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def build_vqa_examples(labels: Sequence[int], indices: Sequence[int] | None = None) -> List[VQAExample]:
    rng = np.random.default_rng(42)
    indices = list(range(len(labels))) if indices is None else list(indices)
    examples: List[VQAExample] = []
    for row, (idx, label) in enumerate(zip(indices, labels)):
        class_name = CLASS_NAMES[int(label)]
        wrong = [name for name in CLASS_NAMES if name != class_name]
        query_class = class_name if row % 2 == 0 else str(rng.choice(wrong))
        rows = [
            ("What shape is in this image?", class_name, "recognition"),
            (f"Is there a {query_class}?", "yes" if query_class == class_name else "no", "binary"),
            ("Geometric or non-geometric?", "geometric" if class_name in GEOMETRIC else "non-geometric", "abstraction"),
            ("How many axes of symmetry?", SYMMETRY_AXES[class_name], "symmetry"),
        ]
        for question, answer, template in rows:
            examples.append(VQAExample(int(idx), int(label), class_name, question, answer, template))
    return examples


def build_imggen_examples(labels: Sequence[int], indices: Sequence[int] | None = None) -> List[ImageGenExample]:
    templates = ["Generate an image of a {}.", "Draw a small {} pattern.", "Create a 16 by 16 image showing a {}."]
    indices = list(range(len(labels))) if indices is None else list(indices)
    examples: List[ImageGenExample] = []
    for row, (idx, label) in enumerate(zip(indices, labels)):
        class_name = CLASS_NAMES[int(label)]
        examples.append(ImageGenExample(int(idx), int(label), class_name, templates[row % len(templates)].format(class_name)))
    return examples


def class_counts(labels: Sequence[int]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in labels:
        name = CLASS_NAMES[int(label)]
        counts[name] = counts.get(name, 0) + 1
    return counts


if __name__ == "__main__":
    import argparse
    try:
        from .config import get_config
    except ImportError:
        from config import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(save_synthetic_cache(get_config(args.smoke), force=args.force))
