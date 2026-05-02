from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .config import VLMConfig
from .utils import format_alpaca, require_torch


CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CAPTION_TEMPLATES = [
    "a photo of a {}.",
    "this is a {}.",
    "the image shows a {}.",
    "an image of a {}.",
    "there is a {} in the picture.",
    "this picture contains a {}.",
]

VQA_TEMPLATE_NAMES = [
    "object_recognition",
    "class_presence",
    "vehicle_or_living",
    "can_fly",
    "is_animal",
]

VEHICLE_CLASSES = {"airplane", "automobile", "ship", "truck"}
ANIMAL_CLASSES = {"bird", "cat", "deer", "dog", "frog", "horse"}
FLYING_CLASSES = {"airplane", "bird"}


@dataclass
class CaptionExample:
    index: int
    label: int
    class_name: str
    caption: str


@dataclass
class VQAExample:
    index: int
    label: int
    class_name: str
    question: str
    answer: str
    template: str


def load_cifar10_subset(cfg: VLMConfig, train: bool, smoke: bool = False) -> Any:
    try:
        from torchvision.datasets import CIFAR10
    except ImportError as exc:
        raise RuntimeError("torchvision is required to load CIFAR-10.") from exc

    per_class = cfg.train_per_class if train else cfg.test_per_class
    if smoke:
        per_class = min(per_class, 2 if train else 1)
    dataset = CIFAR10(root=str(cfg.data_dir), train=train, download=True)
    targets = np.asarray(dataset.targets)
    rng = np.random.default_rng(cfg.seed)
    selected: List[int] = []
    for class_id in range(len(CLASSES)):
        candidates = np.flatnonzero(targets == class_id)
        chosen = rng.choice(candidates, size=per_class, replace=False)
        selected.extend(int(i) for i in chosen)
    selected = sorted(selected)
    return dataset, selected


def build_caption_examples(labels: Sequence[int], indices: Sequence[int] | None = None) -> List[CaptionExample]:
    if indices is None:
        indices = list(range(len(labels)))
    examples: List[CaptionExample] = []
    for row, (feature_idx, label) in enumerate(zip(indices, labels)):
        class_name = CLASSES[int(label)]
        template = CAPTION_TEMPLATES[row % len(CAPTION_TEMPLATES)]
        examples.append(
            CaptionExample(
                index=int(feature_idx),
                label=int(label),
                class_name=class_name,
                caption=template.format(class_name),
            )
        )
    return examples


def build_vqa_examples(labels: Sequence[int], indices: Sequence[int] | None = None) -> List[VQAExample]:
    if indices is None:
        indices = list(range(len(labels)))
    examples: List[VQAExample] = []
    for feature_idx, label in zip(indices, labels):
        class_name = CLASSES[int(label)]
        rows = [
            ("What object is shown?", class_name, "object_recognition"),
            (f"Is there a {class_name}?", "yes", "class_presence"),
            (
                "Vehicle or living thing?",
                "vehicle" if class_name in VEHICLE_CLASSES else "living thing",
                "vehicle_or_living",
            ),
            ("Can it fly?", "yes" if class_name in FLYING_CLASSES else "no", "can_fly"),
            ("Is this an animal?", "yes" if class_name in ANIMAL_CLASSES else "no", "is_animal"),
        ]
        for question, answer, template in rows:
            examples.append(
                VQAExample(
                    index=int(feature_idx),
                    label=int(label),
                    class_name=class_name,
                    question=question,
                    answer=answer,
                    template=template,
                )
            )
    return examples


def load_clip_cache(path: Path) -> Dict[str, Any]:
    torch = require_torch()
    data = torch.load(path, map_location="cpu")
    if "features" not in data or "labels" not in data:
        raise ValueError(f"{path} does not look like a CLIP cache.")
    return data


def save_clip_cache(path: Path, features: Any, labels: Any, indices: Sequence[int]) -> None:
    torch = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features.cpu(),
            "labels": labels.cpu(),
            "indices": list(map(int, indices)),
            "class_names": CLASSES,
        },
        path,
    )


def load_alpaca_replay(limit: int) -> List[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load Alpaca replay data.") from exc
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca = alpaca.select(range(min(limit, len(alpaca))))
    return [format_alpaca(dict(row)) for row in alpaca]


def majority_vote_by_template(vqa_examples: Sequence[VQAExample]) -> Dict[str, str]:
    counts: Dict[str, Dict[str, int]] = {}
    for ex in vqa_examples:
        counts.setdefault(ex.template, {})
        counts[ex.template][ex.answer] = counts[ex.template].get(ex.answer, 0) + 1
    return {template: max(answer_counts.items(), key=lambda x: x[1])[0] for template, answer_counts in counts.items()}


def dataclass_to_dict(items: Sequence[Any]) -> List[Dict[str, Any]]:
    return [item.__dict__.copy() for item in items]
