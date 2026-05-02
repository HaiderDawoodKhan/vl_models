from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import numpy as np


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. Install dependencies with `python3 -m pip install -r requirements.txt`.") from exc
    return torch


def get_device(prefer_mps: bool = False) -> str:
    torch = require_torch()
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch = require_torch()
    except RuntimeError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def save_json(path: Path, data: Dict[str, Any] | List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_params(model: Any) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model: Any) -> Dict[str, int | float]:
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return {"trainable": trainable, "total": total, "percent": 100.0 * trainable / total if total else 0.0}


def chunked(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def cycle_loader(loader: Iterable[Any]) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def normalize_answer(text: str) -> str:
    return " ".join(text.lower().strip().replace(".", "").replace("\n", " ").split())


def perplexity_from_loss(loss: float) -> float:
    return float(math.exp(min(20.0, loss)))


def format_alpaca(example: Dict[str, Any]) -> str:
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


@contextmanager
def timer() -> Iterable[Dict[str, float]]:
    stats = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield stats
    finally:
        stats["seconds"] = time.perf_counter() - start

