from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

try:
    from .config import CLASS_NAMES
    from .synthetic_data import build_imggen_examples, build_vqa_examples
    from .utils import format_alpaca, require_torch
except ImportError:
    from config import CLASS_NAMES
    from synthetic_data import build_imggen_examples, build_vqa_examples
    from utils import format_alpaca, require_torch


torch = require_torch()


@dataclass
class TokenizedExample:
    input_ids: List[int]
    labels: List[int]
    meta: Dict[str, Any]


def visual_ids_from_indices(indices: Any, vtxt: int) -> List[int]:
    return (indices.reshape(-1).long() + vtxt + 2).tolist()


def encode_multimodal(tokenizer: Any, image_id: int, end_image_id: int, question: str, answer: str, visual_ids: Sequence[int]) -> TokenizedExample:
    bos = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id]
    q_ids = tokenizer(question, add_special_tokens=False).input_ids
    a_ids = tokenizer(answer, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    input_ids = bos + [image_id] + list(visual_ids) + [end_image_id] + q_ids + a_ids
    prefix_len = len(bos) + 1 + len(visual_ids) + 1 + len(q_ids)
    labels = [-100] * prefix_len + a_ids
    return TokenizedExample(input_ids, labels, {"mode": "vqa", "question": question, "answer": answer})


def encode_imagegen(tokenizer: Any, image_id: int, end_image_id: int, prompt: str, visual_ids: Sequence[int], supervise_eos: bool = True) -> TokenizedExample:
    bos = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id]
    p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    tail = [end_image_id, tokenizer.eos_token_id] if supervise_eos else [end_image_id]
    input_ids = bos + p_ids + [image_id] + list(visual_ids) + tail
    prefix_len = len(bos) + len(p_ids) + 1
    labels = [-100] * prefix_len + list(visual_ids) + tail
    return TokenizedExample(input_ids, labels, {"mode": "imggen", "prompt": prompt})


def encode_text_replay(tokenizer: Any, text: str, max_length: int = 512) -> TokenizedExample:
    ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length).input_ids
    return TokenizedExample(ids, ids.copy(), {"mode": "text"})


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, examples: Sequence[TokenizedExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TokenizedExample:
        return self.examples[idx]


class Collator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: Sequence[TokenizedExample]) -> Dict[str, Any]:
        max_len = max(len(ex.input_ids) for ex in batch)
        input_ids = []
        labels = []
        for ex in batch:
            pad = max_len - len(ex.input_ids)
            input_ids.append([self.pad_token_id] * pad + ex.input_ids)
            labels.append([-100] * pad + ex.labels)
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return {"input_ids": input_tensor, "labels": label_tensor, "attention_mask": (input_tensor != self.pad_token_id).long(), "meta": [ex.meta for ex in batch]}


def build_tokenized_vqa(tokenizer: Any, image_id: int, end_image_id: int, labels: Any, encoded_indices: Any, vtxt: int) -> List[TokenizedExample]:
    examples = build_vqa_examples(labels.tolist())
    rows: List[TokenizedExample] = []
    for ex in examples:
        vis = visual_ids_from_indices(encoded_indices[ex.index], vtxt)
        row = encode_multimodal(tokenizer, image_id, end_image_id, ex.question, ex.answer, vis)
        row.meta.update(ex.__dict__)
        rows.append(row)
    return rows


def build_tokenized_imggen(tokenizer: Any, image_id: int, end_image_id: int, labels: Any, encoded_indices: Any, vtxt: int) -> List[TokenizedExample]:
    examples = build_imggen_examples(labels.tolist())
    rows: List[TokenizedExample] = []
    for ex in examples:
        vis = visual_ids_from_indices(encoded_indices[ex.index], vtxt)
        row = encode_imagegen(tokenizer, image_id, end_image_id, ex.prompt, vis)
        row.meta.update(ex.__dict__)
        rows.append(row)
    return rows


def build_tokenized_alpaca(tokenizer: Any, limit: int) -> List[TokenizedExample]:
    from datasets import load_dataset

    alpaca = load_dataset("tatsu-lab/alpaca", split="train").select(range(limit))
    return [encode_text_replay(tokenizer, format_alpaca(dict(row))) for row in alpaca]


def token_type_sequence(input_ids: Sequence[int], labels: Sequence[int], cfg: Any, image_id: int, end_image_id: int) -> str:
    parts = []
    for token_id, label in zip(input_ids, labels):
        if token_id == image_id:
            kind = "<image>"
        elif token_id == end_image_id:
            kind = "</image>"
        elif cfg.vtxt + 2 <= token_id < cfg.vtxt + 2 + cfg.k:
            kind = "VIS"
        elif token_id >= cfg.vtxt:
            kind = "NEW"
        else:
            kind = "TXT"
        parts.append(f"{kind}:{'L' if label != -100 else '_'}")
    return " / ".join(parts)
