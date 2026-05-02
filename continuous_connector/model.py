from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .config import VLMConfig
from .utils import ensure_pad_token, require_torch


def _torch():
    return require_torch()


class MLPConnector:
    """Factory wrapper so importing this module does not require torch."""

    @staticmethod
    def build(d_in: int = 768, d_hidden: int = 960, d_out: int = 960) -> Any:
        torch = _torch()
        nn = torch.nn

        class _MLPConnector(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d_in, d_hidden),
                    nn.GELU(),
                    nn.Linear(d_hidden, d_out),
                )
                self.apply(self._init_weights)

            def _init_weights(self, module: Any) -> None:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            def forward(self, z: Any) -> Any:
                return self.net(z)

        return _MLPConnector()


def load_clip_vision(cfg: VLMConfig, device: str) -> Any:
    from transformers import CLIPVisionModel

    clip = CLIPVisionModel.from_pretrained(cfg.clip_model).to(device)
    clip.eval()
    for param in clip.parameters():
        param.requires_grad = False
    return clip


def load_clip_processor(cfg: VLMConfig) -> Any:
    from transformers import CLIPImageProcessor

    return CLIPImageProcessor.from_pretrained(cfg.clip_model)


def load_lm_and_tokenizer(cfg: VLMConfig, device: str, freeze: bool = True) -> Tuple[Any, Any]:
    torch = _torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model, use_fast=True)
    ensure_pad_token(tokenizer)
    lm = AutoModelForCausalLM.from_pretrained(cfg.lm_model, torch_dtype=dtype).to(device)
    if freeze:
        for param in lm.parameters():
            param.requires_grad = False
    return lm, tokenizer


def apply_lora(lm: Any, cfg: VLMConfig) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(lm, config)


def build_connector(cfg: VLMConfig) -> Any:
    return MLPConnector.build(cfg.d_vision, cfg.d_lm, cfg.d_lm)


def select_visual_tokens(features: Any, representation: str = "patches") -> Any:
    if representation == "patches":
        return features[:, 1:, :] if features.shape[1] == 50 else features
    if representation == "cls":
        return features[:, :1, :]
    if representation == "mean_pool":
        patch_features = features[:, 1:, :] if features.shape[1] == 50 else features
        return patch_features.mean(dim=1, keepdim=True)
    raise ValueError(f"Unknown visual representation: {representation}")


def connector_state_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "connector.pt"


def save_checkpoint(checkpoint_dir: Path, connector: Any, lm: Any | None = None, extra: Dict[str, Any] | None = None) -> None:
    torch = _torch()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(connector.state_dict(), connector_state_path(checkpoint_dir))
    if lm is not None and hasattr(lm, "save_pretrained"):
        lm.save_pretrained(checkpoint_dir / "lora")
    if extra:
        torch.save(extra, checkpoint_dir / "training_state.pt")


def load_connector_checkpoint(connector: Any, checkpoint_dir: Path, device: str) -> Any:
    torch = _torch()
    path = connector_state_path(checkpoint_dir)
    connector.load_state_dict(torch.load(path, map_location=device))
    return connector


def build_caption_batch(
    connector: Any,
    lm: Any,
    tokenizer: Any,
    features: Any,
    captions: Sequence[str],
    device: str,
) -> Dict[str, Any]:
    torch = _torch()
    features = features.to(device)
    visual = connector(features).to(lm.dtype if hasattr(lm, "dtype") else torch.float32)
    encoded = tokenizer(list(captions), add_special_tokens=False, padding=True, return_tensors="pt").to(device)
    caption_ids = encoded.input_ids
    caption_mask = encoded.attention_mask
    embed = lm.get_input_embeddings()
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    bos_ids = torch.full((features.shape[0], 1), bos_id, dtype=torch.long, device=device)
    bos_emb = embed(bos_ids)
    text_emb = embed(caption_ids)
    inputs_embeds = torch.cat([bos_emb, visual, text_emb], dim=1)
    prefix_len = 1 + visual.shape[1]
    prefix_labels = torch.full((features.shape[0], prefix_len), -100, dtype=torch.long, device=device)
    caption_labels = caption_ids.clone()
    caption_labels[caption_mask == 0] = -100
    labels = torch.cat([prefix_labels, caption_labels], dim=1)
    attention_mask = torch.cat(
        [
            torch.ones((features.shape[0], prefix_len), dtype=torch.long, device=device),
            caption_mask,
        ],
        dim=1,
    )
    return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask, "labels": labels}


def _append_eos_and_mask(ids: Any, mask: Any, eos_id: int) -> Tuple[Any, Any]:
    torch = _torch()
    eos_col = torch.full((ids.shape[0], 1), eos_id, dtype=ids.dtype, device=ids.device)
    eos_mask = torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)
    return torch.cat([ids, eos_col], dim=1), torch.cat([mask, eos_mask], dim=1)


def build_vqa_batch(
    connector: Any,
    lm: Any,
    tokenizer: Any,
    features: Any,
    questions: Sequence[str],
    answers: Sequence[str],
    device: str,
    include_answer: bool = True,
) -> Dict[str, Any]:
    torch = _torch()
    features = features.to(device)
    visual = connector(features).to(lm.dtype if hasattr(lm, "dtype") else torch.float32)
    q = tokenizer(list(questions), add_special_tokens=False, padding=True, return_tensors="pt").to(device)
    embed = lm.get_input_embeddings()
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    bos_ids = torch.full((features.shape[0], 1), bos_id, dtype=torch.long, device=device)
    bos_emb = embed(bos_ids)
    q_emb = embed(q.input_ids)
    prefix_embeds = torch.cat([bos_emb, visual, q_emb], dim=1)
    prefix_mask = torch.cat(
        [
            torch.ones((features.shape[0], 1 + visual.shape[1]), dtype=torch.long, device=device),
            q.attention_mask,
        ],
        dim=1,
    )
    if not include_answer:
        return {"inputs_embeds": prefix_embeds, "attention_mask": prefix_mask}

    a = tokenizer(list(answers), add_special_tokens=False, padding=True, return_tensors="pt").to(device)
    a_ids, a_mask = _append_eos_and_mask(a.input_ids, a.attention_mask, tokenizer.eos_token_id)
    a_emb = embed(a_ids)
    inputs_embeds = torch.cat([prefix_embeds, a_emb], dim=1)
    answer_labels = a_ids.clone()
    answer_labels[a_mask == 0] = -100
    labels = torch.cat(
        [
            torch.full((features.shape[0], prefix_embeds.shape[1]), -100, dtype=torch.long, device=device),
            answer_labels,
        ],
        dim=1,
    )
    attention_mask = torch.cat([prefix_mask, a_mask], dim=1)
    return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask, "labels": labels}


def build_text_lm_batch(tokenizer: Any, texts: Sequence[str], device: str, max_length: int = 512) -> Dict[str, Any]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    labels = encoded.input_ids.clone()
    labels[encoded.attention_mask == 0] = -100
    return {"input_ids": encoded.input_ids, "attention_mask": encoded.attention_mask, "labels": labels}


def greedy_generate_from_embeds(
    lm: Any,
    tokenizer: Any,
    inputs_embeds: Any,
    attention_mask: Any,
    max_new_tokens: int,
) -> List[str]:
    outputs = lm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def compute_rnorm(connector: Any, lm: Any, features: Any, device: str) -> float:
    torch = _torch()
    with torch.no_grad():
        visual = connector(features.to(device))
        visual_norm = visual.norm(dim=-1).mean()
        text_norm = lm.get_input_embeddings().weight.norm(dim=-1).mean()
    return float((visual_norm / text_norm).detach().cpu())
