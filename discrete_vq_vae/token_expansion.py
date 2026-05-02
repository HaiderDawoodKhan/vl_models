from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from .utils import count_trainable_params, ensure_pad_token, require_torch
except ImportError:
    from utils import count_trainable_params, ensure_pad_token, require_torch


torch = require_torch()
nn = torch.nn
F = torch.nn.functional


def visual_token_strings(k: int) -> List[str]:
    return [f"<vis_{i}>" for i in range(k)]


def add_visual_tokens(tokenizer: Any, lm: Any, cfg: Any) -> Dict[str, int]:
    tokens = ["<image>", "</image>"] + visual_token_strings(cfg.k)
    tokenizer.add_tokens(tokens)
    lm.resize_token_embeddings(len(tokenizer))
    image_id = tokenizer.convert_tokens_to_ids("<image>")
    end_image_id = tokenizer.convert_tokens_to_ids("</image>")
    if len(tokenizer) != cfg.v_total:
        print(f"Warning: tokenizer length is {len(tokenizer)}, expected {cfg.v_total}. Using actual tokenizer length for LM.")
    emb = lm.get_input_embeddings().weight.data
    mean_text = emb[: cfg.vtxt].mean(dim=0)
    emb[image_id].copy_(mean_text)
    emb[end_image_id].copy_(mean_text)
    for token in visual_token_strings(cfg.k):
        emb[tokenizer.convert_tokens_to_ids(token)].zero_()
    return {"image_id": image_id, "end_image_id": end_image_id, "first_visual_id": cfg.vtxt + 2}


class OverlayEmbedding(nn.Module):
    def __init__(self, base_emb: Any, vtxt: int = 49152, new_tokens: int = 258) -> None:
        super().__init__()
        self.base_emb = base_emb
        self.vtxt = vtxt
        self.overlay = nn.Embedding(new_tokens, base_emb.embedding_dim)
        with torch.no_grad():
            text_mean = base_emb.weight[:vtxt].mean(dim=0)
            self.overlay.weight[0].copy_(text_mean)
            self.overlay.weight[1].copy_(text_mean)
            self.overlay.weight[2:].zero_()
        for param in self.base_emb.parameters():
            param.requires_grad = False

    def forward(self, input_ids: Any) -> Any:
        base_ids = input_ids.clamp(max=self.vtxt - 1)
        out = self.base_emb(base_ids)
        mask = input_ids >= self.vtxt
        if mask.any():
            overlay_ids = input_ids[mask] - self.vtxt
            out = out.clone()
            out[mask] = self.overlay(overlay_ids)
        return out


def load_lm_and_tokenizer(cfg: Any, device: str, freeze: bool = True) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model, use_fast=True)
    ensure_pad_token(tokenizer)
    lm = AutoModelForCausalLM.from_pretrained(cfg.lm_model, torch_dtype=dtype).to(device)
    if freeze:
        for param in lm.parameters():
            param.requires_grad = False
    return lm, tokenizer


def build_overlay(lm: Any, cfg: Any, device: str) -> OverlayEmbedding:
    return OverlayEmbedding(lm.get_input_embeddings(), cfg.vtxt, cfg.new_tokens).to(device)


def warmup_projector(
    cfg: Any,
    lm: Any,
    overlay: OverlayEmbedding,
    codebook: Any,
    device: str,
    steps: int | None = None,
    lr: float = 1e-3,
) -> Dict[str, float]:
    projector = nn.Linear(cfg.d_code, cfg.d_lm).to(device)
    nn.init.kaiming_uniform_(projector.weight, a=5**0.5)
    codebook = codebook.detach().to(device)
    text_emb = lm.get_input_embeddings().weight[: cfg.vtxt].detach().to(device)
    text_norm = text_emb.norm(dim=-1).mean()
    text_mean = text_emb.mean(dim=0)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr)
    losses = []
    for _ in range(steps or cfg.projector_steps):
        proj = projector(codebook)
        loss_norm = (proj.norm(dim=-1).mean() - text_norm).pow(2)
        loss_mean = (proj.mean(dim=0) - text_mean).pow(2).mean()
        loss = loss_norm + 0.1 * loss_mean
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        projected = projector(codebook).to(overlay.overlay.weight.dtype)
        overlay.overlay.weight[2 : 2 + cfg.k].copy_(projected)
        visual_norm = overlay.overlay.weight[2 : 2 + cfg.k].norm(dim=-1).mean()
        ratio = visual_norm / text_norm.to(visual_norm.device)
        if ratio < 0.2 or ratio > 5.0:
            overlay.overlay.weight[2 : 2 + cfg.k].mul_(text_norm.to(visual_norm.device) / visual_norm.clamp_min(1e-8))
            visual_norm = overlay.overlay.weight[2 : 2 + cfg.k].norm(dim=-1).mean()
            ratio = visual_norm / text_norm.to(visual_norm.device)
    return {
        "projector_final_loss": losses[-1] if losses else float("nan"),
        "text_norm": float(text_norm.detach().cpu()),
        "visual_norm": float(visual_norm.detach().cpu()),
        "norm_ratio": float(ratio.detach().cpu()),
    }


def apply_lora(lm: Any, cfg: Any) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(lm, config)


def split_trainable_lora_params(lm: Any) -> List[Any]:
    return [param for name, param in lm.named_parameters() if param.requires_grad and "lora_" in name]


def save_overlay(path: Path, overlay: OverlayEmbedding, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"overlay": overlay.state_dict(), "metadata": metadata}, path)


def load_overlay(path: Path, lm: Any, cfg: Any, device: str) -> Tuple[OverlayEmbedding, Dict[str, Any]]:
    data = torch.load(path, map_location=device)
    overlay = build_overlay(lm, cfg, device)
    overlay.load_state_dict(data["overlay"])
    return overlay, data.get("metadata", {})


def trainable_summary(lm: Any, overlay: OverlayEmbedding) -> Dict[str, Any]:
    return {"lm": count_trainable_params(lm), "overlay": count_trainable_params(overlay)}

