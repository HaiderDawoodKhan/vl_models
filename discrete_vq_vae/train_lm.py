from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from .config import get_config
    from .synthetic_data import load_synthetic, save_synthetic_cache
    from .token_expansion import (
        add_visual_tokens,
        apply_lora,
        build_overlay,
        load_lm_and_tokenizer,
        save_overlay,
        split_trainable_lora_params,
        trainable_summary,
        warmup_projector,
    )
    from .tokenization import Collator, TokenDataset, build_tokenized_alpaca, build_tokenized_imggen, build_tokenized_vqa, token_type_sequence
    from .train_vqvae import load_vqvae_checkpoint
    from .utils import cycle_loader, get_device, perplexity_from_loss, require_torch, save_json, set_seed, timer
except ImportError:
    from config import get_config
    from synthetic_data import load_synthetic, save_synthetic_cache
    from token_expansion import (
        add_visual_tokens,
        apply_lora,
        build_overlay,
        load_lm_and_tokenizer,
        save_overlay,
        split_trainable_lora_params,
        trainable_summary,
        warmup_projector,
    )
    from tokenization import Collator, TokenDataset, build_tokenized_alpaca, build_tokenized_imggen, build_tokenized_vqa, token_type_sequence
    from train_vqvae import load_vqvae_checkpoint
    from utils import cycle_loader, get_device, perplexity_from_loss, require_torch, save_json, set_seed, timer


torch = require_torch()


def preencode_images(cfg: Any, vqvae_path: Path | None = None, smoke: bool = False, force: bool = False) -> Dict[str, str]:
    cfg.ensure_dirs()
    save_synthetic_cache(cfg)
    if cfg.encoded_train_path.exists() and cfg.encoded_val_path.exists() and not force:
        train_synth = load_synthetic(cfg.synthetic_train_path)
        val_synth = load_synthetic(cfg.synthetic_val_path)
        train_enc = torch.load(cfg.encoded_train_path, map_location="cpu")
        val_enc = torch.load(cfg.encoded_val_path, map_location="cpu")
        if (
            len(train_enc.get("indices", [])) == len(train_synth["images"])
            and len(val_enc.get("indices", [])) == len(val_synth["images"])
            and int(train_enc.get("k", cfg.k)) == cfg.k
            and int(val_enc.get("k", cfg.k)) == cfg.k
        ):
            return {"train": str(cfg.encoded_train_path), "val": str(cfg.encoded_val_path), "cached": True}
    device = get_device()
    vqvae = load_vqvae_checkpoint(vqvae_path or cfg.vqvae_path, cfg, device)
    vqvae.eval()
    paths = {}
    for split, data_path, out_path in [
        ("train", cfg.synthetic_train_path, cfg.encoded_train_path),
        ("val", cfg.synthetic_val_path, cfg.encoded_val_path),
    ]:
        data = load_synthetic(data_path)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(data["images"]), cfg.vqvae_batch):
                x = data["images"][start : start + cfg.vqvae_batch].to(device)
                chunks.append(vqvae.encode_to_indices(x).cpu())
        encoded = {"indices": torch.cat(chunks, dim=0), "labels": data["labels"], "class_names": data["class_names"], "k": cfg.k}
        torch.save(encoded, out_path)
        paths[split] = str(out_path)
    return paths


def load_encoded(path: Path) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    if "indices" not in data or "labels" not in data:
        raise ValueError(f"{path} is not an encoded-token cache.")
    return data


def make_loader(examples: List[Any], cfg: Any, tokenizer: Any, shuffle: bool = True) -> Any:
    return torch.utils.data.DataLoader(
        TokenDataset(examples),
        batch_size=cfg.lm_batch,
        shuffle=shuffle,
        collate_fn=Collator(tokenizer.pad_token_id),
        num_workers=0,
    )


def forward_loss(lm: Any, overlay: Any, batch: Dict[str, Any], device: str) -> Any:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    inputs_embeds = overlay(input_ids)
    return lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss


def compute_lm_perplexity(lm: Any, overlay: Any, tokenizer: Any, texts: List[Any], device: str, batch_size: int = 4) -> float:
    lm.eval()
    collator = Collator(tokenizer.pad_token_id)
    losses = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = collator(texts[start : start + batch_size])
            losses.append(float(forward_loss(lm, overlay, batch, device).detach().cpu()))
    lm.train()
    return perplexity_from_loss(float(np.mean(losses))) if losses else float("nan")


def build_training_sets(cfg: Any, tokenizer: Any, image_id: int, end_image_id: int) -> Dict[str, List[Any]]:
    train_enc = load_encoded(cfg.encoded_train_path)
    val_enc = load_encoded(cfg.encoded_val_path)
    vqa_train = build_tokenized_vqa(tokenizer, image_id, end_image_id, train_enc["labels"], train_enc["indices"], cfg.vtxt)
    img_train = build_tokenized_imggen(tokenizer, image_id, end_image_id, train_enc["labels"], train_enc["indices"], cfg.vtxt)
    vqa_val = build_tokenized_vqa(tokenizer, image_id, end_image_id, val_enc["labels"], val_enc["indices"], cfg.vtxt)
    img_val = build_tokenized_imggen(tokenizer, image_id, end_image_id, val_enc["labels"], val_enc["indices"], cfg.vtxt)
    text = build_tokenized_alpaca(tokenizer, cfg.alpaca_replay_examples)
    return {"vqa_train": vqa_train, "img_train": img_train, "vqa_val": vqa_val, "img_val": img_val, "text": text}


def print_token_sanity(cfg: Any, sets: Dict[str, List[Any]], image_id: int, end_image_id: int) -> List[Dict[str, str]]:
    rows = []
    for key in ["vqa_train", "img_train"]:
        for ex in sets[key][:3]:
            seq = token_type_sequence(ex.input_ids, ex.labels, cfg, image_id, end_image_id)
            row = {"dataset": key, "sequence": seq}
            print(row)
            rows.append(row)
    return rows


def train_lm(
    smoke: bool = False,
    max_steps: int | None = None,
    vqvae_path: Path | None = None,
    no_projector: bool = False,
    lambda_lm: float | None = None,
    gamma_img: float | None = None,
    lora_r: int | None = None,
) -> Dict[str, Any]:
    cfg = get_config(smoke)
    if lambda_lm is not None:
        cfg.lambda_lm = lambda_lm
    if gamma_img is not None:
        cfg.gamma_img = gamma_img
    if lora_r is not None:
        cfg.lora_r = lora_r
    set_seed(cfg.seed)
    cfg.ensure_dirs()
    preencode_images(cfg, vqvae_path=vqvae_path)
    device = get_device()
    lm, tokenizer = load_lm_and_tokenizer(cfg, device, freeze=True)
    token_ids = add_visual_tokens(tokenizer, lm, cfg)
    overlay = build_overlay(lm, cfg, device)

    vqvae = load_vqvae_checkpoint(vqvae_path or cfg.vqvae_path, cfg, device)
    vqvae.eval()
    if no_projector:
        projector_stats = {"projector": "skipped"}
        torch.nn.init.kaiming_uniform_(overlay.overlay.weight[2:], a=5**0.5)
    else:
        projector_stats = warmup_projector(cfg, lm, overlay, vqvae.quantizer.embedding.weight, device)
    del vqvae
    if device == "cuda":
        torch.cuda.empty_cache()

    sets = build_training_sets(cfg, tokenizer, token_ids["image_id"], token_ids["end_image_id"])
    sanity = print_token_sanity(cfg, sets, token_ids["image_id"], token_ids["end_image_id"])
    ppl0 = compute_lm_perplexity(lm, overlay, tokenizer, sets["text"][: min(100, len(sets["text"]))], device, batch_size=max(1, cfg.lm_batch))

    lm = apply_lora(lm, cfg)
    if hasattr(lm, "gradient_checkpointing_enable"):
        lm.gradient_checkpointing_enable()
    lm.config.use_cache = False
    lora_params = split_trainable_lora_params(lm)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": cfg.lr_lora},
            {"params": overlay.overlay.parameters(), "lr": cfg.lr_vis_emb},
        ]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    vqa_loader = make_loader(sets["vqa_train"], cfg, tokenizer, shuffle=True)
    img_loader = make_loader(sets["img_train"], cfg, tokenizer, shuffle=True)
    text_loader = make_loader(sets["text"], cfg, tokenizer, shuffle=True)
    vqa_iter = cycle_loader(vqa_loader)
    img_iter = cycle_loader(img_loader)
    text_iter = cycle_loader(text_loader)
    steps_per_epoch = max(len(vqa_loader), len(img_loader), len(text_loader))
    total_steps = max_steps or steps_per_epoch
    metrics: List[Dict[str, float]] = []

    with torch.no_grad():
        step0_vqa = float(forward_loss(lm, overlay, next(vqa_iter), device).detach().cpu())
        step0_img = float(forward_loss(lm, overlay, next(img_iter), device).detach().cpu())
    print({"step0_vqa_loss": step0_vqa, "step0_img_loss": step0_img})

    optimizer.zero_grad(set_to_none=True)
    with timer() as elapsed:
        for step in range(1, total_steps + 1):
            lm.train()
            overlay.train()
            batch_vqa = next(vqa_iter)
            batch_img = next(img_iter)
            batch_text = next(text_iter)
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                loss_vqa = forward_loss(lm, overlay, batch_vqa, device)
            scaler.scale(loss_vqa / cfg.grad_accum).backward()
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                loss_img = forward_loss(lm, overlay, batch_img, device)
            scaler.scale((cfg.gamma_img * loss_img) / cfg.grad_accum).backward()
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                loss_lm = forward_loss(lm, overlay, batch_text, device)
            scaler.scale((cfg.lambda_lm * loss_lm) / cfg.grad_accum).backward()

            if step % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params + list(overlay.overlay.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            row = {
                "step": step,
                "loss_vqa": float(loss_vqa.detach().cpu()),
                "loss_img": float(loss_img.detach().cpu()),
                "loss_lm": float(loss_lm.detach().cpu()),
            }
            if step % cfg.eval_interval == 0 or step == total_steps:
                ppl = compute_lm_perplexity(lm, overlay, tokenizer, sets["text"][: min(100, len(sets["text"]))], device, batch_size=max(1, cfg.lm_batch))
                row["ppl"] = ppl
                row["forgetting_ratio"] = ppl / ppl0 if ppl0 and not np.isnan(ppl0) else float("nan")
            metrics.append(row)
            if step % max(1, cfg.eval_interval) == 0 or step == total_steps:
                print(row)

    cfg.lm_lora_dir.mkdir(parents=True, exist_ok=True)
    lm.save_pretrained(cfg.lm_lora_dir)
    tokenizer.save_pretrained(cfg.lm_lora_dir)
    save_overlay(cfg.lm_lora_dir / "overlay.pt", overlay, {"token_ids": token_ids, "projector_stats": projector_stats})
    report = {
        "ppl0": ppl0,
        "step0_vqa_loss": step0_vqa,
        "step0_img_loss": step0_img,
        "projector_stats": projector_stats,
        "token_sanity": sanity,
        "trainable": trainable_summary(lm, overlay),
        "metrics": metrics,
        "seconds": elapsed["seconds"],
        "lora_dir": str(cfg.lm_lora_dir),
    }
    save_json(cfg.outputs_dir / "lm_training_metrics.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--vqvae", type=Path, default=None)
    parser.add_argument("--no-projector", action="store_true")
    parser.add_argument("--lambda-lm", type=float, default=None)
    parser.add_argument("--gamma-img", type=float, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--preencode-only", action="store_true")
    parser.add_argument("--force-preencode", action="store_true")
    args = parser.parse_args()
    cfg = get_config(args.smoke)
    if args.preencode_only:
        print(preencode_images(cfg, args.vqvae, args.smoke, args.force_preencode))
    else:
        print(train_lm(args.smoke, args.max_steps, args.vqvae, args.no_projector, args.lambda_lm, args.gamma_img, args.lora_r))
