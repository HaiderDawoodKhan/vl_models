from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .config import VLMConfig
from .data import CaptionExample, VQAExample, build_caption_examples, build_vqa_examples, load_alpaca_replay, load_clip_cache
from .model import (
    apply_lora,
    build_caption_batch,
    build_connector,
    build_text_lm_batch,
    build_vqa_batch,
    compute_rnorm,
    load_connector_checkpoint,
    load_lm_and_tokenizer,
    save_checkpoint,
    select_visual_tokens,
)
from .utils import (
    chunked,
    count_trainable_params,
    cuda_memory_summary,
    get_device,
    perplexity_from_loss,
    require_torch,
    save_json,
    set_seed,
    timer,
)


def make_optimizer(params: Sequence[Any], lr: float) -> Any:
    torch = require_torch()
    return torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr)


def _batch_features(features: Any, labels: Any, examples: Sequence[Any], batch: Sequence[Any]) -> Any:
    torch = require_torch()
    idx = torch.tensor([ex.index for ex in batch], dtype=torch.long)
    return features[idx], labels[idx]


def _maybe_limit(items: List[Any], limit: int | None) -> List[Any]:
    return items[:limit] if limit is not None else items


def compute_lm_perplexity(lm: Any, tokenizer: Any, texts: Sequence[str], device: str, batch_size: int = 4) -> float:
    torch = require_torch()
    lm.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in chunked(list(texts), batch_size):
            out = lm(**build_text_lm_batch(tokenizer, batch, device))
            losses.append(float(out.loss.detach().cpu()))
    return perplexity_from_loss(float(np.mean(losses))) if losses else float("nan")


def train_phase1(
    cfg: VLMConfig,
    train_cache_path: Path,
    output_dir: Path,
    max_steps: int | None = None,
    max_train_examples: int | None = None,
    representation: str = "patches",
) -> Dict[str, Any]:
    torch = require_torch()
    set_seed(cfg.seed)
    cfg.ensure_dirs()
    device = get_device()
    cache = load_clip_cache(train_cache_path)
    features = select_visual_tokens(cache["features"], representation)
    labels = cache["labels"]
    examples = _maybe_limit(build_caption_examples(labels.tolist()), max_train_examples)

    lm, tokenizer = load_lm_and_tokenizer(cfg, device, freeze=True)
    connector = build_connector(cfg).to(device)
    optimizer = make_optimizer(list(connector.parameters()), cfg.lr_phase1)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    steps = 0
    losses: List[float] = []

    with timer() as elapsed:
        for epoch in itertools.count(1):
            rng = np.random.default_rng(cfg.seed + epoch)
            shuffled = [examples[i] for i in rng.permutation(len(examples))]
            for batch_examples in chunked(shuffled, cfg.batch_phase1):
                batch_features, _ = _batch_features(features, labels, examples, batch_examples)
                captions = [ex.caption for ex in batch_examples]
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=device == "cuda"):
                    batch = build_caption_batch(connector, lm, tokenizer, batch_features, captions, device)
                    loss = lm(**batch).loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
            if max_steps is not None and steps >= max_steps:
                break
            if epoch >= 1 and max_steps is None:
                break

    rnorm = compute_rnorm(connector, lm, features[: min(64, len(features))], device)
    metrics = {
        "phase": "phase1",
        "steps": steps,
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "rnorm": rnorm,
        "trainable_params": count_trainable_params(connector),
        "memory": cuda_memory_summary(),
        "seconds": elapsed["seconds"],
    }
    save_checkpoint(output_dir, connector, extra=metrics)
    save_json(cfg.metrics_dir / "phase1_metrics.json", metrics)
    return metrics


def train_phase2(
    cfg: VLMConfig,
    train_cache_path: Path,
    phase1_dir: Path,
    output_dir: Path,
    lambda_replay: float | None = None,
    max_steps: int | None = None,
    max_train_examples: int | None = None,
    representation: str = "patches",
    norm_weight: float = 0.0,
) -> Dict[str, Any]:
    torch = require_torch()
    set_seed(cfg.seed)
    cfg.ensure_dirs()
    lambda_replay = cfg.lambda_replay if lambda_replay is None else lambda_replay
    device = get_device()
    cache = load_clip_cache(train_cache_path)
    features = select_visual_tokens(cache["features"], representation)
    labels = cache["labels"]
    vqa = _maybe_limit(build_vqa_examples(labels.tolist()), max_train_examples)
    alpaca = load_alpaca_replay(cfg.alpaca_replay_examples)

    lm, tokenizer = load_lm_and_tokenizer(cfg, device, freeze=True)
    lm = apply_lora(lm, cfg)
    connector = build_connector(cfg).to(device)
    load_connector_checkpoint(connector, phase1_dir, device)
    for p in connector.parameters():
        p.requires_grad = True
    optimizer = make_optimizer(list(connector.parameters()) + list(lm.parameters()), cfg.lr_phase2)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    steps = 0
    losses: List[float] = []
    text_cycle = itertools.cycle(chunked(alpaca, max(1, cfg.batch_phase2)))

    with timer() as elapsed:
        for epoch in itertools.count(1):
            rng = np.random.default_rng(cfg.seed + epoch)
            shuffled = [vqa[i] for i in rng.permutation(len(vqa))]
            for batch_examples in chunked(shuffled, cfg.batch_phase2):
                batch_features, _ = _batch_features(features, labels, vqa, batch_examples)
                questions = [ex.question for ex in batch_examples]
                answers = [ex.answer for ex in batch_examples]
                replay_texts = next(text_cycle)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=device == "cuda"):
                    vqa_batch = build_vqa_batch(connector, lm, tokenizer, batch_features, questions, answers, device)
                    loss_vqa = lm(**vqa_batch).loss
                    loss_lm = lm(**build_text_lm_batch(tokenizer, replay_texts, device)).loss
                    loss = loss_vqa + lambda_replay * loss_lm
                    if norm_weight:
                        visual_norm = connector(batch_features.to(device)).norm(dim=-1).mean()
                        text_norm = lm.get_input_embeddings().weight.norm(dim=-1).mean()
                        loss = loss + norm_weight * (visual_norm - text_norm).pow(2)
                scaler.scale(loss / cfg.grad_accum).backward()
                if (steps + 1) % cfg.grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
            if max_steps is not None and steps >= max_steps:
                break
            if epoch >= 1 and max_steps is None:
                break

    rnorm = compute_rnorm(connector, lm, features[: min(64, len(features))], device)
    metrics = {
        "phase": "phase2",
        "steps": steps,
        "lambda_replay": lambda_replay,
        "norm_weight": norm_weight,
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "rnorm": rnorm,
        "trainable_params": count_trainable_params(lm),
        "memory": cuda_memory_summary(),
        "seconds": elapsed["seconds"],
    }
    save_checkpoint(output_dir, connector, lm=lm, extra=metrics)
    save_json(cfg.metrics_dir / f"phase2_lambda_{lambda_replay}_norm_{norm_weight}.json", metrics)
    return metrics


def train_phase3(
    cfg: VLMConfig,
    train_cache_path: Path,
    phase2_dir: Path,
    output_dir: Path,
    max_steps: int | None = None,
    max_train_examples: int | None = 10000,
    representation: str = "patches",
) -> Dict[str, Any]:
    torch = require_torch()
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("peft is required to load the Phase 2 LoRA checkpoint.") from exc
    set_seed(cfg.seed)
    cfg.ensure_dirs()
    device = get_device()
    cache = load_clip_cache(train_cache_path)
    features = select_visual_tokens(cache["features"], representation)
    labels = cache["labels"]
    vqa = _maybe_limit(build_vqa_examples(labels.tolist()), max_train_examples)
    lm, tokenizer = load_lm_and_tokenizer(cfg, device, freeze=True)
    lm = PeftModel.from_pretrained(lm, phase2_dir / "lora", is_trainable=True)
    connector = build_connector(cfg).to(device)
    load_connector_checkpoint(connector, phase2_dir, device)
    optimizer = make_optimizer(list(connector.parameters()) + list(lm.parameters()), cfg.lr_phase3)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    steps = 0
    losses: List[float] = []

    with timer() as elapsed:
        rng = np.random.default_rng(cfg.seed + 3)
        shuffled = [vqa[i] for i in rng.permutation(len(vqa))]
        for batch_examples in chunked(shuffled, cfg.batch_phase3):
            batch_features, _ = _batch_features(features, labels, vqa, batch_examples)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                batch = build_vqa_batch(
                    connector,
                    lm,
                    tokenizer,
                    batch_features,
                    [ex.question for ex in batch_examples],
                    [ex.answer for ex in batch_examples],
                    device,
                )
                loss = lm(**batch).loss
            scaler.scale(loss / cfg.grad_accum).backward()
            if (steps + 1) % cfg.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break

    metrics = {
        "phase": "phase3",
        "steps": steps,
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "rnorm": compute_rnorm(connector, lm, features[: min(64, len(features))], device),
        "trainable_params": count_trainable_params(lm),
        "memory": cuda_memory_summary(),
        "seconds": elapsed["seconds"],
    }
    save_checkpoint(output_dir, connector, lm=lm, extra=metrics)
    save_json(cfg.metrics_dir / "phase3_metrics.json", metrics)
    return metrics
