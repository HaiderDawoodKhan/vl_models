from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .config import VLMConfig
from .data import VQAExample, build_vqa_examples, load_clip_cache, majority_vote_by_template
from .model import build_connector, build_vqa_batch, greedy_generate_from_embeds, load_connector_checkpoint, load_lm_and_tokenizer, select_visual_tokens
from .utils import chunked, normalize_answer, require_torch, save_json, set_seed


def exact_match_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def agg(key: str | None = None) -> Dict[str, float]:
        groups: Dict[str, List[int]] = defaultdict(list)
        for row in rows:
            group = "overall" if key is None else row[key]
            groups[group].append(int(row["correct"]))
        return {name: float(np.mean(vals)) for name, vals in groups.items()}

    return {"overall": agg(), "per_template": agg("template"), "per_class": agg("class_name")}


def evaluate_vqa(
    cfg: VLMConfig,
    cache_path: Path,
    checkpoint_dir: Path | None,
    max_examples: int | None = None,
    text_only: bool = False,
    representation: str = "patches",
) -> Dict[str, Any]:
    torch = require_torch()
    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None
    set_seed(cfg.seed)
    cache = load_clip_cache(cache_path)
    features = select_visual_tokens(cache["features"], representation)
    labels = cache["labels"]
    examples = build_vqa_examples(labels.tolist())
    if max_examples is not None:
        examples = examples[:max_examples]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm, tokenizer = load_lm_and_tokenizer(cfg, device, freeze=True)
    if checkpoint_dir and (checkpoint_dir / "lora").exists() and PeftModel is not None:
        lm = PeftModel.from_pretrained(lm, checkpoint_dir / "lora")
    connector = build_connector(cfg).to(device)
    if checkpoint_dir and (checkpoint_dir / "connector.pt").exists():
        load_connector_checkpoint(connector, checkpoint_dir, device)
    connector.eval()
    lm.eval()

    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch_examples in chunked(examples, cfg.eval_batch_size):
            idx = torch.tensor([ex.index for ex in batch_examples], dtype=torch.long)
            batch_features = features[idx]
            questions = [ex.question for ex in batch_examples]
            if text_only:
                encoded = tokenizer(questions, add_special_tokens=True, padding=True, return_tensors="pt").to(device)
                generated = lm.generate(
                    **encoded,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                preds = tokenizer.batch_decode(generated[:, encoded.input_ids.shape[1] :], skip_special_tokens=True)
            else:
                batch = build_vqa_batch(connector, lm, tokenizer, batch_features, questions, ["" for _ in questions], device, include_answer=False)
                preds = greedy_generate_from_embeds(lm, tokenizer, batch["inputs_embeds"], batch["attention_mask"], cfg.max_new_tokens)
            for ex, pred in zip(batch_examples, preds):
                pred_norm = normalize_answer(pred)
                ans_norm = normalize_answer(ex.answer)
                rows.append(
                    {
                        "index": ex.index,
                        "class_name": ex.class_name,
                        "template": ex.template,
                        "question": ex.question,
                        "answer": ex.answer,
                        "prediction": pred,
                        "correct": pred_norm == ans_norm,
                    }
                )
    metrics = exact_match_metrics(rows)
    metrics["num_examples"] = len(rows)
    metrics["text_only"] = text_only
    save_json(cfg.metrics_dir / ("vqa_text_only.json" if text_only else "vqa_visual.json"), metrics)
    save_json(cfg.examples_dir / ("vqa_text_only_rows.json" if text_only else "vqa_visual_rows.json"), rows)
    return metrics


def evaluate_majority_baseline(cfg: VLMConfig, train_cache_path: Path, val_cache_path: Path) -> Dict[str, Any]:
    train = load_clip_cache(train_cache_path)
    val = load_clip_cache(val_cache_path)
    train_vqa = build_vqa_examples(train["labels"].tolist())
    val_vqa = build_vqa_examples(val["labels"].tolist())
    majority = majority_vote_by_template(train_vqa)
    rows = []
    for ex in val_vqa:
        pred = majority[ex.template]
        rows.append(
            {
                "class_name": ex.class_name,
                "template": ex.template,
                "answer": ex.answer,
                "prediction": pred,
                "correct": normalize_answer(pred) == normalize_answer(ex.answer),
            }
        )
    metrics = exact_match_metrics(rows)
    metrics["majority_answers"] = majority
    save_json(cfg.metrics_dir / "majority_baseline.json", metrics)
    return metrics


def first_token_topk(lm: Any, tokenizer: Any, batch: Dict[str, Any], k: int = 5) -> List[List[Dict[str, Any]]]:
    torch = require_torch()
    with torch.no_grad():
        out = lm(**batch)
        logits = out.logits[:, -1, :]
        vals, ids = torch.topk(logits, k=k, dim=-1)
    decoded: List[List[Dict[str, Any]]] = []
    for row_vals, row_ids in zip(vals, ids):
        decoded.append(
            [
                {"token": tokenizer.decode([int(tok)]), "logit": float(val)}
                for tok, val in zip(row_ids.detach().cpu(), row_vals.detach().cpu())
            ]
        )
    return decoded


def save_qualitative_examples(
    cfg: VLMConfig,
    cache_path: Path,
    checkpoint_dir: Path,
    max_scan: int = 200,
    representation: str = "patches",
) -> List[Dict[str, Any]]:
    torch = require_torch()
    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None

    cache = load_clip_cache(cache_path)
    features = select_visual_tokens(cache["features"], representation)
    labels = cache["labels"]
    examples = build_vqa_examples(labels.tolist())[:max_scan]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm, tokenizer = load_lm_and_tokenizer(cfg, device, freeze=True)
    if (checkpoint_dir / "lora").exists() and PeftModel is not None:
        lm = PeftModel.from_pretrained(lm, checkpoint_dir / "lora")
    connector = build_connector(cfg).to(device)
    load_connector_checkpoint(connector, checkpoint_dir, device)
    connector.eval()
    lm.eval()

    candidates: List[Dict[str, Any]] = []
    with torch.no_grad():
        for ex in examples:
            batch_features = features[torch.tensor([ex.index], dtype=torch.long)]
            prefix = build_vqa_batch(connector, lm, tokenizer, batch_features, [ex.question], [""], device, include_answer=False)
            pred = greedy_generate_from_embeds(lm, tokenizer, prefix["inputs_embeds"], prefix["attention_mask"], cfg.max_new_tokens)[0]
            top5 = first_token_topk(lm, tokenizer, prefix, k=5)[0]
            correct = normalize_answer(pred) == normalize_answer(ex.answer)
            candidates.append(
                {
                    "image_class": ex.class_name,
                    "question": ex.question,
                    "ground_truth": ex.answer,
                    "prediction": pred,
                    "top5_first_token_logits": top5,
                    "correct": correct,
                    "comment": "correct" if correct else "failure",
                }
            )
    correct = [row for row in candidates if row["correct"]]
    failures = [row for row in candidates if not row["correct"]]
    chosen = correct[:4] + failures[:2]
    if len(chosen) < 6:
        chosen = candidates[:6]
    save_json(cfg.examples_dir / "qualitative_examples.json", chosen)
    return chosen
