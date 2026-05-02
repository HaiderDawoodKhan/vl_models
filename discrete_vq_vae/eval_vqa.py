from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from .config import CLASS_NAMES, get_config
    from .synthetic_data import build_vqa_examples, load_synthetic, majority_vote_by_template
except ImportError:
    majority_vote_by_template = None

try:
    from .config import CLASS_NAMES, get_config
    from .synthetic_data import build_vqa_examples, load_synthetic
    from .token_expansion import add_visual_tokens, apply_lora, load_lm_and_tokenizer, load_overlay
    from .tokenization import encode_multimodal, visual_ids_from_indices
    from .train_lm import load_encoded
    from .utils import get_device, normalize_answer, require_torch, save_json, set_seed
except ImportError:
    from config import CLASS_NAMES, get_config
    from synthetic_data import build_vqa_examples, load_synthetic
    from token_expansion import add_visual_tokens, apply_lora, load_lm_and_tokenizer, load_overlay
    from tokenization import encode_multimodal, visual_ids_from_indices
    from train_lm import load_encoded
    from utils import get_device, normalize_answer, require_torch, save_json, set_seed


torch = require_torch()


def load_finetuned(cfg: Any, device: str) -> tuple[Any, Any, Any, Dict[str, int]]:
    from peft import PeftModel

    lm, tokenizer = load_lm_and_tokenizer(cfg, device, freeze=True)
    token_ids = add_visual_tokens(tokenizer, lm, cfg)
    lm = PeftModel.from_pretrained(lm, cfg.lm_lora_dir)
    overlay, meta = load_overlay(cfg.lm_lora_dir / "overlay.pt", lm, cfg, device)
    if "token_ids" in meta:
        token_ids.update(meta["token_ids"])
    lm.eval()
    overlay.eval()
    return lm, tokenizer, overlay, token_ids


def generate_masked(
    lm: Any,
    overlay: Any,
    tokenizer: Any,
    input_ids: List[int],
    device: str,
    max_new_tokens: int,
    text_only_mask: bool,
    cfg: Any,
) -> tuple[str, List[Dict[str, Any]]]:
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    top5_rows: List[Dict[str, Any]] = []
    generated: List[int] = []
    for step in range(max_new_tokens):
        with torch.no_grad():
            out = lm(inputs_embeds=overlay(ids), attention_mask=attention)
            logits = out.logits[:, -1, :]
            if text_only_mask:
                logits[:, cfg.vtxt :] = -1e9
            vals, toks = torch.topk(logits[0], k=5)
            if step == 0:
                top5_rows = [{"token": tokenizer.decode([int(t)]), "id": int(t), "logit": float(v)} for v, t in zip(vals.cpu(), toks.cpu())]
            next_id = int(torch.argmax(logits[0]).detach().cpu())
        if next_id == tokenizer.eos_token_id:
            break
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        attention = torch.ones_like(ids)
    return tokenizer.decode(generated, skip_special_tokens=True), top5_rows


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def by(key: str | None = None) -> Dict[str, float]:
        groups: Dict[str, List[int]] = defaultdict(list)
        for row in rows:
            groups["overall" if key is None else row[key]].append(int(row["correct"]))
        return {name: float(np.mean(vals)) for name, vals in groups.items()}

    return {"overall": by(), "per_template": by("template"), "per_class": by("class_name")}


def majority_baseline(train_labels: Any, val_examples: List[Any]) -> Dict[str, Any]:
    train_examples = build_vqa_examples(train_labels.tolist())
    counts: Dict[str, Dict[str, int]] = {}
    for ex in train_examples:
        counts.setdefault(ex.template, {})
        counts[ex.template][ex.answer] = counts[ex.template].get(ex.answer, 0) + 1
    majority = {template: max(answer_counts.items(), key=lambda x: x[1])[0] for template, answer_counts in counts.items()}
    rows = []
    for ex in val_examples:
        pred = majority[ex.template]
        rows.append({"template": ex.template, "class_name": ex.class_name, "answer": ex.answer, "prediction": pred, "correct": normalize_answer(pred) == normalize_answer(ex.answer)})
    out = aggregate(rows)
    out["majority_answers"] = majority
    return out


def evaluate_vqa(smoke: bool = False, max_examples: int | None = None, text_only: bool = False) -> Dict[str, Any]:
    cfg = get_config(smoke)
    set_seed(cfg.seed)
    device = get_device()
    lm, tokenizer, overlay, token_ids = load_finetuned(cfg, device)
    val_enc = load_encoded(cfg.encoded_val_path)
    val_examples = build_vqa_examples(val_enc["labels"].tolist())
    if max_examples is not None:
        val_examples = val_examples[:max_examples]
    rows: List[Dict[str, Any]] = []
    for ex in val_examples:
        if text_only:
            prefix = [tokenizer.bos_token_id or tokenizer.eos_token_id] + tokenizer(ex.question, add_special_tokens=False).input_ids
        else:
            vis = visual_ids_from_indices(val_enc["indices"][ex.index], cfg.vtxt)
            tok = encode_multimodal(tokenizer, token_ids["image_id"], token_ids["end_image_id"], ex.question, "", vis)
            prefix = tok.input_ids[:-1]
        pred, top5 = generate_masked(lm, overlay, tokenizer, prefix, device, cfg.max_new_tokens, True, cfg)
        rows.append(
            {
                "index": ex.index,
                "class_name": ex.class_name,
                "template": ex.template,
                "question": ex.question,
                "answer": ex.answer,
                "prediction": pred,
                "top5_first_token_logits": top5,
                "correct": normalize_answer(pred) == normalize_answer(ex.answer),
            }
        )
    metrics = aggregate(rows)
    metrics["num_examples"] = len(rows)
    metrics["text_only"] = text_only
    if not text_only:
        train = load_synthetic(cfg.synthetic_train_path)
        metrics["majority_baseline"] = majority_baseline(train["labels"], val_examples)
        save_confusion_matrix(cfg, rows)
        save_qualitative(cfg, rows)
    save_json(cfg.outputs_dir / ("vqa_text_only_metrics.json" if text_only else "vqa_metrics.json"), metrics)
    save_json(cfg.outputs_dir / ("vqa_text_only_rows.json" if text_only else "vqa_rows.json"), rows)
    return metrics


def save_confusion_matrix(cfg: Any, rows: List[Dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    filtered = [r for r in rows if r["template"] == "recognition"]
    if not filtered:
        return
    y_true = [r["answer"] for r in filtered]
    y_pred = [normalize_answer(r["prediction"]) for r in filtered]
    y_pred = [p if p in CLASS_NAMES else "unknown" for p in y_pred]
    labels = CLASS_NAMES + ["unknown"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("What shape? confusion matrix")
    fig.tight_layout()
    fig.savefig(cfg.plots_dir / "vqa_shape_confusion.png", dpi=160)
    plt.close(fig)


def save_qualitative(cfg: Any, rows: List[Dict[str, Any]]) -> None:
    correct = [r for r in rows if r["correct"]][:2]
    failures = [r for r in rows if not r["correct"]][:2]
    save_json(cfg.outputs_dir / "vqa_qualitative_examples.json", correct + failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    print(evaluate_vqa(args.smoke, args.max_examples, args.text_only))

