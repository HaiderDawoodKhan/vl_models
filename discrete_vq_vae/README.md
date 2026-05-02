# Part B Discrete VQ-VAE

This package implements the assignment Part B pipeline:

`synthetic 16x16 image -> VQ-VAE 4x4 code map -> 16 visual token IDs -> SmolLM2 + LoRA`.

Unlike Part A, images are represented as discrete token IDs and trained in the same autoregressive stream as text.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Smoke Workflow

Smoke mode keeps the synthetic dataset and training loops tiny. It still needs model downloads for LM steps.

```bash
python3 synthetic_data.py --smoke --force
python3 train_vqvae.py --smoke --epochs 1 --name vqvae_best
python3 train_lm.py --smoke --max-steps 2
python3 eval_vqa.py --smoke --max-examples 8
python3 eval_vqa.py --smoke --max-examples 8 --text-only
python3 eval_imggen.py --smoke
```

## Full Workflow

```bash
python3 synthetic_data.py --force
python3 train_vqvae.py --ema --name vqvae_best
python3 train_lm.py
python3 eval_vqa.py
python3 eval_vqa.py --text-only
python3 eval_imggen.py
```

## Ablations

```bash
python3 ablations.py --kind vqvae
python3 ablations.py --kind lm-weights
python3 ablations.py --kind no-projector
python3 ablations.py --kind break-protection
```

## Outputs

- `cache/synthetic_train.pt`, `cache/synthetic_val.pt`
- `cache/encoded_train_tokens.pt`, `cache/encoded_val_tokens.pt`
- `weights/vqvae_best.pt`
- `weights/lm_partB_lora/`
- `outputs/synthetic_grid.png`
- `outputs/plots/`
- `outputs/reconstructions/`
- `outputs/token_maps/`
- `outputs/generated_images/`
- `outputs/*metrics*.json`

## Critical Checks

- VQ-VAE encoder output is `[B,64,4,4]`.
- Visual token IDs use `codebook_index + 49152 + 2`.
- SmolLM2 tokenizer is left padded.
- Vocabulary is resized before LoRA is applied.
- VQA labels supervise only answer plus EOS.
- Image-generation labels supervise visual tokens plus `</image>` and EOS.
- VQA generation masks out tokens with ID `>= 49152`.
- Image generation masks in only IDs `49154..49409`.
