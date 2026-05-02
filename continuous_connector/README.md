# Part A Continuous Connector VLMs

This package implements the Part A pipeline:

`CIFAR-10 -> CLIP ViT-B/32 patch tokens -> MLP connector -> SmolLM2 inputs_embeds -> caption/VQA`.

It keeps CLIP frozen, injects continuous visual embeddings through `inputs_embeds`, and does not add image tokens to the tokenizer vocabulary.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Full Run

```bash
python3 -m partA_continuous_vlm.cache_clip --split both
python3 -m partA_continuous_vlm.train_phase1
python3 -m partA_continuous_vlm.train_phase2
python3 -m partA_continuous_vlm.train_phase3
python3 -m partA_continuous_vlm.eval --alpaca-ppl
python3 -m partA_continuous_vlm.eval --checkpoint phase3
python3 -m partA_continuous_vlm.eval --majority
python3 -m partA_continuous_vlm.eval --text-only
python3 -m partA_continuous_vlm.eval --checkpoint phase3 --qualitative
python3 -m partA_continuous_vlm.modality_gap --name initial
python3 -m partA_continuous_vlm.modality_gap --checkpoint phase1
python3 -m partA_continuous_vlm.modality_gap --checkpoint phase2
python3 -m partA_continuous_vlm.modality_gap --checkpoint phase3
```

## Smoke Run

The smoke flags reduce CIFAR subset sizes and batch sizes. They still require the ML dependencies and model downloads.

```bash
python3 -m partA_continuous_vlm.cache_clip --split both --smoke
python3 -m partA_continuous_vlm.train_phase1 --smoke --max-steps 1
python3 -m partA_continuous_vlm.train_phase2 --smoke --max-steps 1 --max-train-examples 4
python3 -m partA_continuous_vlm.train_phase3 --smoke --max-steps 1 --max-train-examples 4
```

## Ablations

```bash
python3 -m partA_continuous_vlm.ablations --kind lambda
python3 -m partA_continuous_vlm.cache_clip --split both --include-cls
python3 -m partA_continuous_vlm.ablations --kind representation --cache cifar_train_clip_with_cls.pt
python3 -m partA_continuous_vlm.ablations --kind norm
```

Outputs are written to `cache/`, `checkpoints/`, and `results/`.
