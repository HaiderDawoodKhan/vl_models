from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class VLMConfig:
    seed: int = 42
    clip_model: str = "openai/clip-vit-base-patch32"
    lm_model: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    d_vision: int = 768
    d_lm: int = 960
    num_patches: int = 49
    train_per_class: int = 1000
    test_per_class: int = 200
    batch_phase1: int = 32
    batch_phase2: int = 4
    batch_phase3: int = 4
    eval_batch_size: int = 8
    lr_phase1: float = 3e-4
    lr_phase2: float = 5e-4
    lr_phase3: float = 2e-4
    lambda_replay: float = 0.2
    grad_accum: int = 4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    alpaca_replay_examples: int = 1000
    eval_interval: int = 300
    max_new_tokens: int = 12
    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")
    checkpoint_dir: Path = Path("checkpoints")
    result_dir: Path = Path("results")

    @property
    def train_cache(self) -> Path:
        return self.cache_dir / "cifar_train_clip.pt"

    @property
    def test_cache(self) -> Path:
        return self.cache_dir / "cifar_test_clip.pt"

    @property
    def metrics_dir(self) -> Path:
        return self.result_dir / "metrics"

    @property
    def plots_dir(self) -> Path:
        return self.result_dir / "plots"

    @property
    def examples_dir(self) -> Path:
        return self.result_dir / "examples"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.cache_dir,
            self.checkpoint_dir,
            self.metrics_dir,
            self.plots_dir,
            self.examples_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_config(smoke: bool = False) -> VLMConfig:
    cfg = VLMConfig()
    if smoke:
        cfg.train_per_class = 2
        cfg.test_per_class = 1
        cfg.batch_phase1 = 2
        cfg.batch_phase2 = 1
        cfg.batch_phase3 = 1
        cfg.eval_batch_size = 1
        cfg.alpaca_replay_examples = 8
        cfg.eval_interval = 2
        cfg.grad_accum = 1
    return cfg
