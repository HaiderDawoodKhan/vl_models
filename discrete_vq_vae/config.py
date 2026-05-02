from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = [
    "spiral",
    "triangle",
    "circle",
    "cross",
    "checkerboard",
    "gradient",
]


@dataclass
class PartBConfig:
    seed: int = 42
    lm_model: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    img_size: int = 16
    num_classes: int = 6
    n_per_class: int = 1000
    train_frac: float = 0.8
    k: int = 256
    d_code: int = 64
    beta: float = 0.25
    ema_decay: float = 0.99
    ema_eps: float = 1e-5
    dead_code_threshold: float = 1.0
    vqvae_batch: int = 64
    vqvae_lr: float = 3e-4
    vqvae_epochs: int = 80
    vtxt: int = 49152
    d_lm: int = 960
    lambda_lm: float = 0.2
    gamma_img: float = 0.5
    lm_batch: int = 16
    grad_accum: int = 4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr_lora: float = 5e-4
    lr_vis_emb: float = 5e-5
    alpaca_replay_examples: int = 1000
    projector_steps: int = 500
    eval_interval: int = 300
    max_new_tokens: int = 12
    cache_dir: Path = Path("cache")
    weights_dir: Path = Path("weights")
    outputs_dir: Path = Path("outputs")

    @property
    def new_tokens(self) -> int:
        return 2 + self.k

    @property
    def v_total(self) -> int:
        return self.vtxt + self.new_tokens

    @property
    def synthetic_train_path(self) -> Path:
        return self.cache_dir / "synthetic_train.pt"

    @property
    def synthetic_val_path(self) -> Path:
        return self.cache_dir / "synthetic_val.pt"

    @property
    def encoded_train_path(self) -> Path:
        return self.cache_dir / "encoded_train_tokens.pt"

    @property
    def encoded_val_path(self) -> Path:
        return self.cache_dir / "encoded_val_tokens.pt"

    @property
    def vqvae_path(self) -> Path:
        return self.weights_dir / "vqvae_best.pt"

    @property
    def lm_lora_dir(self) -> Path:
        return self.weights_dir / "lm_partB_lora"

    @property
    def plots_dir(self) -> Path:
        return self.outputs_dir / "plots"

    @property
    def recon_dir(self) -> Path:
        return self.outputs_dir / "reconstructions"

    @property
    def token_maps_dir(self) -> Path:
        return self.outputs_dir / "token_maps"

    @property
    def gen_dir(self) -> Path:
        return self.outputs_dir / "generated_images"

    @property
    def metrics_path(self) -> Path:
        return self.outputs_dir / "metrics.json"

    def ensure_dirs(self) -> None:
        for path in (
            self.cache_dir,
            self.weights_dir,
            self.outputs_dir,
            self.plots_dir,
            self.recon_dir,
            self.token_maps_dir,
            self.gen_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_config(smoke: bool = False) -> PartBConfig:
    cfg = PartBConfig()
    if smoke:
        cfg.n_per_class = 12
        cfg.vqvae_batch = 12
        cfg.vqvae_epochs = 2
        cfg.lm_batch = 2
        cfg.grad_accum = 1
        cfg.alpaca_replay_examples = 8
        cfg.projector_steps = 5
        cfg.eval_interval = 2
    return cfg

