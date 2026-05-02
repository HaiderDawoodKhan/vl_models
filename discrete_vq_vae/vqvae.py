from __future__ import annotations

from typing import Any, Dict, Tuple

try:
    from .utils import require_torch
except ImportError:
    from utils import require_torch

torch = require_torch()
nn = torch.nn
F = torch.nn.functional


class Encoder(nn.Module):
    def __init__(self, d_code: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, d_code, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, d_code),
            nn.ReLU(),
        )

    def forward(self, x: Any) -> Any:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, d_code: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(d_code, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: Any) -> Any:
        return self.net(z)


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        k: int = 256,
        d_code: int = 64,
        beta: float = 0.25,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        dead_code_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.k = k
        self.d_code = d_code
        self.beta = beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.dead_code_threshold = dead_code_threshold
        self.embedding = nn.Embedding(k, d_code)
        self.embedding.weight.data.uniform_(-1.0 / k, 1.0 / k)
        self.register_buffer("ema_cluster_size", torch.zeros(k))
        self.register_buffer("ema_weight", self.embedding.weight.detach().clone())

    def _nearest(self, flat: Any) -> Tuple[Any, Any, Any]:
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * flat @ self.embedding.weight.t()
        )
        indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(indices, self.k).type(flat.dtype)
        quantized = encodings @ self.embedding.weight
        return quantized, indices, encodings

    @torch.no_grad()
    def ema_update(self, flat: Any, encodings: Any) -> None:
        if not self.use_ema or not self.training:
            return
        counts = encodings.sum(dim=0)
        embed_sum = encodings.t() @ flat
        self.ema_cluster_size.mul_(self.ema_decay).add_(counts, alpha=1.0 - self.ema_decay)
        self.ema_weight.mul_(self.ema_decay).add_(embed_sum, alpha=1.0 - self.ema_decay)
        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + self.ema_eps) / (n + self.k * self.ema_eps) * n
        updated = self.ema_weight / cluster_size.unsqueeze(1).clamp_min(self.ema_eps)

        dead = self.ema_cluster_size < self.dead_code_threshold
        if dead.any() and flat.shape[0] > 0:
            restart_idx = torch.randint(0, flat.shape[0], (int(dead.sum()),), device=flat.device)
            updated[dead] = flat[restart_idx]
            self.ema_cluster_size[dead] = self.dead_code_threshold + 1.0
            self.ema_weight[dead] = updated[dead] * self.ema_cluster_size[dead].unsqueeze(1)
        self.embedding.weight.data.copy_(updated)

    def forward(self, ze: Any) -> Tuple[Any, Dict[str, Any]]:
        b, c, h, w = ze.shape
        flat = ze.permute(0, 2, 3, 1).contiguous().view(-1, c)
        zq_flat, indices_flat, encodings = self._nearest(flat)
        if self.use_ema:
            self.ema_update(flat.detach(), encodings.detach())
        zq = zq_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(zq, ze.detach())
        commit_loss = F.mse_loss(ze, zq.detach())
        if self.use_ema:
            codebook_loss = torch.zeros((), device=ze.device, dtype=ze.dtype)
        zq_st = ze + (zq - ze).detach()
        avg_probs = encodings.float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        usage = encodings.sum(dim=0)
        info = {
            "indices": indices_flat.view(b, h, w),
            "codebook_loss": codebook_loss,
            "commit_loss": commit_loss,
            "perplexity": perplexity,
            "usage": usage.detach(),
            "dead_codes": int((usage == 0).sum().detach().cpu()),
            "zq": zq,
        }
        return zq_st, info


class VQVAE(nn.Module):
    def __init__(
        self,
        k: int = 256,
        d_code: int = 64,
        beta: float = 0.25,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        dead_code_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(d_code)
        self.quantizer = VectorQuantizer(k, d_code, beta, use_ema, ema_decay, ema_eps, dead_code_threshold)
        self.decoder = Decoder(d_code)
        self.beta = beta

    def forward(self, x: Any) -> Tuple[Any, Dict[str, Any]]:
        ze = self.encoder(x)
        zq_st, info = self.quantizer(ze)
        recon = self.decoder(zq_st)
        info["ze"] = ze
        return recon, info

    @torch.no_grad()
    def encode_to_indices(self, x: Any) -> Any:
        was_training = self.training
        self.eval()
        ze = self.encoder(x)
        _, info = self.quantizer(ze)
        if was_training:
            self.train()
        return info["indices"]

    @torch.no_grad()
    def decode_indices(self, indices: Any) -> Any:
        b, h, w = indices.shape
        zq = self.quantizer.embedding(indices.reshape(-1)).view(b, h, w, self.quantizer.d_code)
        zq = zq.permute(0, 3, 1, 2).contiguous()
        return self.decoder(zq)

    def loss(self, recon: Any, x: Any, info: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        recon_loss = F.mse_loss(recon, x)
        total = recon_loss + info["codebook_loss"] + self.beta * info["commit_loss"]
        logs = {
            "loss": float(total.detach().cpu()),
            "recon_loss": float(recon_loss.detach().cpu()),
            "codebook_loss": float(info["codebook_loss"].detach().cpu()),
            "commit_loss": float(info["commit_loss"].detach().cpu()),
            "perplexity": float(info["perplexity"].detach().cpu()),
            "dead_codes": float(info["dead_codes"]),
        }
        return total, logs


def build_vqvae_from_cfg(cfg: Any, use_ema: bool = False, k: int | None = None, beta: float | None = None) -> VQVAE:
    return VQVAE(
        k=cfg.k if k is None else k,
        d_code=cfg.d_code,
        beta=cfg.beta if beta is None else beta,
        use_ema=use_ema,
        ema_decay=cfg.ema_decay,
        ema_eps=cfg.ema_eps,
        dead_code_threshold=cfg.dead_code_threshold,
    )
