"""
src/training/diffusion_schedule.py
====================================
DDPM noise schedule + v-prediction math (LAND paper Section 2 "Diffusion process"):
  - T = 1000, linear beta schedule, beta_1=1e-4, beta_T=0.02
  - v-prediction (Salimans & Ho, 2022): the model predicts velocity v_t = alpha_t*eps - sigma_t*x0
    (alpha_t = sqrt(alpha_bar_t), sigma_t = sqrt(1-alpha_bar_t)), rather than predicting eps or x0
    directly. Two things fall out of that choice, both implemented here:
      1. `velocity_target` / `add_noise` -- the forward-process formulas needed to build training
         targets from a clean latent x0 and sampled noise eps.
      2. `predict_x0_from_v` -- the exact algebraic inverse, needed at sampling time later to
         recover x0 (and hence continue the reverse process) from a predicted v.
  - Min-SNR-gamma (Hang et al., 2023) loss weighting: caps the effective loss weight at high-SNR
    (low-noise, easy) timesteps so training doesn't spend disproportionate gradient signal on
    trivial near-x0 timesteps at the expense of harder high-noise ones. gamma=5 per
    configs/diffusion_config.yaml (the value the Min-SNR paper itself validates; LAND's paper
    cites the technique but doesn't restate gamma explicitly).  [INFERRED: gamma value only]

This module has NO torch.nn dependency (no model, no U-Net) -- it's pure tensor math, which is
what makes it possible to test independently of the (much more expensive) actual model forward
pass, exactly like sliding_window.py's separation from the VAE itself.
"""
from __future__ import annotations

import torch


class VPredictionSchedule:
    """
    All the noise-schedule buffers and forward/inverse formulas needed for v-prediction
    training and (later) sampling. Kept as a plain class holding tensors, not an nn.Module --
    schedules aren't learned parameters.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1.0e-4,
        beta_end: float = 0.02,
        min_snr_gamma: float = 5.0,
        device: str | torch.device = "cpu",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.min_snr_gamma = min_snr_gamma

        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)  # alpha_bar_t = prod_{s<=t} alpha_s

        self.betas = betas.to(device=device, dtype=torch.float32)
        self.alpha_bars = alpha_bars.to(device=device, dtype=torch.float32)
        self.sqrt_alpha_bars = alpha_bars.sqrt().to(device=device, dtype=torch.float32)
        self.sqrt_one_minus_alpha_bars = (1.0 - alpha_bars).sqrt().to(device=device, dtype=torch.float32)
        # SNR(t) = alpha_bar_t / (1 - alpha_bar_t). float64 upstream to avoid precision loss at
        # very small (1-alpha_bar_t) near t=0, where SNR is large; cast down only at the end.
        self.snr = (alpha_bars / (1.0 - alpha_bars)).to(device=device, dtype=torch.float32)

    def _gather(self, buf: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """buf: (T,), t: (B,) int64 timestep indices -> (B, 1, 1, ..., 1) for broadcasting
        against an (B, C, D, H, W)-shaped latent (ndim = latent.ndim)."""
        out = buf.to(t.device)[t]
        return out.view(-1, *([1] * (ndim - 1)))

    def add_noise(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps."""
        alpha_t = self._gather(self.sqrt_alpha_bars, t, x0.ndim)
        sigma_t = self._gather(self.sqrt_one_minus_alpha_bars, t, x0.ndim)
        return alpha_t * x0 + sigma_t * eps

    def velocity_target(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """v-prediction training target: v_t = sqrt(alpha_bar_t) * eps - sqrt(1 - alpha_bar_t) * x0."""
        alpha_t = self._gather(self.sqrt_alpha_bars, t, x0.ndim)
        sigma_t = self._gather(self.sqrt_one_minus_alpha_bars, t, x0.ndim)
        return alpha_t * eps - sigma_t * x0

    def predict_x0_from_v(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Exact algebraic inverse of (add_noise, velocity_target): given x_t and a predicted v,
        recover x0. Needed at sampling time (not used during training itself, but training-time
        code and sampling-time code share this schedule object, so it lives here rather than
        being duplicated later).

        Derivation: x_t = alpha*x0 + sigma*eps  and  v = alpha*eps - sigma*x0
          => eps = alpha*v + sigma*x_t   (substituting and solving the 2x2 linear system)
          => x0  = alpha*x_t - sigma*v
        (using alpha^2 + sigma^2 = 1, i.e. alpha_bar_t + (1-alpha_bar_t) = 1)
        """
        alpha_t = self._gather(self.sqrt_alpha_bars, t, x_t.ndim)
        sigma_t = self._gather(self.sqrt_one_minus_alpha_bars, t, x_t.ndim)
        return alpha_t * x_t - sigma_t * v

    def min_snr_loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Min-SNR-gamma weight for v-prediction (Hang et al. 2023, Table/Eq. for v-parameterization):
            weight(t) = min(SNR(t), gamma) / (SNR(t) + 1)
        This is NOT the same formula as for epsilon-prediction (min(SNR,gamma)/SNR) -- using the
        eps-prediction formula here would systematically over/under-weight timesteps since v and
        eps have a different relationship to the loss at each SNR.  [INFERRED: gamma=5 from
        config; the v-prediction-specific weighting formula itself is standard, not LAND-specific]

        Returns: (B,) per-sample scalar weight, to be broadcast against a per-sample loss before
        reduction (mean).
        """
        snr_t = self.snr.to(t.device)[t]
        return torch.clamp(snr_t, max=self.min_snr_gamma) / (snr_t + 1.0)

    def sample_timesteps(self, batch_size: int, device: str | torch.device) -> torch.Tensor:
        """Uniform random timestep per training example, as standard DDPM training does."""
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device, dtype=torch.long)
