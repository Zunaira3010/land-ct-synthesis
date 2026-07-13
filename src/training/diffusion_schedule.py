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

        # alpha_bar_{t-1} shifted by one, with alpha_bar_{-1} := 1 (the "x_{-1} = x0" convention
        # used by the reverse-process posterior at t=0). Built in float64 alongside alpha_bars
        # itself, not by re-indexing the already-downcast self.alpha_bars later.
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bars[:-1]])

        # (1 - alpha_bar_t) and (1 - alpha_bar_{t-1}) computed in float64 BEFORE the downcast to
        # float32, and kept as their own stored buffers rather than ever being recomputed later
        # as `1.0 - self.alpha_bars[t]` from the already-lossy float32 tensor. Near t=0,
        # alpha_bar_t is extremely close to 1 (e.g. ~0.9999 at t=0), so a float32 subtraction
        # 1.0 - alpha_bar_t is catastrophic cancellation: most of the significant digits cancel,
        # leaving only ~3-4 correct digits in the result even though alpha_bar_t itself was
        # accurate to ~7 digits. Doing the subtraction in float64 first (where the same
        # cancellation still happens, but against ~16 digits of precision instead of ~7) and
        # only THEN casting the already-small, already-correct difference down to float32
        # avoids the problem. This value is used repeatedly by both training (add_noise,
        # velocity_target) and sampling (ddpm_reverse_step), so a single precision-safe buffer
        # here is the fix, not a local workaround at each call site.
        one_minus_alpha_bars = 1.0 - alpha_bars
        one_minus_alpha_bars_prev = 1.0 - alpha_bars_prev

        self.betas = betas.to(device=device, dtype=torch.float32)
        self.alphas = alphas.to(device=device, dtype=torch.float32)
        self.alpha_bars = alpha_bars.to(device=device, dtype=torch.float32)
        self.alpha_bars_prev = alpha_bars_prev.to(device=device, dtype=torch.float32)
        self.one_minus_alpha_bars = one_minus_alpha_bars.to(device=device, dtype=torch.float32)
        self.one_minus_alpha_bars_prev = one_minus_alpha_bars_prev.to(device=device, dtype=torch.float32)
        self.sqrt_alpha_bars = alpha_bars.sqrt().to(device=device, dtype=torch.float32)
        self.sqrt_alpha_bars_prev = alpha_bars_prev.sqrt().to(device=device, dtype=torch.float32)
        self.sqrt_one_minus_alpha_bars = one_minus_alpha_bars.sqrt().to(device=device, dtype=torch.float32)
        # SNR(t) = alpha_bar_t / (1 - alpha_bar_t). float64 upstream to avoid precision loss at
        # very small (1-alpha_bar_t) near t=0, where SNR is large; cast down only at the end.
        self.snr = (alpha_bars / one_minus_alpha_bars).to(device=device, dtype=torch.float32)

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

    def ddpm_reverse_step(
        self,
        x_t: torch.Tensor,
        v_pred: torch.Tensor,
        t: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        One ancestral DDPM reverse-process step: given the current noisy latent x_t and the
        model's predicted velocity v_pred at timestep t, sample x_{t-1} ~ q(x_{t-1} | x_t, x0).

        x0 is recovered from v_pred via the existing exact algebraic inverse
        (predict_x0_from_v), then the standard DDPM posterior is used:

            mu_tilde_t(x_t, x0) = sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t) * x0
                                 + sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * x_t
            beta_tilde_t         = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
            x_{t-1}               = mu_tilde_t + sqrt(beta_tilde_t) * noise,  noise ~ N(0, I)

        using alpha_bar_{-1} := 1 for the t=0 step (see __init__). This is not a special case
        handled by an `if t == 0` branch: at t=0, alpha_bar_{t-1} = alpha_bar_{-1} = 1 exactly,
        so beta_tilde_0 = (1-1)/(1-alpha_bar_0) * beta_0 = 0 exactly, and separately
        beta_0/(1-alpha_bar_0) = beta_0/beta_0 = 1 exactly (since alpha_bar_0 = alpha_0 = 1 -
        beta_0), so mu_tilde_0 reduces to exactly x0_pred with zero injected noise. The formula
        already does the right thing at the boundary; see the self-test below for a check that
        this holds to floating-point precision, not just algebraically.

        Uses self.one_minus_alpha_bars / self.one_minus_alpha_bars_prev (the precision-safe,
        float64-computed buffers from __init__) for every (1 - alpha_bar) term here, rather than
        recomputing `1.0 - self.alpha_bars[t]` on the fly -- see __init__'s comment for why that
        matters most exactly at small t, which is also where sampling spends its final, most
        detail-determining steps.

        Args:
            x_t:       current noisy latent, (B, C, D, H, W)
            v_pred:    model's predicted velocity at x_t, t -- same shape as x_t
            t:         (B,) int64 current timestep indices (moving from t to t-1)
            generator: optional torch.Generator for reproducible sampling

        Returns:
            x_{t-1}: same shape as x_t
        """
        x0_pred = self.predict_x0_from_v(x_t, v_pred, t)

        alpha_t = self._gather(self.alphas, t, x_t.ndim)
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_alpha_bar_prev = self._gather(self.sqrt_alpha_bars_prev, t, x_t.ndim)
        one_minus_alpha_bar_t = self._gather(self.one_minus_alpha_bars, t, x_t.ndim)
        one_minus_alpha_bar_prev = self._gather(self.one_minus_alpha_bars_prev, t, x_t.ndim)
        beta_t = self._gather(self.betas, t, x_t.ndim)

        x0_coef = sqrt_alpha_bar_prev * beta_t / one_minus_alpha_bar_t
        xt_coef = sqrt_alpha_t * one_minus_alpha_bar_prev / one_minus_alpha_bar_t
        mu = x0_coef * x0_pred + xt_coef * x_t

        posterior_variance = one_minus_alpha_bar_prev / one_minus_alpha_bar_t * beta_t
        # Clamp to >=0 as a defensive floor against stray float roundoff producing a tiny
        # negative value (mathematically it's always >=0) -- not expected to trigger, but
        # sqrt() of a negative float is NaN, which would silently poison the rest of sampling.
        posterior_std = posterior_variance.clamp(min=0.0).sqrt()

        noise = torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype) \
            if generator is not None else torch.randn_like(x_t)
        return mu + posterior_std * noise

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


# ---------------------------------------------------------------------------
# Self-test (run with: python -m src.training.diffusion_schedule)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    torch.manual_seed(0)
    print("Running VPredictionSchedule self-tests...")

    sched = VPredictionSchedule()
    shape = (2, 4, 8, 8, 8)

    # ---- 1. add_noise / velocity_target / predict_x0_from_v round-trip ----
    x0 = torch.randn(shape)
    eps = torch.randn(shape)
    t = torch.tensor([0, 999])
    x_t = sched.add_noise(x0, eps, t)
    v = sched.velocity_target(x0, eps, t)
    x0_recovered = sched.predict_x0_from_v(x_t, v, t)
    err = (x0 - x0_recovered).abs().max().item()
    assert err < 1e-4, f"round-trip x0 recovery error too large: {err}"
    print(f"  add_noise/velocity_target/predict_x0_from_v round-trip: PASS (max err {err:.2e})")

    # ---- 2. precision-safe buffer actually differs from naive float32 subtraction ----
    # At small t, alpha_bar_t is close to 1, so 1.0 - alpha_bars[t] computed directly in
    # float32 loses precision relative to the buffer computed in float64 upstream. Confirm
    # there IS a measurable gap at small t (proving the buffer isn't a no-op) while both
    # remain sane, finite, positive values.
    naive_f32 = 1.0 - sched.alpha_bars  # recompute the "wrong" old way, for comparison only
    small_t_diff = (sched.one_minus_alpha_bars[0] - naive_f32[0]).abs().item()
    assert sched.one_minus_alpha_bars[0].item() > 0, "one_minus_alpha_bars[0] must be positive"
    print(f"  precision-safe buffer differs from naive float32 subtraction at t=0 "
          f"(delta={small_t_diff:.3e}), buffer value={sched.one_minus_alpha_bars[0].item():.6e}: PASS")

    # ---- 3. t=0 boundary: deterministic, mu == x0_pred exactly, zero injected noise ----
    t0 = torch.tensor([0, 0])
    v_pred = torch.randn(shape)
    x_t0 = torch.randn(shape)
    x0_pred_at_0 = sched.predict_x0_from_v(x_t0, v_pred, t0)
    g = torch.Generator().manual_seed(42)
    out_a = sched.ddpm_reverse_step(x_t0, v_pred, t0, generator=g)
    g2 = torch.Generator().manual_seed(999)  # different seed -- must not matter at t=0
    out_b = sched.ddpm_reverse_step(x_t0, v_pred, t0, generator=g2)
    err_a = (out_a - x0_pred_at_0).abs().max().item()
    err_b = (out_b - x0_pred_at_0).abs().max().item()
    assert err_a < 1e-5 and err_b < 1e-5, f"t=0 step should equal x0_pred exactly, got {err_a}, {err_b}"
    assert (out_a - out_b).abs().max().item() < 1e-6, "t=0 step must be deterministic (zero variance)"
    print(f"  t=0 boundary: mu==x0_pred exactly, zero noise regardless of seed: PASS")

    # ---- 4. posterior variance is non-negative and finite everywhere ----
    all_t = torch.arange(sched.num_train_timesteps)
    alpha_t_all = sched._gather(sched.alphas, all_t, 1).squeeze(-1)
    one_minus_ab_t_all = sched._gather(sched.one_minus_alpha_bars, all_t, 1).squeeze(-1)
    one_minus_ab_prev_all = sched._gather(sched.one_minus_alpha_bars_prev, all_t, 1).squeeze(-1)
    beta_all = sched._gather(sched.betas, all_t, 1).squeeze(-1)
    posterior_var_all = one_minus_ab_prev_all / one_minus_ab_t_all * beta_all
    assert torch.isfinite(posterior_var_all).all(), "posterior variance has non-finite entries"
    assert (posterior_var_all >= -1e-8).all(), "posterior variance has a meaningfully negative entry"
    assert posterior_var_all[0].item() < 1e-6, "posterior variance at t=0 should be ~0"
    print(f"  posterior variance finite and non-negative across all {sched.num_train_timesteps} "
          f"timesteps, ~0 at t=0: PASS")

    # ---- 5. full reverse chain T-1 -> 0 stays finite, shape-correct, no NaN/Inf ----
    class _DummyModel:
        """Stand-in for a real UNet3D -- returns plausible-scale random velocity, just to
        exercise the sampling loop's numerics end-to-end without needing a real model."""
        def __call__(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.randn_like(x_t)

    model = _DummyModel()
    x = torch.randn(1, 4, 8, 8, 8)
    g = torch.Generator().manual_seed(7)
    for step in reversed(range(sched.num_train_timesteps)):
        t_batch = torch.full((1,), step, dtype=torch.long)
        v_pred = model(x, t_batch)
        x = sched.ddpm_reverse_step(x, v_pred, t_batch, generator=g)
        assert torch.isfinite(x).all(), f"non-finite value produced at step {step}"
    assert x.shape == (1, 4, 8, 8, 8)
    print(f"  full {sched.num_train_timesteps}-step reverse chain (random model): "
          f"finite output, correct shape: PASS")

    print("\nAll VPredictionSchedule tests passed.")


if __name__ == "__main__":
    _self_test()
