"""
src/losses/vae_loss.py
======================
Combined VAE training loss:

    L_VAE = L_MAE(x, x̂) + L_LPIPS(x, x̂) + L_ADV(x, x̂) + L_KL(E(x))

All four terms are exactly as stated in the LAND paper (Section 2, "3D VAE"):
  - L_MAE   : pixel-wise L1 reconstruction loss          [PAPER]
  - L_LPIPS : perceptual similarity loss (Zhang et al.)  [PAPER, ref 25]
  - L_ADV   : adversarial loss (PatchGAN)                [PAPER, ref 8, 7]
  - L_KL    : KL divergence regularisation               [PAPER, ref 16]

3-D LPIPS ("2.5D LPIPS")
--------------------------
LPIPS has no native 3D network.  We run it on 2D slices sampled from each of
the three anatomical axes (axial / coronal / sagittal) and average.  This is
the standard workaround used in MAISI and other 3D medical-image VAEs.
[INFERRED — vae_config.yaml: perceptual_2p5d: true]

Adversarial loss
----------------
We use the non-saturating GAN loss (hinge variant is also available):

  Generator (VAE encoder+decoder):
    L_G = -mean(D(x̂))   ← make fake look real

  Discriminator:
    L_D = mean(ReLU(1 - D(x))) + mean(ReLU(1 + D(x̂)))   ← hinge loss
    or standard BCE:
    L_D = BCE(D(x), 1) + BCE(D(x̂.detach()), 0)

We implement hinge as the default (matches MAISI convention).  [INFERRED]
The discriminator is trained separately from the VAE — the caller must run
disc_loss() and step the discriminator optimiser before gen_adv_loss().

Discriminator warm-up
---------------------
The adversarial weight ramps from 0 to its target value linearly over
`discriminator_warmup_epochs` to let the encoder/decoder stabilise first.
[INFERRED — vae_config.yaml: discriminator_warmup_epochs: 10]

Usage (training loop sketch)
-----------------------------
    from src.losses.vae_loss import VAELoss

    loss_fn = VAELoss.from_config(cfg.vae)

    # VAE step:
    recon, posterior = vae(x)
    fake_logits = discriminator(recon)
    vae_loss, log = loss_fn.generator_loss(x, recon, posterior, fake_logits, epoch=e)
    vae_loss.backward(); vae_optim.step()

    # Discriminator step:
    real_logits = discriminator(x.detach())
    fake_logits = discriminator(recon.detach())
    d_loss = loss_fn.discriminator_loss(real_logits, fake_logits)
    d_loss.backward(); disc_optim.step()
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.vae import DiagonalGaussian


# ---------------------------------------------------------------------------
# LPIPS 2.5D helper
# ---------------------------------------------------------------------------

class LPIPS25D(nn.Module):
    """
    Apply torchmetrics / lpips-pytorch LPIPS on 2D slices extracted from
    three axes of a 3D volume and return the mean.

    The underlying LPIPS network expects:
      - input range [-1, 1]  (we normalise from the CT's [-1, 1] or [0, 1])
      - shape (B, 3, H, W)   (we repeat single channel → 3 channels)

    Args
    ----
    net_type        : 'alex' (default, lighter) or 'vgg' (higher quality).
    num_slices_per_axis : how many random slices per axis to sample.
                          More → better estimate but slower.  Default 5.
    input_range     : '[-1,1]' (default) or '[0,1]' — whether to rescale before LPIPS.
    """

    def __init__(
        self,
        net_type: str = "alex",
        num_slices_per_axis: int = 5,
        input_range: str = "[-1,1]",
    ) -> None:
        super().__init__()
        try:
            import lpips as _lpips_lib
            self._lpips = _lpips_lib.LPIPS(net=net_type, verbose=False)
        except ImportError as exc:
            raise ImportError(
                "lpips package not found. Install with: pip install lpips"
            ) from exc

        self.num_slices = num_slices_per_axis
        self.input_range = input_range

    def _to_lpips_input(self, vol: torch.Tensor, axis: int, idx: int) -> torch.Tensor:
        """
        Extract a 2D slice from `vol` along `axis` at position `idx`,
        expand single channel to 3 channels, and ensure shape (B, 3, H, W).
        """
        # vol: (B, 1, D, H, W)
        if axis == 0:   # axial    → slice along D
            sl = vol[:, :, idx, :, :]          # (B, 1, H, W)
        elif axis == 1: # coronal  → slice along H
            sl = vol[:, :, :, idx, :]          # (B, 1, D, W)
        else:           # sagittal → slice along W
            sl = vol[:, :, :, :, idx]          # (B, 1, D, H)

        # Repeat single channel → 3 channels for LPIPS
        sl = sl.repeat(1, 3, 1, 1)             # (B, 3, H, W)

        # LPIPS expects [-1, 1]
        if self.input_range == "[0,1]":
            sl = sl * 2.0 - 1.0

        return sl

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x, x_hat : (B, 1, D, H, W) — both on the same device.

        Returns:
            scalar mean LPIPS distance (tensor, grad-attached to x_hat).
        """
        device = x.device
        self._lpips = self._lpips.to(device)

        dims = [x.shape[2], x.shape[3], x.shape[4]]  # D, H, W
        losses: list[torch.Tensor] = []

        for axis in range(3):
            depth = dims[axis]
            indices = random.sample(range(depth), min(self.num_slices, depth))
            for idx in indices:
                sl_real = self._to_lpips_input(x,     axis, idx)
                sl_fake = self._to_lpips_input(x_hat, axis, idx)
                losses.append(self._lpips(sl_real, sl_fake).mean())

        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Hinge adversarial losses
# ---------------------------------------------------------------------------

def _hinge_gen_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Generator hinge loss: L_G = -mean(D(x̂))."""
    return -fake_logits.mean()


def _hinge_disc_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Discriminator hinge loss:
        L_D = mean(ReLU(1 - D(x))) + mean(ReLU(1 + D(x̂)))
    Encourages real logits > 1 and fake logits < -1.
    """
    loss_real = F.relu(1.0 - real_logits).mean()
    loss_fake = F.relu(1.0 + fake_logits).mean()
    return 0.5 * (loss_real + loss_fake)


# ---------------------------------------------------------------------------
# VAELoss
# ---------------------------------------------------------------------------

class VAELoss(nn.Module):
    """
    Combines all four VAE loss components per the LAND paper.

    Attributes
    ----------
    mae_weight           : weight for L_MAE  (default 1.0)
    lpips_weight         : weight for L_LPIPS (default 1.0)
    adversarial_weight   : weight for L_ADV  (default 0.1)
    kl_weight            : weight for L_KL   (default 1e-6)
    warmup_epochs        : L_ADV weight is 0 for the first N epochs  (default 10)
    use_lpips            : False disables the perceptual term (useful for quick runs
                           without the lpips package installed)
    """

    def __init__(
        self,
        mae_weight: float        = 1.0,
        lpips_weight: float      = 1.0,
        adversarial_weight: float= 0.1,
        kl_weight: float         = 1.0e-6,
        warmup_epochs: int       = 10,
        use_lpips: bool          = True,
        lpips_net: str           = "alex",
        lpips_slices_per_axis: int = 5,
    ) -> None:
        super().__init__()

        self.mae_weight        = mae_weight
        self.lpips_weight      = lpips_weight
        self.adversarial_weight= adversarial_weight
        self.kl_weight         = kl_weight
        self.warmup_epochs     = warmup_epochs
        self.use_lpips         = use_lpips

        if use_lpips:
            self.lpips = LPIPS25D(
                net_type=lpips_net,
                num_slices_per_axis=lpips_slices_per_axis,
            )
        else:
            self.lpips = None   # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Generator (VAE) loss
    # ------------------------------------------------------------------

    def generator_loss(
        self,
        x:           torch.Tensor,        # (B,1,D,H,W) — real CT
        recon:       torch.Tensor,        # (B,1,D,H,W) — reconstruction
        posterior:   DiagonalGaussian,    # from vae.encode(x)
        fake_logits: Optional[torch.Tensor] = None,  # D(recon), may be None during warmup
        epoch:       int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the full VAE generator loss.

        Returns
        -------
        total_loss : scalar tensor (backward-compatible)
        log        : dict of individual loss values for logging
        """
        # L_MAE  ─────────────────────────────────────────────────────────
        l_mae = F.l1_loss(recon, x)

        # L_LPIPS ─────────────────────────────────────────────────────────
        if self.use_lpips and self.lpips is not None:
            l_lpips = self.lpips(x, recon)
        else:
            l_lpips = torch.tensor(0.0, device=x.device)

        # L_KL  ───────────────────────────────────────────────────────────
        l_kl = posterior.kl().mean()

        # L_ADV  ──────────────────────────────────────────────────────────
        adv_weight_effective = self._adv_weight(epoch)
        if fake_logits is not None and adv_weight_effective > 0.0:
            l_adv = _hinge_gen_loss(fake_logits)
        else:
            l_adv = torch.tensor(0.0, device=x.device)

        # Total  ──────────────────────────────────────────────────────────
        total = (
            self.mae_weight   * l_mae
          + self.lpips_weight * l_lpips
          + adv_weight_effective * l_adv
          + self.kl_weight    * l_kl
        )

        log = {
            "loss/total"  : total.item(),
            "loss/mae"    : l_mae.item(),
            "loss/lpips"  : l_lpips.item(),
            "loss/adv_gen": l_adv.item(),
            "loss/kl"     : l_kl.item(),
            "loss/adv_w"  : adv_weight_effective,
        }
        return total, log

    # ------------------------------------------------------------------
    # Discriminator loss
    # ------------------------------------------------------------------

    def discriminator_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Hinge discriminator loss.

        real_logits : D(x.detach())
        fake_logits : D(recon.detach())  — must be detached from VAE graph

        Returns (loss_tensor, log_dict).
        """
        loss = _hinge_disc_loss(real_logits, fake_logits)
        log  = {"loss/disc": loss.item()}
        return loss, log

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _adv_weight(self, epoch: int) -> float:
        """
        Linear warm-up: adversarial weight ramps from 0 to self.adversarial_weight
        over `warmup_epochs` then stays constant.
        """
        if self.warmup_epochs <= 0:
            return self.adversarial_weight
        ramp = min(1.0, epoch / max(self.warmup_epochs, 1))
        return self.adversarial_weight * ramp

    @classmethod
    def from_config(cls, vae_cfg) -> "VAELoss":
        """
        Build VAELoss from OmegaConf vae_config.yaml.

        Usage:
            cfg     = load_config(vae="configs/vae_config.yaml")
            loss_fn = VAELoss.from_config(cfg.vae)
        """
        loss_cfg  = vae_cfg.loss
        train_cfg = vae_cfg.training
        return cls(
            mae_weight        = float(loss_cfg.mae_weight),
            lpips_weight      = float(loss_cfg.lpips_weight),
            adversarial_weight= float(loss_cfg.adversarial_weight),
            kl_weight         = float(loss_cfg.kl_weight),
            warmup_epochs     = int(train_cfg.discriminator_warmup_epochs),
            use_lpips         = bool(loss_cfg.get("perceptual_2p5d", True)),
        )


# ---------------------------------------------------------------------------
# Quick unit tests  (python -m src.losses.vae_loss)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from src.models.vae import VAE
    from src.models.discriminator import PatchGAN3D

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running VAELoss self-test on {device} (no LPIPS to keep it fast)…")

    # Build minimal models
    vae  = VAE(channels=[16, 32, 64], latent_channels=4,
               norm_num_groups=16).to(device)
    disc = PatchGAN3D(in_channels=1, base_channels=8, num_layers=2).to(device)
    loss_fn = VAELoss(
        mae_weight=1.0, lpips_weight=0.0, adversarial_weight=0.1,
        kl_weight=1e-6, warmup_epochs=10, use_lpips=False,
    )

    # Tiny CT: 32³ → latent 8³
    x = torch.randn(1, 1, 32, 32, 32, device=device)

    # ------------------------------------------------------------------
    # VAE forward
    recon, posterior = vae(x)

    # Discriminator on the reconstruction
    fake_logits = disc(recon)
    real_logits = disc(x.detach())

    # Generator loss (epoch=5 < warmup=10 → adv_w = 0.05)
    gen_loss, log_gen = loss_fn.generator_loss(x, recon, posterior, fake_logits, epoch=5)
    print(f"  generator_loss() at epoch 5:  {gen_loss.item():.4f}")
    print(f"    adv_weight: {log_gen['loss/adv_w']:.3f}  (expected 0.05)")
    assert abs(log_gen["loss/adv_w"] - 0.05) < 1e-6

    gen_loss.backward()
    print("  generator backward():          PASS")

    # Discriminator loss
    disc_loss, log_disc = loss_fn.discriminator_loss(real_logits, fake_logits.detach())
    print(f"  discriminator_loss():          {disc_loss.item():.4f}")
    disc_loss.backward()
    print("  discriminator backward():      PASS")

    # Generator loss after warmup (epoch=100 → adv_w = 0.1)
    _, log_warm = loss_fn.generator_loss(x, recon.detach(), posterior, fake_logits.detach(), epoch=100)
    assert abs(log_warm["loss/adv_w"] - 0.1) < 1e-6
    print(f"  adv_weight at epoch 100:       {log_warm['loss/adv_w']:.3f}  (expected 0.1)  PASS")

    print("\nAll VAELoss tests passed.")
    sys.exit(0)
