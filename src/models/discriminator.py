"""
src/models/discriminator.py
===========================
3D PatchGAN discriminator for the adversarial loss term L_ADV in the VAE.

Design decisions
----------------
Paper (Section 2, "3D VAE"):
  "an adversarial loss L_ADV ... prevents unrealistic artifacts [8, 7]"
  No discriminator architecture is specified. We adopt a 3D PatchGAN following
  MAISI [8] and Rombach et al. (Stable Diffusion) [22].  [INFERRED — open question #4]

Architecture
------------
A stack of Conv3d(stride=2) layers with LeakyReLU(0.2) (no GroupNorm on the
first layer, then use InstanceNorm3d — same convention as the original PatchGAN
paper and MAISI):

  input  (B, C_in, D, H, W)
    └─ Conv3d(C_in, base_ch, k=4, s=2, p=1)  LeakyReLU   ← no norm
    └─ Conv3d(base_ch,   2*base_ch, k=4, s=2, p=1)  InstanceNorm  LeakyReLU
    └─ Conv3d(2*base_ch, 4*base_ch, k=4, s=2, p=1)  InstanceNorm  LeakyReLU
    └─ Conv3d(4*base_ch, 8*base_ch, k=4, s=1, p=1)  InstanceNorm  LeakyReLU
    └─ Conv3d(8*base_ch, 1,         k=4, s=1, p=1)             ← logit map

Output is a spatial logit map (not a scalar), so the loss is computed over
a 3D patch grid — hence "PatchGAN".  No sigmoid here; BCEWithLogitsLoss
is used in the training loop (numerically more stable).

config keys used
----------------
  vae_config.yaml → discriminator.base_channels   (default 32)
  vae_config.yaml → discriminator.num_layers       (default 3)

num_layers controls how many strided layers are stacked.  Total receptive field
grows with more layers.  3 layers ≅ 22³ voxel receptive field (at base_ch=32).

References
----------
- MAISI: Guo et al., WACV 2025 (ref [8] in LAND paper)
- Rombach et al., LDM, CVPR 2022 (ref [22] in LAND paper)
- Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks",
  CVPR 2017 — original PatchGAN.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Discriminator building blocks
# ---------------------------------------------------------------------------

def _disc_block(
    in_ch: int,
    out_ch: int,
    stride: int,
    use_norm: bool = True,
) -> list[nn.Module]:
    """
    One PatchGAN conv block: Conv3d → [InstanceNorm3d] → LeakyReLU(0.2).
    stride=2 halves spatial dims; stride=1 keeps them.
    """
    layers: list[nn.Module] = [
        nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not use_norm)
    ]
    if use_norm:
        layers.append(nn.InstanceNorm3d(out_ch, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


# ---------------------------------------------------------------------------
# PatchGAN3D
# ---------------------------------------------------------------------------

class PatchGAN3D(nn.Module):
    """
    3D PatchGAN discriminator.

    Outputs a 3D logit map (B, 1, d, h, w) rather than a scalar.
    BCEWithLogitsLoss is applied over the full spatial logit map in the
    training loop — each "patch" of the input is judged real/fake independently.

    Args
    ----
    in_channels   : number of input channels (1 for single-channel CT).
    base_channels : number of filters in the first conv layer (32 per config).
    num_layers    : number of strided downsampling layers (3 per config).
                    After num_layers strided layers, two more stride-1 layers
                    refine the features before the final logit projection.
    """

    def __init__(
        self,
        in_channels: int   = 1,
        base_channels: int = 32,
        num_layers: int    = 3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        # ── First block: no norm ────────────────────────────────────────────
        layers.extend(_disc_block(in_channels, base_channels, stride=2, use_norm=False))

        # ── Strided downsampling blocks ─────────────────────────────────────
        in_ch  = base_channels
        out_ch = base_channels * 2
        for _ in range(num_layers - 1):
            layers.extend(_disc_block(in_ch, out_ch, stride=2, use_norm=True))
            in_ch   = out_ch
            out_ch  = min(out_ch * 2, base_channels * 8)   # cap at 8× base

        # ── Stride-1 refinement block ───────────────────────────────────────
        layers.extend(_disc_block(in_ch, out_ch, stride=1, use_norm=True))
        in_ch = out_ch

        # ── Final logit projection: no norm, no activation ─────────────────
        layers.append(nn.Conv3d(in_ch, 1, kernel_size=4, stride=1, padding=1, bias=True))

        self.model = nn.Sequential(*layers)

        # Weight initialisation: Gaussian(0, 0.02) — PatchGAN convention
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d) and m.weight is not None:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CT volume or reconstruction, shape (B, C_in, D, H, W).

        Returns:
            Logit map, shape (B, 1, d', h', w').
            Each spatial element is the raw (pre-sigmoid) real/fake score
            for a local patch of the input.
        """
        return self.model(x)

    @classmethod
    def from_config(cls, vae_cfg) -> "PatchGAN3D":
        """
        Build discriminator from OmegaConf vae_config.yaml.

        Usage:
            cfg  = load_config(vae="configs/vae_config.yaml")
            disc = PatchGAN3D.from_config(cfg.vae)
        """
        d_cfg = vae_cfg.discriminator
        arch  = vae_cfg.architecture
        return cls(
            in_channels  = arch.in_channels,
            base_channels= d_cfg.base_channels,
            num_layers   = d_cfg.num_layers,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Quick unit tests  (python -m src.models.discriminator)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running discriminator self-test on {device}…")

    disc = PatchGAN3D(in_channels=1, base_channels=32, num_layers=3).to(device)
    print(f"  Parameters: {disc.count_parameters():,}")

    # Use a small 64³ volume so it fits on any GPU
    x = torch.randn(1, 1, 64, 64, 64, device=device)
    logits = disc(x)
    print(f"  Input  shape: {tuple(x.shape)}")
    print(f"  Output shape: {tuple(logits.shape)}  (PatchGAN logit map)")
    assert logits.shape[0] == 1  and logits.shape[1] == 1, \
        f"Unexpected output shape: {logits.shape}"
    assert logits.ndim == 5, "Expected 5-D logit map"
    print("  Shape check:  PASS")

    # Backward
    loss = logits.mean()
    loss.backward()
    print("  backward():   PASS")

    # BCEWithLogitsLoss sanity
    criterion = nn.BCEWithLogitsLoss()
    real_labels = torch.ones_like(logits)
    fake_labels = torch.zeros_like(logits)
    loss_real = criterion(logits, real_labels)
    loss_fake = criterion(torch.randn_like(logits), fake_labels)
    assert loss_real.ndim == 0 and loss_fake.ndim == 0, "Losses should be scalars"
    print(f"  BCEWithLogitsLoss: PASS  (real={loss_real.item():.4f}, fake={loss_fake.item():.4f})")

    print("\nAll discriminator tests passed.")
    sys.exit(0)
