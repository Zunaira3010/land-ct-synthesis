"""
src/models/vae.py
=================
3D Variational Autoencoder matching the LAND paper spec.

Architecture
------------
Paper (Section 2, "3D VAE"):
  - Input:  256×256×256×1  CT volume
  - Latent: 64×64×64×4     (4× spatial compression, 4× channel expansion)
  - Lightweight variant of MAISI [8]
  - 3 resolution levels, 1 residual block per level  [PAPER]
  - Same number of channels in encoder and decoder   [PAPER]
  - Channel widths [64, 128, 256]                    [INFERRED from MAISI lightweight]
  - No self-attention in the VAE                     [INFERRED from MAISI lightweight]
  - GroupNorm + SiLU throughout                      [INFERRED from MAISI/LDM convention]

Latent parameterisation
-----------------------
The encoder outputs 2 × latent_channels feature maps (mean μ and log-variance log σ²).
Sampling: z = μ + σ · ε,  ε ~ N(0, I)  during training.
At inference (encode only): the mean μ is returned (no noise).

Public API
----------
  vae = VAE.from_config(cfg.vae)

  # Training forward pass (returns reconstruction + KL term):
  recon, posterior = vae(x)   # x: (B,1,256,256,256)

  # Encode only (returns DiagonalGaussian):
  posterior = vae.encode(x)
  z = posterior.sample()      # or .mode() for deterministic

  # Decode only:
  recon = vae.decode(z)       # z: (B,4,64,64,64)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .blocks import ResBlock, EncoderLevel, DecoderLevel, _norm, _act, _conv3d


# ---------------------------------------------------------------------------
# Diagonal Gaussian posterior
# ---------------------------------------------------------------------------

class DiagonalGaussian:
    """
    Represents q(z|x) = N(μ, σ²I) encoded as (mean, logvar) tensors.

    KL divergence against N(0,I):
        KL = -0.5 * mean(1 + logvar - mean² - exp(logvar))
    This is the standard VAE KL term (Kingma & Welling, 2013).  [PAPER — cites ref 16]
    """

    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor) -> None:
        self.mean   = mean
        self.logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        self.std    = torch.exp(0.5 * self.logvar)
        self.var    = torch.exp(self.logvar)

    def sample(self) -> torch.Tensor:
        """Reparameterised sample: z = μ + σ·ε."""
        eps = torch.randn_like(self.mean)
        return self.mean + self.std * eps

    def mode(self) -> torch.Tensor:
        """Deterministic MAP estimate (no noise)."""
        return self.mean

    def kl(self) -> torch.Tensor:
        """
        Per-element KL divergence, shape = mean.shape.
        Caller should .mean() this before adding to the loss.
        """
        return -0.5 * (1.0 + self.logvar - self.mean.pow(2) - self.var)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    3-level encoder:
      input conv → [EncoderLevel × 3] → bottleneck ResBlock → output conv

    Level   in_ch   out_ch   spatial
      0       1      64      256 → 128
      1      64     128      128 → 64
      2     128     256       64 → 64  (NO downsample at deepest level)
    bottleneck ResBlock(256, 256)
    output conv: 256 → 2 × latent_channels  (mean + logvar)

    The last level does NOT downsample so that the bottleneck spatial size
    equals input_size / 2^(num_levels-1) = 256 / 4 = 64.  [INFERRED]
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        channels: list[int],           # e.g. [64, 128, 256]
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.input_conv = _conv3d(in_channels, channels[0])

        levels: list[nn.Module] = []
        num_levels = len(channels)
        for i in range(num_levels):
            inc  = channels[i - 1] if i > 0 else channels[0]
            outc = channels[i]
            # Downsample at all levels EXCEPT the last
            levels.append(EncoderLevel(
                in_channels=inc,
                out_channels=outc,
                num_res_blocks=num_res_blocks,
                norm_num_groups=norm_num_groups,
                use_checkpointing=use_checkpointing,
                downsample=(i < num_levels - 1),
            ))
        self.levels = nn.ModuleList(levels)

        # Bottleneck ResBlock at the deepest spatial resolution
        self.bottleneck = ResBlock(
            channels[-1], channels[-1],
            norm_num_groups=norm_num_groups,
            use_checkpointing=use_checkpointing,
        )

        # Final norm+act+conv → 2×latent_channels (μ and log σ²)
        self.norm_out = _norm(norm_num_groups, channels[-1])
        self.act_out  = _act()
        self.conv_out = _conv3d(channels[-1], 2 * latent_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw (mean, logvar) concatenated along channel dim."""
        x = self.input_conv(x)
        for level in self.levels:
            x = level(x)
        x = self.bottleneck(x)
        x = self.conv_out(self.act_out(self.norm_out(x)))
        return x   # shape: (B, 2*latent_channels, D/4, H/4, W/4)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    3-level decoder — mirror of Encoder:
      input conv → bottleneck ResBlock → [DecoderLevel × 3] → output conv

    Level   in_ch   out_ch   spatial
      0     256     256       64 → 64  (NO upsample — mirrors encoder's no-downsample)
      1     256     128       64 → 128
      2     128      64      128 → 256
    output conv: 64 → out_channels (1 for CT)

    Paper: "the same number of channels in both encoder and decoder"  [PAPER]
    """

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        channels: list[int],           # e.g. [64, 128, 256] — reversed internally
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        rev_channels = list(reversed(channels))   # [256, 128, 64]

        self.input_conv = _conv3d(latent_channels, rev_channels[0])

        # Bottleneck ResBlock before upsampling starts
        self.bottleneck = ResBlock(
            rev_channels[0], rev_channels[0],
            norm_num_groups=norm_num_groups,
            use_checkpointing=use_checkpointing,
        )

        levels: list[nn.Module] = []
        num_levels = len(rev_channels)
        for i in range(num_levels):
            inc  = rev_channels[i]
            outc = rev_channels[i + 1] if i < num_levels - 1 else rev_channels[-1]
            # Upsample at all levels EXCEPT the first (mirrors encoder)
            levels.append(DecoderLevel(
                in_channels=inc,
                out_channels=outc,
                num_res_blocks=num_res_blocks,
                norm_num_groups=norm_num_groups,
                use_checkpointing=use_checkpointing,
                upsample=(i > 0),
            ))
        self.levels = nn.ModuleList(levels)

        # Final norm+act+conv → output CT volume
        self.norm_out = _norm(norm_num_groups, rev_channels[-1])
        self.act_out  = _act()
        self.conv_out = _conv3d(rev_channels[-1], out_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(z)
        x = self.bottleneck(x)
        for level in self.levels:
            x = level(x)
        x = self.conv_out(self.act_out(self.norm_out(x)))
        return x   # shape: (B, out_channels, D*4, H*4, W*4)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """
    3D Variational Autoencoder (LAND paper, Section 2).

    Forward pass returns (reconstruction, posterior).
    Use posterior.kl().mean() as L_KL.
    Use posterior.sample() when you need a stochastic latent during training.
    Use posterior.mode() for deterministic encoding at inference.

    Example
    -------
    >>> vae = VAE.from_config(cfg.vae)
    >>> x   = torch.randn(1, 1, 256, 256, 256)   # CT volume
    >>> recon, post = vae(x)
    >>> recon.shape   # (1, 1, 256, 256, 256)
    >>> post.mean.shape   # (1, 4, 64, 64, 64)
    """

    def __init__(
        self,
        in_channels: int       = 1,
        out_channels: int      = 1,
        latent_channels: int   = 4,
        channels: list[int]    | None = None,
        num_res_blocks: int    = 1,
        norm_num_groups: int   = 32,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [64, 128, 256]

        self.latent_channels = latent_channels

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            channels=channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            use_checkpointing=use_checkpointing,
        )

        self.decoder = Decoder(
            out_channels=out_channels,
            latent_channels=latent_channels,
            channels=channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            use_checkpointing=use_checkpointing,
        )

    # ------------------------------------------------------------------
    # Encode / decode / forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> DiagonalGaussian:
        """
        Encode a CT volume to a DiagonalGaussian posterior over the latent.

        x: (B, 1, D, H, W)
        returns DiagonalGaussian with .mean and .logvar of shape (B, 4, D/4, H/4, W/4)
        """
        h = self.encoder(x)                          # (B, 2*C_z, d, h, w)
        mean, logvar = h.chunk(2, dim=1)             # each (B, C_z, d, h, w)
        return DiagonalGaussian(mean, logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent sample to a CT volume.

        z: (B, 4, d, h, w)
        returns: (B, 1, D, H, W)
        """
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
    ) -> tuple[torch.Tensor, DiagonalGaussian]:
        """
        Full encode → sample → decode pass used during VAE training.

        Args:
            x:                 Input CT volume (B, 1, D, H, W)
            sample_posterior:  If True, sample z ~ q(z|x); if False, use mean.
                               Set False for validation to get deterministic reconstructions.

        Returns:
            recon:     Reconstructed volume (B, 1, D, H, W)
            posterior: DiagonalGaussian — use .kl() to get L_KL
        """
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        recon = self.decode(z)
        return recon, posterior

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, vae_cfg) -> "VAE":
        """
        Build a VAE from the OmegaConf vae_config.yaml architecture block.

        Usage:
            cfg = load_config(vae="configs/vae_config.yaml")
            vae = VAE.from_config(cfg.vae)
        """
        arch = vae_cfg.architecture
        return cls(
            in_channels      = arch.in_channels,
            out_channels     = arch.out_channels,
            latent_channels  = arch.latent_channels,
            channels         = list(arch.channels),
            num_res_blocks   = arch.num_res_blocks_per_level,
            norm_num_groups  = arch.norm_num_groups,
            use_checkpointing= arch.use_checkpointing,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        return {"encoder": enc, "decoder": dec, "total": enc + dec}


# ---------------------------------------------------------------------------
# Quick unit tests (run with:  python -m src.models.vae)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running VAE self-test on {device}…")
    print("(Using small spatial dim=32 to keep RAM reasonable in the test)")

    # Build a small VAE with the same channel structure but tiny spatial dims
    vae = VAE(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        channels=[64, 128, 256],
        num_res_blocks=1,
        norm_num_groups=32,
        use_checkpointing=False,
    ).to(device)

    params = vae.count_parameters()
    print(f"  Parameters — encoder: {params['encoder']:,}  decoder: {params['decoder']:,}  "
          f"total: {params['total']:,}")

    # Tiny volume: 32³ → latent 8³
    x = torch.randn(1, 1, 32, 32, 32, device=device)

    # Encode
    post = vae.encode(x)
    assert post.mean.shape   == (1, 4, 8, 8, 8), f"mean shape: {post.mean.shape}"
    assert post.logvar.shape == (1, 4, 8, 8, 8), f"logvar shape: {post.logvar.shape}"
    print(f"  encode():            PASS  posterior.mean {tuple(post.mean.shape)}")

    # KL
    kl = post.kl().mean()
    assert kl.ndim == 0, "KL should be scalar after .mean()"
    print(f"  posterior.kl():      PASS  kl={kl.item():.4f}")

    # Sample
    z = post.sample()
    assert z.shape == (1, 4, 8, 8, 8), f"sample shape: {z.shape}"
    print(f"  posterior.sample():  PASS  z {tuple(z.shape)}")

    # Decode
    recon = vae.decode(z)
    assert recon.shape == x.shape, f"recon shape: {recon.shape}"
    print(f"  decode():            PASS  recon {tuple(recon.shape)}")

    # Full forward
    recon2, post2 = vae(x)
    assert recon2.shape == x.shape
    print(f"  forward():           PASS  recon {tuple(recon2.shape)}")

    # Gradient check (tiny backward)
    recon2.mean().backward()
    print("  backward():          PASS")

    print(f"\nAll VAE tests passed.")
    sys.exit(0)
