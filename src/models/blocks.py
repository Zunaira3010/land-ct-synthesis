"""
src/models/blocks.py
====================
Primitive 3D building blocks shared by the VAE encoder/decoder.

Design decisions
----------------
- All convolutions are 3D (nn.Conv3d), kernel_size=3, padding=1 (same-padding).
- Normalisation: GroupNorm(num_groups=norm_num_groups, num_channels=C).
  Paper is silent; MAISI lightweight variant uses GroupNorm throughout.  [INFERRED]
- Activation: SiLU (Swish). Standard in MAISI / LDM codebases.  [INFERRED]
- Downsampling: strided Conv3d(stride=2). Clean, avoids aliasing artefacts
  compared with MaxPool when used inside an encoder.  [INFERRED]
- Upsampling: nearest-neighbour interpolation followed by a 3x3x3 conv.
  Avoids checkerboard artefacts common with ConvTranspose3d.  [INFERRED]
- Gradient checkpointing is threaded through ResBlock so it can be toggled
  per the `use_checkpointing` flag in vae_config.yaml.

References
----------
- LAND paper, Section 2, "3D VAE":
    "lightweight variant of the MAISI architecture [8], using 3 resolution levels
     with one residual block per level"
- vae_config.yaml:
    channels: [64, 128, 256]
    norm_num_groups: 32
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(num_groups: int, num_channels: int) -> nn.GroupNorm:
    """GroupNorm with zero-init bias and unit-init weight (PyTorch default)."""
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)


def _act() -> nn.SiLU:
    return nn.SiLU(inplace=True)


def _conv3d(in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1) -> nn.Conv3d:
    """3×3×3 (or 1×1×1) conv with same-padding for stride=1."""
    padding = kernel_size // 2
    return nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=True)


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    3D residual block: Norm → Act → Conv → Norm → Act → Conv + skip.

    If in_channels != out_channels a 1×1×1 projection shortcut is used.
    Supports gradient checkpointing to reduce activation memory.

    Paper: "one residual block per level" in the VAE.  [PAPER]
    Architecture follows MAISI's lightweight ResBlock convention.  [INFERRED]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int = 32,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.norm1 = _norm(norm_num_groups, in_channels)
        self.act1  = _act()
        self.conv1 = _conv3d(in_channels, out_channels)

        self.norm2 = _norm(norm_num_groups, out_channels)
        self.act2  = _act()
        self.conv2 = _conv3d(out_channels, out_channels)

        # zero-initialise the last conv so the block starts as identity  [INFERRED]
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        # Projection shortcut when channel counts differ
        self.skip_proj: nn.Module
        if in_channels != out_channels:
            self.skip_proj = _conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            return grad_checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


# ---------------------------------------------------------------------------
# Downsample block
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    """
    Halve spatial resolution with a strided 3×3×3 convolution (stride=2).

    Using a learnable strided conv rather than pooling keeps the encoder
    fully trainable end-to-end and is standard in MAISI/LDM encoders.  [INFERRED]
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            channels, channels,
            kernel_size=3, stride=2, padding=1, bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Upsample block
# ---------------------------------------------------------------------------

class Upsample(nn.Module):
    """
    Double spatial resolution: nearest-neighbour × 2, then refine with 3×3×3 conv.

    Nearest + conv avoids the checkerboard artefacts of ConvTranspose3d.  [INFERRED]
    This is the standard decoder upsampler in MAISI and Stable Diffusion.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = _conv3d(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder level (one resolution stage)
# ---------------------------------------------------------------------------

class EncoderLevel(nn.Module):
    """
    One resolution level of the encoder:
      ResBlock(in_ch → out_ch)  [× num_res_blocks]
      Downsample(out_ch)        [omitted at the bottleneck level]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        use_checkpointing: bool = False,
        downsample: bool = True,
    ) -> None:
        super().__init__()

        blocks: list[nn.Module] = []
        for i in range(num_res_blocks):
            blocks.append(ResBlock(
                in_channels  if i == 0 else out_channels,
                out_channels,
                norm_num_groups=norm_num_groups,
                use_checkpointing=use_checkpointing,
            ))
        self.res_blocks = nn.Sequential(*blocks)

        self.down: nn.Module = Downsample(out_channels) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_blocks(x)
        return self.down(x)


# ---------------------------------------------------------------------------
# Decoder level (one resolution stage)
# ---------------------------------------------------------------------------

class DecoderLevel(nn.Module):
    """
    One resolution level of the decoder:
      Upsample(in_ch)             [omitted at the bottleneck level]
      ResBlock(in_ch → out_ch)   [× num_res_blocks]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        use_checkpointing: bool = False,
        upsample: bool = True,
    ) -> None:
        super().__init__()

        self.up: nn.Module = Upsample(in_channels) if upsample else nn.Identity()

        blocks: list[nn.Module] = []
        for i in range(num_res_blocks):
            blocks.append(ResBlock(
                in_channels  if i == 0 else out_channels,
                out_channels,
                norm_num_groups=norm_num_groups,
                use_checkpointing=use_checkpointing,
            ))
        self.res_blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.res_blocks(x)


# ---------------------------------------------------------------------------
# Quick unit tests (run with:  python -m src.models.blocks)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running blocks self-test on {device}…")

    # --- ResBlock: same channels ---
    x = torch.randn(1, 64, 32, 32, 32, device=device)
    y = ResBlock(64, 64).to(device)(x)
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    print("  ResBlock(same ch):   PASS")

    # --- ResBlock: channel change ---
    x = torch.randn(1, 64, 32, 32, 32, device=device)
    y = ResBlock(64, 128).to(device)(x)
    assert y.shape == (1, 128, 32, 32, 32), f"Got {y.shape}"
    print("  ResBlock(ch change): PASS")

    # --- Downsample ---
    x = torch.randn(1, 128, 32, 32, 32, device=device)
    y = Downsample(128).to(device)(x)
    assert y.shape == (1, 128, 16, 16, 16), f"Got {y.shape}"
    print("  Downsample:          PASS")

    # --- Upsample ---
    x = torch.randn(1, 128, 16, 16, 16, device=device)
    y = Upsample(128).to(device)(x)
    assert y.shape == (1, 128, 32, 32, 32), f"Got {y.shape}"
    print("  Upsample:            PASS")

    # --- EncoderLevel ---
    x = torch.randn(1, 64, 64, 64, 64, device=device)
    y = EncoderLevel(64, 128, downsample=True).to(device)(x)
    assert y.shape == (1, 128, 32, 32, 32), f"Got {y.shape}"
    print("  EncoderLevel:        PASS")

    # --- DecoderLevel ---
    x = torch.randn(1, 128, 32, 32, 32, device=device)
    y = DecoderLevel(128, 64, upsample=True).to(device)(x)
    assert y.shape == (1, 64, 64, 64, 64), f"Got {y.shape}"
    print("  DecoderLevel:        PASS")

    print("All block tests passed.")
    sys.exit(0)
