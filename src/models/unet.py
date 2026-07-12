"""
src/models/unet.py
===================
3D Diffusion U-Net (denoising network) matching the LAND paper spec.

Architecture
------------
Paper (Section 2, "3D U-Net"):
  - Input / output: 64x64x64x4 latent (matches VAE latent shape)          [PAPER]
  - 5 resolution levels, 2 residual blocks per level                     [PAPER]
  - Skip connections are ADDITIVE, not concatenation (cites PatchDDM,
    ref [2], motivated by memory reduction)                              [PAPER]
  - Conditioning: cross-attention, re-injected at multiple resolution
    levels                                                               [PAPER]
  - Channel widths [64, 128, 256, 384, 512]                              [INFERRED, standard LDM progression]
  - Attention at the coarsest 3 of 5 levels                              [INFERRED, standard LDM practice]
  - Prediction target: velocity (v-prediction)                           [PAPER]
  - Timestep embedding: sinusoidal + 2-layer MLP, injected additively
    into every ResBlock                                                  [INFERRED — paper doesn't
    describe this, but it's a structural necessity for any diffusion
    U-Net conditioned on t, not an optional design choice]

Conditioning mask (docs/01_architecture.md, docs/06_open_questions.md #1, #9)
-------------------------------------------------------------------------
Paper, Section 2, verbatim: masks are "normalized to [0,1], downsampled four times via
3D max pooling, concatenated with the noisy latent, and injected into U-Net cross-attention
layers." This is BOTH concatenation AND cross-attention — unusual, but explicit.  [PAPER]

  1. Concatenation: the downsampled mask (B,1,64,64,64) is concatenated channel-wise with
     the noisy latent (B,4,64,64,64) -> (B,5,64,64,64) before the input conv.
  2. Cross-attention: the SAME mask is separately flattened into a token sequence and
     linear-projected to `cross_attention_dim`, then used as the key/value context at every
     attended resolution level. This is Q#9's resolution — **Option A** (flatten -> linear
     project), chosen over a small-CNN front end (Option B) because the mask is small and
     semantically simple (3 values: 0 / 0.5 / texture÷5) and spatial location IS the
     information; a learned CNN front end adds complexity with no clear benefit.  [INFERRED]

Known open item carried forward from Option A: flattening the full 64^3 mask gives 262,144
context tokens for cross-attention at every attended level. This is cheap in parameter count
but the attention matrix at the widest attended level (finest of the 3 attended levels) is the
memory bottleneck to watch once training is attempted — if it doesn't fit, the fallback is
subsampling/pooling the context tokens further, NOT silently switching to Option B.

Public API
----------
  unet = UNet3D.from_config(cfg.diffusion)

  # v: predicted velocity, same shape as the input latent
  v_pred = unet(z_t, timesteps, mask)   # z_t: (B,4,64,64,64), timesteps: (B,), mask: (B,1,64,64,64)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .blocks import _norm, _act, _conv3d, Downsample, Upsample


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Standard transformer-style sinusoidal embedding of the diffusion timestep.
    Not paper-specified; this is the near-universal choice across diffusion U-Nets
    (DDPM, LDM, MAISI) so treated as a structural default rather than a design
    decision needing its own open-question entry.  [INFERRED]
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=timesteps.device).float() / half
        )
        args = timesteps.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding -> 2-layer MLP, standard LDM timestep conditioning path."""

    def __init__(self, sinusoidal_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.sinusoidal = SinusoidalTimestepEmbedding(sinusoidal_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(timesteps))


# ---------------------------------------------------------------------------
# Mask token encoder feeding cross-attention  (Q#9 — Option A, resolved)
# ---------------------------------------------------------------------------

class MaskTokenEncoder(nn.Module):
    """
    Flattens the downsampled conditioning mask directly into a token sequence and
    linear-projects each token to `cross_attention_dim`. No CNN front end (Option A,
    see module docstring and docs/06_open_questions.md #9).

    mask: (B, mask_channels, D, H, W) -> tokens: (B, D*H*W, cross_attention_dim)
    """

    def __init__(self, mask_channels: int, cross_attention_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(mask_channels, cross_attention_dim)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = mask.shape
        tokens = mask.flatten(2).transpose(1, 2)   # (B, N=D*H*W, C=mask_channels)
        return self.proj(tokens)                    # (B, N, cross_attention_dim)


# ---------------------------------------------------------------------------
# Cross-attention block
# ---------------------------------------------------------------------------

class CrossAttention3D(nn.Module):
    """
    Standard multi-head cross-attention: spatial feature map (flattened to tokens) as
    query, mask tokens (from MaskTokenEncoder) as key/value context. Residual, with the
    output projection zero-initialised so each block starts as identity — same
    zero-init-last-layer convention used in blocks.ResBlock.  [INFERRED, standard LDM practice]
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int,
        num_head_channels: int = 64,
        norm_num_groups: int = 32,
    ) -> None:
        super().__init__()
        num_heads = max(1, query_dim // num_head_channels)
        inner_dim = num_heads * num_head_channels
        self.num_heads = num_heads
        self.head_dim = num_head_channels

        self.norm = _norm(norm_num_groups, query_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)   context: (B, N_ctx, cross_attention_dim)
        b, c, d, h, w = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)   # (B, N, C)

        q = self._split_heads(self.to_q(tokens), b)
        k = self._split_heads(self.to_k(context), b)
        v = self._split_heads(self.to_v(context), b)

        out = F.scaled_dot_product_attention(q, k, v)       # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(b, -1, self.num_heads * self.head_dim)
        out = self.to_out(out).transpose(1, 2).view(b, c, d, h, w)
        return x + out   # residual; zero-init means this starts as a no-op

    def _split_heads(self, t: torch.Tensor, batch: int) -> torch.Tensor:
        n = t.shape[1]
        return t.view(batch, n, self.num_heads, self.head_dim).transpose(1, 2)


# ---------------------------------------------------------------------------
# Time-conditioned ResBlock
# ---------------------------------------------------------------------------

class ResBlockTime(nn.Module):
    """
    Same Norm->Act->Conv->Norm->Act->Conv (+skip) shape as blocks.ResBlock, with a
    timestep-embedding projection added in between the two convs — standard LDM
    ResBlock convention.  [INFERRED]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        norm_num_groups: int = 32,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.norm1 = _norm(norm_num_groups, in_channels)
        self.act1 = _act()
        self.conv1 = _conv3d(in_channels, out_channels)

        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = _norm(norm_num_groups, out_channels)
        self.act2 = _act()
        self.conv2 = _conv3d(out_channels, out_channels)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self.skip_proj: nn.Module
        if in_channels != out_channels:
            self.skip_proj = _conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def _forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip_proj(x)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            return grad_checkpoint(self._forward, x, t_emb, use_reentrant=False)
        return self._forward(x, t_emb)


# ---------------------------------------------------------------------------
# Encoder / decoder levels
# ---------------------------------------------------------------------------

class UNetEncoderLevel(nn.Module):
    """
    One resolution level: [ResBlockTime (+ optional CrossAttention3D)] x num_res_blocks,
    then Downsample (omitted at the bottleneck level). Returns the pre-downsample
    activation as `skip` for the matching decoder level's additive skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int,
        use_attention: bool,
        cross_attention_dim: int,
        num_head_channels: int,
        norm_num_groups: int,
        use_checkpointing: bool,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList() if use_attention else None
        for i in range(num_res_blocks):
            inc = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResBlockTime(
                inc, out_channels, time_emb_dim, norm_num_groups, use_checkpointing,
            ))
            if use_attention:
                self.attn_blocks.append(CrossAttention3D(
                    out_channels, cross_attention_dim, num_head_channels, norm_num_groups,
                ))

        self.down: nn.Module = Downsample(out_channels) if downsample else nn.Identity()

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, context: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for i, res in enumerate(self.res_blocks):
            x = res(x, t_emb)
            if self.use_attention:
                x = self.attn_blocks[i](x, context)
        skip = x
        x = self.down(x)
        return x, skip


class UNetDecoderLevel(nn.Module):
    """
    Mirror of UNetEncoderLevel: Upsample, ADDITIVE skip connection (not concat — see
    module docstring), then [ResBlockTime (+ optional CrossAttention3D)] x num_res_blocks.

    Note on channel handling: an additive skip requires the upsampled activation and the
    skip tensor to already have the SAME channel count *before* any ResBlock runs (unlike
    the VAE decoder, where concatenation isn't used and ResBlocks are free to change
    channel count on their own first block). blocks.Upsample only preserves channel count,
    so the in_channels -> out_channels projection happens here, fused into the upsample
    step itself (nearest-interp + a conv that changes channels), rather than being left to
    the first ResBlock as in blocks.DecoderLevel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int,
        use_attention: bool,
        cross_attention_dim: int,
        num_head_channels: int,
        norm_num_groups: int,
        use_checkpointing: bool,
        upsample: bool,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.upsample = upsample
        if upsample:
            # nearest-neighbour x2 (applied in forward) + conv that also changes channels
            self.up_conv = _conv3d(in_channels, out_channels)
        else:
            # Bottleneck level: no upsample, so in_channels must already equal out_channels.
            assert in_channels == out_channels, (
                "Non-upsampling decoder level (bottleneck) requires in_channels == "
                f"out_channels, got {in_channels} != {out_channels}"
            )
            self.up_conv = nn.Identity()

        # Channel transition already happened above, so every ResBlockTime here runs at a
        # constant out_channels -> out_channels (unlike blocks.DecoderLevel, which lets its
        # first ResBlock absorb the channel change).
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList() if use_attention else None
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResBlockTime(
                out_channels, out_channels, time_emb_dim, norm_num_groups, use_checkpointing,
            ))
            if use_attention:
                self.attn_blocks.append(CrossAttention3D(
                    out_channels, cross_attention_dim, num_head_channels, norm_num_groups,
                ))

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: torch.Tensor,
        context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.up_conv(x)
        x = x + skip   # additive skip connection  [PAPER — cites PatchDDM, ref [2]]
        for i, res in enumerate(self.res_blocks):
            x = res(x, t_emb)
            if self.use_attention:
                x = self.attn_blocks[i](x, context)
        return x


# ---------------------------------------------------------------------------
# UNet3D
# ---------------------------------------------------------------------------

class UNet3D(nn.Module):
    """
    3D denoising U-Net (LAND paper, Section 2).

    Forward pass:
        v_pred = unet(z_t, timesteps, mask)

    Args to forward():
        z_t:        noisy latent, (B, in_channels, D, H, W) -- e.g. (B, 4, 64, 64, 64)
        timesteps:  (B,) integer or float diffusion timesteps
        mask:       downsampled conditioning mask, (B, mask_channels, D, H, W),
                    same spatial size as z_t -- e.g. (B, 1, 64, 64, 64)

    Returns:
        v_pred: predicted velocity, same shape as z_t

    Example
    -------
    >>> unet = UNet3D.from_config(cfg.diffusion)
    >>> z_t  = torch.randn(1, 4, 64, 64, 64)
    >>> mask = torch.rand(1, 1, 64, 64, 64)
    >>> t    = torch.randint(0, 1000, (1,))
    >>> v_pred = unet(z_t, t, mask)
    >>> v_pred.shape   # (1, 4, 64, 64, 64)
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        channels: list[int] | None = None,
        num_res_blocks: int = 2,
        attention_levels: list[bool] | None = None,
        mask_channels: int = 1,
        cross_attention_dim: int = 64,
        num_head_channels: int = 64,
        norm_num_groups: int = 32,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [64, 128, 256, 384, 512]
        if attention_levels is None:
            attention_levels = [False, False, True, True, True]
        assert len(channels) == len(attention_levels), \
            "channels and attention_levels must have the same number of resolution levels"

        num_levels = len(channels)
        time_emb_dim = channels[0] * 4   # [INFERRED] standard LDM convention (base_ch * 4)

        self.time_embedding = TimestepEmbedding(sinusoidal_dim=channels[0], emb_dim=time_emb_dim)
        self.mask_encoder = MaskTokenEncoder(mask_channels, cross_attention_dim)

        # Concatenate the mask (mask_channels) with the noisy latent (in_channels) before
        # the input conv -- the "concatenated with the noisy latent" half of the paper's
        # dual concat+cross-attention conditioning.  [PAPER]
        self.input_conv = _conv3d(in_channels + mask_channels, channels[0])

        encoder_levels: list[nn.Module] = []
        for i in range(num_levels):
            inc = channels[0] if i == 0 else channels[i - 1]
            encoder_levels.append(UNetEncoderLevel(
                in_channels=inc,
                out_channels=channels[i],
                time_emb_dim=time_emb_dim,
                num_res_blocks=num_res_blocks,
                use_attention=attention_levels[i],
                cross_attention_dim=cross_attention_dim,
                num_head_channels=num_head_channels,
                norm_num_groups=norm_num_groups,
                use_checkpointing=use_checkpointing,
                downsample=(i < num_levels - 1),   # no downsample at the bottleneck level
            ))
        self.encoder_levels = nn.ModuleList(encoder_levels)

        rev_channels = list(reversed(channels))
        rev_attention = list(reversed(attention_levels))
        decoder_levels: list[nn.Module] = []
        for i in range(num_levels):
            # Level 0 is the bottleneck: in == out == deepest channel width, no upsample.
            # Every level after that takes the PREVIOUS level's out_channels as input and
            # projects down to this level's target width during the upsample step (see
            # UNetDecoderLevel docstring) so the result matches this level's skip tensor.
            inc = rev_channels[0] if i == 0 else rev_channels[i - 1]
            outc = rev_channels[i]
            decoder_levels.append(UNetDecoderLevel(
                in_channels=inc,
                out_channels=outc,
                time_emb_dim=time_emb_dim,
                num_res_blocks=num_res_blocks,
                use_attention=rev_attention[i],
                cross_attention_dim=cross_attention_dim,
                num_head_channels=num_head_channels,
                norm_num_groups=norm_num_groups,
                use_checkpointing=use_checkpointing,
                upsample=(i > 0),   # no upsample at the bottleneck level
            ))
        self.decoder_levels = nn.ModuleList(decoder_levels)

        # Project the last decoder level's channels back down to channels[0] before the
        # output conv, since additive skips keep the decoder at full width throughout.
        self.pre_out_proj = _conv3d(rev_channels[-1], channels[0], kernel_size=1)
        self.norm_out = _norm(norm_num_groups, channels[0])
        self.act_out = _act()
        self.conv_out = _conv3d(channels[0], out_channels)
        # Zero-init the final conv so the model initially predicts zero velocity --
        # standard diffusion-U-Net practice for training stability.  [INFERRED]
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embedding(timesteps)
        context = self.mask_encoder(mask)

        x = torch.cat([z_t, mask], dim=1)
        x = self.input_conv(x)

        skips: list[torch.Tensor] = []
        for level in self.encoder_levels:
            x, skip = level(x, t_emb, context)
            skips.append(skip)

        for level, skip in zip(self.decoder_levels, reversed(skips)):
            x = level(x, skip, t_emb, context)

        x = self.pre_out_proj(x)
        x = self.conv_out(self.act_out(self.norm_out(x)))
        return x

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, diffusion_cfg) -> "UNet3D":
        """
        Build a UNet3D from the OmegaConf diffusion_config.yaml architecture/conditioning
        blocks.

        Usage:
            cfg = load_config(diffusion="configs/diffusion_config.yaml")
            unet = UNet3D.from_config(cfg.diffusion)
        """
        arch = diffusion_cfg.architecture
        cond = diffusion_cfg.conditioning
        return cls(
            in_channels          = arch.in_channels,
            out_channels         = arch.out_channels,
            channels             = list(arch.channels),
            num_res_blocks       = arch.num_res_blocks_per_level,
            attention_levels     = list(arch.attention_levels),
            mask_channels        = cond.mask_channels,
            cross_attention_dim  = arch.cross_attention_dim,
            num_head_channels    = arch.num_head_channels,
            norm_num_groups      = arch.norm_num_groups,
            use_checkpointing    = getattr(arch, "use_checkpointing", False),
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict[str, int]:
        enc = sum(p.numel() for lvl in self.encoder_levels for p in lvl.parameters())
        dec = sum(p.numel() for lvl in self.decoder_levels for p in lvl.parameters())
        other = sum(p.numel() for n, p in self.named_parameters()
                    if not n.startswith("encoder_levels") and not n.startswith("decoder_levels"))
        return {"encoder": enc, "decoder": dec, "other": other, "total": enc + dec + other}


# ---------------------------------------------------------------------------
# Quick unit tests (run with:  python -m src.models.unet)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running UNet3D self-test on {device}…")
    print("(Using a small 2-level config at spatial dim=8 to keep this fast/light)")

    # Small config: 2 levels instead of 5, tiny spatial size, so the self-test runs in
    # a couple seconds on CPU. Real config is exercised separately via from_config()
    # once the diffusion training script drives it end-to-end.
    unet = UNet3D(
        in_channels=4,
        out_channels=4,
        channels=[8, 16],
        num_res_blocks=2,
        attention_levels=[False, True],
        mask_channels=1,
        cross_attention_dim=4,
        num_head_channels=4,
        norm_num_groups=4,
        use_checkpointing=False,
    ).to(device)

    params = unet.count_parameters()
    print(f"  Parameters — encoder: {params['encoder']:,}  decoder: {params['decoder']:,}  "
          f"other: {params['other']:,}  total: {params['total']:,}")

    z_t = torch.randn(1, 4, 8, 8, 8, device=device)
    mask = torch.rand(1, 1, 8, 8, 8, device=device)
    t = torch.randint(0, 1000, (1,), device=device)

    v_pred = unet(z_t, t, mask)
    assert v_pred.shape == z_t.shape, f"Expected {z_t.shape}, got {v_pred.shape}"
    print(f"  forward():            PASS  v_pred {tuple(v_pred.shape)}")

    # Batch > 1 sanity check
    z_t2 = torch.randn(2, 4, 8, 8, 8, device=device)
    mask2 = torch.rand(2, 1, 8, 8, 8, device=device)
    t2 = torch.randint(0, 1000, (2,), device=device)
    v_pred2 = unet(z_t2, t2, mask2)
    assert v_pred2.shape == z_t2.shape, f"Got {v_pred2.shape}"
    print(f"  forward(batch=2):     PASS  v_pred {tuple(v_pred2.shape)}")

    # Gradient check
    v_pred.mean().backward()
    print("  backward():           PASS")

    print("\nAll UNet3D tests passed.")
    sys.exit(0)
