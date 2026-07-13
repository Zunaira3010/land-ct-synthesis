"""
src/inference/sample.py
========================
Generation pipeline: noise -> reverse diffusion -> VAE decode -> CT volume.

Mirrors scripts/precompute_latents.py's sliding-window strategy, but in the opposite
direction for decoding: that script slides a 96^3 CT-space window across the full volume to
ENCODE (since a naive full-256^3 vae.encode() needs ~54GB on the first conv), and this module
slides the same-sized window to DECODE a full 64^3 latent back into a full 256^3 CT volume, for
the same underlying reason -- the decoder's upsampling path has the same kind of memory wall as
the encoder's downsampling path, just approached from the other end.

Diffusion sampling itself, unlike encode/decode, is NOT patch-tiled: the paper's own reference
setup uses full 64^3-latent diffusion (7.5GB claimed on an A100 for inference, since inference
is forward-passes-only -- no gradients, optimizer state, or grad-accum buffers, unlike
training), and the U-Net needs the full spatial context at every step for global coherence
across the volume, which patch-tiling the *diffusion* process itself would undermine. Only the
VAE decode step at the very end is tiled, exactly mirroring precompute_latents.py's encode.

Usage
-----
See scripts/generate_samples.py for the CLI entry point. This module is the reusable core;
generate_samples.py is a thin argument-parsing wrapper around it, in keeping with the repo's
existing pattern of keeping library logic separate from CLI scripts (e.g. precompute_latents.py
vs. its own encode_full_volume, or check_vae_reconstruction.py's structure).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.masks import downsample_mask_maxpool, make_healthy_mask
from src.data.preprocessing import DEFAULT_HU_WINDOW, DEFAULT_OUT_RANGE
from src.data.sliding_window import sliding_window_average
from src.models.unet import UNet3D
from src.models.vae import VAE
from src.training.diffusion_schedule import VPredictionSchedule

# Same CT-space / latent-space window sizes as scripts/precompute_latents.py, reused here for
# the decode direction so the encode and decode passes are symmetric (a window size that's
# already proven safe to encode is also safe to decode -- same conv workspace cost either way).
CT_PATCH_SIZE = 96
MASK_DOWNSAMPLE_FACTOR = 4
LATENT_PATCH_SIZE = CT_PATCH_SIZE // MASK_DOWNSAMPLE_FACTOR   # 24
LATENT_STRIDE = LATENT_PATCH_SIZE - 8                          # 16, 8-voxel overlap in latent space
CT_STRIDE = LATENT_STRIDE * MASK_DOWNSAMPLE_FACTOR              # 64, 32-voxel overlap in CT space


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_unet_from_checkpoint(
    checkpoint_path: str,
    diffusion_cfg,
    device: str,
    use_ema: bool = True,
) -> UNet3D:
    """
    Build a UNet3D from config and load weights from a src.training.train_diffusion checkpoint.

    Real checkpoints are saved as {"global_step", "unet_state", "ema_state", "optim_state",
    "scaler_state", "best_val_loss"} (see train_diffusion.py's _save_checkpoint) -- NOT under a
    "model" key, and NOT with weights_only=True (matches how train_diffusion.py itself loads
    checkpoints for --resume).

    use_ema=True (default) loads the EMA shadow weights rather than the raw (noisier,
    single-step) optimizer weights -- standard practice for generation/evaluation with a
    diffusion model, since EMA is what training accumulates specifically to smooth out
    per-step optimizer noise for exactly this use case.
    """
    unet = UNet3D.from_config(diffusion_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_key = "ema_state" if use_ema else "unet_state"
    if state_key not in ckpt:
        raise KeyError(
            f"Checkpoint {checkpoint_path} has no '{state_key}' key -- found keys: "
            f"{list(ckpt.keys())}. If this is an early/smoke-test checkpoint saved before EMA "
            f"was wired in, pass use_ema=False to fall back to raw weights."
        )
    # EMA shadow is stored in float32 for accumulation precision (see train_diffusion.py's EMA
    # class); cast to match the freshly-constructed model's own parameter dtypes before loading,
    # exactly like EMA.copy_to does during training-time validation.
    model_dtype = next(unet.parameters()).dtype
    shadow = {k: v.to(dtype=model_dtype) for k, v in ckpt[state_key].items()}
    unet.load_state_dict(shadow)
    unet.eval()
    step = ckpt.get("global_step", "?")
    print(f"Loaded UNet3D from {checkpoint_path} (global_step={step}, weights={state_key})")
    return unet


def load_vae_from_checkpoint(checkpoint_path: str, vae_cfg, device: str) -> VAE:
    """Build a VAE from config and load weights -- same checkpoint format/convention as
    scripts/precompute_latents.py and scripts/check_vae_reconstruction.py use."""
    vae = VAE.from_config(vae_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["vae_state"])
    vae.eval()
    epoch = ckpt.get("epoch", "?")
    best_val = ckpt.get("best_val_loss", float("nan"))
    print(f"Loaded VAE from {checkpoint_path} (epoch={epoch}, best_val_loss={best_val:.4f})")
    return vae


# ---------------------------------------------------------------------------
# Conditioning mask
# ---------------------------------------------------------------------------

def build_conditioning_mask(
    mode: str,
    device: str,
    lung_mask_256: np.ndarray | None = None,
    encoded_mask_256: np.ndarray | None = None,
) -> torch.Tensor:
    """
    Build a (1, 1, 64, 64, 64) conditioning mask tensor at diffusion-latent resolution.

    mode="reuse_encoded": encoded_mask_256 is an already lung/nodule-value-encoded 256^3 mask
      pulled straight from a cached patient's .npz (the same array LIDCVolumeDataset reads,
      per lidc_dataset.py's load_cached_npz docstring) -- just needs downsampling, no
      re-encoding. This is the easiest way to get a real, paper-faithful conditioning signal
      for a first generation test.
    mode="healthy_from_lung_mask": lung_mask_256 is a boolean lung segmentation (no nodule
      info) -- encodes it via masks.make_healthy_mask (lung=0.5, background=0), then
      downsamples. Produces a nodule-free conditioning signal.
    mode="unconditional": an all-zero mask -- only meaningful if the model was actually trained
      with conditioning dropout (configs/diffusion_config.yaml's cfg_dropout_prob), since
      that's the only way it ever saw an all-zero mask during training.
    """
    if mode == "reuse_encoded":
        if encoded_mask_256 is None:
            raise ValueError("mode='reuse_encoded' requires encoded_mask_256")
        mask_256 = encoded_mask_256
    elif mode == "healthy_from_lung_mask":
        if lung_mask_256 is None:
            raise ValueError("mode='healthy_from_lung_mask' requires lung_mask_256")
        mask_256 = make_healthy_mask(lung_mask_256)
    elif mode == "unconditional":
        mask_256 = np.zeros((256, 256, 256), dtype=np.float32)
    else:
        raise ValueError(f"unknown mask mode: {mode!r}")

    mask_64 = downsample_mask_maxpool(mask_256.astype(np.float32), factor=MASK_DOWNSAMPLE_FACTOR)
    return torch.from_numpy(mask_64).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Reverse diffusion (full 64^3 latent -- not patch-tiled, see module docstring)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_diffusion_sampling(
    unet: UNet3D,
    schedule: VPredictionSchedule,
    mask: torch.Tensor,
    latent_shape: tuple[int, int, int, int] = (1, 4, 64, 64, 64),
    num_inference_steps: int | None = None,
    guidance_scale: float = 1.0,
    device: str = "cpu",
    seed: int | None = None,
    progress_every: int = 100,
) -> torch.Tensor:
    """
    Full ancestral DDPM reverse process, from pure noise x_T down to x_0, at full latent
    resolution (no sliding-window tiling -- see module docstring for why).

    guidance_scale: 1.0 = no classifier-free guidance (single conditioned forward pass per
    step, fastest). >1.0 runs BOTH a conditioned and an unconditional (all-zero-mask) forward
    pass per step and extrapolates away from the unconditional prediction -- meaningful only if
    the checkpoint was trained with conditioning dropout (cfg_dropout_prob), since that's the
    only way the model ever learned to produce a sensible unconditional prediction to
    extrapolate away from. Doubles compute per step when enabled.

    num_inference_steps: defaults to schedule.num_train_timesteps (paper: "inference uses the
    same number of steps to prioritize sample quality" -- i.e. full 1000-step ancestral
    sampling, not a strided/DDIM-style subsample). Fewer steps trades quality for speed by
    literally skipping timesteps in the reverse loop; NOT the same schedule the model was
    trained against, so treat this as an explicit, non-paper-faithful speed/quality trade if
    used, not a free lunch.
    """
    if num_inference_steps is None:
        num_inference_steps = schedule.num_train_timesteps

    # Generator must live on the same device as the tensors it's seeding noise for --
    # torch.randn(..., device="cuda", generator=<cpu generator>) raises a RuntimeError rather
    # than silently working, so this can't be a CPU generator whenever device="cuda".
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    x = torch.randn(latent_shape, generator=generator, device=device)
    zero_mask = torch.zeros_like(mask)

    # Evenly-spaced subset of the full [0, num_train_timesteps) range, walked in reverse.
    # With num_inference_steps == num_train_timesteps (the paper-faithful default) this is
    # just every integer timestep, i.e. ordinary full ancestral sampling.
    step_indices = torch.linspace(
        0, schedule.num_train_timesteps - 1, num_inference_steps
    ).round().long().unique()
    step_indices = torch.flip(step_indices, dims=[0])

    for i, step in enumerate(step_indices):
        t_batch = torch.full((latent_shape[0],), int(step.item()), dtype=torch.long, device=device)

        v_cond = unet(x, t_batch, mask)
        if guidance_scale != 1.0:
            v_uncond = unet(x, t_batch, zero_mask)
            v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v_pred = v_cond

        x = schedule.ddpm_reverse_step(x, v_pred, t_batch, generator=generator)

        if progress_every and (i % progress_every == 0 or i == len(step_indices) - 1):
            print(f"  sampling step {i + 1}/{len(step_indices)} (t={int(step.item())})", flush=True)

    return x


# ---------------------------------------------------------------------------
# Sliding-window VAE decode (64^3 latent -> 256^3 CT), mirrors precompute_latents.py's encode
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_full_volume(
    vae: VAE,
    latent: np.ndarray,
    device: str,
    ct_shape: tuple[int, int, int] = (256, 256, 256),
) -> np.ndarray:
    """
    Decode a full (latent_channels, 64, 64, 64) latent to a full (256, 256, 256) CT volume via
    sliding-window VAE decoding, blended with sliding_window_average -- the exact inverse of
    scripts/precompute_latents.py's encode_full_volume, using the same window/stride constants
    so every window this decodes is the same size the encoder was ever asked to handle (a
    96^3-in/24^3-out patch, just run through vae.decode() instead of vae.encode()).

    latent: (C, 64, 64, 64) numpy array (the mean/x0 estimate from diffusion sampling, moved
    to CPU/numpy -- decoding patch-by-patch doesn't need it on GPU as one large tensor anyway).
    """
    def decode_patch_fn(d0: int, h0: int, w0: int) -> np.ndarray:
        # CT-space origin -> latent-space origin: exact scale-down (inverse of
        # precompute_latents.py's encode_patch_fn, which scales latent origins UP to CT
        # origins). CT_STRIDE/MASK_DOWNSAMPLE_FACTOR == LATENT_STRIDE by construction, so this
        # always lands on an integer latent index.
        ld0 = d0 // MASK_DOWNSAMPLE_FACTOR
        lh0 = h0 // MASK_DOWNSAMPLE_FACTOR
        lw0 = w0 // MASK_DOWNSAMPLE_FACTOR
        z_patch = latent[:, ld0:ld0 + LATENT_PATCH_SIZE, lh0:lh0 + LATENT_PATCH_SIZE, lw0:lw0 + LATENT_PATCH_SIZE]
        z = torch.from_numpy(z_patch[np.newaxis]).to(device=device, dtype=torch.float32)
        ct_patch = vae.decode(z)
        return ct_patch.squeeze(0).cpu().numpy().astype(np.float32)

    return sliding_window_average(
        volume_shape=ct_shape,
        patch_size=CT_PATCH_SIZE,
        stride=CT_STRIDE,
        encode_patch_fn=decode_patch_fn,
        out_channels=1,
    )[0]  # single output channel -> (256, 256, 256)


def denormalize_hu(
    ct_normalized: np.ndarray,
    hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
    out_range: tuple[float, float] = DEFAULT_OUT_RANGE,
) -> np.ndarray:
    """Exact inverse of src.data.preprocessing.clip_and_normalize_hu: maps the VAE's
    [-1, 1]-ish output back to the original HU window it was trained against. Values outside
    [-1, 1] (the decoder isn't constrained to stay in-range) map outside [hu_window] rather
    than being clamped -- left as-is here so the caller can inspect out-of-range voxels rather
    than having them silently hidden by a clamp."""
    lo, hi = hu_window
    out_lo, out_hi = out_range
    normalized_01 = (ct_normalized - out_lo) / (out_hi - out_lo)
    return (normalized_01 * (hi - lo) + lo).astype(np.float32)


# ---------------------------------------------------------------------------
# Self-test (run with: python -m src.inference.sample)
# Exercises every function against tiny synthetic models/data -- no real checkpoint needed,
# same "prove the wiring before trusting it against real weights" approach as
# scripts/precompute_latents.py's own module docstring describes.
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import sys
    from src.data.preprocessing import clip_and_normalize_hu

    device = "cpu"
    print("Running src.inference.sample self-tests...")

    # ---- 1. build_conditioning_mask, all three modes ----
    small_shape = (32, 32, 32)  # divisible by MASK_DOWNSAMPLE_FACTOR=4, tiny for speed
    lung = np.zeros(small_shape, dtype=bool)
    lung[8:24, 8:24, 8:24] = True
    m1 = build_conditioning_mask("healthy_from_lung_mask", device, lung_mask_256=lung)
    assert m1.shape == (1, 1, 8, 8, 8), f"got {m1.shape}"
    assert torch.isclose(m1.max(), torch.tensor(0.5)), "lung value should downsample to 0.5"

    encoded = np.zeros(small_shape, dtype=np.float32)
    encoded[8:24, 8:24, 8:24] = 0.5
    encoded[14:18, 14:18, 14:18] = 0.6  # fake nodule value
    m2 = build_conditioning_mask("reuse_encoded", device, encoded_mask_256=encoded)
    assert m2.shape == (1, 1, 8, 8, 8)
    assert m2.max().item() > 0.5, "max-pool downsample should preserve the nodule's peak value"

    m3 = build_conditioning_mask("unconditional", device)
    assert torch.all(m3 == 0), "unconditional mask must be all-zero"
    print("  build_conditioning_mask (healthy/reuse_encoded/unconditional): PASS")

    # ---- 2. run_diffusion_sampling with a tiny UNet, both with and without CFG ----
    tiny_unet = UNet3D(
        in_channels=4, out_channels=4, channels=[8, 16], num_res_blocks=2,
        attention_levels=[False, True], mask_channels=1, cross_attention_dim=4,
        num_head_channels=4, norm_num_groups=4,
    ).to(device).eval()
    schedule = VPredictionSchedule()
    mask_8 = torch.rand(1, 1, 8, 8, 8)

    x_final = run_diffusion_sampling(
        tiny_unet, schedule, mask_8, latent_shape=(1, 4, 8, 8, 8),
        num_inference_steps=20, guidance_scale=1.0, device=device, seed=0, progress_every=0,
    )
    assert x_final.shape == (1, 4, 8, 8, 8)
    assert torch.isfinite(x_final).all(), "sampling produced non-finite values"
    print(f"  run_diffusion_sampling (20 steps, no CFG): PASS, output finite, shape correct")

    # NOTE: can't test guidance_scale's effect using tiny_unet directly -- UNet3D zero-inits
    # its final conv (see unet.py: "Zero-init the final conv so the model initially predicts
    # zero velocity", standard diffusion-training-stability practice), so an UNTRAINED UNet3D
    # predicts exactly zero for every input regardless of the mask, making v_cond == v_uncond
    # == 0 and guidance_scale a no-op -- correctly, not a bug. That's a real, useful fact for
    # anyone sampling from an early-training-stage checkpoint too: classifier-free guidance's
    # effect is proportional to how far the final conv has moved from zero-init, so it will be
    # weak on a lightly-trained model even once the checkpoint is real. To test the CFG
    # ARITHMETIC itself (independent of any particular model's weights), use a hand-built fake
    # model that returns distinguishable outputs for conditioned vs. unconditional input.
    class _FakeCondSensitiveModel:
        def __call__(self, x_t: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            # Deliberately mask-dependent: all-zero mask -> all-zero output (mimics a real
            # model's unconditional behavior), non-zero mask -> a distinguishable non-zero
            # constant offset, so v_cond != v_uncond and guidance_scale's effect is checkable.
            return torch.ones_like(x_t) * 0.1 if mask.abs().sum() > 0 else torch.zeros_like(x_t)

    fake_model = _FakeCondSensitiveModel()
    x_final_cfg = run_diffusion_sampling(
        fake_model, schedule, mask_8, latent_shape=(1, 4, 8, 8, 8),
        num_inference_steps=20, guidance_scale=1.5, device=device, seed=0, progress_every=0,
    )
    x_final_nocfg = run_diffusion_sampling(
        fake_model, schedule, mask_8, latent_shape=(1, 4, 8, 8, 8),
        num_inference_steps=20, guidance_scale=1.0, device=device, seed=0, progress_every=0,
    )
    assert x_final_cfg.shape == (1, 4, 8, 8, 8)
    assert torch.isfinite(x_final_cfg).all(), "CFG sampling produced non-finite values"
    assert not torch.allclose(x_final_nocfg, x_final_cfg), \
        "guidance_scale had no effect on output even with a mask-sensitive model -- real bug"
    print(f"  run_diffusion_sampling guidance_scale (1.0 vs 1.5, mask-sensitive fake model): "
          f"PASS, outputs differ as expected")

    # ---- 3. decode_full_volume with a tiny VAE, single-patch and multi-patch (blended) cases ----
    # 3 channel levels -> 4x downsample (256->64 in the real config), matching
    # MASK_DOWNSAMPLE_FACTOR=4 which this module's decode math assumes throughout. A 2-level
    # VAE would only be a 2x downsample and silently produce the wrong output size here.
    tiny_vae = VAE(
        in_channels=1, out_channels=1, latent_channels=4, channels=[8, 16, 32], num_res_blocks=1,
        norm_num_groups=4,
    ).to(device).eval()

    # Single-patch case: ct_shape exactly equals CT_PATCH_SIZE, so the sliding-window grid is
    # just one window -- no blending involved, isolates the encode/decode plumbing itself.
    latent_single = np.random.randn(4, LATENT_PATCH_SIZE, LATENT_PATCH_SIZE, LATENT_PATCH_SIZE).astype(np.float32)
    ct_single = decode_full_volume(tiny_vae, latent_single, device, ct_shape=(CT_PATCH_SIZE,) * 3)
    assert ct_single.shape == (CT_PATCH_SIZE,) * 3, f"got {ct_single.shape}"
    assert np.isfinite(ct_single).all()
    print(f"  decode_full_volume (single window, no blending): PASS, shape {ct_single.shape}")

    # Multi-patch case: one axis long enough to need 2 overlapping windows -> exercises the
    # actual tapered-weight blending path in sliding_window_average.
    multi_ct_shape = (CT_PATCH_SIZE + CT_STRIDE, CT_PATCH_SIZE, CT_PATCH_SIZE)
    multi_latent_shape = tuple(s // MASK_DOWNSAMPLE_FACTOR for s in multi_ct_shape)
    latent_multi = np.random.randn(4, *multi_latent_shape).astype(np.float32)
    ct_multi = decode_full_volume(tiny_vae, latent_multi, device, ct_shape=multi_ct_shape)
    assert ct_multi.shape == multi_ct_shape, f"got {ct_multi.shape}"
    assert np.isfinite(ct_multi).all()
    print(f"  decode_full_volume (2 overlapping windows, blended): PASS, shape {ct_multi.shape}")

    # ---- 4. denormalize_hu is the exact inverse of clip_and_normalize_hu (in-range values) ----
    hu_original = np.array([-1000.0, -500.0, 0.0, 400.0], dtype=np.float32)
    normalized = clip_and_normalize_hu(hu_original, hu_window=DEFAULT_HU_WINDOW, out_range=DEFAULT_OUT_RANGE)
    recovered = denormalize_hu(normalized, hu_window=DEFAULT_HU_WINDOW, out_range=DEFAULT_OUT_RANGE)
    err = np.abs(hu_original - recovered).max()
    assert err < 1e-2, f"denormalize_hu round-trip error too large: {err}"
    print(f"  denormalize_hu round-trips clip_and_normalize_hu (max err {err:.2e} HU): PASS")

    print("\nAll src.inference.sample tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    _self_test()
