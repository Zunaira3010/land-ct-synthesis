"""
scripts/generate_samples.py
=============================
CLI entry point for generation: noise -> reverse diffusion -> VAE decode -> CT volume.
Thin argument-parsing wrapper around src.inference.sample -- library logic lives there (see
that module's docstring), this script just wires it to argparse/config/checkpoint loading,
matching the repo's existing pattern (e.g. scripts/precompute_latents.py vs.
src's actual encode/decode functions, scripts/check_vae_reconstruction.py's own structure).

Usage
-----
    # Condition on a real val patient's cached mask (recommended first thing to try -- reuses
    # a real, paper-faithful conditioning signal rather than synthesizing one):
    python -m scripts.generate_samples \\
        --unet-checkpoint checkpoints/diffusion/latest.pt \\
        --vae-checkpoint checkpoints/vae/vae_epoch0084_best.pt \\
        --mask-mode reuse_encoded --reference-patient-id <patient_id> \\
        --out-dir checkpoints/diffusion/samples

    # Unconditional (only meaningful if the checkpoint was trained with conditioning dropout):
    python -m scripts.generate_samples \\
        --unet-checkpoint checkpoints/diffusion/latest.pt \\
        --vae-checkpoint checkpoints/vae/vae_epoch0084_best.pt \\
        --mask-mode unconditional --out-dir checkpoints/diffusion/samples_unconditional

    # Faster/cheaper look with fewer inference steps (explicitly NOT paper-faithful --
    # see src.inference.sample.run_diffusion_sampling's docstring):
    python -m scripts.generate_samples ... --num-inference-steps 100

Output: <out-dir>/sample_<seed>.npz (arrays "ct_hu" float32 (256,256,256), "mask_64" float32
(64,64,64)) plus <out-dir>/sample_<seed>_slices.png (axial/coronal/sagittal center-slice
preview) for a fast visual look without needing a separate viewer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Must be set before matplotlib is imported for plotting -- headless, no display on the lab PC,
# same workaround check_vae_reconstruction.py already applies.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.lidc_dataset import load_cached_npz, LIDCSample
from src.data.masks import LUNG_VALUE
from src.inference.sample import (
    build_conditioning_mask,
    decode_full_volume,
    denormalize_hu,
    load_unet_from_checkpoint,
    load_vae_from_checkpoint,
    run_diffusion_sampling,
)
from src.training.diffusion_schedule import VPredictionSchedule
from src.utils.config import get_device, load_config


def _load_reference_mask(manifest_path: str, cache_dir: str, patient_id: str) -> np.ndarray:
    """Pull the already-encoded 256^3 mask straight out of a cached patient's .npz, per
    lidc_dataset.py's load_cached_npz docstring (reads "ct"/"mask" arrays written by the
    preprocessing/caching script) -- no re-encoding needed, this is the real training-time
    conditioning signal for that patient."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    entry = None
    for split in ("train", "val", "test"):
        for row in manifest.get(split, []):
            if row["patient_id"] == patient_id:
                entry = row
                break
        if entry:
            break
    if entry is None:
        raise ValueError(f"patient_id {patient_id!r} not found in {manifest_path}")

    sample = LIDCSample(
        patient_id=entry["patient_id"],
        scan_dicom_dir=entry["scan_dicom_dir"],
        is_healthy=entry["is_healthy"],
        nodule_texture_score=entry.get("nodule_texture_score"),
    )
    _, mask = load_cached_npz(sample, cache_dir)
    return mask


def _save_slice_preview(ct_hu: np.ndarray, mask_64: np.ndarray, out_path: Path, title: str) -> None:
    """Center axial/coronal/sagittal slices of the generated CT, plus the conditioning mask at
    its native 64^3 resolution -- fast visual sanity check, same spirit as
    check_vae_reconstruction.py's comparison figure but for a generated (no ground truth to
    diff against) rather than reconstructed volume."""
    d, h, w = ct_hu.shape
    axes_specs = [
        ("axial",    ct_hu[d // 2, :, :],    mask_64[mask_64.shape[0] // 2, :, :]),
        ("coronal",  ct_hu[:, h // 2, :],    mask_64[:, mask_64.shape[1] // 2, :]),
        ("sagittal", ct_hu[:, :, w // 2],    mask_64[:, :, mask_64.shape[2] // 2]),
    ]
    fig, axarr = plt.subplots(2, 3, figsize=(13, 9))
    for col, (axis_name, ct_slice, mask_slice) in enumerate(axes_specs):
        # Lung window for display: [-1000, 400] HU covers the same range training normalized
        # against, so this is a like-for-like visual comparison with real CT lung windowing.
        axarr[0, col].imshow(ct_slice, cmap="gray", vmin=-1000, vmax=400)
        axarr[0, col].set_title(f"Generated CT ({axis_name})")
        axarr[1, col].imshow(mask_slice, cmap="viridis", vmin=0, vmax=1)
        axarr[1, col].set_title(f"Conditioning mask ({axis_name}, 64^3)")
        for ax in (axarr[0, col], axarr[1, col]):
            ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unet-checkpoint", type=str, required=True,
                   help="e.g. checkpoints/diffusion/latest.pt")
    p.add_argument("--vae-checkpoint", type=str, required=True,
                   help="e.g. checkpoints/vae/vae_epoch0084_best.pt")
    p.add_argument("--diffusion-config", type=str, default="configs/diffusion_config.yaml")
    p.add_argument("--vae-config", type=str, default="configs/vae_config.yaml")
    p.add_argument("--out-dir", type=str, default="checkpoints/diffusion/samples")
    p.add_argument("--mask-mode", type=str, default="reuse_encoded",
                   choices=["reuse_encoded", "healthy_from_lung_mask", "unconditional"])
    p.add_argument("--reference-patient-id", type=str, default=None,
                   help="Required for --mask-mode reuse_encoded/healthy_from_lung_mask")
    p.add_argument("--manifest", type=str, default="data/processed/land/manifest.json")
    p.add_argument("--cache-dir", type=str, default="data/processed/land/cache")
    p.add_argument("--num-inference-steps", type=int, default=None,
                   help="Default: full schedule length (1000, paper-faithful). Fewer steps is "
                        "an explicit, non-paper-faithful speed/quality trade -- see "
                        "src.inference.sample.run_diffusion_sampling's docstring.")
    p.add_argument("--guidance-scale", type=float, default=1.0,
                   help="1.0 = no classifier-free guidance (default, single forward/step). "
                        ">1.0 doubles compute per step; only meaningful if the checkpoint was "
                        "trained with conditioning dropout.")
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="use_ema", action="store_false",
                   help="Use raw (non-EMA) weights -- e.g. for an early smoke-test checkpoint "
                        "that predates EMA being wired in.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--checkpoint-every-n-steps", type=int, default=25,
                   help="Save resumable sampling state to <out-dir>/.resume_seed<seed>.pt "
                        "every N steps, so a crash or Ctrl+C loses at most this many steps of "
                        "progress -- same spirit as train_diffusion.py's own "
                        "checkpoint_every_n_steps. Set to 0 to disable.")
    args = p.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    cfg = load_config(diffusion=args.diffusion_config, vae=args.vae_config)

    unet = load_unet_from_checkpoint(args.unet_checkpoint, cfg.diffusion, device, use_ema=args.use_ema)
    vae = load_vae_from_checkpoint(args.vae_checkpoint, cfg.vae, device)
    schedule = VPredictionSchedule(
        num_train_timesteps=cfg.diffusion.diffusion_process.num_train_timesteps,
        beta_start=cfg.diffusion.diffusion_process.beta_start,
        beta_end=cfg.diffusion.diffusion_process.beta_end,
        min_snr_gamma=cfg.diffusion.diffusion_process.min_snr_gamma,
        device=device,
    )

    # ---- Build the conditioning mask ----
    if args.mask_mode in ("reuse_encoded", "healthy_from_lung_mask"):
        if args.reference_patient_id is None:
            raise SystemExit(f"--mask-mode {args.mask_mode} requires --reference-patient-id")
        encoded_mask_256 = _load_reference_mask(args.manifest, args.cache_dir, args.reference_patient_id)
        if args.mask_mode == "reuse_encoded":
            mask = build_conditioning_mask("reuse_encoded", device, encoded_mask_256=encoded_mask_256)
        else:
            lung_mask_256 = (encoded_mask_256 > 0)  # any nonzero voxel (lung or nodule) is lung tissue
            mask = build_conditioning_mask("healthy_from_lung_mask", device, lung_mask_256=lung_mask_256)
    else:
        mask = build_conditioning_mask("unconditional", device)
    print(f"Conditioning mask: mode={args.mask_mode}"
          + (f", reference_patient={args.reference_patient_id}" if args.reference_patient_id else ""))
    expected_mask_shape = tuple(cfg.diffusion.conditioning.mask_spatial_shape)
    actual_mask_shape = tuple(mask.shape[2:])
    if actual_mask_shape != expected_mask_shape:
        raise SystemExit(
            f"Mask spatial shape mismatch: build_conditioning_mask (using a fixed "
            f"MASK_DOWNSAMPLE_FACTOR={4}, i.e. assuming a 256^3 cached volume) produced "
            f"{actual_mask_shape}, but {args.diffusion_config}'s conditioning.mask_spatial_shape "
            f"says {expected_mask_shape}. This means the config and the actual downsample factor "
            f"have drifted apart -- fix one or the other before sampling, since a silent mismatch "
            f"here would otherwise surface as a confusing shape error inside the U-Net forward pass."
        )

    # ---- Reverse diffusion (full 64^3 latent, see src.inference.sample module docstring) ----
    latent_shape = (
        1,
        cfg.diffusion.architecture.in_channels,
        *cfg.diffusion.conditioning.mask_spatial_shape,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Named by seed (not by e.g. mask_mode/patient) so a resumed run is only ever resumed under
    # the exact same run identity -- run_diffusion_sampling's own mismatch check backs this up
    # by refusing to resume if seed/shape/steps/guidance don't match, but keeping the filename
    # itself seed-scoped means two different seeds run back-to-back never even look at each
    # other's resume file in the first place.
    resume_path = out_dir / f".resume_seed{args.seed}.pt"
    if resume_path.exists():
        print(f"Found existing resume state at {resume_path} -- will continue from there "
              f"(will error out if --seed/--num-inference-steps/--guidance-scale don't match "
              f"what produced it; delete the file to force a fresh start instead).")

    print(f"Sampling: latent_shape={latent_shape}, "
          f"num_inference_steps={args.num_inference_steps or schedule.num_train_timesteps}, "
          f"guidance_scale={args.guidance_scale}, seed={args.seed}")
    print(f"If this run is interrupted (crash, Ctrl+C, closed terminal), just re-run the exact "
          f"same command -- it will pick up from {resume_path.name} instead of starting over "
          f"(loses at most --checkpoint-every-n-steps={args.checkpoint_every_n_steps} steps of progress).")
    x0_latent = run_diffusion_sampling(
        unet, schedule, mask, latent_shape=latent_shape,
        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
        device=device, seed=args.seed,
        resume_path=resume_path, checkpoint_every_n_steps=args.checkpoint_every_n_steps,
    )

    # ---- Decode: sliding-window VAE decode, then denormalize back to HU ----
    print("Decoding latent to full-resolution CT via sliding-window VAE decode...")
    latent_np = x0_latent.squeeze(0).cpu().numpy().astype(np.float32)
    ct_normalized = decode_full_volume(vae, latent_np, device)
    ct_hu = denormalize_hu(ct_normalized)

    # ---- Save ----
    mask_64_np = mask.squeeze(0).squeeze(0).cpu().numpy()

    npz_path = out_dir / f"sample_seed{args.seed}.npz"
    np.savez_compressed(npz_path, ct_hu=ct_hu, mask_64=mask_64_np)
    print(f"Saved: {npz_path}")

    png_path = out_dir / f"sample_seed{args.seed}_slices.png"
    title = (f"Generated sample (seed={args.seed}, mask_mode={args.mask_mode}, "
             f"guidance_scale={args.guidance_scale})")
    _save_slice_preview(ct_hu, mask_64_np, png_path, title)
    print(f"Saved: {png_path}")

    print(
        "\nDone. This is a single unconditional-quality sanity check, not a validated result -- "
        "open the PNG and look for rough lung-shaped structure and, if using reuse_encoded, "
        "whether density is roughly concentrated where the mask says lung/nodule tissue should "
        "be. A checkpoint trained for only a few thousand steps (vs. the paper's 500k) should "
        "be expected to look noisy/blob-like rather than anatomically clean -- see "
        "docs/04_training.md's step-count deviation note for why milestone-based evaluation "
        "(this script) rather than a fixed target step count is the plan."
    )


if __name__ == "__main__":
    main()
