"""
scripts/check_vae_reconstruction.py
====================================
Stage 5: VAE reconstruction quality check.

Loads a trained VAE checkpoint, runs a handful of real val-set patches through
encode -> decode, and saves side-by-side (real | reconstructed | abs. difference)
axial/coronal/sagittal slice comparisons as PNGs -- plus per-patch quantitative
MAE/MSE/PSNR so you have numbers alongside the pictures.

This does NOT touch unet.py, diffusion training, or anything downstream. Its only
job is to answer: "did the VAE actually learn something useful, or did it just
memorize a low val loss number?"

Usage (run from the repo root, e.g. C:\\Users\\24COBEA\\Desktop\\land_ct_synthesis\\land-ct-synthesis):

    python scripts/check_vae_reconstruction.py \
        --checkpoint checkpoints/vae/vae_epoch0084_best.pt \
        --manifest data/processed/land/manifest.json \
        --cache-dir data/processed/land/cache \
        --out-dir checkpoints/vae/recon_check \
        --num-patches 6

Notes:
  - Uses the SAME val_seed=42 the real training run used for its val split
    (see train_vae.py's `_build_dataset(..., val_seed=42, ...)` call), so the
    patches you inspect here are the exact same ones the checkpoint's reported
    val loss was computed against -- no discrepancy between the number and the
    picture.
  - Uses --patch-size 96 by default to match the run that actually produced
    vae_epoch0084_best.pt. If you trained at a different patch size, pass that
    --patch-size instead.
  - Deliberately picks a mix of healthy and tumor-bearing patients (if both are
    present in val) so you can eyeball nodule-region reconstruction quality
    specifically, not just generic lung tissue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless -- no display needed on the lab PC
import matplotlib.pyplot as plt

from src.utils.config import load_config, get_device
from src.models.vae import VAE
from src.data.lidc_dataset import LIDCVolumeDataset, LIDCSample
from src.data.masks import LUNG_VALUE


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(real: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    """real/recon are float32 arrays in the model's native [-1, 1] range."""
    diff = real - recon
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    # data range for this normalization is 2.0 (from -1 to 1)
    psnr = float("inf") if mse == 0 else 10.0 * np.log10((2.0 ** 2) / mse)
    return {"mae": mae, "mse": mse, "psnr_db": psnr}


# ---------------------------------------------------------------------------
# Slice picking
# ---------------------------------------------------------------------------

def _mid_slices(volume: np.ndarray, nodule_mask: np.ndarray | None) -> dict[str, int]:
    """
    Pick one slice index per axis (axial/coronal/sagittal). If a nodule mask with
    any positive voxels is available, center on the nodule's centroid instead of
    the geometric middle -- otherwise a small nodule can easily be missed by a
    generic mid-volume slice.
    """
    d, h, w = volume.shape
    if nodule_mask is not None and nodule_mask.any():
        zz, yy, xx = np.nonzero(nodule_mask)
        return {"axial": int(zz.mean()), "coronal": int(yy.mean()), "sagittal": int(xx.mean())}
    return {"axial": d // 2, "coronal": h // 2, "sagittal": w // 2}


def _save_comparison_figure(
    real: np.ndarray,
    recon: np.ndarray,
    nodule_mask: np.ndarray | None,
    patient_id: str,
    is_healthy: bool,
    metrics: dict[str, float],
    out_path: Path,
) -> None:
    slices = _mid_slices(real, nodule_mask)
    axes_specs = [
        ("axial",    lambda v, i: v[i, :, :]),
        ("coronal",  lambda v, i: v[:, i, :]),
        ("sagittal", lambda v, i: v[:, :, i]),
    ]

    fig, axarr = plt.subplots(3, 3, figsize=(13, 12))
    for row, (axis_name, slicer) in enumerate(axes_specs):
        i = slices[axis_name]
        r_slice = slicer(real, i)
        c_slice = slicer(recon, i)
        d_slice = np.abs(r_slice - c_slice)

        axarr[row, 0].imshow(r_slice, cmap="gray", vmin=-1, vmax=1)
        axarr[row, 0].set_title(f"Real ({axis_name}, idx={i})")
        axarr[row, 1].imshow(c_slice, cmap="gray", vmin=-1, vmax=1)
        axarr[row, 1].set_title(f"Reconstructed ({axis_name})")
        im = axarr[row, 2].imshow(d_slice, cmap="inferno", vmin=0, vmax=0.5)
        axarr[row, 2].set_title(f"|diff| ({axis_name})")
        for ax in axarr[row]:
            ax.axis("off")

    fig.colorbar(im, ax=axarr[:, 2], shrink=0.6, label="|real - recon|  (data range [-1,1])")

    kind = "healthy" if is_healthy else "tumor-bearing"
    fig.suptitle(
        f"Patient {patient_id}  ({kind})   "
        f"MAE={metrics['mae']:.4f}  MSE={metrics['mse']:.4f}  PSNR={metrics['psnr_db']:.1f} dB",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Stage 5: VAE reconstruction quality check")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="e.g. checkpoints/vae/vae_epoch0084_best.pt -- use the BEST checkpoint, "
                        "not necessarily the final one")
    p.add_argument("--manifest", type=str, default="data/processed/land/manifest.json")
    p.add_argument("--cache-dir", type=str, default="data/processed/land/cache")
    p.add_argument("--vae-config", type=str, default="configs/vae_config.yaml")
    p.add_argument("--data-config", type=str, default="configs/data_config.yaml")
    p.add_argument("--out-dir", type=str, default="checkpoints/vae/recon_check")
    p.add_argument("--patch-size", type=int, default=96,
                   help="Must match the patch size the checkpoint was trained at")
    p.add_argument("--num-patches", type=int, default=6,
                   help="How many val patients to check (val split has 7 total)")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    cfg = load_config(vae=args.vae_config, data=args.data_config)

    # ---- Build VAE from the real config, load the checkpoint's weights ----
    vae = VAE.from_config(cfg.vae).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vae.load_state_dict(ckpt["vae_state"])
    vae.eval()
    epoch = ckpt.get("epoch", "?")
    best_val_loss = ckpt.get("best_val_loss", float("nan"))
    print(f"Loaded checkpoint from epoch {epoch} (best_val_loss={best_val_loss:.4f}): "
          f"{args.checkpoint}")

    # ---- Build the exact same val dataset the training run used ----
    with open(args.manifest) as f:
        splits = json.load(f)
    val_entries = splits["val"]
    print(f"Val split has {len(val_entries)} patients "
          f"({sum(not e['is_healthy'] for e in val_entries)} tumor / "
          f"{sum(e['is_healthy'] for e in val_entries)} healthy)")

    samples = [
        LIDCSample(
            patient_id=e["patient_id"],
            scan_dicom_dir=e["scan_dicom_dir"],
            is_healthy=e["is_healthy"],
            nodule_texture_score=e.get("nodule_texture_score"),
        )
        for e in val_entries
    ]
    val_dataset = LIDCVolumeDataset(
        samples,
        cache_dir=args.cache_dir,
        patch_size=args.patch_size,
        nodule_centered_prob=0.9,  # bias toward nodule-containing crops for THIS check --
                                   # we want to see nodule regions, not just random lung tissue
        val_seed=42,               # same seed train_vae.py used -- reproducible, comparable patches
    )

    n = min(args.num_patches, len(val_dataset))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    print(f"\nRunning {n} val patches through encode -> decode...\n")

    with torch.no_grad():
        for idx in range(n):
            batch = val_dataset[idx]
            ct = torch.from_numpy(batch["ct"]).unsqueeze(0).to(device)      # (1,1,P,P,P)
            mask_np = batch["mask"][0]                                      # downsampled mask, (P/4)^3
            patient_id = batch["patient_id"]
            is_healthy = batch["is_healthy"]

            posterior = vae.encode(ct)
            z = posterior.mode()          # deterministic reconstruction (no sampling noise)
            recon = vae.decode(z)

            real_np = ct[0, 0].cpu().numpy()
            recon_np = recon[0, 0].cpu().numpy()

            m = _metrics(real_np, recon_np)
            all_metrics.append(m)

            # Upsample the downsampled mask back to patch resolution just for slice-picking
            # (nearest-neighbor is fine here -- only used to locate the nodule, not for math)
            factor = real_np.shape[0] // mask_np.shape[0]
            mask_full_res = np.repeat(np.repeat(np.repeat(
                mask_np, factor, axis=0), factor, axis=1), factor, axis=2)
            # Same nodule-voxel test the repo itself uses (lidc_dataset.py::_nodule_voxel_indices):
            # background=0.0, lung-only=LUNG_VALUE (0.5), nodule voxels are neither -- so
            # "not 0.0 and not LUNG_VALUE" reliably isolates nodule voxels with no separate mask.
            nodule_mask = (mask_full_res != 0.0) & (~np.isclose(mask_full_res, LUNG_VALUE))

            fname = out_dir / f"recon_{idx:02d}_{patient_id}.png"
            _save_comparison_figure(real_np, recon_np, nodule_mask, patient_id, is_healthy, m, fname)

            kind = "healthy" if is_healthy else "tumor"
            print(f"  [{idx+1}/{n}] {patient_id} ({kind:6s})  "
                  f"MAE={m['mae']:.4f}  MSE={m['mse']:.4f}  PSNR={m['psnr_db']:.1f} dB  "
                  f"-> {fname.name}")

    # ---- Summary ----
    mean_mae = float(np.mean([m["mae"] for m in all_metrics]))
    mean_psnr = float(np.mean([m["psnr_db"] for m in all_metrics]))
    print(f"\nDone. {n} comparison figures saved to: {out_dir}/")
    print(f"Mean MAE across checked patches: {mean_mae:.4f}")
    print(f"Mean PSNR across checked patches: {mean_psnr:.1f} dB")
    print(
        "\nThese numbers are a sanity check, not a pass/fail threshold -- there's no paper-stated "
        "target to compare against. What matters most is the PNGs: open a few and look at "
        "whether lung structure stays coherent and whether the nodule region (see the |diff| "
        "panel -- it should be small and diffuse, not a bright blob outlining the whole nodule) "
        "looks like blur/smoothing rather than the nodule disappearing or morphing into something "
        "anatomically wrong."
    )


if __name__ == "__main__":
    main()
