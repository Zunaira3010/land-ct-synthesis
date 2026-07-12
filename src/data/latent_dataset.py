"""
src/data/latent_dataset.py
============================
PyTorch Dataset for diffusion training: reads the precomputed full 64^3x4 latents (see
scripts/precompute_latents.py) and crops smaller sub-patches out of them for training -- the
same idea as LIDCVolumeDataset (src/data/lidc_dataset.py) cropping CT-space patches for VAE
training, just operating one compression level up, in latent space instead of voxel space.

Why crop again, if the cached latents are already only 64^3? Because the diffusion U-Net's
channel progression (64->128->256->384->512 across 5 levels, with self/cross-attention at the
3 coarsest) doesn't fit an 8GB card at the full 64^3 latent even at batch size 1 with
gradient checkpointing -- the paper's own more-optimized config reports needing 10-16GB just
for that. So diffusion training also runs on patches, just smaller ones (32^3 latent by
default, i.e. corresponding to a 128^3 region of the original scan) -- and critically, starting
from an already-cached 64^3 latent load rather than needing to encode a 256^3 CT volume on
the fly, which is the whole reason precompute_latents.py exists as a separate one-time step.

One simplification versus the CT-space case: the mask here is already at 64^3 resolution
(precompute_latents.py downsamples it once, at precompute time), so a latent patch and its
mask patch are a direct, aligned 1:1 crop -- no separate mask_downsample_factor bookkeeping
inside this Dataset (contrast with LIDCVolumeDataset.__getitem__, which crops the mask at
full CT resolution and downsamples afterward, every __getitem__ call).
"""
from __future__ import annotations

import random
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from src.data.lidc_dataset import LIDCSample, sample_patch_origin

FULL_LATENT_SIZE = 64   # matches configs/diffusion_config.yaml conditioning.mask_spatial_shape


def load_cached_latent(sample: LIDCSample, latent_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Default volume_loader: reads a precomputed <patient_id>.npz from scripts/precompute_latents.py.

    Expects arrays "latent" (float32, C x 64 x 64 x 64) and "mask" (float32, 64 x 64 x 64,
    already lung/nodule-value-encoded AND already downsampled -- see that script's docstring).
    """
    cache_path = Path(latent_dir) / f"{sample.patient_id}.npz"
    data = np.load(cache_path)
    return data["latent"].astype(np.float32), data["mask"].astype(np.float32)


class LatentVolumeDataset(Dataset):
    """Returns dicts of {"latent", "mask", "is_healthy", "texture_score", "patient_id"} for
    diffusion training.

    patch_size: side length of the cropped cube, in LATENT voxels (default 32 -- half the
    full 64^3 latent). Unlike LIDCVolumeDataset there's no divisibility-against-a-downsample-
    factor constraint, since the mask here is already at latent resolution; any
    1 <= patch_size <= 64 works, though very small values give the U-Net little spatial
    context per training example. Set to None to return full, uncropped 64^3 latents.
    nodule_centered_prob / val_seed: same semantics as LIDCVolumeDataset (see that class's
    docstring in lidc_dataset.py) -- reuses `sample_patch_origin` completely unmodified,
    since that function only needs a volume shape plus an optional mask to bias against, and
    doesn't care what physical resolution either one is defined at.
    """

    def __init__(
        self,
        manifest: list[LIDCSample],
        latent_dir: str,
        volume_loader: Optional[Callable[[LIDCSample], tuple]] = None,
        patch_size: Optional[int] = 32,
        nodule_centered_prob: float = 0.0,
        val_seed: Optional[int] = None,
    ):
        if patch_size is not None and not (1 <= patch_size <= FULL_LATENT_SIZE):
            raise ValueError(
                f"patch_size ({patch_size}) must be in [1, {FULL_LATENT_SIZE}]"
            )
        if not (0.0 <= nodule_centered_prob <= 1.0):
            raise ValueError(f"nodule_centered_prob must be in [0, 1], got {nodule_centered_prob}")

        self.manifest = manifest
        self.latent_dir = latent_dir
        self.patch_size = patch_size
        self.nodule_centered_prob = nodule_centered_prob
        self.val_seed = val_seed
        # functools.partial, not a lambda/closure -- same picklability reasoning as
        # LIDCVolumeDataset: Windows DataLoader(num_workers>0) uses spawn and must pickle the
        # whole dataset (including this loader) to hand it to worker processes.
        self._volume_loader = volume_loader or partial(load_cached_latent, latent_dir=self.latent_dir)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict:
        sample = self.manifest[index]
        latent, mask = self._volume_loader(sample)

        if self.patch_size is not None:
            # Healthy samples have no nodule voxels to center on -- always random crop, same
            # rule as LIDCVolumeDataset.
            nodule_prob = 0.0 if sample.is_healthy else self.nodule_centered_prob
            rng = random.Random(self.val_seed + index) if self.val_seed is not None else None
            # Crop origin computed against the mask's spatial shape (64,64,64); latent has an
            # extra leading channel dim so its crop uses the same origin, applied one axis over.
            d0, h0, w0 = sample_patch_origin(
                mask.shape, self.patch_size, mask=mask, nodule_centered_prob=nodule_prob, rng=rng
            )
            p = self.patch_size
            latent = latent[:, d0:d0 + p, h0:h0 + p, w0:w0 + p]
            mask = mask[d0:d0 + p, h0:h0 + p, w0:w0 + p]

        return {
            "latent": latent.astype(np.float32),
            "mask": mask[np.newaxis, ...].astype(np.float32),
            "is_healthy": sample.is_healthy,
            "texture_score": sample.nodule_texture_score or 0,
            "patient_id": sample.patient_id,
        }


# ---------------------------------------------------------------------------
# Self-test (run with: python -m src.data.latent_dataset)
# Uses a synthetic in-memory volume_loader -- no real cached latents or torch/VAE needed,
# same pattern lidc_dataset.py's own tests use (see tests/test_stage2_preprocessing.py).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Running latent_dataset self-tests...")

    rng_np = np.random.RandomState(0)
    LATENT_CHANNELS = 4

    fake_latents: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def make_fake_patient(patient_id: str, is_healthy: bool) -> LIDCSample:
        latent = rng_np.randn(LATENT_CHANNELS, 64, 64, 64).astype(np.float32)
        mask = np.zeros((64, 64, 64), dtype=np.float32)
        mask[20:40, 20:40, 20:40] = 0.5  # "lung"
        if not is_healthy:
            mask[28:32, 28:32, 28:32] = 0.8  # "nodule" (texture 4/5)
        fake_latents[patient_id] = (latent, mask)
        return LIDCSample(
            patient_id=patient_id,
            scan_dicom_dir="unused",
            is_healthy=is_healthy,
            nodule_texture_score=None if is_healthy else 4,
        )

    def fake_loader(sample: LIDCSample) -> tuple[np.ndarray, np.ndarray]:
        return fake_latents[sample.patient_id]

    manifest = [
        make_fake_patient("P_HEALTHY_0", is_healthy=True),
        make_fake_patient("P_TUMOR_0", is_healthy=False),
    ]

    # Test 1: uncropped (patch_size=None) returns full 64^3 latent/mask untouched.
    ds_full = LatentVolumeDataset(manifest, latent_dir="unused", volume_loader=fake_loader, patch_size=None)
    item = ds_full[0]
    assert item["latent"].shape == (LATENT_CHANNELS, 64, 64, 64), item["latent"].shape
    assert item["mask"].shape == (1, 64, 64, 64), item["mask"].shape
    print("  patch_size=None (full volume): PASS")

    # Test 2: cropped patch shapes are correct, and latent/mask crops are spatially aligned
    # (same origin applied to both).
    ds_patch = LatentVolumeDataset(
        manifest, latent_dir="unused", volume_loader=fake_loader, patch_size=32, nodule_centered_prob=1.0
    )
    item = ds_patch[1]  # tumor sample, nodule_centered_prob=1.0 -> should always center on the nodule
    assert item["latent"].shape == (LATENT_CHANNELS, 32, 32, 32), item["latent"].shape
    assert item["mask"].shape == (1, 32, 32, 32), item["mask"].shape
    # With nodule_centered_prob=1.0 and the nodule block at [28:32]^3 well inside the volume,
    # a 32-wide crop centered on a nodule voxel must fully contain the 4-voxel nodule block.
    assert (item["mask"] > 0.5).any(), "expected the centered crop to contain nodule voxels"
    print("  patch_size=32, nodule_centered_prob=1.0 (tumor sample): PASS (nodule voxels present in crop)")

    # Test 3: healthy sample never gets nodule-centered (no nodule voxels exist to center on
    # anyway) -- just check it doesn't error and shapes are right.
    item_healthy = ds_patch[0]
    assert item_healthy["latent"].shape == (LATENT_CHANNELS, 32, 32, 32)
    print("  patch_size=32 on healthy sample (no nodule to center on): PASS")

    # Test 4: val_seed determinism -- same index, same seed, must give the identical crop
    # location every call (same reasoning as LIDCVolumeDataset's val_seed).
    ds_val = LatentVolumeDataset(
        manifest, latent_dir="unused", volume_loader=fake_loader, patch_size=16,
        nodule_centered_prob=0.5, val_seed=123,
    )
    item_a = ds_val[1]
    item_b = ds_val[1]
    assert np.array_equal(item_a["latent"], item_b["latent"]), "val_seed should make repeated __getitem__ calls deterministic"
    print("  val_seed determinism (same index -> same crop across calls): PASS")

    # Test 5: without val_seed, training-mode randomness actually varies crop location across
    # many calls (sanity check that it's NOT accidentally deterministic).
    ds_train = LatentVolumeDataset(
        manifest, latent_dir="unused", volume_loader=fake_loader, patch_size=16, nodule_centered_prob=0.0
    )
    seen = {tuple(ds_train[1]["latent"][0, 0, 0, :3]) for _ in range(20)}
    assert len(seen) > 1, "expected random crops to vary across repeated calls without val_seed"
    print("  training-mode randomness (varies without val_seed): PASS")

    print("\nALL LATENT_DATASET TESTS PASSED.")
    sys.exit(0)
