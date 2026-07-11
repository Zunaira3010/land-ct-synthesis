"""
PyTorch Dataset for LAND training samples.

See docs/02_dataset_pipeline.md for the manifest/sample construction rules (healthy vs tumor
samples) and docs/01_architecture.md for why the mask is downsampled 4x (256^3 CT -> 64^3 mask,
matching the VAE's latent spatial resolution).

The actual disk I/O (loading cached .npz preprocessed volumes) is injected via `volume_loader`
so this module stays testable without any real data or cache directory -- see
tests/test_stage2_preprocessing.py for the synthetic-loader pattern. The default loader
(`load_cached_npz`) is what training actually uses.

PATCH SAMPLING (added after the 256^3 full-volume encoder OOM -- see NOTES.md): the first
ResBlock conv at full 256^3 resolution needs 54GB of workspace, which no 8GB dev card can hold,
even with gradient checkpointing (checkpointing only helps the backward pass; this OOM happens
on the very first forward conv). So training runs on smaller 3D crops instead of full volumes.
Crops are optionally biased toward nodule-containing regions (`nodule_centered_prob`) rather than
pure random crops, because nodules are a small fraction of total lung volume -- plain random
crops at 96-128^3 would very often contain zero nodule voxels, which is a real risk for a model
whose whole point is eventually generating convincing synthetic nodules. Full-volume inference
still needs sliding-window stitching later (not implemented here yet -- see NOTES.md).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from src.data.masks import LUNG_VALUE, downsample_mask_maxpool

MASK_DOWNSAMPLE_FACTOR = 4


@dataclass
class LIDCSample:
    """One manifest entry: a single patient/scan, plus whether it's a healthy or tumor sample.

    scan_dicom_dir: path to the raw DICOM series (used by the caching/preprocessing scripts to
    build the .npz cache the first time; not read directly by the Dataset at train time).
    nodule_texture_score: 1-5, required iff is_healthy is False. Multiple-nodule scans should be
    split into multiple LIDCSample tumor entries (one dominant nodule's texture per sample), or
    aggregated upstream in the manifest-building script -- see docs/02_dataset_pipeline.md.
    """

    patient_id: str
    scan_dicom_dir: str
    is_healthy: bool
    nodule_texture_score: Optional[int] = None

    def __post_init__(self):
        if not self.is_healthy and self.nodule_texture_score is None:
            raise ValueError(
                f"{self.patient_id}: tumor sample (is_healthy=False) requires "
                "nodule_texture_score"
            )
        if self.is_healthy and self.nodule_texture_score is not None:
            raise ValueError(
                f"{self.patient_id}: healthy sample (is_healthy=True) must not carry a "
                "nodule_texture_score"
            )


def load_cached_npz(sample: LIDCSample, cache_dir: str):
    """Default volume_loader: reads the preprocessed CT + encoded mask from the .npz cache.

    Expects `<cache_dir>/<patient_id>.npz` with arrays "ct" (float16, 256^3) and "mask"
    (float32, 256^3, already lung/nodule-value-encoded per masks.py) -- written by the
    preprocessing/caching script described in docs/03_preprocessing.md step 8.
    """
    cache_path = Path(cache_dir) / f"{sample.patient_id}.npz"
    data = np.load(cache_path)
    ct = data["ct"].astype(np.float32)
    mask = data["mask"].astype(np.float32)
    return ct, mask


def _nodule_voxel_indices(mask: np.ndarray) -> np.ndarray:
    """Indices (N, 3) of voxels that are nodule (not background 0.0, not plain lung LUNG_VALUE).

    Mask encoding (masks.py): background=0.0, lung-only=LUNG_VALUE (0.5), nodule voxels overwrite
    the lung value with texture_score/5.0 (never exactly LUNG_VALUE, since texture_score is an
    integer 1-5). So "not 0.0 and not LUNG_VALUE" reliably picks out nodule voxels without
    needing a separate nodule-only mask array.
    """
    nodule_voxels = (mask != 0.0) & (mask != LUNG_VALUE)
    return np.argwhere(nodule_voxels)


def _crop_bounds_1d(size: int, patch: int, center: Optional[int], rng: random.Random) -> int:
    """Start index for a 1D crop of length `patch` out of an axis of length `size`.

    If `center` is given, the crop is centered on it but clipped so the whole patch stays
    in-bounds (so a nodule near the volume edge still gets a full-size patch, just off-center).
    If `center` is None, the start index is uniform-random over all valid positions.
    """
    max_start = size - patch
    if max_start <= 0:
        return 0
    if center is None:
        return rng.randint(0, max_start)
    start = center - patch // 2
    return max(0, min(start, max_start))


def sample_patch_origin(
    volume_shape: tuple[int, int, int],
    patch_size: int,
    mask: Optional[np.ndarray] = None,
    nodule_centered_prob: float = 0.0,
    rng: Optional[random.Random] = None,
) -> tuple[int, int, int]:
    """Pick a (d0, h0, w0) crop origin for a `patch_size`^3 patch out of `volume_shape`.

    With probability `nodule_centered_prob` (only if `mask` has any nodule voxels), the patch is
    centered on a randomly chosen nodule voxel; otherwise it's a plain uniform-random crop. This
    is a per-sample decision, not "random crops now, nodule-aware later" -- see module docstring.

    rng: an explicit random.Random instance for deterministic sampling (e.g. a per-sample seeded
    instance for a validation set, so val loss is comparable epoch-to-epoch instead of jumping
    around with whatever patch happened to get drawn). Falls back to the global `random` module
    if not given, which is what training should use (nondeterministic -- acts as augmentation).
    """
    r = rng if rng is not None else random
    center = None
    if mask is not None and nodule_centered_prob > 0 and r.random() < nodule_centered_prob:
        candidates = _nodule_voxel_indices(mask)
        if len(candidates) > 0:
            center = candidates[r.randrange(len(candidates))]

    origin = tuple(
        _crop_bounds_1d(volume_shape[axis], patch_size, None if center is None else int(center[axis]), r)
        for axis in range(3)
    )
    return origin  # type: ignore[return-value]


class LIDCVolumeDataset(Dataset):
    """Returns dicts of {"ct", "mask", "is_healthy", "texture_score"} for VAE/diffusion training.

    patch_size: if set, each __getitem__ crops a random patch_size^3 sub-volume out of the full
    256^3 scan instead of returning the whole thing (see module docstring for why). Must divide
    evenly by mask_downsample_factor. If None (default), full volumes are returned -- the
    original behavior, which will OOM on an 8GB card past the first encoder conv.
    nodule_centered_prob: for tumor-bearing samples only (healthy samples have no nodule voxels
    to center on), the probability a given patch is centered on a real nodule voxel rather than
    placed uniformly at random. 0.0 disables nodule-centering entirely (pure random crops).
    val_seed: if set, each sample gets a deterministic patch location (same crop every epoch)
    instead of a fresh random one per __getitem__ call. Intended for validation splits -- without
    this, val loss is noisy from epoch to epoch purely because a different random patch got drawn,
    which makes it hard to tell "loss went up because the model got worse" from "loss went up
    because this epoch's val patches happened to be harder." Leave unset for the training split,
    where patch randomness is free augmentation and determinism would remove that benefit.
    """

    def __init__(
        self,
        manifest: list[LIDCSample],
        cache_dir: str,
        volume_loader: Optional[Callable[[LIDCSample], tuple]] = None,
        mask_downsample_factor: int = MASK_DOWNSAMPLE_FACTOR,
        patch_size: Optional[int] = None,
        nodule_centered_prob: float = 0.0,
        val_seed: Optional[int] = None,
    ):
        if patch_size is not None and patch_size % mask_downsample_factor != 0:
            raise ValueError(
                f"patch_size ({patch_size}) must be divisible by mask_downsample_factor "
                f"({mask_downsample_factor}) so the downsampled mask patch has an integer shape"
            )
        if not (0.0 <= nodule_centered_prob <= 1.0):
            raise ValueError(f"nodule_centered_prob must be in [0, 1], got {nodule_centered_prob}")

        self.manifest = manifest
        self.cache_dir = cache_dir
        self.mask_downsample_factor = mask_downsample_factor
        self.patch_size = patch_size
        self.nodule_centered_prob = nodule_centered_prob
        self.val_seed = val_seed
        # Default loader is bound to cache_dir; tests inject their own single-arg loader.
        # Uses functools.partial (not a lambda/closure) so the dataset stays picklable --
        # required on Windows, where DataLoader(num_workers>0) uses spawn and must pickle
        # the whole dataset object to hand it to worker processes.
        self._volume_loader = volume_loader or partial(
            load_cached_npz, cache_dir=self.cache_dir
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict:
        sample = self.manifest[index]
        ct, mask = self._volume_loader(sample)

        if self.patch_size is not None:
            # Healthy samples have no nodule voxels to center on -- always random crop for them.
            nodule_prob = 0.0 if sample.is_healthy else self.nodule_centered_prob
            # A fresh local Random per call (not module-global) when val_seed is set, so seeding
            # the val split's patches doesn't perturb the global `random` state anything else in
            # the training loop might depend on (e.g. any other random.* calls elsewhere).
            rng = random.Random(self.val_seed + index) if self.val_seed is not None else None
            d0, h0, w0 = sample_patch_origin(
                ct.shape, self.patch_size, mask=mask, nodule_centered_prob=nodule_prob, rng=rng
            )
            p = self.patch_size
            ct   = ct[d0:d0 + p, h0:h0 + p, w0:w0 + p]
            mask = mask[d0:d0 + p, h0:h0 + p, w0:w0 + p]

        mask_down = downsample_mask_maxpool(mask, factor=self.mask_downsample_factor)

        return {
            "ct": ct[np.newaxis, ...].astype(np.float32),
            "mask": mask_down[np.newaxis, ...].astype(np.float32),
            "is_healthy": sample.is_healthy,
            "texture_score": sample.nodule_texture_score or 0,
            "patient_id": sample.patient_id,
        }
