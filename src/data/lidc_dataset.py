"""
PyTorch Dataset for LAND training samples.

See docs/02_dataset_pipeline.md for the manifest/sample construction rules (healthy vs tumor
samples) and docs/01_architecture.md for why the mask is downsampled 4x (256^3 CT -> 64^3 mask,
matching the VAE's latent spatial resolution).

The actual disk I/O (loading cached .npz preprocessed volumes) is injected via `volume_loader`
so this module stays testable without any real data or cache directory -- see
tests/test_stage2_preprocessing.py for the synthetic-loader pattern. The default loader
(`load_cached_npz`) is what training actually uses.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from src.data.masks import downsample_mask_maxpool

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


class LIDCVolumeDataset(Dataset):
    """Returns dicts of {"ct", "mask", "is_healthy", "texture_score"} for VAE/diffusion training."""

    def __init__(
        self,
        manifest: list[LIDCSample],
        cache_dir: str,
        volume_loader: Optional[Callable[[LIDCSample], tuple]] = None,
        mask_downsample_factor: int = MASK_DOWNSAMPLE_FACTOR,
    ):
        self.manifest = manifest
        self.cache_dir = cache_dir
        self.mask_downsample_factor = mask_downsample_factor
        # Default loader is bound to cache_dir; tests inject their own single-arg loader.
        self._volume_loader = volume_loader or (
            lambda s: load_cached_npz(s, cache_dir=self.cache_dir)
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict:
        sample = self.manifest[index]
        ct, mask = self._volume_loader(sample)

        mask_down = downsample_mask_maxpool(mask, factor=self.mask_downsample_factor)

        return {
            "ct": ct[np.newaxis, ...].astype(np.float32),
            "mask": mask_down[np.newaxis, ...].astype(np.float32),
            "is_healthy": sample.is_healthy,
            "texture_score": sample.nodule_texture_score or 0,
            "patient_id": sample.patient_id,
        }
