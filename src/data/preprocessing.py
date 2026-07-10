"""
CT / mask preprocessing pipeline for LAND.

Implements docs/03_preprocessing.md steps 2-5:
    resample -> HU clip -> normalize -> crop/pad
Step 1 (DICOM load) lives in io.py. Steps 6-8 (mask generation via lungmask/pylidc, encoding,
caching) live in masks.py / lidc_dataset.py / the caching scripts.

NOTE (see docs/06_open_questions.md #10): the HU window below is our best-documented
placeholder pending verification against the WDM paper's preprocessing script. Don't treat it
as paper-confirmed; it's easy to override via configs/data_config.yaml.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import SimpleITK as sitk

DEFAULT_HU_WINDOW = (-1000.0, 400.0)
DEFAULT_OUT_RANGE = (-1.0, 1.0)
DEFAULT_TARGET_SHAPE = (256, 256, 256)
DEFAULT_TARGET_SPACING_MM = (1.0, 1.0, 1.0)


def resample_to_spacing(
    image: sitk.Image,
    target_spacing_mm: Sequence[float] = DEFAULT_TARGET_SPACING_MM,
    interpolator: int = sitk.sitkLinear,
) -> sitk.Image:
    """Resample a SimpleITK image to the given isotropic (or anisotropic) spacing.

    Paper: "1 mm isotropic resolution" -- use interpolator=sitk.sitkLinear for CT intensities,
    sitk.sitkNearestNeighbor for co-registered label/mask volumes (call sites choose this).
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing_mm)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(s) for s in target_spacing_mm))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)


def clip_and_normalize_hu(
    volume: np.ndarray,
    hu_window: Sequence[float] = DEFAULT_HU_WINDOW,
    out_range: Sequence[float] = DEFAULT_OUT_RANGE,
) -> np.ndarray:
    """Clip HU values to hu_window, then linearly rescale to out_range."""
    lo, hi = float(hu_window[0]), float(hu_window[1])
    out_lo, out_hi = float(out_range[0]), float(out_range[1])

    clipped = np.clip(volume.astype(np.float32), lo, hi)
    normalized = (clipped - lo) / (hi - lo)  # -> [0, 1]
    rescaled = normalized * (out_hi - out_lo) + out_lo  # -> [out_lo, out_hi]
    return rescaled.astype(np.float32)


def crop_or_pad_to_shape(
    volume: np.ndarray,
    target_shape: Sequence[int] = DEFAULT_TARGET_SHAPE,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Center-crop axes that are too large, symmetric zero/pad-value-pad axes too small.

    Handles a mix of crop-needed and pad-needed axes in the same call.
    """
    target_shape = tuple(target_shape)
    volume = np.asarray(volume)

    # Step 1: center-crop any axis larger than target.
    cropped = volume
    for axis, (cur, tgt) in enumerate(zip(cropped.shape, target_shape)):
        if cur > tgt:
            start = (cur - tgt) // 2
            slicer = [slice(None)] * cropped.ndim
            slicer[axis] = slice(start, start + tgt)
            cropped = cropped[tuple(slicer)]

    # Step 2: symmetric pad any axis smaller than target.
    pad_width = []
    for cur, tgt in zip(cropped.shape, target_shape):
        if cur < tgt:
            total_pad = tgt - cur
            before = total_pad // 2
            after = total_pad - before
            pad_width.append((before, after))
        else:
            pad_width.append((0, 0))

    if any(p != (0, 0) for p in pad_width):
        cropped = np.pad(cropped, pad_width, mode="constant", constant_values=pad_value)

    return cropped


def preprocess_ct_volume(
    image: sitk.Image,
    target_shape: Sequence[int] = DEFAULT_TARGET_SHAPE,
    target_spacing_mm: Sequence[float] = DEFAULT_TARGET_SPACING_MM,
    hu_window: Sequence[float] = DEFAULT_HU_WINDOW,
    out_range: Sequence[float] = DEFAULT_OUT_RANGE,
) -> np.ndarray:
    """Full CT preprocessing: resample -> array -> HU clip/normalize -> crop/pad.

    Returns a float32 numpy array of shape target_shape, values within out_range.
    """
    resampled = resample_to_spacing(image, target_spacing_mm, interpolator=sitk.sitkLinear)
    arr = sitk.GetArrayFromImage(resampled)  # z, y, x
    normalized = clip_and_normalize_hu(arr, hu_window=hu_window, out_range=out_range)
    return crop_or_pad_to_shape(normalized, target_shape=target_shape, pad_value=out_range[0])


def preprocess_mask_volume(
    image: sitk.Image,
    target_shape: Sequence[int] = DEFAULT_TARGET_SHAPE,
    target_spacing_mm: Sequence[float] = DEFAULT_TARGET_SPACING_MM,
) -> np.ndarray:
    """Full mask preprocessing: resample (nearest-neighbor, label-safe) -> crop/pad.

    Does NOT apply the lung/nodule value encoding (see masks.py) -- this only handles the
    geometric resampling + crop/pad so a raw binary/label mask stays aligned with its CT volume.
    """
    resampled = resample_to_spacing(
        image, target_spacing_mm, interpolator=sitk.sitkNearestNeighbor
    )
    arr = sitk.GetArrayFromImage(resampled)
    return crop_or_pad_to_shape(arr, target_shape=target_shape, pad_value=0)
