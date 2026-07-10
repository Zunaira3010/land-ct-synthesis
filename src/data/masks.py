"""
Conditioning mask encoding for LAND.

Paper (Section 2, Method): "Spatial and textural cues are encoded by assigning lungs a value
of 0.5 and nodules 1-5 (non-solid to solid). Masks are normalized to [0,1], downsampled four
times via 3D max pooling, concatenated with the noisy latent, and injected into U-Net
cross-attention layers."

This module implements the [PAPER]/[INFERRED] mask semantics logged in
docs/06_open_questions.md item #1: lung voxels get a constant 0.5; nodule voxels overwrite that
with texture_score / 5.0; everything else (background) is 0.0.
"""
from __future__ import annotations

import numpy as np

LUNG_VALUE = 0.5

TEXTURE_LABELS = {
    1: "non-solid",
    2: "non-solid-mixed",
    3: "part-solid",
    4: "solid-mixed",
    5: "solid",
}

NODULE_TEXTURE_SCALE_MAX = 5.0


def texture_label_name(score: int) -> str:
    """Human-readable name for a LIDC-IDRI nodule texture score (1-5)."""
    if score not in TEXTURE_LABELS:
        raise ValueError(f"texture score must be in 1..5, got {score}")
    return TEXTURE_LABELS[score]


def make_healthy_mask(lung_mask: np.ndarray) -> np.ndarray:
    """Build a conditioning mask for a nodule-free ('healthy') sample.

    Only two values appear: 0.0 (background) and LUNG_VALUE (0.5) inside the lungs.
    """
    lung_mask = lung_mask.astype(bool)
    encoded = np.zeros(lung_mask.shape, dtype=np.float32)
    encoded[lung_mask] = LUNG_VALUE
    return encoded


def encode_conditioning_mask(
    lung_mask: np.ndarray,
    nodule_mask: np.ndarray,
    nodule_texture_score: int | None,
) -> np.ndarray:
    """Build the full lung+nodule conditioning mask for a tumor-bearing sample.

    lung_mask / nodule_mask: boolean arrays, same shape, nodule_mask should be a subset of
    lung_mask (nodules occur inside lungs), but we don't hard-enforce that here.
    nodule_texture_score: int 1-5, required whenever nodule_mask has any True voxels.

    Encoding: background=0.0, lung-only voxels=LUNG_VALUE (0.5), nodule voxels overwrite the
    lung value with texture_score / 5.0.
    """
    lung_mask = lung_mask.astype(bool)
    nodule_mask = nodule_mask.astype(bool)

    if nodule_mask.any():
        if nodule_texture_score is None:
            raise ValueError(
                "nodule_texture_score is required when nodule_mask contains any nodule voxels"
            )
        if not (1 <= nodule_texture_score <= 5):
            raise ValueError(
                f"nodule_texture_score must be in 1..5, got {nodule_texture_score}"
            )

    encoded = make_healthy_mask(lung_mask)
    if nodule_mask.any():
        encoded[nodule_mask] = nodule_texture_score / NODULE_TEXTURE_SCALE_MAX

    return encoded


def downsample_mask_maxpool(mask: np.ndarray, factor: int = 4) -> np.ndarray:
    """3D max-pool downsample by an integer factor along every axis.

    Paper: masks are "downsampled four times via 3D max pooling" before being concatenated
    with the noisy latent (which lives at 64^3 for a 256^3 input, i.e. factor=4).
    """
    if any(dim % factor != 0 for dim in mask.shape):
        raise ValueError(
            f"mask shape {mask.shape} must be divisible by downsample factor {factor}"
        )
    d, h, w = mask.shape
    reshaped = mask.reshape(d // factor, factor, h // factor, factor, w // factor, factor)
    return reshaped.max(axis=(1, 3, 5))
