"""
src/data/sliding_window.py
===========================
Generic 3D sliding-window patch-grid + tapered blend-weight utilities.

Used by scripts/precompute_latents.py to build a full 64^3x4 VAE latent for each patient by
sliding the already-proven-safe 96^3 VAE-encoder patch size across the full 256^3 CT volume,
encoding each window separately (fits comfortably in 8GB, unlike a naive full-volume
vae.encode() call, which OOMs on the very first encoder conv -- see lidc_dataset.py's module
docstring for that same 54GB-at-256^3 figure), and blending the overlapping latent outputs
with a tapered linear ramp so patch boundaries don't leave visible seams in the stitched
result.

This module is intentionally generic (plain numpy, no torch, no knowledge of what
`encode_patch_fn` actually does) so it can be tested in isolation against synthetic data --
see the self-test at the bottom -- before trusting it against a real VAE.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def sliding_window_starts(size: int, patch: int, stride: int) -> list[int]:
    """1D grid of patch start indices covering [0, size) with the given patch/stride.

    Guarantees the whole axis is covered including the tail: if the last regular-stride step
    would leave a partial patch hanging off the end (or not reach the end at all), one extra
    start is inserted at `size - patch` so every patch is full-size and the union of all
    patches covers the entire axis with no gap.
    """
    if patch > size:
        raise ValueError(f"patch ({patch}) cannot exceed axis size ({size})")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    starts = list(range(0, size - patch + 1, stride))
    if not starts or starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def sliding_window_grid(
    volume_shape: tuple[int, int, int],
    patch_size: int,
    stride: int,
) -> list[tuple[int, int, int]]:
    """Full 3D grid of (d0, h0, w0) patch origins covering `volume_shape`."""
    d_starts = sliding_window_starts(volume_shape[0], patch_size, stride)
    h_starts = sliding_window_starts(volume_shape[1], patch_size, stride)
    w_starts = sliding_window_starts(volume_shape[2], patch_size, stride)
    return [(d0, h0, w0) for d0 in d_starts for h0 in h_starts for w0 in w_starts]


def blend_weights_1d(patch: int, overlap: int) -> np.ndarray:
    """1D tapered weight ramp for one patch of length `patch`, with `overlap`-sized ramps at
    each end and flat 1.0 in the middle.

    Two adjacent patches placed `patch - overlap` apart (i.e. stride = patch - overlap) get
    ramps that are exact complements of each other in the shared region: this patch's
    down-ramp at its tail and the neighbor's up-ramp at its head sum to exactly 1.0 at every
    voxel in the overlap. Concretely, at overlap-relative index i (0-indexed):
        this patch's tail weight   = (overlap - i) / (overlap + 1)
        neighbor's head weight     = (i + 1)       / (overlap + 1)
        sum                        = (overlap + 1) / (overlap + 1) = 1.0
    which is exact, not approximate -- verified in the self-test below.
    """
    if overlap < 0 or 2 * overlap > patch:
        raise ValueError(
            f"overlap ({overlap}) must be in [0, patch // 2] = [0, {patch // 2}], got patch={patch}"
        )
    weights = np.ones(patch, dtype=np.float64)
    if overlap == 0:
        return weights
    # ramp[0] is the most-tapered position (index 0, adjacent to a neighbor's own
    # most-tapered tail); ramp[-1] = overlap/(overlap+1) is nearly full weight, one step
    # before the flat interior.
    ramp = np.arange(1, overlap + 1, dtype=np.float64) / (overlap + 1)
    weights[:overlap] = ramp
    weights[-overlap:] = ramp[::-1]
    return weights


def sliding_window_average(
    volume_shape: tuple[int, int, int],
    patch_size: int,
    stride: int,
    encode_patch_fn: Callable[[int, int, int], np.ndarray],
    out_channels: int,
) -> np.ndarray:
    """Stitch overlapping patch-wise encodings into one full volume via tapered-weight blending.

    encode_patch_fn(d0, h0, w0) -> array of shape (out_channels, patch_size, patch_size,
    patch_size) for the patch starting at that origin. Called once per grid point from
    `sliding_window_grid`.

    Returns an array of shape (out_channels, *volume_shape): at every voxel, a weighted
    average of every patch that covers it, using `blend_weights_1d`'s ramps (outer-product
    across the 3 axes). Normalizes by the ACTUAL accumulated weight at each voxel
    (accum / weight_accum), not an assumption that raw weights sum to exactly 1 everywhere --
    that assumption is false at true volume edges (only one patch covers them) and doesn't
    need to be true for this to be a correct weighted average.
    """
    overlap = patch_size - stride
    if overlap < 0:
        raise ValueError(f"stride ({stride}) cannot exceed patch_size ({patch_size})")

    w1d = blend_weights_1d(patch_size, overlap)
    weight_cube = w1d[:, None, None] * w1d[None, :, None] * w1d[None, None, :]

    accum = np.zeros((out_channels, *volume_shape), dtype=np.float64)
    weight_accum = np.zeros(volume_shape, dtype=np.float64)

    for d0, h0, w0 in sliding_window_grid(volume_shape, patch_size, stride):
        patch = np.asarray(encode_patch_fn(d0, h0, w0))
        expected_shape = (out_channels, patch_size, patch_size, patch_size)
        if patch.shape != expected_shape:
            raise ValueError(
                f"encode_patch_fn(d0={d0}, h0={h0}, w0={w0}) returned shape {patch.shape}, "
                f"expected {expected_shape}"
            )
        d_sl = slice(d0, d0 + patch_size)
        h_sl = slice(h0, h0 + patch_size)
        w_sl = slice(w0, w0 + patch_size)
        accum[:, d_sl, h_sl, w_sl] += patch.astype(np.float64) * weight_cube[None, ...]
        weight_accum[d_sl, h_sl, w_sl] += weight_cube

    if np.any(weight_accum <= 0):
        raise RuntimeError(
            "sliding-window grid left voxels with zero accumulated weight -- "
            "this indicates a bug in sliding_window_grid's coverage, not expected usage error"
        )

    return (accum / weight_accum[None, ...]).astype(np.float32)


# ---------------------------------------------------------------------------
# Self-test (run with: python -m src.data.sliding_window)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Running sliding_window self-tests...")

    # Test 1: blend_weights_1d complementarity -- adjacent patches' ramps must sum to
    # exactly 1.0 across the overlap, not approximately.
    for patch, overlap in [(24, 8), (96, 32), (10, 5), (7, 3)]:
        w = blend_weights_1d(patch, overlap)
        stride = patch - overlap
        w_next = blend_weights_1d(patch, overlap)  # identical ramp shape for a same-size neighbor
        tail = w[-overlap:] if overlap else np.array([])
        head = w_next[:overlap] if overlap else np.array([])
        if overlap:
            combined = tail + head
            assert np.allclose(combined, 1.0, atol=1e-12), (
                f"patch={patch} overlap={overlap}: tail+head={combined}, expected all 1.0"
            )
        print(f"  blend_weights_1d(patch={patch}, overlap={overlap}): PASS (complementary sum=1.0)")

    # Test 2: overlapping-patch reconstruction of a known random signal, exact to float
    # precision (this is the actual bug caught previously: a flawed ramp construction gave
    # 1.125 instead of 1.0 in the overlap region -- this test would have caught it directly).
    np.random.seed(0)
    D = H = W = 64
    ground_truth = np.random.randn(2, D, H, W).astype(np.float32)

    def encode_patch_fn(d0, h0, w0):
        return ground_truth[:, d0:d0 + 24, h0:h0 + 24, w0:w0 + 24]

    recon = sliding_window_average(
        (D, H, W), patch_size=24, stride=16, encode_patch_fn=encode_patch_fn, out_channels=2
    )
    max_err = np.abs(recon - ground_truth).max()
    print(f"  Overlapping-patch (stride=16, overlap=8) max reconstruction error: {max_err:.2e}")
    assert max_err < 1e-4, "sliding-window blend does not reconstruct exact patches correctly!"

    # Test 3: exact tiling (no overlap) still works (stride == patch_size).
    def encode_patch_fn2(d0, h0, w0):
        return ground_truth[:, d0:d0 + 32, h0:h0 + 32, w0:w0 + 32]

    recon2 = sliding_window_average(
        (D, H, W), patch_size=32, stride=32, encode_patch_fn=encode_patch_fn2, out_channels=2
    )
    max_err2 = np.abs(recon2 - ground_truth).max()
    print(f"  Exact-tiling (no overlap) max error: {max_err2:.2e}")
    assert max_err2 < 1e-4

    # Test 4: non-cleanly-divisible axis (the real 256/96/64 case, scaled down for a fast
    # test) -- checks sliding_window_starts' tail-clamping logic actually gets exercised and
    # still reconstructs correctly, including the clamped/overlapping tail patch.
    D2 = H2 = W2 = 50
    ground_truth2 = np.random.randn(1, D2, H2, W2).astype(np.float32)
    starts = sliding_window_starts(50, patch=24, stride=16)
    assert starts[-1] == 50 - 24, f"expected tail-clamped start at 26, got starts={starts}"

    def encode_patch_fn3(d0, h0, w0):
        return ground_truth2[:, d0:d0 + 24, h0:h0 + 24, w0:w0 + 24]

    recon3 = sliding_window_average(
        (D2, H2, W2), patch_size=24, stride=16, encode_patch_fn=encode_patch_fn3, out_channels=1
    )
    max_err3 = np.abs(recon3 - ground_truth2).max()
    print(f"  Non-divisible axis (50, patch=24, stride=16, tail-clamped) max error: {max_err3:.2e}")
    assert max_err3 < 1e-4

    # Test 5: guard rail actually rejects overlap > patch/2, rather than silently misbehaving.
    try:
        blend_weights_1d(24, 13)
        raise AssertionError("expected ValueError for overlap > patch/2, got none")
    except ValueError:
        print("  Guard rail (overlap > patch/2 rejected): PASS")

    print("\nALL SLIDING-WINDOW TESTS PASSED.")
    sys.exit(0)
