"""
Stage 2 sanity checks, using synthetic data throughout — no real LIDC-IDRI download or pylidc/
lungmask installation required. Validates the pure logic in preprocessing.py, masks.py, and the
Dataset __getitem__ contract in lidc_dataset.py against what docs/02_dataset_pipeline.md and
docs/03_preprocessing.md specify.

Run with: pytest tests/test_stage2_preprocessing.py -v
"""
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import (
    resample_to_spacing,
    clip_and_normalize_hu,
    crop_or_pad_to_shape,
    preprocess_ct_volume,
    preprocess_mask_volume,
)
from src.data.masks import (
    encode_conditioning_mask,
    make_healthy_mask,
    downsample_mask_maxpool,
    texture_label_name,
    LUNG_VALUE,
)
from src.data.lidc_dataset import LIDCVolumeDataset, LIDCSample


# ---------------------------------------------------------------------------
# preprocessing.py
# ---------------------------------------------------------------------------

def _make_synthetic_sitk_volume(size=(100, 100, 60), spacing=(0.7, 0.7, 1.5), fill_value=-500):
    arr = np.full((size[2], size[1], size[0]), fill_value, dtype=np.int16)  # sitk is z,y,x
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing(spacing)
    return image


def test_resample_to_1mm_changes_size_correctly():
    image = _make_synthetic_sitk_volume(size=(100, 100, 60), spacing=(0.7, 0.7, 1.5))
    resampled = resample_to_spacing(image, target_spacing_mm=(1.0, 1.0, 1.0))

    expected_size = (70, 70, 90)  # size * spacing / target_spacing, rounded
    assert resampled.GetSize() == expected_size
    assert resampled.GetSpacing() == (1.0, 1.0, 1.0)


def test_clip_and_normalize_hu_range():
    volume = np.array([-2000, -1000, -300, 400, 2000], dtype=np.float32)
    normalized = clip_and_normalize_hu(volume, hu_window=(-1000, 400), out_range=(-1.0, 1.0))

    assert normalized.min() >= -1.0 - 1e-6
    assert normalized.max() <= 1.0 + 1e-6
    # values outside the window should clip to the range endpoints
    assert np.isclose(normalized[0], -1.0)   # -2000 clips to -1000 -> -1.0
    assert np.isclose(normalized[-1], 1.0)   # 2000 clips to 400 -> 1.0
    assert np.isclose(normalized[1], -1.0)   # exactly at lo
    assert np.isclose(normalized[3], 1.0)    # exactly at hi


def test_crop_or_pad_shrinks_correctly():
    volume = np.zeros((300, 300, 300), dtype=np.float32)
    result = crop_or_pad_to_shape(volume, target_shape=(256, 256, 256))
    assert result.shape == (256, 256, 256)


def test_crop_or_pad_grows_correctly():
    volume = np.ones((200, 200, 200), dtype=np.float32)
    result = crop_or_pad_to_shape(volume, target_shape=(256, 256, 256), pad_value=-1.0)
    assert result.shape == (256, 256, 256)
    # center should still be the original data (value 1.0), edges should be pad value
    assert result[128, 128, 128] == 1.0
    assert result[0, 0, 0] == -1.0


def test_crop_or_pad_mixed_axes():
    """One axis needs cropping, another needs padding -- exercises both branches at once."""
    volume = np.ones((300, 200, 256), dtype=np.float32)
    result = crop_or_pad_to_shape(volume, target_shape=(256, 256, 256), pad_value=-1.0)
    assert result.shape == (256, 256, 256)


def test_full_ct_preprocessing_pipeline_shape():
    image = _make_synthetic_sitk_volume(size=(180, 180, 150), spacing=(1.0, 1.0, 1.0), fill_value=-1000)
    result = preprocess_ct_volume(image, target_shape=(256, 256, 256))
    assert result.shape == (256, 256, 256)
    assert result.dtype == np.float32
    assert result.min() >= -1.0 - 1e-6
    assert result.max() <= 1.0 + 1e-6


def test_full_mask_preprocessing_pipeline_shape():
    image = _make_synthetic_sitk_volume(size=(180, 180, 150), spacing=(1.0, 1.0, 1.0), fill_value=0)
    result = preprocess_mask_volume(image, target_shape=(256, 256, 256))
    assert result.shape == (256, 256, 256)


# ---------------------------------------------------------------------------
# masks.py
# ---------------------------------------------------------------------------

def test_encode_healthy_mask_has_only_background_and_lung_values():
    lung = np.zeros((10, 10, 10), dtype=bool)
    lung[2:8, 2:8, 2:8] = True

    encoded = make_healthy_mask(lung)
    unique_values = set(np.unique(encoded).tolist())
    assert unique_values == {0.0, LUNG_VALUE}


def test_encode_tumor_mask_texture_scaling():
    lung = np.zeros((10, 10, 10), dtype=bool)
    lung[1:9, 1:9, 1:9] = True
    nodule = np.zeros((10, 10, 10), dtype=bool)
    nodule[4:6, 4:6, 4:6] = True

    for score in range(1, 6):
        encoded = encode_conditioning_mask(lung, nodule, nodule_texture_score=score)
        nodule_values = encoded[nodule]
        assert np.allclose(nodule_values, score / 5.0)
        # non-nodule lung voxels should remain at LUNG_VALUE
        non_nodule_lung = lung & ~nodule
        assert np.allclose(encoded[non_nodule_lung], LUNG_VALUE)
        # everything is within [0, 1] per the paper's stated normalization
        assert encoded.min() >= 0.0
        assert encoded.max() <= 1.0


def test_encode_mask_rejects_missing_texture_score():
    lung = np.ones((5, 5, 5), dtype=bool)
    nodule = np.zeros((5, 5, 5), dtype=bool)
    nodule[2, 2, 2] = True
    try:
        encode_conditioning_mask(lung, nodule, nodule_texture_score=None)
        assert False, "expected ValueError for missing texture score"
    except ValueError:
        pass


def test_encode_mask_rejects_out_of_range_texture_score():
    lung = np.ones((5, 5, 5), dtype=bool)
    nodule = np.zeros((5, 5, 5), dtype=bool)
    nodule[2, 2, 2] = True
    for bad_score in (0, 6, -1):
        try:
            encode_conditioning_mask(lung, nodule, nodule_texture_score=bad_score)
            assert False, f"expected ValueError for score {bad_score}"
        except ValueError:
            pass


def test_downsample_maxpool_shape_and_values():
    mask = np.zeros((8, 8, 8), dtype=np.float32)
    mask[0, 0, 0] = 0.8  # one hot voxel in the first pooling block

    downsampled = downsample_mask_maxpool(mask, factor=4)
    assert downsampled.shape == (2, 2, 2)
    assert downsampled[0, 0, 0] == 0.8  # max-pool should preserve the hot voxel's value
    assert downsampled[1, 1, 1] == 0.0


def test_downsample_matches_paper_resolution():
    """256^3 mask downsampled 4x should exactly match the 64^3 latent spatial resolution."""
    mask = np.zeros((256, 256, 256), dtype=np.float32)
    downsampled = downsample_mask_maxpool(mask, factor=4)
    assert downsampled.shape == (64, 64, 64)


def test_texture_label_names():
    assert texture_label_name(1) == "non-solid"
    assert texture_label_name(5) == "solid"
    try:
        texture_label_name(6)
        assert False
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# lidc_dataset.py -- Dataset contract, with an injected synthetic loader
# ---------------------------------------------------------------------------

def _synthetic_loader(sample: LIDCSample):
    """Stand-in for the real .npz cache loader -- returns plausible-shaped fake data."""
    ct = np.random.uniform(-1, 1, size=(256, 256, 256)).astype(np.float32)
    mask = np.zeros((256, 256, 256), dtype=np.float32)
    mask[64:192, 64:192, 64:192] = LUNG_VALUE
    if not sample.is_healthy:
        mask[120:136, 120:136, 120:136] = (sample.nodule_texture_score or 1) / 5.0
    return ct, mask


def test_dataset_getitem_shapes_and_keys():
    manifest = [
        LIDCSample(patient_id="P001", scan_dicom_dir="/fake", is_healthy=True),
        LIDCSample(patient_id="P002", scan_dicom_dir="/fake", is_healthy=False, nodule_texture_score=4),
    ]
    dataset = LIDCVolumeDataset(
        manifest=manifest,
        cache_dir="/fake/cache",
        volume_loader=_synthetic_loader,
    )

    assert len(dataset) == 2

    healthy_item = dataset[0]
    assert healthy_item["ct"].shape == (1, 256, 256, 256)
    assert healthy_item["mask"].shape == (1, 64, 64, 64)   # 256/4 per docs/01_architecture.md
    assert healthy_item["is_healthy"] is True
    assert healthy_item["texture_score"] == 0

    tumor_item = dataset[1]
    assert tumor_item["is_healthy"] is False
    assert tumor_item["texture_score"] == 4
    assert tumor_item["mask"].max() <= 1.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
