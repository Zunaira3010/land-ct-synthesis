"""
Run the full LAND preprocessing pipeline over every manifest entry and cache the result.

For each scan: load raw DICOM -> segment lungs (lungmask) -> rasterize nodule mask + texture
score (pylidc, tumor scans only) -> encode conditioning mask (masks.py) -> resample CT+mask to
1mm together -> HU clip/normalize CT -> crop/pad both to 256^3 -> save as
<cache_dir>/<patient_id>.npz with keys "ct" (float16) and "mask" (float32), matching what
src/data/lidc_dataset.py::load_cached_npz expects.

Usage:
    python -m scripts.preprocess_dataset \
        --manifest data/processed/land/manifest.json \
        --cache-dir data/processed/land/cache \
        [--limit 5]   # smoke-test a handful of patients first -- do this before a full run
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.io import get_scan_for_patient, build_nodule_mask, load_dicom_series
from src.data.lung_mask import segment_lungs
from src.data.masks import encode_conditioning_mask, make_healthy_mask
from src.data.preprocessing import preprocess_ct_volume, preprocess_mask_volume


def process_one(entry: dict, cache_dir: Path, target_shape=(256, 256, 256)) -> None:
    patient_id = entry["patient_id"]
    dicom_dir = entry["scan_dicom_dir"]

    image = load_dicom_series(dicom_dir)  # native resolution, HU values, sitk.Image
    native_arr_shape = sitk.GetArrayFromImage(image).shape  # z, y, x -- matches pylidc's array

    lung_mask = segment_lungs(image)

    if entry["is_healthy"]:
        encoded_mask = make_healthy_mask(lung_mask)
    else:
        scan = get_scan_for_patient(patient_id)
        nodule_mask, texture_score = build_nodule_mask(scan, volume_shape=native_arr_shape)
        if texture_score is None:
            # Manifest said tumor but pylidc found nothing -- data drifted, fail loudly rather
            # than silently mis-labeling a training sample.
            raise RuntimeError(
                f"{patient_id}: manifest says is_healthy=False but pylidc found no nodules"
            )
        encoded_mask = encode_conditioning_mask(lung_mask, nodule_mask, texture_score)

    # Wrap the encoded mask as a sitk image sharing the CT's geometry so resampling keeps both
    # volumes aligned (see docs/03_preprocessing.md step 6).
    mask_image = sitk.GetImageFromArray(encoded_mask.astype(np.float32))
    mask_image.CopyInformation(image)

    ct_processed = preprocess_ct_volume(image, target_shape=target_shape)
    mask_processed = preprocess_mask_volume(mask_image, target_shape=target_shape)

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{patient_id}.npz"
    np.savez_compressed(
        out_path,
        ct=ct_processed.astype(np.float16),
        mask=mask_processed.astype(np.float32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N entries across all splits -- smoke test")
    args = parser.parse_args()

    with open(args.manifest) as f:
        splits = json.load(f)

    all_entries = splits["train"] + splits["val"] + splits["test"]
    if args.limit:
        all_entries = all_entries[:args.limit]

    cache_dir = Path(args.cache_dir)
    failures = []
    for i, entry in enumerate(all_entries):
        patient_id = entry["patient_id"]
        print(f"[{i+1}/{len(all_entries)}] {patient_id} (healthy={entry['is_healthy']}) ...", flush=True)
        try:
            process_one(entry, cache_dir)
        except Exception as exc:  # noqa: BLE001 -- keep going, report all failures at the end
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            failures.append((patient_id, str(exc)))

    print(f"\nDone. {len(all_entries) - len(failures)}/{len(all_entries)} succeeded.")
    if failures:
        print(f"{len(failures)} failures:")
        for patient_id, reason in failures:
            print(f"  {patient_id}: {reason}")


if __name__ == "__main__":
    main()
