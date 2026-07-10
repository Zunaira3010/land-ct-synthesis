"""
Raw data I/O: DICOM series loading + LIDC-IDRI nodule annotation parsing.

Two separate concerns live here on purpose:

1. `load_dicom_series` -- plain DICOM -> SimpleITK image, via SimpleITK's own series reader.
   Works for any standard DICOM series (LIDC-IDRI *or* NLST), no LIDC-specific metadata needed.

2. `get_scan_for_patient` / `build_nodule_mask` -- LIDC-IDRI ONLY. These go through `pylidc`,
   which reads the LIDC XML annotation files (radiologist-drawn nodule contours + texture
   ratings) and needs its own one-time setup (see `check_pylidc_setup` below). This is the ONLY
   source of nodule masks + texture scores for LIDC-IDRI -- it does NOT work from a NIfTI
   conversion or any other derived volume, only from the original raw DICOM directory tree,
   because pylidc matches annotations to DICOM SOPInstanceUIDs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk


def load_dicom_series(dicom_dir: str) -> sitk.Image:
    """Load a single DICOM series directory into a 3D SimpleITK image (in HU for CT).

    Raises RuntimeError if the directory contains zero or >1 series (point this at a leaf
    series folder, e.g. LIDC-IDRI-0001/<study>/<series>/, not a patient-level folder).
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))

    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {dicom_dir}")
    if len(series_ids) > 1:
        raise RuntimeError(
            f"{dicom_dir} contains {len(series_ids)} series, expected exactly 1. "
            f"Point this at a single series' leaf directory."
        )

    file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(file_names)
    image = reader.Execute()
    return image


def check_pylidc_setup() -> None:
    """Sanity-check that pylidc's config file exists and points at a real directory.

    pylidc needs a one-time config file (~/.pylidcrc on Linux/Mac, %USERPROFILE%\\.pylidcrc on
    Windows) with:
        [dicom]
        path = /absolute/path/to/LIDC-IDRI
        warn = True
    where that path is the folder containing LIDC-IDRI-0001/, LIDC-IDRI-0002/, ... exactly as
    downloaded by the NBIA Data Retriever. Raises RuntimeError with setup instructions if
    missing/misconfigured, rather than a cryptic pylidc internal error.
    """
    import configparser

    cfg_path = Path.home() / ".pylidcrc"
    if not cfg_path.exists():
        raise RuntimeError(
            f"pylidc config not found at {cfg_path}. Create it with:\n\n"
            "[dicom]\n"
            "path = /absolute/path/to/LIDC-IDRI\n"
            "warn = True\n\n"
            "...where the path is the folder containing LIDC-IDRI-0001/, LIDC-IDRI-0002/ etc, "
            "exactly as downloaded by the NBIA Data Retriever."
        )

    parser = configparser.ConfigParser()
    parser.read(cfg_path)
    dicom_path = Path(parser.get("dicom", "path", fallback=""))
    if not dicom_path.is_dir():
        raise RuntimeError(
            f"pylidc config at {cfg_path} points to '{dicom_path}', which doesn't exist or "
            "isn't a directory. Fix the 'path' entry under [dicom]."
        )


def get_scan_for_patient(patient_id: str):
    """Return the pylidc Scan object for a given LIDC-IDRI patient ID (e.g. 'LIDC-IDRI-0001').

    Requires pylidc's one-time DB build to have already run (it builds a local sqlite cache
    from the XML annotations the first time any pylidc query is made -- can take a while the
    very first time, is fast afterwards).
    """
    check_pylidc_setup()
    import pylidc as pl

    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
    if not scans:
        raise ValueError(f"No pylidc Scan found for patient_id={patient_id}")
    if len(scans) > 1:
        raise ValueError(
            f"{len(scans)} scans found for patient_id={patient_id}, expected exactly 1. "
            "Pick the correct scan explicitly (e.g. by series_instance_uid)."
        )
    return scans[0]


def build_nodule_mask(scan, volume_shape: tuple[int, int, int]) -> tuple[np.ndarray, Optional[int]]:
    """Rasterize a pylidc Scan's consensus nodule annotations into a binary mask + texture score.

    volume_shape must match the scan's *native* (un-resampled) array shape -- run this BEFORE
    resample_to_spacing() in the preprocessing pipeline (see docs/03_preprocessing.md step 6,
    which explicitly resamples mask/CT together afterward so both stay aligned).

    Returns (nodule_mask, texture_score):
      - nodule_mask: bool array, shape == volume_shape, True where a consensus nodule exists.
      - texture_score: int 1-5 (rounded mean across radiologists who annotated the dominant/
        largest nodule cluster), or None if the scan has zero annotated nodules (healthy scan).

    A scan may have several distinct nodules; this returns only the largest cluster's mask +
    score, matching LAND's "one dominant nodule per tumor sample" assumption
    (docs/02_dataset_pipeline.md). For multi-nodule training samples, call this per-cluster
    using `scan.cluster_annotations()` directly instead.
    """
    clusters = scan.cluster_annotations()
    if not clusters:
        return np.zeros(volume_shape, dtype=bool), None

    # Pick the cluster with the most radiologist agreement (most annotations), tie-broken by
    # largest resulting volume.
    largest_cluster = max(clusters, key=len)
# pylidc's Annotation.bbox()/boolean_mask() are defined relative to scan.to_volume(),
    # whose array axis order is (row, col, slice) -- i.e. the z axis is LAST. This is the
    # opposite of SimpleITK's GetArrayFromImage() order (z, y, x) used everywhere else in
    # this pipeline (volume_shape). Build the mask in pylidc's native orientation first,
    # then transpose into (z, y, x) so it aligns with the CT array.
    pylidc_shape = scan.to_volume().shape
    pylidc_mask = np.zeros(pylidc_shape, dtype=bool)
    texture_scores = []
    for annotation in largest_cluster:
        ann_mask = annotation.boolean_mask()
        bbox = annotation.bbox()
        pylidc_mask[bbox] |= ann_mask
        texture_scores.append(annotation.texture)

    # (row, col, slice) -> (slice, row, col) == (z, y, x)
    mask = np.transpose(pylidc_mask, (2, 0, 1))
    if mask.shape != volume_shape:
        raise RuntimeError(
            f"Nodule mask shape {mask.shape} does not match expected CT volume shape "
            f"{volume_shape} for scan {scan.patient_id} -- check DICOM series consistency."
        )

    texture_score = int(round(float(np.mean(texture_scores))))
    texture_score = min(max(texture_score, 1), 5)
    return mask, texture_score
