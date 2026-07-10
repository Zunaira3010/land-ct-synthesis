"""
Build the LAND training manifest from LIDC-IDRI via pylidc.

For every patient scan: pylidc tells us whether it has any annotated nodules (-> tumor sample,
with a texture score) or none (-> healthy sample). Splits patients 85/10/5 train/val/test
(seed=42), per configs/data_config.yaml, at the PATIENT level to avoid leakage.

Usage:
    python -m scripts.build_manifest \
        --out data/processed/land/manifest.json \
        [--limit 20]   # for a quick smoke-test subset before running on all 1,010 patients

Output JSON shape:
    {
      "train": [{"patient_id": ..., "scan_dicom_dir": ..., "is_healthy": ..., "nodule_texture_score": ...}, ...],
      "val":   [...],
      "test":  [...]
    }
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.io import check_pylidc_setup


def build_manifest(limit: int | None = None) -> list[dict]:
    check_pylidc_setup()
    import pylidc as pl

    scans = pl.query(pl.Scan).all()
    if limit:
        scans = scans[:limit]

    entries = []
    skipped = []
    for scan in scans:
        try:
            dicom_dir = scan.get_path_to_dicom_files()
        except Exception as exc:  # noqa: BLE001 -- log and skip, don't kill the whole run
            skipped.append((scan.patient_id, f"no dicom files found: {exc}"))
            continue

        clusters = scan.cluster_annotations()
        if not clusters:
            entries.append({
                "patient_id": scan.patient_id,
                "scan_dicom_dir": dicom_dir,
                "is_healthy": True,
                "nodule_texture_score": None,
            })
        else:
            largest_cluster = max(clusters, key=len)
            texture_scores = [a.texture for a in largest_cluster]
            texture_score = int(round(sum(texture_scores) / len(texture_scores)))
            texture_score = min(max(texture_score, 1), 5)
            entries.append({
                "patient_id": scan.patient_id,
                "scan_dicom_dir": dicom_dir,
                "is_healthy": False,
                "nodule_texture_score": texture_score,
            })

    if skipped:
        print(f"WARNING: skipped {len(skipped)} scans with no locatable DICOM files:")
        for patient_id, reason in skipped[:10]:
            print(f"  {patient_id}: {reason}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    n_healthy = sum(e["is_healthy"] for e in entries)
    print(f"Built manifest: {len(entries)} scans ({n_healthy} healthy, {len(entries) - n_healthy} tumor)")
    return entries


def split_manifest(entries: list[dict], train_frac=0.85, val_frac=0.10, seed=42) -> dict:
    rng = random.Random(seed)
    shuffled = entries[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/processed/land/manifest.json")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N scans -- use for a smoke test")
    args = parser.parse_args()

    entries = build_manifest(limit=args.limit)
    splits = split_manifest(entries)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(f"Wrote manifest to {out_path}")
