# 05 — Inference Specification

## Sampling procedure

**[PAPER]**: "Inference uses the same number of steps [T=1000] to prioritize sample quality."
So unlike many diffusion deployments that use a fast sampler (DDIM, 20–50 steps) for speed, LAND
does full 1000-step ancestral sampling at inference too. This is slower but matches the paper —
we implement full T=1000 sampling as the default, with an optional faster DDIM path for rapid
iteration/debugging clearly labeled as *not* paper-faithful.

## Generating the two dataset classes

```
Healthy CT:  lung_mask (from a real or template lung shape) + nodule_channel = all zeros
Tumor CT:    lung_mask + nodule_mask (chosen location/size) + texture_score in {1..5}
```

Lung masks for generation come from real patient anatomy (held-out LIDC-IDRI lung masks, or
NLST masks per `02_dataset_pipeline.md`'s NLST caveat) — LAND has no lung-shape generator of its
own; anatomical diversity comes entirely from which real lung masks you feed it. This matches
the paper's own framing (Conclusion: "adding a mask generation module to enhance anatomical
diversity" is listed as *future* work, i.e. not part of LAND itself).

## Reported memory footprint

**[PAPER]** (published version): <8GB GPU memory during inference. This is within reach of the
local 8GB dev GPU — inference should be runnable locally once a trained checkpoint exists,
unlike training.

## Output format

Save as NIfTI (`.nii.gz`) by default — universal in the medical imaging Python ecosystem
(nibabel/SimpleITK/3D Slicer all read it natively), plus optional DICOM re-encoding if the
downstream use case needs it (e.g. feeding into existing DICOM-only tooling).

## Batch generation for dataset construction

For your actual goal (a synthetic dataset), `inference/generate.py` should support:
- A manifest-driven batch mode: CSV/JSON listing `{lung_mask_path, nodule_mask_path_or_none,
  texture_score_or_none, output_name}` rows, so large healthy+tumor batches can be scripted
  rather than generated one-by-one interactively.
- Deterministic seeding per sample for reproducibility.
