# 03 — Preprocessing Specification

**[PAPER]** Section 3: "All scans were preprocessed as in [6]." Ref [6] = Friedrich et al., WDM
(3D wavelet diffusion models paper). **[OPEN — action item]**: we have not yet pulled WDM's
actual preprocessing script to confirm exact values; the steps below are our best-documented
placeholder using standard chest-CT preprocessing conventions until that's verified. Do not
treat the specific HU window as paper-confirmed.

## Pipeline

1. **Load** DICOM series → 3D volume + spacing metadata (SimpleITK).
2. **Resample** to 1mm isotropic spacing — **[PAPER]**, explicit: "1 mm isotropic resolution."
   Linear interpolation for the image, nearest-neighbor for any co-registered mask.
3. **HU windowing / clip**: clip to a lung-relevant Hounsfield Unit range before normalization.
   **[INFERRED, pending WDM verification]**: [-1000, 400] HU (standard "lung window" covering
   air through soft tissue/early calcification, wide enough to preserve solid nodules).
4. **Normalize** clipped HU values to [-1, 1] (standard for diffusion models, which typically
   assume roughly Gaussian/[-1,1]-ish input ranges going into the VAE encoder).
5. **Crop or pad** to exactly 256×256×256 — **[PAPER]** explicit target shape. Center-crop if
   larger, symmetric zero-pad (at the clipped-HU "air" value, i.e. -1.0 post-normalization) if
   smaller.
6. **Mask generation**, run on the *resampled, un-cropped* volume so lung/nodule geometry stays
   consistent with the CT before the final crop/pad step is applied identically to both:
   - Lung mask via `lungmask` (Hofmanninger U-Net).
   - Nodule mask + texture score via `pylidc` (LIDC-IDRI only).
7. **Mask encoding** per `01_architecture.md` Section 3 (lung=0.5, nodule=texture/5, background=0).
8. **Cache** CT volume + mask tensor to `.npz` per `02_dataset_pipeline.md`.

## Open verification item

Before Stage 3 (VAE training) begins in earnest, pull the WDM repo's actual preprocessing code
(https://github.com/pfriedri/wdm-3d, referenced as [6] in the paper) and diff it against steps
3–4 above. If the HU window or normalization range differs, our VAE will learn a different
intensity distribution than the paper's, which would silently bias every downstream result.
