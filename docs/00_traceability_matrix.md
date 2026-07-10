# Traceability Matrix

Maps every component described in the paper to (a) where it's specified in `docs/`, (b) where
it's implemented in code, and (c) its current status. Update this as each stage progresses —
it's the single place to check "is X done, and does X actually correspond to something the paper
says, or did we invent it?"

| Component | Paper section | Spec doc | Implementation file | Status |
|---|---|---|---|---|
| Overall pipeline | Fig. 1, Sec. 2 | `docs/01_architecture.md` | — | Documented |
| 3D VAE encoder/decoder | Sec. 2 "3D VAE" | `docs/01_architecture.md` | `src/models/vae.py` | **Not started** |
| VAE loss (L1+LPIPS+ADV+KL) | Sec. 2 | `docs/01_architecture.md` | `src/utils/losses.py` | **Not started** |
| Discriminator (for L_ADV) | Sec. 2 (implied) | `docs/01_architecture.md` #4 | `src/models/discriminator.py` | **Not started** |
| Diffusion 3D U-Net | Sec. 2 "3D U-Net" | `docs/01_architecture.md` | `src/models/unet.py` | **Not started** |
| Cross-attention conditioning | Sec. 2 | `docs/01_architecture.md` §3 | `src/models/unet.py` | **Not started** |
| Mask encoding (lung=0.5, nodule=texture) | Sec. 2 | `docs/01_architecture.md` §3, `docs/06_open_questions.md` #1 | `src/data/masks.py` | **Not started** |
| Velocity (v-)prediction | Sec. 2 | `docs/01_architecture.md` | `src/models/diffusion.py` | **Not started** |
| Min-SNR-γ loss | Sec. 2 | `docs/01_architecture.md` | `src/utils/losses.py` | **Not started** |
| Noise scheduler (linear, T=1000) | Sec. 2 | `docs/01_architecture.md` | `src/models/scheduler.py` | **Not started** |
| DICOM loading | Sec. 3 | `docs/02_dataset_pipeline.md` | `src/data/lidc_dataset.py` | **In progress** — manifest builder + Dataset class written, untested against real DICOM (no local LIDC-IDRI download yet) |
| Resample / HU clip / normalize / crop | Sec. 3 (partial) | `docs/03_preprocessing.md` | `src/data/preprocessing.py` | ✅ **Done, tested** (synthetic volumes; HU window itself still [INFERRED], pending WDM check) |
| Mask encoding (lung=0.5, nodule=texture/5) + 4x max-pool downsample | Sec. 2 | `docs/01_architecture.md` §3 | `src/data/masks.py` | ✅ **Done, tested** |
| Lung segmentation (Hofmanninger U-Net) | Sec. 3, ref [13] | `docs/02_dataset_pipeline.md` | `src/data/lung_segmentation.py` | **Not started** — will wrap `lungmask` package directly |
| Nodule mask + texture (pylidc) | Sec. 3 | `docs/02_dataset_pipeline.md` | `src/data/lidc_dataset.py::build_manifest_from_pylidc` | **In progress** — written, untested against real pylidc DB |
| Healthy/tumor sample construction | Sec. 1 (abstract) | `docs/02_dataset_pipeline.md` | `src/data/lidc_dataset.py` | ✅ **Done, tested** (Dataset __getitem__ contract; manifest-building from real data still untested) |
| Train/val/test split | — (not paper-specified) | `docs/02_dataset_pipeline.md` | `src/data/lidc_dataset.py` | **Not started** |
| Precompute cache script | — (our tooling) | `docs/02_dataset_pipeline.md` | `scripts/build_cache.py` | **Not started** — needed before real training can run |
| VAE training loop | Sec. 3 "Implementation Details" | `docs/04_training.md` | `src/training/train_vae.py` | **Not started** |
| Diffusion training loop | Sec. 3 "Implementation Details" | `docs/04_training.md` | `src/training/train_diffusion.py` | **Not started** |
| Inference / sampling | Sec. 3 (T=1000 at inference) | `docs/05_inference.md` | `src/inference/generate.py` | **Not started** |
| FID evaluation (Med3D ResNet-50) | Sec. 3 | `docs/06_open_questions.md` #14 | `src/utils/metrics.py` | **Not started** |
| MS-SSIM evaluation | Sec. 3 | — | `src/utils/metrics.py` | **Not started** |
| Config system | — (our tooling) | — | `src/utils/config.py`, `configs/*.yaml` | ✅ **Done, tested** |
| Project scaffold | — (our tooling) | — | folder structure, `environment.yml` | ✅ **Done, tested** |

## Legend
- ✅ **Done, tested** — implemented and has a passing test validating it against a paper-stated
  fact (not just "code runs").
- **Not started** — no code yet.
- **In progress** — partial implementation.
