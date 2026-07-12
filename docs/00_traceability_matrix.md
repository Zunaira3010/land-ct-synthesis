# Traceability Matrix

Maps every component described in the paper to (a) where it's specified in `docs/`, (b) where
it's implemented in code, and (c) its current status. Update this as each stage progresses —
it's the single place to check "is X done, and does X actually correspond to something the paper
says, or did we invent it?"

**Updated 2026-07-12** against the actual repo file listing (not just chat claims) — several
rows were stale (VAE/discriminator/mask-encoding marked "Not started" despite being done) and a
few "Implementation file" paths didn't match what's actually on disk; both are corrected below.

| Component | Paper section | Spec doc | Implementation file | Status |
|---|---|---|---|---|
| Overall pipeline | Fig. 1, Sec. 2 | `docs/01_architecture.md` | — | Documented |
| 3D VAE encoder/decoder | Sec. 2 "3D VAE" | `docs/01_architecture.md` | `src/models/vae.py`, `src/models/blocks.py` | ✅ **Code done.** 100-epoch/72-patient training run in progress (`--patch-size 96`); reconstruction quality not yet checked |
| VAE loss (L1+LPIPS+ADV+KL) | Sec. 2 | `docs/01_architecture.md` | `src/losses/vae_loss.py` | ✅ **Done** (file path corrected — lives under `src/losses/`, not `src/utils/losses.py`) |
| Discriminator (for L_ADV) | Sec. 2 (implied) | `docs/01_architecture.md` #4 | `src/models/discriminator.py` | ✅ **Done** — 3D PatchGAN, in active use in the current training run |
| Diffusion 3D U-Net | Sec. 2 "3D U-Net" | `docs/01_architecture.md` | `src/models/unet.py` | **Not started** — confirmed absent from repo; next stage after VAE reconstruction check |
| Cross-attention conditioning | Sec. 2 | `docs/01_architecture.md` §3 | `src/models/unet.py` | **Not started** (blocked on unet.py) — design settled: Option A, flatten 64³ mask → linear-project to `cross_attention_dim`, see `docs/06_open_questions.md` #9 |
| Mask encoding (lung=0.5, nodule=texture) | Sec. 2 | `docs/01_architecture.md` §3, `docs/06_open_questions.md` #1 | `src/data/masks.py` | ✅ **Done, tested** |
| Velocity (v-)prediction | Sec. 2 | `docs/01_architecture.md` | `src/models/diffusion.py` | **Not started** |
| Min-SNR-γ loss | Sec. 2 | `docs/01_architecture.md` | `src/losses/` (file TBD) | **Not started** |
| Noise scheduler (linear, T=1000) | Sec. 2 | `docs/01_architecture.md` | `src/models/scheduler.py` | **Not started** |
| DICOM loading | Sec. 3 | `docs/02_dataset_pipeline.md` | `src/data/lidc_dataset.py`, `src/data/io.py` | ✅ **Done, tested** — verified end-to-end on real patient LIDC-IDRI-0049, 72/72 patients cached |
| Resample / HU clip / normalize / crop | Sec. 3 (partial) | `docs/03_preprocessing.md` | `src/data/preprocessing.py` | ✅ **Done, tested.** HU window [-1000, 400] cross-checked against WDM paper (ref [6]) Sec. 4.1.1 — accepted, see `docs/06_open_questions.md` #10 |
| Mask encoding (lung=0.5, nodule=texture/5) + 4x max-pool downsample | Sec. 2 | `docs/01_architecture.md` §3 | `src/data/masks.py` | ✅ **Done, tested** |
| Lung segmentation | Sec. 3, ref [13] | `docs/02_dataset_pipeline.md` | `src/data/lung_mask.py` | ✅ **Done** — wraps the `lungmask` package as planned (file path corrected — `lung_mask.py`, not `lung_segmentation.py`) |
| Nodule mask + texture (pylidc) | Sec. 3 | `docs/02_dataset_pipeline.md` | `src/data/masks.py`, `scripts/build_manifest.py` | ✅ **Done, tested** — verified against real pylidc DB during caching |
| Healthy/tumor sample construction | Sec. 1 (abstract) | `docs/02_dataset_pipeline.md` | `src/data/lidc_dataset.py` | ✅ **Done, tested** — 72 patients = 68 nodule / 4 healthy |
| Train/val/test split | — (not paper-specified) | `docs/02_dataset_pipeline.md` | `scripts/build_manifest.py::split_manifest` | ✅ **Done, tested** — patient-level, 85/10/5, seed=42 (file path corrected — lives in `scripts/build_manifest.py`, not `lidc_dataset.py`) |
| Precompute cache script | — (our tooling) | `docs/02_dataset_pipeline.md` | `scripts/preprocess_dataset.py` | ✅ **Done, tested** — 72/72 patients cached, zero failures (file path corrected — `preprocess_dataset.py`, not `build_cache.py`) |
| VAE training loop | Sec. 3 "Implementation Details" | `docs/04_training.md` | `src/training/train_vae.py` | ✅ **Code done, running now** — patch-based (96³), checkpoint/resume execution-tested on real hardware, live ETA reporting |
| Diffusion training loop | Sec. 3 "Implementation Details" | `docs/04_training.md` | `src/training/train_diffusion.py` | **Not started** |
| Inference / sampling | Sec. 3 (T=1000 at inference) | `docs/05_inference.md` | `src/inference/generate.py` | **Not started** |
| FID evaluation (Med3D ResNet-50) | Sec. 3 | `docs/06_open_questions.md` #14 | `src/utils/metrics.py` | **Not started** — blocks Evaluation stage only, not training |
| MS-SSIM evaluation | Sec. 3 | — | `src/utils/metrics.py` | **Not started** |
| Config system | — (our tooling) | — | `src/utils/config.py`, `configs/*.yaml` | ✅ **Done, tested** |
| Project scaffold | — (our tooling) | — | folder structure, `environment.yml` | ✅ **Done, tested** |

## Legend
- ✅ **Done, tested** — implemented and has a passing test validating it against a paper-stated
  fact (not just "code runs").
- **Not started** — no code yet.
- **In progress** — partial implementation.
