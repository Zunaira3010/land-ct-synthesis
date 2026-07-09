# LAND reimplementation — Lung and Nodule Diffusion for 3D Chest CT Synthesis

Unofficial reimplementation of:

> Oliveras, A., Marí, R., Redondo, R. et al. "Anatomically guided latent diffusion for
> high-resolution 3D chest CT synthesis." *Scientific Reports* (2026).
> https://doi.org/10.1038/s41598-026-51634-4
> (arXiv preprint: [2510.18446](https://arxiv.org/abs/2510.18446))

No official code, weights, or training scripts have been released by the authors (verified —
see project history / NOTES.md). This repo reimplements the architecture described in Section 2
of the paper from scratch, using MONAI's generative-model building blocks as scaffolding.

**Read `NOTES.md` before trusting any specific hyperparameter** — several details aren't fully
specified in the paper and had to be inferred. Every assumption is logged there and flagged
inline in the relevant config file.

## Goal of this project

Generate a synthetic chest CT dataset containing both:
- **Healthy lung volumes** — lung mask + empty (all-zero) nodule channel
- **Tumor-bearing lung volumes** — lung mask + nodule mask with a chosen texture (1=non-solid
  through 5=solid)

## Project structure

```
land-ct-synthesis/
├── configs/                  # all hyperparameters live here, not hardcoded in code
│   ├── data_config.yaml      # dataset paths, volume shape, mask encoding scheme
│   ├── vae_config.yaml       # 3D VAE architecture + training config
│   └── diffusion_config.yaml # 3D diffusion U-Net architecture + training config
├── src/
│   ├── data/                 # DICOM loading, resampling, mask generation, PyTorch Dataset
│   ├── models/                # VAE and diffusion U-Net definitions
│   ├── training/               # train_vae.py, train_diffusion.py
│   ├── inference/              # generate.py — sample healthy/tumor CTs from a trained model
│   └── utils/                 # config loader, loss functions, metrics
├── scripts/                  # one-off setup / data-download helper scripts
├── tests/                    # shape/sanity tests to run after each stage
├── checkpoints/               # trained model weights land here (gitignored)
├── logs/                     # tensorboard logs (gitignored)
├── environment.yml
└── NOTES.md                  # assumption log — read this
```

This mirrors the paper's own structure: a VAE (Section 2, "3D VAE") that compresses
256³ CT volumes into 64×64×64×4 latents, and a conditional diffusion U-Net (Section 2, "3D
U-Net") that denoises in that latent space, conditioned on lung + nodule masks.

## Setup

```bash
conda env create -f environment.yml
conda activate land-ct
```

GPU note: training needs ~10-16GB VRAM per the published paper's figures (the arXiv preprint's
higher 20GB figure was revised down in the peer-reviewed version). An 8GB card can run inference
on an already-trained model but cannot train from scratch — see project history for the compute
plan.

## Stage roadmap

- [x] **Stage 1 — Project setup** (this commit): folder structure, environment, config system
- [ ] **Stage 2 — Dataset pipeline**: LIDC-IDRI download, DICOM→volume, lung/nodule mask
      generation, healthy-sample construction, train/val/test split
- [ ] **Stage 3 — 3D VAE**: implement + train, verify 64×64×64×4 latent shape and
      reconstruction quality
- [ ] **Stage 4 — Diffusion U-Net**: implement cross-attention conditioning, v-prediction,
      Min-SNR-γ loss
- [ ] **Stage 5 — Training**: full training runs on rented GPU
- [ ] **Stage 6 — Inference**: generate healthy + tumor CT datasets at scale

## Config usage

All scripts load configs via `src/utils/config.py`, which merges the relevant yaml files and
allows CLI overrides without editing files:

```bash
python -m src.training.train_vae vae.training.batch_size=2 data.paths.processed_dir=/mnt/data
```

## License note

The LAND paper is published under CC BY-NC-ND 4.0 (non-commercial, no derivatives of the paper
itself). This is an independent reimplementation of the *method described*, not a derivative of
any paper text or figures — standard practice for reproducing published architectures — but keep
the non-commercial constraint in mind for how the resulting synthetic dataset gets used.
