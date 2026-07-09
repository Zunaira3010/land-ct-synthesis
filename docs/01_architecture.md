# 01 — Architecture Specification

Source: LAND paper (arXiv:2510.18446 / Sci Rep s41598-026-51634-4), Section 2 "Method".
Every claim below is tagged **[PAPER]** (stated explicitly) or **[INFERRED]** (our engineering
decision where the paper is silent — cross-reference `06_open_questions.md`).

## Overall pipeline

```
LIDC-IDRI DICOM series
        │
        ▼
Preprocessing (resample -> HU clip -> normalize -> crop/pad)      [PAPER, partially — see 03]
        │
        ▼
256 x 256 x 256 CT volume  ──────────────┐
        │                                 │
        ▼                                 ▼
   3D VAE Encoder                 Lung mask (Hofmanninger U-Net, ref [13])
        │                         Nodule mask + texture score (pylidc, ref [1])
        ▼                                 │
64 x 64 x 64 x 4 latent  <────  concat/cross-attn conditioning [PAPER]
        │                                 │
        ▼                                 │
  Diffusion U-Net (denoising, T=1000) <───┘
        │
        ▼
  Denoised latent (64x64x64x4)
        │
        ▼
   3D VAE Decoder
        │
        ▼
  Synthetic CT volume (256^3)
```
**[PAPER]** Fig. 1 confirms this exact pipeline shape.

## 1. 3D VAE

| Property | Value | Source |
|---|---|---|
| Input | 256×256×256×1 | [PAPER] |
| Output latent | 64×64×64×4 | [PAPER] — "compressing 4× spatial resolution and expanding 4× feature dimensionality" |
| Spatial compression factor | 4 | [PAPER] |
| Resolution levels | 3 | [PAPER] |
| Residual blocks / level | 1 | [PAPER] |
| Channel symmetry | encoder channels == decoder channels | [PAPER] (exact values not given) |
| Channel widths | [64, 128, 256] | [INFERRED] from MAISI's lightweight config, ref [8] |
| Base architecture | "lightweight variant of the MAISI architecture" | [PAPER] |
| Attention in VAE | none | [INFERRED] — MAISI's lightweight variant omits VAE self-attention |
| Normalization | GroupNorm | [INFERRED] — MONAI/MAISI default |

**Loss**: `L_VAE = L_MAE(x, x̂) + L_LPIPS(x, x̂) + L_ADV(x, x̂) + L_KL(E(x))` — **[PAPER]**, exact formula given.
- `L_MAE`: pixelwise L1 — [PAPER]
- `L_LPIPS`: perceptual similarity — [PAPER] cites ref [25] (Zhang et al.). LPIPS has no native
  3D network; **[INFERRED]** we apply LPIPS per 2D slice across all 3 axes and average ("2.5D LPIPS").
- `L_ADV`: adversarial loss vs a discriminator, prevents unrealistic artifacts — [PAPER] cites
  refs [8, 7]. Discriminator architecture **[INFERRED]**: 3D PatchGAN, per MAISI/Rombach convention.
- `L_KL`: standard VAE KL regularization on `E(x)` — [PAPER]. KL weight **[INFERRED]** (small,
  1e-6, standard for KL-regularized/"KL-VAE" autoencoders as opposed to VQ-VAEs).

## 2. Diffusion U-Net

| Property | Value | Source |
|---|---|---|
| Resolution levels | 5 | [PAPER] |
| Residual blocks / level | 2 | [PAPER] |
| Skip connections | additive (not concatenation) | [PAPER] — explicitly cites PatchDDM ref [2] for this choice, motivated by memory reduction |
| Conditioning mechanism | cross-attention, re-injected at multiple resolution levels | [PAPER] |
| Channel widths | [64, 128, 256, 384, 512] | [INFERRED] |
| Attention placement | coarser 3 of 5 levels | [INFERRED], standard LDM practice |
| Prediction target | velocity (v-prediction) | [PAPER] cites refs [12, 23] |
| Loss weighting | Min-SNR-γ | [PAPER] cites ref [9]; exact formula reproduced below |
| Noise schedule | linear, β₁=1e-4 → β_T=0.02 | [PAPER] |
| Timesteps (train & inference) | T = 1000 | [PAPER] — "Inference uses the same number of steps to prioritize sample quality" |

**Min-SNR-γ loss** (reproduced verbatim from paper, Section 2):

```
L_min-SNR = γ(SNR_t) * |v̂_t(z_t, m) - v_t|²
```
where `z_t` is the noisy latent, `m` the conditioning mask, `γ(·)` the Min-SNR weight, and
`v_t, v̂_t` the target/predicted velocities. γ value itself (commonly 5 in the original Min-SNR
paper) is **[INFERRED]** since LAND doesn't restate it.

## 3. Conditioning masks

**[PAPER]**, Section 2, verbatim: *"Spatial and textural cues are encoded by assigning lungs a
value of 0.5 and nodules 1–5 (non-solid to solid). Masks are normalized to [0,1], downsampled
four times via 3D max pooling, concatenated with the noisy latent, and injected into U-Net
cross-attention layers."*

Concretely:
- Background voxels: 0
- Lung voxels (non-nodule): 0.5
- Nodule voxels: raw texture score (int, 1–5)
- Then the whole mask tensor is normalized into [0,1].

**[INFERRED — see 06_open_questions.md item 1]**: exact normalization formula. We use:
`lung = 0.5` (already in [0,1], left as-is), `nodule = texture_score / 5.0` (maps 1→0.2 ... 5→1.0),
overwriting the lung value at nodule voxel locations. Empty/healthy nodule channel = all zeros.

Mask injection is **both** concatenation with the noisy latent AND cross-attention — [PAPER]
states both explicitly, which is somewhat unusual (most LDM conditioning uses one or the other).
Mask is spatially downsampled 4x via max pooling to match the 64³ latent resolution before
concatenation — [PAPER].

## 4. Ablation findings that constrain design choices

**[PAPER]**, Section 3 Discussion / Table 1:
- Nodule-mask-only conditioning → nodules placed outside lungs (anatomically wrong). This is why
  lung mask is mandatory, not optional, in the conditioning tensor.
- nodule+lung+texture gives control over solidity without retraining (Fig. 3) — confirms texture
  score is a first-class conditioning signal, not just metadata.
