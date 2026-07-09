# Reimplementation notes & open assumptions

This file exists because LAND has no released code, so several implementation details had to
be inferred rather than copied. Every assumption below is called out at its source in the
relevant config file too (`configs/*.yaml`). Treat this as a running log — update it whenever
a stage is validated or an assumption turns out to be wrong.

## Confirmed from the paper (high confidence)

- VAE: 3 resolution levels, 1 residual block/level, 4x spatial compression, 4 latent channels,
  same channel count encoder/decoder, loss = L_MAE + L_LPIPS + L_ADV + L_KL.
- U-Net: 5 resolution levels, 2 residual blocks/level, additive skip connections, cross-attention
  conditioning, v-prediction, Min-SNR-γ weighting, linear noise schedule β₁=1e-4 → β_T=0.02, T=1000.
- Mask semantics: lungs = 0.5, nodules = texture score 1 (non-solid) to 5 (solid), masks
  normalized to [0,1], downsampled 4x via 3D max pooling before injection.
- Training: VAE 100 epochs, U-Net 500k steps, both AdamW batch size 1, VAE lr=1e-4, U-Net lr=1e-5.
- Data: LIDC-IDRI (1,010 volumes) for training, NLST subset (881 volumes) for unseen-mask
  inference only. Lung segmentation via the pretrained Hofmanninger U-Net (ref [13], = `lungmask`
  package). Nodule textures scored 1-5 by four LIDC-IDRI radiologists.
- Hardware: arXiv v1 says trained on a single 20GB A100; published Sci Rep version revises this
  down to 10-16GB training / <8GB inference.

## Assumptions we had to make (LOW confidence — validate before trusting downstream results)

1. **Exact nodule value normalization.** Paper says lungs=0.5 and nodules 1-5, "normalized to
   [0,1]," but never gives the exact formula relating the two. We assume: lung voxels = 0.5
   constant; nodule voxels = texture_score / 5.0, overwriting lung value at nodule locations;
   empty nodule channel = 0.0 for healthy samples. This is the single most important assumption
   in the whole pipeline since it directly controls conditioning — worth explicitly asking the
   corresponding author about if we get a reply.

2. **VAE channel widths.** Paper says "same number of channels in both encoder and decoder" but
   never lists the actual channel list. We started from MAISI's published lightweight VAE config
   ([64, 128, 256]) since LAND explicitly says it's a lightweight MAISI variant. Revisit if
   reconstruction quality (Stage 3) doesn't qualitatively match the paper's Fig. 2 samples.

3. **Discriminator architecture for L_ADV.** Not specified. Using a standard 3D PatchGAN as used
   in MAISI / Rombach et al. Adversarial loss weight (0.1) and warmup schedule are also our
   choices, not the paper's.

4. **U-Net channel widths and attention placement.** Not specified beyond "5 resolution levels,
   2 residual blocks/level." We used a standard LDM-style progression ([64,128,256,384,512]) with
   cross-attention only at the coarser 3 levels — common practice, not confirmed against LAND.

5. **Preprocessing pipeline "as in [6]" (WDM paper).** We haven't yet pulled WDM's actual
   preprocessing script to confirm the exact HU windowing / resampling order. Current HU window
   ([-1000, 400]) is a standard lung window, not verified against [6] yet — do this before
   Stage 2 preprocessing is finalized.

6. **cross_attention_dim / mask encoder.** The paper injects the mask via both concatenation
   *and* cross-attention ("concatenated with the noisy latent, and injected into U-Net
   cross-attention layers") — slightly unusual to do both. We've configured for both but the
   exact mask encoder architecture feeding the cross-attention layers isn't specified in the
   paper. Needs a design decision in Stage 4, flagged there.

7. **EMA, gradient checkpointing, mixed precision, gradient accumulation, CFG dropout.** None of
   these are mentioned in the paper. They're standard tools needed to fit training into memory
   budgets we actually have access to (rented 24GB card vs. their institutional 20GB A100), and
   to stabilize training generally. They shouldn't change what the model learns in principle, but
   they're not part of a "faithful" reproduction in the strictest sense — flagging so nobody
   mistakes them for paper-sourced choices later.

## Known unknowns still to resolve

- Exact evaluation protocol details for FID (paper uses a Med3D ResNet-50 feature extractor,
  ref [4] — need to locate/reimplement that specific feature extractor rather than using generic
  Inception features, or FID numbers won't be comparable to Table 1).
- No confirmed anonymized code repo exists (checked the OpenReview submission's listed link —
  dead / placeholder, see conversation history).
