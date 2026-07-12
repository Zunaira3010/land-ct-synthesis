# 06 — Open Questions / Assumption Registry

This is the authoritative, structured version of what `NOTES.md` first logged informally.
Every row here is a place the paper is silent and we made an engineering call. **Update the
Status column as each gets validated** (e.g. by qualitative comparison against Fig. 2/3 of the
paper, or by author response).

| # | Question | Our resolution | Resolved via | Status |
|---|---|---|---|---|
| 1 | Exact nodule-value normalization formula | lung=0.5 constant; nodule=texture/5.0, overwrites lung at nodule voxels; empty=0.0 | Closest literal reading of paper's wording. WDM paper (ref [6]) is unconditional (no masks at all) so it can't corroborate this directly; cross-checked instead against the LAND paper's own wording ("lungs=0.5, nodules 1–5 normalized to [0,1]"), and this is the only sensible reading of that spec | **Accepted** — no better source exists to verify further; revisit only if reconstruction/generation quality on nodule regions looks wrong |
| 2 | VAE channel widths | [64, 128, 256] | MAISI lightweight config (paper explicitly names MAISI as base) | Unverified |
| 3 | VAE attention | None | MAISI lightweight variant convention | Unverified |
| 4 | Discriminator architecture (L_ADV) | 3D PatchGAN, base_channels=32, 3 layers | MAISI/Rombach convention | Unverified |
| 5 | KL loss weight | 1e-6 | Standard KL-VAE convention | Unverified |
| 6 | U-Net channel widths | [64,128,256,384,512] | Standard LDM progression over 5 levels | Unverified |
| 7 | U-Net attention placement | Coarsest 3 of 5 levels | Standard LDM practice | Unverified |
| 8 | Min-SNR γ value | 5.0 | Original Min-SNR-γ paper's proposed value (ref [9]) | Unverified |
| 9 | Mask encoder feeding cross-attention | Option A: flatten the 64³ downsampled mask directly to a token sequence, linear-project to `cross_attention_dim`. (Rejected Option B: small CNN encoder first, then flatten — adds complexity with no clear benefit for a 3-value, spatially-simple mask.) | Design decision — mask is small and semantically simple (0 / 0.5 / texture÷5), so spatial location itself carries the signal; no evidence a learned CNN front-end would help | **Resolved** — ready to implement in `unet.py` |
| 10 | HU window for preprocessing | [-1000, 400] | WDM paper (Friedrich et al., MICCAI Workshops 2024), Section 4.1.1: clips below −1000 HU and above the upper 0.1 percentile intensity, then normalizes to [−1, 1]. Our −1000 lower bound matches exactly; our fixed +400 upper bound approximates their data-driven 0.1-percentile cutoff (empirically ~400–500 HU for LIDC-IDRI, i.e. bone/implants) | **Accepted** — difference is minor and limited to dense bone/metal voxels, irrelevant to lung tissue and nodules; cached data is not wrong enough to justify re-caching |
| 11 | Train/val/test split ratio | 85/10/5, patient-level, seed=42 | Not specified by paper at all; our own choice | N/A — genuinely our decision, no paper answer expected |
| 12 | NLST nodule-mask generation ("ad-hoc U-Net") | Not reproduced — using held-out LIDC-IDRI instead for generalization testing | Paper's NLST mask-generation model was never released | Accepted deviation, documented |
| 13 | EMA / grad checkpointing / mixed precision / grad accumulation | Added, not paper-specified | Standard training stabilization tooling | Accepted as non-architectural tooling, not a fidelity concern |
| 14 | FID feature extractor | Paper uses Med3D ResNet-50 (ref [4]) pretrained on 23 medical imaging datasets — need to locate/reproduce this specific checkpoint | Not yet started | **Open — blocks Stage "Evaluation" only, not training** |
| 15 | LR scheduler | None (constant LR) | Literal reading — paper doesn't mention one | Unverified |

## Priority order for resolution

**#1, #9, and #10 are resolved** (see table above — nodule normalization accepted as the only
sensible reading of the paper's wording, HU window cross-checked against the actual WDM paper
text, mask encoder design decided as Option A). None of these block current work.

Remaining unresolved rows (#2–8, #15) are architectural detail that primarily affects sample
quality/compute efficiency rather than correctness — reasonable to proceed with current choices
and revisit only if Stage 5 (VAE reconstruction check) or later diffusion-stage qualitative
results look clearly wrong. **#14** (FID feature extractor) is the only other row with an
explicit blocker note — it blocks the Evaluation stage only, not training.
