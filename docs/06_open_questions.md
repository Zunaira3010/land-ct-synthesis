# 06 — Open Questions / Assumption Registry

This is the authoritative, structured version of what `NOTES.md` first logged informally.
Every row here is a place the paper is silent and we made an engineering call. **Update the
Status column as each gets validated** (e.g. by qualitative comparison against Fig. 2/3 of the
paper, or by author response).

| # | Question | Our resolution | Resolved via | Status |
|---|---|---|---|---|
| 1 | Exact nodule-value normalization formula | lung=0.5 constant; nodule=texture/5.0, overwrites lung at nodule voxels; empty=0.0 | Closest literal reading of paper's wording | **Unverified** — highest priority to confirm |
| 2 | VAE channel widths | [64, 128, 256] | MAISI lightweight config (paper explicitly names MAISI as base) | Unverified |
| 3 | VAE attention | None | MAISI lightweight variant convention | Unverified |
| 4 | Discriminator architecture (L_ADV) | 3D PatchGAN, base_channels=32, 3 layers | MAISI/Rombach convention | Unverified |
| 5 | KL loss weight | 1e-6 | Standard KL-VAE convention | Unverified |
| 6 | U-Net channel widths | [64,128,256,384,512] | Standard LDM progression over 5 levels | Unverified |
| 7 | U-Net attention placement | Coarsest 3 of 5 levels | Standard LDM practice | Unverified |
| 8 | Min-SNR γ value | 5.0 | Original Min-SNR-γ paper's proposed value (ref [9]) | Unverified |
| 9 | Mask encoder feeding cross-attention | TBD in Stage 4 implementation | — | **Open — needs design decision, not yet resolved** |
| 10 | HU window for preprocessing | [-1000, 400] | Standard lung window, placeholder pending WDM repo check | **Open — action item, see 03_preprocessing.md** |
| 11 | Train/val/test split ratio | 85/10/5, patient-level, seed=42 | Not specified by paper at all; our own choice | N/A — genuinely our decision, no paper answer expected |
| 12 | NLST nodule-mask generation ("ad-hoc U-Net") | Not reproduced — using held-out LIDC-IDRI instead for generalization testing | Paper's NLST mask-generation model was never released | Accepted deviation, documented |
| 13 | EMA / grad checkpointing / mixed precision / grad accumulation | Added, not paper-specified | Standard training stabilization tooling | Accepted as non-architectural tooling, not a fidelity concern |
| 14 | FID feature extractor | Paper uses Med3D ResNet-50 (ref [4]) pretrained on 23 medical imaging datasets — need to locate/reproduce this specific checkpoint | Not yet started | **Open — blocks Stage "Evaluation" only, not training** |
| 15 | LR scheduler | None (constant LR) | Literal reading — paper doesn't mention one | Unverified |

## Priority order for resolution

1. **#1 (nodule normalization)** — wrong here silently corrupts every training sample's
   conditioning signal. Resolve before Stage 3 training, ideally by author email response.
2. **#10 (HU window)** — wrong here shifts the whole intensity distribution the VAE learns.
   Resolve by pulling WDM's actual preprocessing code before Stage 2 finalizes.
3. **#9 (mask encoder)** — blocks Stage 4 implementation directly, needs a decision either way.
4. Everything else is architectural detail that primarily affects sample quality/compute
   efficiency rather than correctness — reasonable to proceed with current choices and revisit
   only if Stage 3/4 qualitative results look clearly wrong.
