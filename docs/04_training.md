# 04 — Training Specification

## Stage A — VAE (trained independently, first)

**[PAPER]** Section 3 "Implementation Details":
- Epochs: 100
- Optimizer: AdamW
- Learning rate: 1e-4
- Batch size: 1
- Hardware used by authors: single Nvidia Grid A100-20C (20GB)

**[INFERRED]** additions needed to actually fit training on hardware we have access to (paper
doesn't mention these, they're standard tooling, not a deviation from the *model*):
- Gradient checkpointing: on
- Mixed precision: bf16
- Gradient accumulation: 4 steps (effective batch stays 1 sample/step, just spread over more
  forward/backward passes before the optimizer step, to reduce peak memory)
- LR scheduler: none stated by paper → constant LR, matching literal paper description, unless
  training instability forces a change (would be logged here if so)

## Stage B — Diffusion U-Net (trained on frozen VAE's latents)

**[PAPER]**:
- Steps: 500,000
- Optimizer: AdamW
- Learning rate: 1e-5
- Batch size: 1
- Timesteps: T=1000, linear schedule β₁=1e-4 → β_T=0.02

VAE weights are frozen during this stage — **[INFERRED but standard]**: paper describes them as
trained independently, and doesn't mention joint fine-tuning, so we freeze the VAE encoder before
diffusion training starts.

**[INFERRED]** EMA (decay 0.9999) on U-Net weights for sampling — standard diffusion-training
practice, not mentioned by paper. If this measurably changes results relative to a no-EMA run,
that's worth noting since it wouldn't be a "paper-faithful" result strictly speaking.

## What "faithful reproduction" means operationally here

We will run two variants where it's cheap to do so:
1. **Literal**: exactly the settings above, no extras (no EMA, no LR schedule, no CFG dropout).
2. **Practical**: literal settings + the [INFERRED] stabilization tools above.

If (1) fails to train stably (a real risk at batch size 1 with no LR warmup over 500k steps),
we fall back to (2) and document that we deviated, rather than silently only ever running (2)
and calling it a reproduction.

## Compute plan

- Paper's reference hardware: single 20GB A100 (arXiv) / 10–16GB per the published Sci Rep
  revision.
- Available hardware: local GTX 1080 (8GB) for debugging/dev only — cannot fit training.
- Training requires a rented GPU (24GB card comfortably covers both stages). Budget roughly:
  VAE 100 epochs + diffusion 500k steps at batch 1 is a substantial compute bill — will scope an
  MVP run (reduced epochs/steps on a data subset) before committing to a full-scale run, to
  validate the pipeline is correct before spending the bulk of the compute budget.
