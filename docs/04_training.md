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
- **UPDATED — this section was stale.** It previously assumed the local GTX 1080 (8GB) could
  only be used for debugging/dev, and that training would require a rented GPU. That's been
  disproven directly: Stage A (VAE) completed a full, real 100-epoch training run on this exact
  card (best val loss 0.0725 at epoch 84), and Stage B (diffusion) has run real, successful
  smoke-test training steps on it too (see below). No rented GPU has been needed for either
  stage. The patch-based training strategy (Stage A: 96³ CT patches; Stage B: patch-cropped
  latents, see below) is *why* this fits — not a fallback that was avoided.

## Stage B step count — [DEVIATION] from the paper's literal 500,000 steps

**[PAPER]** specifies 500,000 training steps, batch size 1, on a 20GB A100.

**Measured on the actual hardware (GTX 1080, 8GB, patch-cropped 32³ latents out of the full
64³, gradient checkpointing + bf16 + grad-accum=4):** ~10s/step. At the literal 500k-step
target, that's **~58 days of continuous, uninterrupted training** — not viable on a card shared
with a labmate, and arguably not a meaningful "faithful reproduction" target anyway, since:
  1. The paper's compute budget was tuned for its own (larger) training dataset; we have 61
     training patients, a small dataset by comparison, where 500k steps risks overfitting long
     before it risks undertraining.
  2. A secondary finding worth recording: per-step time only grew ~3.6x (2.8s → 10s) going from
     patch_size=16 to patch_size=32 (8x more voxels), suggesting a meaningful chunk of per-step
     cost is fixed overhead (EMA update over all 124M params, optimizer step, data loading) —
     not pure conv compute. This means further shrinking patch_size is unlikely to buy
     proportional speedup, and isn't the lever to pull if more speed is needed later.

**Practical deviation adopted:** rather than committing to a fixed step count up front (neither
the literal 500k, nor an arbitrarily-smaller guessed number), train in checkpointed milestones
and evaluate actual generated-sample quality at each one, stopping when quality plateaus or
starts degrading (the standard small-dataset overfitting signal) rather than at a predetermined
step count. First milestone: ~20,000–30,000 steps (~2.3–3.5 days at the measured rate) as an
initial look, not a target believed sufficient on its own.

This mirrors the same kind of hardware-driven, explicitly-documented call already made at Stage
A (patch-based training instead of full-volume) and in HU-window verification (Q#10) — a
reasoned adaptation to real constraints, logged here rather than silently deviating and calling
the result a literal reproduction.
