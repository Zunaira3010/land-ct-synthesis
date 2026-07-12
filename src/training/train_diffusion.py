"""
src/training/train_diffusion.py
=================================
Diffusion U-Net training loop for LAND (Stage B) -- consumes the latents precomputed by
scripts/precompute_latents.py, never touches the VAE itself.

Paper (Section 3, "Implementation Details" + Section 2 "Diffusion process"):
  - T = 1000, linear beta schedule, v-prediction, Min-SNR-gamma (gamma=5) loss weighting
  - AdamW, lr = 1e-5, weight_decay = 1e-5, batch size = 1, 500k steps
  - Mask conditioning: concat + cross-attention (already wired inside UNet3D itself -- this
    script just has to pass `mask` into the forward call, not implement the conditioning logic)

Our additions (not paper-specified, needed to fit an 8GB dev card -- same reasoning as
train_vae.py, one level further downstream):
  - Patch-based latent training (default 32^3 out of the full 64^3 latent) -- see
    src/data/latent_dataset.py's module docstring for why even a full 64^3 latent forward+
    backward doesn't fit here (paper's own reference config needs 10-16GB minimum at full res).
  - Mixed precision (bf16), gradient checkpointing, gradient accumulation (4 steps) --
    identical reasoning and identical pattern to train_vae.py.
  - EMA of model weights (decay=0.9999) -- standard diffusion-training practice, used for
    validation and (later) sampling; not explicitly stated in the paper but near-universal for
    diffusion models of this style.  [INFERRED]
  - Classifier-free-guidance-style conditioning dropout (cfg_dropout_prob=0.1): the mask is
    zeroed out entirely for a training example with this probability, so the model also learns
    to generate unconditionally -- enables classifier-free guidance at sampling time later.
    Not stated in the paper; configs/diffusion_config.yaml flags it as optional.  [INFERRED]

Usage
-----
    # Full run (500k steps, all cached latents, patch settings from diffusion_config.yaml):
    python -m src.training.train_diffusion \\
        --manifest data/processed/land/manifest.json \\
        --latent-dir data/processed/land/latents \\
        --out-dir checkpoints/diffusion

    # Quick smoke-test (few hundred steps, first 4 patients, small patch -- confirms plumbing
    # works before committing to a long real run):
    python -m src.training.train_diffusion \\
        --manifest data/processed/land/manifest.json \\
        --latent-dir data/processed/land/latents \\
        --out-dir checkpoints/diffusion_smoke \\
        --num-steps 200 --limit 4 --patch-size 16 --log-every 10 --val-every 100 --checkpoint-every 100

    # Resume an interrupted run:
    python -m src.training.train_diffusion ... --resume checkpoints/diffusion/latest.pt
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.lidc_dataset import LIDCSample
from src.data.latent_dataset import LatentVolumeDataset
from src.models.unet import UNet3D
from src.training.diffusion_schedule import VPredictionSchedule
from src.utils.config import load_config, get_device


# ---------------------------------------------------------------------------
# Time formatting -- duplicated from train_vae.py rather than imported, since importing that
# module drags in its (unrelated) TensorBoard dependency just for these two small functions.
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _format_clock_eta(seconds_from_now: float) -> str:
    import datetime
    eta = datetime.datetime.now() + datetime.timedelta(seconds=max(0, seconds_from_now))
    return eta.strftime("%a %H:%M")


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate(batch: list[dict]) -> dict:
    latent = torch.from_numpy(np.stack([b["latent"] for b in batch])).float()
    mask = torch.from_numpy(np.stack([b["mask"] for b in batch])).float()
    return {
        "latent": latent,
        "mask": mask,
        "is_healthy": [b["is_healthy"] for b in batch],
        "patient_id": [b["patient_id"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Manifest -> dataset (real manifest.json format: dict keyed by split)
# ---------------------------------------------------------------------------

def _build_dataset(
    manifest_path: str,
    latent_dir: str,
    split: str,
    limit: int | None,
    patch_size: int | None,
    nodule_centered_prob: float,
    val_seed: int | None,
) -> LatentVolumeDataset:
    with open(manifest_path) as f:
        splits = json.load(f)
    entries = splits[split]
    if limit:
        entries = entries[:limit]
    samples = [
        LIDCSample(
            patient_id=e["patient_id"],
            scan_dicom_dir=e["scan_dicom_dir"],
            is_healthy=e["is_healthy"],
            nodule_texture_score=e.get("nodule_texture_score"),
        )
        for e in entries
    ]
    return LatentVolumeDataset(
        samples,
        latent_dir=latent_dir,
        patch_size=patch_size,
        nodule_centered_prob=nodule_centered_prob,
        val_seed=val_seed,
    )


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model parameters. Holds its own shadow copy of the
    state_dict; update() is called once per optimiser step (not per micro-batch, if using
    gradient accumulation -- see the training loop)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()  # non-float buffers (e.g. int counters): just copy

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, state: dict) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> dict:
        """Returns the model's ORIGINAL state_dict (so the caller can restore it after using EMA
        weights temporarily, e.g. for validation), and loads EMA weights into `model` in place."""
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # Cast each EMA shadow tensor back to the model's own dtype before loading -- the shadow
        # is kept in float32 for accumulation precision, but model buffers/params may be a
        # different dtype (e.g. a training run using bf16-cast weights, though normally these
        # stay float32 and only activations are cast under autocast).
        cast_shadow = {k: v.to(dtype=original[k].dtype) for k, v in self.shadow.items()}
        model.load_state_dict(cast_shadow)
        return original


# ---------------------------------------------------------------------------
# Graceful-interrupt handling -- identical pattern to train_vae.py
# ---------------------------------------------------------------------------

_shutdown_requested = [False]


def _request_shutdown(signum, frame) -> None:
    if not _shutdown_requested[0]:
        print(f"\n  [signal {signum} received] Saving a checkpoint and stopping after the "
              f"current step (send the signal again to force-quit without saving)...", flush=True)
    _shutdown_requested[0] = True


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _request_shutdown)
    try:
        signal.signal(signal.SIGTERM, _request_shutdown)
    except (ValueError, AttributeError, OSError):
        pass
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _request_shutdown)


class TrainingInterrupted(Exception):
    pass


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    out_dir: Path,
    global_step: int,
    unet: UNet3D,
    ema: EMA,
    optim: torch.optim.Optimizer,
    scaler: GradScaler,
    best_val_loss: float,
    filename: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / filename
    tmp_path = out_dir / (filename + ".tmp")
    torch.save(
        {
            "global_step": global_step,
            "unet_state": unet.state_dict(),
            "ema_state": ema.state_dict(),
            "optim_state": optim.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_loss": best_val_loss,
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)
    print(f"  Saved checkpoint: {final_path}", flush=True)


def _load_checkpoint(
    ckpt_path: str,
    unet: UNet3D,
    ema: EMA,
    optim: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
) -> tuple[int, float]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    unet.load_state_dict(ckpt["unet_state"])
    ema.load_state_dict(ckpt["ema_state"])
    if optim is not None and "optim_state" in ckpt:
        optim.load_state_dict(ckpt["optim_state"])
    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["global_step"], ckpt.get("best_val_loss", float("inf"))


# ---------------------------------------------------------------------------
# Infinite dataloader cycling (step-based training, not epoch-based)
# ---------------------------------------------------------------------------

def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Loss computation (shared by train step and val)
# ---------------------------------------------------------------------------

def _compute_loss(
    unet: UNet3D,
    schedule: VPredictionSchedule,
    latent: torch.Tensor,
    mask: torch.Tensor,
    device: str,
    cfg_dropout_prob: float,
) -> torch.Tensor:
    b = latent.shape[0]
    eps = torch.randn_like(latent)
    t = schedule.sample_timesteps(b, device=device)
    z_t = schedule.add_noise(latent, eps, t)
    v_target = schedule.velocity_target(latent, eps, t)

    if cfg_dropout_prob > 0.0:
        drop = (torch.rand(b, device=device) < cfg_dropout_prob).view(-1, 1, 1, 1, 1)
        mask = torch.where(drop, torch.zeros_like(mask), mask)

    v_pred = unet(z_t, t, mask)

    per_sample_loss = ((v_pred - v_target) ** 2).flatten(1).mean(dim=1)  # (B,)
    weight = schedule.min_snr_loss_weight(t)  # (B,)
    return (per_sample_loss * weight).mean()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--latent-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--diffusion-config", default="configs/diffusion_config.yaml")
    p.add_argument("--patch-size", type=int, default=32, help="Latent-space patch size (out of the full 64^3)")
    p.add_argument("--nodule-centered-prob", type=float, default=0.7)
    p.add_argument("--limit", type=int, default=None, help="Only use the first N train patients (debugging)")
    p.add_argument("--num-steps", type=int, default=None, help="Override config's training.num_steps")
    p.add_argument("--log-every", type=int, default=None, help="Override config's training.log_every_n_steps")
    p.add_argument("--val-every", type=int, default=None, help="Override config's training.val_every_n_steps")
    p.add_argument("--checkpoint-every", type=int, default=None, help="Override config's training.checkpoint_every_n_steps")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    cfg = load_config(diffusion=args.diffusion_config)
    dcfg = cfg.diffusion

    num_steps = args.num_steps or dcfg.training.num_steps
    log_every = args.log_every or dcfg.training.log_every_n_steps
    val_every = args.val_every or dcfg.training.val_every_n_steps
    checkpoint_every = args.checkpoint_every or dcfg.training.checkpoint_every_n_steps
    grad_accum = dcfg.training.gradient_accumulation_steps
    cfg_dropout_prob = dcfg.training.cfg_dropout_prob

    train_ds = _build_dataset(
        args.manifest, args.latent_dir, split="train", limit=args.limit,
        patch_size=args.patch_size, nodule_centered_prob=args.nodule_centered_prob, val_seed=None,
    )
    val_ds = _build_dataset(
        args.manifest, args.latent_dir, split="val", limit=None,
        patch_size=args.patch_size, nodule_centered_prob=args.nodule_centered_prob, val_seed=42,
    )
    print(f"Train patients: {len(train_ds)}  Val patients: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=dcfg.training.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=dcfg.training.batch_size, shuffle=False,
        collate_fn=_collate, num_workers=0,
    )
    train_iter = _cycle(train_loader)

    unet = UNet3D.from_config(dcfg).to(device)
    if dcfg.training.gradient_checkpointing:
        # from_config only reads use_checkpointing from the `architecture:` block, which
        # doesn't define that key -- gradient_checkpointing actually lives under `training:`
        # in diffusion_config.yaml. use_checkpointing is a plain per-module attribute (checked
        # at forward time, not baked into the graph at construction), so it's safe to flip on
        # every submodule after the fact rather than needing to touch unet.py itself.
        n_toggled = 0
        for module in unet.modules():
            if hasattr(module, "use_checkpointing"):
                module.use_checkpointing = True
                n_toggled += 1
        print(f"Gradient checkpointing enabled on {n_toggled} submodules "
              f"(training.gradient_checkpointing=true in {args.diffusion_config})")
    ema = EMA(unet, decay=dcfg.training.ema_decay)
    optim = torch.optim.AdamW(
        unet.parameters(), lr=dcfg.training.learning_rate, weight_decay=dcfg.training.weight_decay,
    )
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and dcfg.training.mixed_precision == "bf16") else torch.float32
    scaler = GradScaler(enabled=use_amp)

    schedule = VPredictionSchedule(
        num_train_timesteps=dcfg.diffusion_process.num_train_timesteps,
        beta_start=dcfg.diffusion_process.beta_start,
        beta_end=dcfg.diffusion_process.beta_end,
        min_snr_gamma=dcfg.diffusion_process.min_snr_gamma,
        device=device,
    )

    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        start_step, best_val_loss = _load_checkpoint(args.resume, unet, ema, optim, scaler)
        print(f"Resumed from step {start_step} (best_val_loss={best_val_loss:.4f}): {args.resume}")

    _install_signal_handlers()
    out_dir = Path(args.out_dir)

    print(f"\n{'='*60}\n  LAND Diffusion Training\n  device={device}  num_steps={num_steps}  amp={use_amp}\n"
          f"  grad_accum={grad_accum}  patch_size={args.patch_size}  cfg_dropout={cfg_dropout_prob}\n{'='*60}\n")

    run_start = time.time()
    step = start_step
    optim.zero_grad(set_to_none=True)

    try:
        while step < num_steps:
            accum_loss = 0.0
            for micro in range(grad_accum):
                batch = next(train_iter)
                latent = batch["latent"].to(device)
                mask = batch["mask"].to(device)

                with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                     dtype=amp_dtype, enabled=use_amp):
                    loss = _compute_loss(unet, schedule, latent, mask, device, cfg_dropout_prob)
                    loss_to_backward = loss / grad_accum

                scaler.scale(loss_to_backward).backward()
                accum_loss += loss.item()

            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            ema.update(unet)
            step += 1
            avg_loss = accum_loss / grad_accum

            if step % log_every == 0 or step == 1:
                elapsed = time.time() - run_start
                steps_done_this_run = step - start_step
                rate = elapsed / max(steps_done_this_run, 1)
                eta_seconds = rate * (num_steps - step)
                print(f"  step {step}/{num_steps} | loss {avg_loss:.4f} | "
                      f"elapsed {_format_duration(elapsed)} | "
                      f"ETA {_format_duration(eta_seconds)} | "
                      f"finish {_format_clock_eta(eta_seconds)}", flush=True)

            if step % val_every == 0:
                original_state = ema.copy_to(unet)
                unet.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        latent = batch["latent"].to(device)
                        mask = batch["mask"].to(device)
                        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                             dtype=amp_dtype, enabled=use_amp):
                            val_losses.append(
                                _compute_loss(unet, schedule, latent, mask, device, cfg_dropout_prob=0.0).item()
                            )
                val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
                unet.load_state_dict(original_state)  # restore non-EMA weights for continued training
                unet.train()
                print(f"  [val @ step {step}] EMA val loss: {val_loss:.4f}", flush=True)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(out_dir, step, unet, ema, optim, scaler, best_val_loss,
                                      filename=f"diffusion_step{step:07d}_best.pt")

            if step % checkpoint_every == 0:
                _save_checkpoint(out_dir, step, unet, ema, optim, scaler, best_val_loss,
                                  filename="latest.pt")

            if _shutdown_requested[0]:
                _save_checkpoint(out_dir, step, unet, ema, optim, scaler, best_val_loss, filename="latest.pt")
                raise TrainingInterrupted(f"Stopped by signal at step {step}")

    except TrainingInterrupted as e:
        print(f"\n{e}\nSafe to resume with --resume {out_dir}/latest.pt")
        return

    _save_checkpoint(out_dir, step, unet, ema, optim, scaler, best_val_loss, filename="latest.pt")
    _save_checkpoint(out_dir, step, unet, ema, optim, scaler, best_val_loss,
                      filename=f"diffusion_step{step:07d}_final.pt")
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}\nCheckpoints in: {out_dir}")


if __name__ == "__main__":
    main()
