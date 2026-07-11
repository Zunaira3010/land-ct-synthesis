"""
src/training/train_vae.py
=========================
VAE training loop for LAND (Stage A).

Paper (Section 3, "Implementation Details"):
  - Trained independently for 100 epochs
  - AdamW, lr = 1e-4, batch size = 1
  - Loss: L_MAE + L_LPIPS + L_ADV + L_KL
  - Hardware: single Nvidia Grid A100-20C (20GB)

Our additions (not paper-specified, but needed to fit on real hardware):
  - Mixed precision (bf16)                  [INFERRED]
  - Gradient checkpointing                  [INFERRED]
  - Gradient accumulation (4 steps)         [INFERRED — for smaller GPUs]
  - Discriminator warm-up (10 epochs)       [INFERRED]
  - Checkpoint + val every N epochs         [INFERRED]
  - TensorBoard logging                     [INFERRED]
  - Patch-based training (default 128^3, nodule-aware sampling) [ADDED — full 256^3 OOMs on the
    first encoder conv at 54GB, checkpointing doesn't help since that's a forward-pass op; see
    configs/data_config.yaml `patch:` and NOTES.md]

See docs/04_training.md for the "literal vs practical" discussion.

Usage
-----
    # Full run (100 epochs, all data, patch settings from data_config.yaml -- default 128^3):
    python -m src.training.train_vae \
        --manifest data/processed/land/manifest.json \
        --cache-dir data/processed/land/cache \
        --out-dir checkpoints/vae

    # Quick smoke-test (2 epochs, first 8 patients, 128^3 patches -- fits on an 8GB card):
    python -m src.training.train_vae \
        --manifest data/processed/land/manifest.json \
        --cache-dir data/processed/land/cache \
        --out-dir checkpoints/vae_smoke \
        --epochs 2 --limit 8 --no-lpips --patch-size 128

    # Even tighter memory budget:
    python -m src.training.train_vae ... --patch-size 96

    # Override any config value from the CLI (OmegaConf dotlist):
    python -m src.training.train_vae ... vae.training.learning_rate=5e-5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.lidc_dataset import LIDCSample, LIDCVolumeDataset
from src.models.vae import VAE
from src.models.discriminator import PatchGAN3D
from src.losses.vae_loss import VAELoss
from src.utils.config import load_config, save_resolved_config, get_device


# ---------------------------------------------------------------------------
# Collate: Dataset returns numpy dicts; we need torch tensors
# ---------------------------------------------------------------------------

def _collate(batch: list[dict]) -> dict:
    import numpy as np
    ct   = torch.from_numpy(np.stack([b["ct"]   for b in batch])).float()
    mask = torch.from_numpy(np.stack([b["mask"] for b in batch])).float()
    return {
        "ct":            ct,
        "mask":          mask,
        "is_healthy":    [b["is_healthy"]    for b in batch],
        "texture_score": [b["texture_score"] for b in batch],
        "patient_id":    [b["patient_id"]    for b in batch],
    }


# ---------------------------------------------------------------------------
# Manifest → Dataset
# ---------------------------------------------------------------------------

def _build_dataset(
    manifest_path: str,
    cache_dir: str,
    split: str = "train",
    limit: int | None = None,
    patch_size: int | None = None,
    nodule_centered_prob: float = 0.0,
    val_seed: int | None = None,
) -> LIDCVolumeDataset:
    with open(manifest_path) as f:
        splits = json.load(f)

    entries = splits[split]
    if limit:
        entries = entries[:limit]

    samples = [
        LIDCSample(
            patient_id           = e["patient_id"],
            scan_dicom_dir       = e["scan_dicom_dir"],
            is_healthy           = e["is_healthy"],
            nodule_texture_score = e.get("nodule_texture_score"),
        )
        for e in entries
    ]
    return LIDCVolumeDataset(
        samples,
        cache_dir=cache_dir,
        patch_size=patch_size,
        nodule_centered_prob=nodule_centered_prob,
        val_seed=val_seed,
    )


# ---------------------------------------------------------------------------
# Graceful-interrupt handling (shared-machine safety)
# ---------------------------------------------------------------------------
#
# IMPORTANT LIMITATION: this only helps for a *graceful* interrupt -- Ctrl+C (SIGINT), or on
# some setups a plain `kill <pid>` / closing a terminal cleanly (SIGTERM). It does NOT and
# CANNOT help if the process is force-killed (`taskkill /F`, Task Manager "End Task", or VS Code
# forcibly closing a terminal's process tree) -- the OS terminates the process immediately in
# that case and no Python code gets to run, signal handler or not. The real defense against a
# hard kill is the periodic step-level checkpoint below (`checkpoint_every_n_steps`), which
# bounds how much progress you can lose to *any* kind of kill, graceful or not, to at most that
# many optimiser steps -- not this signal handler.

_shutdown_requested = [False]


def _request_shutdown(signum, frame) -> None:
    # Signal handlers should do as little as possible -- just set a flag. The actual checkpoint
    # save happens back in normal Python code (the training loop), not here.
    if not _shutdown_requested[0]:
        print(f"\n  [signal {signum} received] Saving a checkpoint and stopping after the "
              f"current step (send the signal again to force-quit without saving)...", flush=True)
    _shutdown_requested[0] = True


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _request_shutdown)   # Ctrl+C, all platforms
    try:
        signal.signal(signal.SIGTERM, _request_shutdown)   # graceful kill; POSIX-reliable
    except (ValueError, AttributeError, OSError):
        pass
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _request_shutdown)  # Windows Ctrl+Break


class TrainingInterrupted(Exception):
    """Raised internally to unwind cleanly out of the epoch loop after an interrupt-save."""


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    out_dir: Path,
    epoch: int,
    vae: VAE,
    discriminator: PatchGAN3D,
    vae_optim: torch.optim.Optimizer,
    disc_optim: torch.optim.Optimizer,
    scaler_vae: GradScaler,
    scaler_disc: GradScaler,
    global_step: int,
    best_val_loss: float,
    epoch_complete: bool,
    tag: str = "",
    filename: str | None = None,
) -> None:
    """Save full training state. `epoch_complete=False` marks a mid-epoch safety checkpoint
    (periodic or interrupt-triggered) -- see `_load_checkpoint` for how that affects resume.

    Writes to a temp file and atomically renames it into place (`Path.replace`, atomic on both
    POSIX and Windows when src/dst are on the same volume), so a kill that lands mid-`torch.save`
    corrupts only the .tmp file, never the checkpoint you'd actually try to resume from.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = filename or f"vae_epoch{epoch:04d}{('_' + tag) if tag else ''}.pt"
    final_path = out_dir / fname
    tmp_path = out_dir / (fname + ".tmp")
    torch.save(
        {
            "epoch":          epoch,
            "epoch_complete": epoch_complete,
            "global_step":    global_step,
            "vae_state":      vae.state_dict(),
            "disc_state":     discriminator.state_dict(),
            "vae_optim":      vae_optim.state_dict(),
            "disc_optim":     disc_optim.state_dict(),
            "scaler_vae":     scaler_vae.state_dict(),
            "scaler_disc":    scaler_disc.state_dict(),
            "best_val_loss":  best_val_loss,
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)
    print(f"  Saved checkpoint: {final_path}"
          f"{' (mid-epoch safety checkpoint)' if not epoch_complete else ''}", flush=True)


def _load_checkpoint(
    ckpt_path: str,
    vae: VAE,
    discriminator: PatchGAN3D,
    vae_optim: torch.optim.Optimizer | None = None,
    disc_optim: torch.optim.Optimizer | None = None,
    scaler_vae: GradScaler | None = None,
    scaler_disc: GradScaler | None = None,
) -> tuple[int, int, float]:
    """Returns (start_epoch, global_step, best_val_loss).

    If the checkpoint was saved mid-epoch (epoch_complete=False -- a periodic `latest.pt` safety
    checkpoint, or one saved on interrupt), start_epoch is the SAME epoch it was interrupted in,
    so that epoch gets a full, fresh pass over the data rather than trying to resume partway
    through a DataLoader iterator (not easily resumable, especially with num_workers>0 and
    randomized patch sampling). You may redo a handful of already-applied optimiser steps this
    way, but you never lose the model/optimiser weights already achieved within that epoch --
    only the bookkeeping of exactly which batches were already seen.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vae.load_state_dict(ckpt["vae_state"])
    discriminator.load_state_dict(ckpt["disc_state"])
    if vae_optim is not None and "vae_optim" in ckpt:
        vae_optim.load_state_dict(ckpt["vae_optim"])
    if disc_optim is not None and "disc_optim" in ckpt:
        disc_optim.load_state_dict(ckpt["disc_optim"])
    if scaler_vae is not None and "scaler_vae" in ckpt:
        scaler_vae.load_state_dict(ckpt["scaler_vae"])
    if scaler_disc is not None and "scaler_disc" in ckpt:
        scaler_disc.load_state_dict(ckpt["scaler_disc"])

    # Checkpoints saved before this feature existed have no "epoch_complete" key -- they were
    # always full-epoch saves, so default True keeps old checkpoints resuming the same way they
    # always did.
    epoch_complete = ckpt.get("epoch_complete", True)
    start_epoch = ckpt["epoch"] + 1 if epoch_complete else ckpt["epoch"]
    global_step = ckpt.get("global_step", 0)
    best_val_loss = ckpt.get("best_val_loss", math.inf)

    status = "complete" if epoch_complete else "IN-PROGRESS (resuming this same epoch fresh)"
    print(f"  Resumed from checkpoint: epoch {ckpt['epoch']} ({status}), "
          f"global_step={global_step}, best val loss={best_val_loss:.4f}")
    return start_epoch, global_step, best_val_loss


# ---------------------------------------------------------------------------
# Single train epoch
# ---------------------------------------------------------------------------

def _train_epoch(
    epoch: int,
    vae: VAE,
    discriminator: PatchGAN3D,
    train_loader: DataLoader,
    vae_optim: torch.optim.Optimizer,
    disc_optim: torch.optim.Optimizer,
    loss_fn: VAELoss,
    scaler_vae: GradScaler,
    scaler_disc: GradScaler,
    device: str,
    grad_accum_steps: int,
    use_amp: bool,
    writer: SummaryWriter | None,
    global_step: list[int],   # mutable int wrapped in list so we can update it
    out_dir: Path,
    checkpoint_every_n_steps: int | None,
    best_val_loss: float,
) -> dict[str, float]:

    vae.train()
    discriminator.train()

    amp_dtype = torch.bfloat16 if (use_amp and device == "cuda") else torch.float32
    accum_log: dict[str, list[float]] = {}

    def _save_latest(reason: str) -> None:
        _save_checkpoint(
            out_dir, epoch, vae, discriminator, vae_optim, disc_optim,
            scaler_vae, scaler_disc, global_step[0], best_val_loss,
            epoch_complete=False, filename="latest.pt",
        )
        print(f"    (saved because: {reason})", flush=True)

    vae_optim.zero_grad(set_to_none=True)
    disc_optim.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader):
        ct   = batch["ct"].to(device)    # (B, 1, 256, 256, 256)
        is_last_accum = ((step + 1) % grad_accum_steps == 0) or (step + 1 == len(train_loader))

        # ── VAE forward ─────────────────────────────────────────────────
        with torch.autocast(device_type=device if device != "mps" else "cpu",
                            dtype=amp_dtype, enabled=use_amp):
            recon, posterior = vae(ct, sample_posterior=True)
            fake_logits = discriminator(recon)

            gen_loss, log_gen = loss_fn.generator_loss(
                x           = ct,
                recon       = recon,
                posterior   = posterior,
                fake_logits = fake_logits,
                epoch       = epoch,
            )
            gen_loss_scaled = gen_loss / grad_accum_steps

        scaler_vae.scale(gen_loss_scaled).backward()

        # ── Discriminator forward ────────────────────────────────────────
        with torch.autocast(device_type=device if device != "mps" else "cpu",
                            dtype=amp_dtype, enabled=use_amp):
            real_logits = discriminator(ct.detach())
            fake_logits_d = discriminator(recon.detach())
            disc_loss, log_disc = loss_fn.discriminator_loss(real_logits, fake_logits_d)
            disc_loss_scaled = disc_loss / grad_accum_steps

        scaler_disc.scale(disc_loss_scaled).backward()

        # ── Accumulate logs ──────────────────────────────────────────────
        for k, v in {**log_gen, **log_disc}.items():
            accum_log.setdefault(k, []).append(v)

        # ── Optimiser step every grad_accum_steps ────────────────────────
        if is_last_accum:
            scaler_vae.unscale_(vae_optim)
            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            scaler_vae.step(vae_optim)
            scaler_vae.update()
            vae_optim.zero_grad(set_to_none=True)

            scaler_disc.unscale_(disc_optim)
            nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            scaler_disc.step(disc_optim)
            scaler_disc.update()
            disc_optim.zero_grad(set_to_none=True)

            # ── TensorBoard step log ─────────────────────────────────────
            if writer:
                for k, vs in accum_log.items():
                    writer.add_scalar(f"train/{k}", sum(vs) / len(vs), global_step[0])
            global_step[0] += 1
            accum_log = {}

            # ── Periodic safety checkpoint (protects against ANY kill, graceful or not) ──
            if checkpoint_every_n_steps and global_step[0] % checkpoint_every_n_steps == 0:
                _save_latest(f"periodic, every {checkpoint_every_n_steps} steps")

        # ── Graceful-interrupt check (Ctrl+C / SIGTERM -- NOT a hard kill, see note above) ──
        if _shutdown_requested[0]:
            _save_latest("graceful interrupt signal received")
            raise TrainingInterrupted(f"Stopped by signal during epoch {epoch}, step {step}")

    # Epoch-mean across all steps
    return {}   # per-step logging is done above; caller computes epoch means from epoch val


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def _val_epoch(
    vae: VAE,
    val_loader: DataLoader,
    loss_fn: VAELoss,
    device: str,
    use_amp: bool,
) -> dict[str, float]:
    vae.eval()
    amp_dtype = torch.bfloat16 if (use_amp and device == "cuda") else torch.float32
    sums: dict[str, float] = {}
    count = 0

    for batch in val_loader:
        ct = batch["ct"].to(device)
        with torch.autocast(device_type=device if device != "mps" else "cpu",
                            dtype=amp_dtype, enabled=use_amp):
            recon, posterior = vae(ct, sample_posterior=False)   # deterministic at val
            # No discriminator during val — just reconstruction quality
            _, log = loss_fn.generator_loss(
                x=ct, recon=recon, posterior=posterior,
                fake_logits=None, epoch=9999,   # epoch=9999 → full adv weight, but no fake_logits passed
            )
        for k, v in log.items():
            sums[k] = sums.get(k, 0.0) + v
        count += 1

    return {k: v / max(count, 1) for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    manifest_path: str,
    cache_dir: str,
    out_dir: str,
    vae_config_path: str  = "configs/vae_config.yaml",
    data_config_path: str = "configs/data_config.yaml",
    resume_from: str | None = None,
    epochs: int | None = None,
    limit: int | None = None,
    no_lpips: bool = False,
    device_override: str | None = None,
    patch_size: int | None = None,
    nodule_centered_prob: float | None = None,
    checkpoint_every_n_steps: int | None = None,
) -> None:
    """
    Main VAE training entry point.

    Args
    ----
    manifest_path    : path to manifest JSON (from scripts/build_manifest.py)
    cache_dir        : directory of preprocessed .npz files (from scripts/preprocess_dataset.py)
    out_dir          : where to save checkpoints + TensorBoard logs
    vae_config_path  : path to vae_config.yaml
    data_config_path : path to data_config.yaml
    resume_from      : optional path to a .pt checkpoint to resume from (use
                        `<out_dir>/latest.pt` to resume from the most recent safety checkpoint,
                        which is at most `checkpoint_every_n_steps` steps stale)
    epochs           : override number of training epochs (default: from vae_config.yaml)
    limit            : only use the first N training samples (smoke-test)
    no_lpips         : disable LPIPS loss (faster, useful for debugging)
    device_override  : force a specific device ('cpu', 'cuda', 'mps')
    patch_size       : crop each scan to this cube size for train+val instead of using the full
                        256^3 volume (default: from data_config.yaml `patch.size`, which is None
                        i.e. full-volume -- override here or via --patch-size). Needed on 8GB
                        cards: the first encoder conv at full 256^3 needs 54GB, see NOTES.md.
    nodule_centered_prob : probability a tumor-sample patch is centered on a real nodule voxel
                        instead of a uniform-random crop (default: from data_config.yaml
                        `patch.nodule_centered_prob`). Ignored if patch_size is None.
    checkpoint_every_n_steps : save a `latest.pt` safety checkpoint (full state: weights,
                        optimisers, AMP scalers, epoch, global_step) every N optimiser steps, in
                        addition to the normal end-of-epoch checkpoints (default: from
                        vae_config.yaml `training.checkpoint_every_n_steps`). This -- not the
                        Ctrl+C signal handler -- is what actually bounds your worst-case lost
                        progress on a shared machine, since a hard kill (closed terminal window,
                        `taskkill /F`) can't be intercepted at all.
    """
    # ── Config ──────────────────────────────────────────────────────────────
    cfg = load_config(vae=vae_config_path, data=data_config_path)
    device = get_device(device_override)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, str(out_dir / "resolved_config.yaml"))

    train_cfg = cfg.vae.training
    n_epochs  = epochs if epochs is not None else int(train_cfg.epochs)
    grad_accum= int(train_cfg.gradient_accumulation_steps)
    use_amp   = (train_cfg.mixed_precision == "bf16") and (device == "cuda")

    patch_cfg = getattr(cfg.data, "patch", None)
    resolved_patch_size = patch_size if patch_size is not None else (
        int(patch_cfg.size) if patch_cfg is not None and patch_cfg.size is not None else None
    )
    resolved_nodule_prob = nodule_centered_prob if nodule_centered_prob is not None else (
        float(patch_cfg.nodule_centered_prob) if patch_cfg is not None else 0.0
    )

    print(f"\n{'='*60}")
    print(f"  LAND VAE Training")
    print(f"  device={device}  epochs={n_epochs}  amp={use_amp}")
    print(f"  grad_accum={grad_accum}  limit={limit}")
    print(f"  patch_size={resolved_patch_size or 'full-volume (256^3)'}  "
          f"nodule_centered_prob={resolved_nodule_prob}")
    print(f"{'='*60}\n")

    # ── Dataset / DataLoader ─────────────────────────────────────────────────
    train_dataset = _build_dataset(
        manifest_path, cache_dir, split="train", limit=limit,
        patch_size=resolved_patch_size, nodule_centered_prob=resolved_nodule_prob,
    )
    val_dataset   = _build_dataset(
        manifest_path, cache_dir, split="val",
        limit=max(limit // 4, 1) if limit else None,
        patch_size=resolved_patch_size, nodule_centered_prob=resolved_nodule_prob,
        val_seed=42,  # deterministic per-sample patch -> val loss comparable across epochs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = int(cfg.data.dataloader.batch_size),
        shuffle     = True,
        num_workers = int(cfg.data.dataloader.num_workers),
        pin_memory  = bool(cfg.data.dataloader.pin_memory) and (device == "cuda"),
        collate_fn  = _collate,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = 2,
        collate_fn  = _collate,
    )

    print(f"  Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}")
    print(f"  Steps per epoch: {len(train_loader)}  "
          f"Effective optimiser steps: {len(train_loader) // grad_accum}\n")

    # ── Models ──────────────────────────────────────────────────────────────
    vae           = VAE.from_config(cfg.vae).to(device)
    discriminator = PatchGAN3D.from_config(cfg.vae).to(device)

    vae_params   = vae.count_parameters()
    disc_params  = discriminator.count_parameters()
    print(f"  VAE:           {vae_params['total']:,} parameters")
    print(f"  Discriminator: {disc_params:,} parameters\n")

    # ── Optimisers ──────────────────────────────────────────────────────────
    vae_optim = torch.optim.AdamW(
        vae.parameters(),
        lr=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
    )
    disc_optim = torch.optim.AdamW(
        discriminator.parameters(),
        lr=float(train_cfg.discriminator_learning_rate),
        weight_decay=float(train_cfg.weight_decay),
    )

    # ── Loss ────────────────────────────────────────────────────────────────
    use_lpips_effective = (not no_lpips) and bool(cfg.vae.loss.perceptual_2p5d)
    loss_fn = VAELoss(
        mae_weight         = float(cfg.vae.loss.mae_weight),
        lpips_weight       = float(cfg.vae.loss.lpips_weight) if use_lpips_effective else 0.0,
        adversarial_weight = float(cfg.vae.loss.adversarial_weight),
        kl_weight          = float(cfg.vae.loss.kl_weight),
        warmup_epochs      = int(train_cfg.discriminator_warmup_epochs),
        use_lpips          = use_lpips_effective,
    )

    # ── AMP scalers ─────────────────────────────────────────────────────────
    scaler_vae  = GradScaler(enabled=use_amp)
    scaler_disc = GradScaler(enabled=use_amp)

    # ── Shared-machine safety: catch Ctrl+C / graceful kill (NOT a hard kill -- see the note
    # above _request_shutdown). Installed before the loop so an early interrupt is still caught.
    _install_signal_handlers()

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch     = 0
    best_val_loss   = math.inf
    resumed_step    = 0
    if resume_from:
        start_epoch, resumed_step, best_val_loss = _load_checkpoint(
            resume_from, vae, discriminator, vae_optim, disc_optim, scaler_vae, scaler_disc
        )

    # ── TensorBoard ─────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs"))

    # ── Training loop ────────────────────────────────────────────────────────
    global_step = [resumed_step]   # mutable container so _train_epoch can increment it
    val_every   = int(train_cfg.val_every_n_epochs)
    ckpt_every  = int(train_cfg.checkpoint_every_n_epochs)
    ckpt_every_n_steps = (
        checkpoint_every_n_steps if checkpoint_every_n_steps is not None
        else (int(train_cfg.checkpoint_every_n_steps)
              if getattr(train_cfg, "checkpoint_every_n_steps", None) else None)
    )
    print(f"  checkpoint_every_n_steps={ckpt_every_n_steps or 'disabled'}  "
          f"(resume from: {out_dir / 'latest.pt'})\n")

    try:
        for epoch in range(start_epoch, n_epochs):
            t0 = time.time()
            print(f"Epoch {epoch+1}/{n_epochs}", flush=True)

            _train_epoch(
                epoch=epoch,
                vae=vae,
                discriminator=discriminator,
                train_loader=train_loader,
                vae_optim=vae_optim,
                disc_optim=disc_optim,
                loss_fn=loss_fn,
                scaler_vae=scaler_vae,
                scaler_disc=scaler_disc,
                device=device,
                grad_accum_steps=grad_accum,
                use_amp=use_amp,
                writer=writer,
                global_step=global_step,
                out_dir=out_dir,
                checkpoint_every_n_steps=ckpt_every_n_steps,
                best_val_loss=best_val_loss,
            )

            elapsed = time.time() - t0
            print(f"  Epoch time: {elapsed:.1f}s", flush=True)

            # ── Validation ──────────────────────────────────────────────────
            if (epoch + 1) % val_every == 0 or epoch == n_epochs - 1:
                val_log = _val_epoch(vae, val_loader, loss_fn, device, use_amp)
                val_loss = val_log.get("loss/total", math.inf)
                print(f"  Val loss: {val_loss:.4f}  "
                      f"(mae={val_log.get('loss/mae',0):.4f}  "
                      f"kl={val_log.get('loss/kl',0):.6f})", flush=True)

                for k, v in val_log.items():
                    writer.add_scalar(f"val/{k}", v, epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(
                        out_dir, epoch, vae, discriminator, vae_optim, disc_optim,
                        scaler_vae, scaler_disc, global_step[0], best_val_loss,
                        epoch_complete=True, tag="best",
                    )

            # ── Regular end-of-epoch checkpoint ───────────────────────────────
            if (epoch + 1) % ckpt_every == 0:
                _save_checkpoint(
                    out_dir, epoch, vae, discriminator, vae_optim, disc_optim,
                    scaler_vae, scaler_disc, global_step[0], best_val_loss,
                    epoch_complete=True,
                )

        # ── Final checkpoint ─────────────────────────────────────────────────
        _save_checkpoint(
            out_dir, n_epochs - 1, vae, discriminator, vae_optim, disc_optim,
            scaler_vae, scaler_disc, global_step[0], best_val_loss,
            epoch_complete=True, tag="final",
        )
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    except TrainingInterrupted as e:
        # _train_epoch already saved out_dir/latest.pt before raising this -- just exit cleanly
        # instead of printing a scary traceback for what is, on a shared machine, routine.
        print(f"\nTraining stopped early: {e}")
        print(f"Resume any time with: --resume {out_dir / 'latest.pt'}")

    writer.close()
    print(f"Checkpoints and logs in: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the LAND 3D VAE")
    p.add_argument("--manifest",  type=str, required=True,
                   help="Path to manifest JSON (from scripts/build_manifest.py)")
    p.add_argument("--cache-dir", type=str, required=True,
                   help="Directory of preprocessed .npz files")
    p.add_argument("--out-dir",   type=str, default="checkpoints/vae",
                   help="Directory to save checkpoints and TensorBoard logs")
    p.add_argument("--vae-config",  type=str, default="configs/vae_config.yaml")
    p.add_argument("--data-config", type=str, default="configs/data_config.yaml")
    p.add_argument("--resume",    type=str, default=None,
                   help="Path to a checkpoint .pt file to resume from")
    p.add_argument("--epochs",    type=int, default=None,
                   help="Override number of epochs from config")
    p.add_argument("--limit",     type=int, default=None,
                   help="Only use first N training samples (smoke-test)")
    p.add_argument("--no-lpips",  action="store_true",
                   help="Disable LPIPS loss (faster, good for quick debug runs)")
    p.add_argument("--device",    type=str, default=None,
                   help="Force device: cuda / cpu / mps")
    p.add_argument("--patch-size", type=int, default=None,
                   help="Crop scans to this cube size (train+val) instead of full 256^3 "
                        "(default: data_config.yaml patch.size). Use e.g. 128 or 96 on an "
                        "8GB card -- full 256^3 OOMs on the first encoder conv (needs 54GB).")
    p.add_argument("--nodule-centered-prob", type=float, default=None,
                   help="Probability a tumor-sample patch is centered on a real nodule voxel "
                        "instead of a uniform-random crop (default: data_config.yaml "
                        "patch.nodule_centered_prob). Ignored if --patch-size is unset.")
    p.add_argument("--checkpoint-every-n-steps", type=int, default=None,
                   help="Save a full-state 'latest.pt' safety checkpoint every N optimiser "
                        "steps, in addition to end-of-epoch checkpoints (default: "
                        "vae_config.yaml training.checkpoint_every_n_steps, 50). Recommended on "
                        "a shared machine -- bounds worst-case lost progress to this many steps "
                        "even on a hard kill. 0 disables. Resume with --resume <out-dir>/latest.pt")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        manifest_path    = args.manifest,
        cache_dir        = args.cache_dir,
        out_dir          = args.out_dir,
        vae_config_path  = args.vae_config,
        data_config_path = args.data_config,
        resume_from      = args.resume,
        epochs           = args.epochs,
        limit            = args.limit,
        no_lpips         = args.no_lpips,
        device_override  = args.device,
        patch_size       = args.patch_size,
        nodule_centered_prob = args.nodule_centered_prob,
        checkpoint_every_n_steps = args.checkpoint_every_n_steps,
    )
