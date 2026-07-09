"""
Lightweight config loading built on OmegaConf.

Usage:
    from src.utils.config import load_config

    cfg = load_config(
        data="configs/data_config.yaml",
        vae="configs/vae_config.yaml",
    )
    print(cfg.vae.architecture.latent_channels)   # -> 4

Any config value can be overridden from the CLI without touching the yaml files, e.g.:
    python -m src.training.train_vae vae.training.batch_size=2 data.paths.processed_dir=/mnt/data
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf, DictConfig


def load_config(**named_yaml_paths: str) -> DictConfig:
    """
    Load one or more yaml files into a single namespaced DictConfig, then apply any
    `key.path=value` overrides found in sys.argv (dotlist CLI overrides).

    Each keyword becomes a top-level namespace, e.g. load_config(data=..., vae=...)
    produces cfg.data.* and cfg.vae.*
    """
    merged = {}
    for namespace, path in named_yaml_paths.items():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found for namespace '{namespace}': {path}")
        merged[namespace] = OmegaConf.load(path)

    cfg = OmegaConf.create(merged)

    # allow `python -m ... key.subkey=value` CLI overrides, ignoring the script path itself
    cli_overrides = [arg for arg in sys.argv[1:] if "=" in arg and not arg.startswith("-")]
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))

    OmegaConf.set_readonly(cfg, False)
    return cfg


def save_resolved_config(cfg: DictConfig, out_path: str) -> None:
    """Dump the fully-resolved config actually used for a run, for reproducibility."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


def get_device(preferred: Optional[str] = None) -> str:
    import torch

    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
