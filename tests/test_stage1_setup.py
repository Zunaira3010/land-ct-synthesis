"""
Stage 1 sanity checks: configs load, and the numbers in them are internally consistent
with what the paper states. Run with: pytest tests/test_stage1_setup.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config


def _load_all():
    return load_config(
        data="configs/data_config.yaml",
        vae="configs/vae_config.yaml",
        diffusion="configs/diffusion_config.yaml",
    )


def test_configs_load():
    cfg = _load_all()
    assert cfg.data is not None
    assert cfg.vae is not None
    assert cfg.diffusion is not None


def test_vae_latent_shape_matches_paper():
    """Paper: 256^3 volume -> 64x64x64x4 latent, i.e. 4x spatial compression, 4 latent channels."""
    cfg = _load_all()
    vol_shape = cfg.data.volume.target_shape
    compression = cfg.vae.architecture.spatial_compression
    latent_channels = cfg.vae.architecture.latent_channels

    expected_latent_spatial = [d // compression for d in vol_shape]
    assert expected_latent_spatial == [64, 64, 64], (
        f"Expected 64^3 latent spatial dims, got {expected_latent_spatial}"
    )
    assert latent_channels == 4, f"Paper specifies 4 latent channels, got {latent_channels}"


def test_mask_downsample_matches_latent_resolution():
    """Paper: mask is downsampled 4x via max pooling to match the latent resolution before
    concatenation. Mask spatial shape in the diffusion config should equal the VAE's latent
    spatial shape."""
    cfg = _load_all()
    vol_shape = cfg.data.volume.target_shape
    mask_downsample = cfg.data.mask_encoding.mask_downsample_factor
    mask_shape_expected = [d // mask_downsample for d in vol_shape]

    assert list(cfg.diffusion.conditioning.mask_spatial_shape) == mask_shape_expected


def test_unet_resolution_levels_match_paper():
    cfg = _load_all()
    assert cfg.diffusion.architecture.num_resolution_levels == 5
    assert cfg.diffusion.architecture.num_res_blocks_per_level == 2
    assert len(cfg.diffusion.architecture.channels) == 5
    assert len(cfg.diffusion.architecture.attention_levels) == 5


def test_vae_resolution_levels_match_paper():
    cfg = _load_all()
    assert cfg.vae.architecture.num_resolution_levels == 3
    assert cfg.vae.architecture.num_res_blocks_per_level == 1
    assert len(cfg.vae.architecture.channels) == 3


def test_training_hyperparams_match_paper():
    cfg = _load_all()
    assert cfg.vae.training.epochs == 100
    assert cfg.vae.training.batch_size == 1
    assert cfg.vae.training.learning_rate == 1.0e-4

    assert cfg.diffusion.training.num_steps == 500000
    assert cfg.diffusion.training.batch_size == 1
    assert cfg.diffusion.training.learning_rate == 1.0e-5


def test_diffusion_schedule_matches_paper():
    cfg = _load_all()
    dp = cfg.diffusion.diffusion_process
    assert dp.num_train_timesteps == 1000
    assert dp.beta_start == 1.0e-4
    assert dp.beta_end == 0.02
    assert dp.prediction_type == "v_prediction"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
