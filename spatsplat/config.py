from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    bin_size: int = 64
    mask_grid_size: int = 128
    qv_threshold: int = 20
    cell_mask_min_radius: float = 1.0


@dataclass(frozen=True)
class SplatConfig:
    num_splats: int = 4096
    latent_dim: int = 32
    hidden_dim: int = 128
    init_log_scale: float = -4.5
    kernel_sharpness: float = 0.5
    top_k_splats: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    baseline_epochs: int = 10
    splat_epochs: int = 20
    baseline_batch_size: int = 4096
    bin_batch_size: int = 2048
    cell_batch_size: int = 2048
    off_tissue_batch_size: int = 2048
    bin_loss_weight: float = 0.05
    cell_loss_weight: float = 1.0
    off_tissue_weight: float = 0.05
    center_support_weight: float = 100.0
    splat_means_lr: float = 5e-3
    splat_log_scales_lr: float = 1e-3
    splat_rotation_lr: float = 1e-3
    splat_opacity_lr: float = 1e-3
    splat_latents_lr: float = 1e-3
    splat_decoder_lr: float = 1e-3
    splat_amplitude_lr: float = 1e-3
