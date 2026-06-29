import torch
import torch.nn.functional as F


def mae_count_loss(log_mu: torch.Tensor, target_counts: torch.Tensor) -> torch.Tensor:
    mu = torch.exp(log_mu).clamp_min(1e-6)
    target_counts = target_counts.to(mu.dtype)
    return (mu - target_counts).abs().mean()


def log_total_count_loss(log_mu: torch.Tensor, target_counts: torch.Tensor) -> torch.Tensor:
    mu = torch.exp(log_mu).clamp_min(1e-6)
    target_counts = target_counts.to(mu.dtype)
    pred_total = torch.log1p(mu.sum(dim=1))
    target_total = torch.log1p(target_counts.sum(dim=1))
    return (pred_total - target_total).abs().mean()


def composition_cosine_loss(
    pred_counts: torch.Tensor,
    target_counts: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_comp = pred_counts / pred_counts.sum(dim=1, keepdim=True).clamp_min(eps)
    target_comp = target_counts.to(pred_counts.dtype)
    target_comp = target_comp / target_comp.sum(dim=1, keepdim=True).clamp_min(eps)
    cosine = F.cosine_similarity(pred_comp, target_comp, dim=1, eps=eps)
    return (1.0 - cosine).mean()


def sample_tissue_support(points_xy: torch.Tensor, tissue_mask_tensor: torch.Tensor) -> torch.Tensor:
    grid = points_xy.view(1, -1, 1, 2) * 2.0 - 1.0
    sampled = F.grid_sample(
        tissue_mask_tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.view(-1).clamp(0.0, 1.0)


def predict_cell_log_rates(
    model,
    cell_centers: torch.Tensor,
    cell_spans: torch.Tensor | None = None,
    *,
    batch_size: int = 1024,
) -> torch.Tensor:
    all_log_rates = []

    for start in range(0, len(cell_centers), batch_size):
        all_log_rates.append(model(cell_centers[start : start + batch_size]))

    return torch.cat(all_log_rates, dim=0)


def evaluate_count_model(
    model,
    features: torch.Tensor,
    targets: torch.Tensor,
    split_indices: torch.Tensor,
    *,
    batch_size: int = 4096,
    loss_fn=None,
):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_seen = 0

    with torch.no_grad():
        for start in range(0, len(split_indices), batch_size):
            idx = split_indices[start : start + batch_size]
            pred_log_rate = model(features[idx])
            if loss_fn is None:
                loss = mae_count_loss(pred_log_rate, targets[idx])
            else:
                loss = loss_fn(pred_log_rate, targets[idx])
            pred_counts = torch.exp(pred_log_rate)
            mae = (pred_counts - targets[idx]).abs().mean()

            total_loss += loss.item() * len(idx)
            total_mae += mae.item() * len(idx)
            total_seen += len(idx)

    if was_training:
        model.train()

    return {"loss": total_loss / total_seen, "mae": total_mae / total_seen, "n": total_seen}


def evaluate_cell_readout(
    model,
    cell_centers: torch.Tensor,
    cell_spans: torch.Tensor,
    targets: torch.Tensor,
    split_indices: torch.Tensor,
    *,
    batch_size: int = 1024,
    loss_fn=None,
):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_seen = 0

    with torch.no_grad():
        for start in range(0, len(split_indices), batch_size):
            idx = split_indices[start : start + batch_size]
            pred_log_rate = predict_cell_log_rates(
                model,
                cell_centers[idx],
                cell_spans[idx],
                batch_size=batch_size,
            )
            if loss_fn is None:
                loss = mae_count_loss(pred_log_rate, targets[idx])
            else:
                loss = loss_fn(pred_log_rate, targets[idx])
            pred_counts = torch.exp(pred_log_rate)
            mae = (pred_counts - targets[idx]).abs().mean()

            total_loss += loss.item() * len(idx)
            total_mae += mae.item() * len(idx)
            total_seen += len(idx)

    if was_training:
        model.train()

    return {"loss": total_loss / total_seen, "mae": total_mae / total_seen, "n": total_seen}


def create_amplitude_splat_optimizer(
    model,
    *,
    means_lr: float = 5e-3,
    log_scales_lr: float = 1e-3,
    rotation_lr: float = 1e-3,
    opacity_lr: float = 1e-3,
    latents_lr: float = 1e-3,
    log_theta_lr: float = 1e-3,
    decoder_lr: float = 1e-3,
    amplitude_lr: float = 1e-3,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [
            {"params": model.means, "lr": means_lr},
            {"params": model.log_scales, "lr": log_scales_lr},
            {"params": model.rotation, "lr": rotation_lr},
            {"params": model.opacity_logits, "lr": opacity_lr},
            {"params": model.latents, "lr": latents_lr},
            {"params": model.log_theta, "lr": log_theta_lr},
            {"params": model.decoder.parameters(), "lr": decoder_lr},
            {"params": model.log_amplitudes, "lr": amplitude_lr},
        ]
    )
