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


def local_difference_loss(
    pred_counts: torch.Tensor,
    target_counts: torch.Tensor,
    query_xy: torch.Tensor,
    *,
    num_neighbors: int = 8,
    eps: float = 1e-8,
) -> torch.Tensor:
    if pred_counts.ndim != 2 or target_counts.ndim != 2:
        raise RuntimeError("pred_counts and target_counts must be rank-2 tensors")
    if len(pred_counts) != len(target_counts) or len(pred_counts) != len(query_xy):
        raise RuntimeError("pred_counts, target_counts, and query_xy must have matching leading dimensions")
    if len(pred_counts) < 2:
        return pred_counts.new_tensor(0.0)

    k = min(max(int(num_neighbors), 1), len(pred_counts) - 1)
    dist = torch.cdist(query_xy.to(pred_counts.dtype), query_xy.to(pred_counts.dtype))
    knn_idx = torch.topk(dist, k=k + 1, largest=False).indices[:, 1:]

    pred_comp = pred_counts / pred_counts.sum(dim=1, keepdim=True).clamp_min(eps)
    target_comp = target_counts.to(pred_counts.dtype)
    target_comp = target_comp / target_comp.sum(dim=1, keepdim=True).clamp_min(eps)

    pred_center = pred_comp.unsqueeze(1)
    pred_neighbors = pred_comp[knn_idx]
    target_center = target_comp.unsqueeze(1)
    target_neighbors = target_comp[knn_idx]

    pred_diff = pred_center - pred_neighbors
    target_diff = target_center - target_neighbors
    return (pred_diff - target_diff).abs().mean()


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


def create_splat_optimizer(
    model,
    *,
    means_lr: float = 5e-3,
    log_scales_lr: float = 1e-3,
    rotation_lr: float = 1e-3,
    opacity_lr: float = 1e-3,
    latents_lr: float = 1e-3,
    log_theta_lr: float = 1e-3,
    decoder_lr: float = 1e-3,
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
        ]
    )


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


def train_splat_epoch_joint(
    model,
    optimizer: torch.optim.Optimizer,
    *,
    bin_features: torch.Tensor,
    bin_targets: torch.Tensor,
    bin_train_idx: torch.Tensor,
    cell_features: torch.Tensor,
    cell_spans: torch.Tensor,
    cell_targets: torch.Tensor,
    cell_train_idx: torch.Tensor,
    off_tissue_points: torch.Tensor,
    tissue_mask_tensor: torch.Tensor,
    device: str,
    bin_batch_size: int = 2048,
    cell_batch_size: int = 2048,
    off_tissue_batch_size: int = 2048,
    bin_loss_weight: float = 0.1,
    cell_loss_weight: float = 1.0,
    off_tissue_weight: float = 0.05,
    center_support_weight: float = 0.1,
):
    model.train()
    bin_perm = bin_train_idx[torch.randperm(len(bin_train_idx), device=device)]
    cell_perm = cell_train_idx[torch.randperm(len(cell_train_idx), device=device)]
    off_perm = torch.randperm(len(off_tissue_points), device=device)
    num_steps = max(
        (len(bin_perm) + bin_batch_size - 1) // bin_batch_size,
        (len(cell_perm) + cell_batch_size - 1) // cell_batch_size,
    )
    running_loss = 0.0
    running_bin_loss = 0.0
    running_cell_loss = 0.0
    running_off_tissue_loss = 0.0
    running_center_support_loss = 0.0

    for step in range(num_steps):
        bin_start = (step * bin_batch_size) % len(bin_perm)
        cell_start = (step * cell_batch_size) % len(cell_perm)
        off_start = (step * off_tissue_batch_size) % len(off_perm)
        bin_idx = bin_perm[bin_start : bin_start + bin_batch_size]
        cell_idx = cell_perm[cell_start : cell_start + cell_batch_size]
        off_idx = off_perm[off_start : off_start + off_tissue_batch_size]

        if len(bin_idx) == 0:
            bin_idx = bin_perm[:bin_batch_size]
        if len(cell_idx) == 0:
            cell_idx = cell_perm[:cell_batch_size]
        if len(off_idx) == 0:
            off_idx = off_perm[:off_tissue_batch_size]

        pred_bin_log_rate = model(bin_features[bin_idx])
        pred_cell_log_rate = predict_cell_log_rates(model, cell_features[cell_idx], cell_spans[cell_idx])
        off_tissue_intensity = model.total_intensity(off_tissue_points[off_idx])
        center_support = sample_tissue_support(model.means, tissue_mask_tensor)

        bin_loss = mae_count_loss(pred_bin_log_rate, bin_targets[bin_idx])
        cell_loss = mae_count_loss(pred_cell_log_rate, cell_targets[cell_idx])
        off_tissue_loss = off_tissue_intensity.mean()
        center_support_loss = (1.0 - center_support).mean()
        loss = (
            bin_loss_weight * bin_loss
            + cell_loss_weight * cell_loss
            + off_tissue_weight * off_tissue_loss
            + center_support_weight * center_support_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            model.means.clamp_(0.0, 1.0)
            model.log_scales.clamp_(-8.0, -2.0)

        running_loss += loss.item()
        running_bin_loss += bin_loss.item()
        running_cell_loss += cell_loss.item()
        running_off_tissue_loss += off_tissue_loss.item()
        running_center_support_loss += center_support_loss.item()

    return {
        "joint_loss": running_loss / num_steps,
        "bin_loss": running_bin_loss / num_steps,
        "cell_loss": running_cell_loss / num_steps,
        "off_tissue_loss": running_off_tissue_loss / num_steps,
        "center_support_loss": running_center_support_loss / num_steps,
    }
