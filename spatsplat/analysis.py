import numpy as np
import torch


def capture_initial_splat_state(model) -> dict[str, torch.Tensor]:
    return {
        "means": model.means.detach().clone(),
        "log_scales": model.log_scales.detach().clone(),
        "rotation": model.rotation.detach().clone(),
        "opacity_logits": model.opacity_logits.detach().clone(),
    }


def summarize_tensor(tensor: torch.Tensor) -> dict[str, float]:
    flat = tensor.detach().reshape(-1).float()
    q = torch.quantile(flat, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=flat.device))
    return {
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "min": q[0].item(),
        "q25": q[1].item(),
        "median": q[2].item(),
        "q75": q[3].item(),
        "max": q[4].item(),
    }


def compute_splat_weights(model, query_xy: torch.Tensor) -> torch.Tensor:
    return model.compute_weights(query_xy)


def usage_statistics(weights: torch.Tensor) -> dict[str, object]:
    mean_usage = weights.mean(dim=0)
    peak_usage = weights.max(dim=0).values
    topk_per_query = (weights > 0.01).sum(dim=1).float()
    return {
        "active_fraction": (mean_usage > 1e-4).float().mean().item(),
        "dominant_fraction": (peak_usage > 0.05).float().mean().item(),
        "avg_splats_per_query_gt_1pct": topk_per_query.mean().item(),
        "mean_usage_summary": summarize_tensor(mean_usage),
        "peak_usage_summary": summarize_tensor(peak_usage),
    }


def sample_numpy_grid(points_xy: np.ndarray, grid_2d: np.ndarray) -> np.ndarray:
    h, w = grid_2d.shape
    x = np.clip(points_xy[:, 0] * (w - 1), 0, w - 1)
    y = np.clip(points_xy[:, 1] * (h - 1), 0, h - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = x - x0
    wy = y - y0

    v00 = grid_2d[y0, x0]
    v01 = grid_2d[y0, x1]
    v10 = grid_2d[y1, x0]
    v11 = grid_2d[y1, x1]
    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v01
        + (1.0 - wx) * wy * v10
        + wx * wy * v11
    )


def summarize_error_slices(err_values: np.ndarray, distances: np.ndarray) -> dict[str, float]:
    positive = distances[distances > 0]
    if len(positive) == 0:
        raise RuntimeError("No positive support distances")
    edge_thresh = np.quantile(positive, 0.25)
    interior_thresh = np.quantile(positive, 0.75)
    edge_mask = distances <= edge_thresh
    interior_mask = distances >= interior_thresh
    return {
        "edge_thresh": float(edge_thresh),
        "interior_thresh": float(interior_thresh),
        "edge_mae": float(err_values[edge_mask].mean()),
        "interior_mae": float(err_values[interior_mask].mean()),
        "overall_mae": float(err_values.mean()),
        "edge_n": int(edge_mask.sum()),
        "interior_n": int(interior_mask.sum()),
    }


def summarize_activity_slices(signed_err: np.ndarray, true_values: np.ndarray) -> dict[str, float]:
    true_np = np.asarray(true_values)
    signed_np = np.asarray(signed_err)
    active_mask = true_np > 0
    inactive_mask = ~active_mask
    return {
        "active_n": int(active_mask.sum()),
        "inactive_n": int(inactive_mask.sum()),
        "active_true_mean": float(true_np[active_mask].mean()) if active_mask.any() else float("nan"),
        "inactive_true_mean": float(true_np[inactive_mask].mean()) if inactive_mask.any() else float("nan"),
        "active_signed_error_mean": float(signed_np[active_mask].mean()) if active_mask.any() else float("nan"),
        "inactive_signed_error_mean": float(signed_np[inactive_mask].mean()) if inactive_mask.any() else float("nan"),
        "active_abs_error_mean": float(np.abs(signed_np[active_mask]).mean()) if active_mask.any() else float("nan"),
        "inactive_abs_error_mean": float(np.abs(signed_np[inactive_mask]).mean()) if inactive_mask.any() else float("nan"),
    }


def farthest_point_sample_indices(
    points_xy: torch.Tensor,
    num_samples: int,
    *,
    start_idx: int | None = None,
    candidate_limit: int | None = None,
    random_state: int = 0,
) -> torch.Tensor:
    coords = points_xy.detach()
    if coords.ndim != 2 or coords.shape[1] < 1:
        raise RuntimeError(f"Expected points with shape [N, D], got {tuple(coords.shape)}")

    total_points = coords.shape[0]
    if total_points == 0:
        raise RuntimeError("Cannot sample from an empty point set")
    if num_samples <= 0:
        raise RuntimeError("num_samples must be positive")
    if num_samples >= total_points and candidate_limit is None:
        return torch.arange(total_points, device=coords.device, dtype=torch.long)

    if candidate_limit is not None and total_points > candidate_limit:
        generator = torch.Generator(device=coords.device if coords.is_cuda else "cpu")
        generator.manual_seed(random_state)
        candidate_idx = torch.randperm(total_points, generator=generator, device=coords.device)[:candidate_limit]
        candidate_coords = coords[candidate_idx]
    else:
        candidate_idx = torch.arange(total_points, device=coords.device, dtype=torch.long)
        candidate_coords = coords

    num_candidates = candidate_coords.shape[0]
    if num_samples >= num_candidates:
        return candidate_idx

    if start_idx is None:
        center = candidate_coords.mean(dim=0, keepdim=True)
        start_local = torch.argmin(((candidate_coords - center) ** 2).sum(dim=1))
    else:
        start_tensor = torch.as_tensor(start_idx, device=coords.device, dtype=torch.long)
        match = torch.nonzero(candidate_idx == start_tensor, as_tuple=False).reshape(-1)
        start_local = match[0] if match.numel() > 0 else torch.tensor(0, device=coords.device, dtype=torch.long)

    selected_local = torch.empty(num_samples, device=coords.device, dtype=torch.long)
    selected_local[0] = start_local

    min_dist2 = ((candidate_coords - candidate_coords[start_local]) ** 2).sum(dim=1)
    min_dist2[start_local] = -1.0

    for i in range(1, num_samples):
        next_local = torch.argmax(min_dist2)
        selected_local[i] = next_local
        dist2 = ((candidate_coords - candidate_coords[next_local]) ** 2).sum(dim=1)
        min_dist2 = torch.minimum(min_dist2, dist2)
        min_dist2[selected_local[: i + 1]] = -1.0

    return candidate_idx[selected_local]
