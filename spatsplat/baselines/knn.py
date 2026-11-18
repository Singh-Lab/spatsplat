import torch
import torch.nn as nn


class KNNCountRegressor(nn.Module):
    def __init__(self, train_features: torch.Tensor, train_targets: torch.Tensor, *, k: int = 32, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("train_features", train_features)
        self.register_buffer("train_targets", train_targets)
        self.k = k
        self.eps = eps
        mean_target = train_targets.mean(dim=0).clamp_min(1e-6)
        self.register_buffer("fallback_log_rate", mean_target.log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, self.train_features)
        k = min(self.k, self.train_features.shape[0])
        knn_dists, knn_idx = torch.topk(dists, k=k, largest=False, dim=1)
        weights = 1.0 / (knn_dists + self.eps)
        weights = weights / weights.sum(dim=1, keepdim=True)
        neighbor_targets = self.train_targets[knn_idx]
        pred_counts = (weights.unsqueeze(-1) * neighbor_targets).sum(dim=1)
        pred_counts = torch.where(
            torch.isfinite(pred_counts),
            pred_counts,
            self.fallback_log_rate.exp().expand_as(pred_counts),
        )
        return pred_counts.clamp_min(1e-6).log()
