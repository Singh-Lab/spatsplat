import torch
import torch.nn as nn


class MeanProfileBaseline(nn.Module):
    def __init__(self, target_counts: torch.Tensor):
        super().__init__()
        mean_profile = target_counts.mean(dim=0, keepdim=True).clamp_min(1e-6)
        self.register_buffer("log_rate", mean_profile.log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_rate.expand(x.shape[0], -1)
