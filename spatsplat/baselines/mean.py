import torch
import torch.nn as nn


class MeanProfileBaseline(nn.Module):
    def __init__(self, target_counts: torch.Tensor):
        super().__init__()
        mean_profile = target_counts.mean(dim=0, keepdim=True).clamp_min(1e-6)
        self.register_buffer("log_rate", mean_profile.log())
        self.log_theta = nn.Parameter(torch.zeros(target_counts.shape[1], device=target_counts.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_rate.expand(x.shape[0], -1)

    def get_log_theta(self) -> torch.Tensor:
        return self.log_theta
