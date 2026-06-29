import torch
import torch.nn as nn


class CountMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierFeatureMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        num_frequencies: int = 32,
        sigma: float = 10.0,
    ):
        super().__init__()
        self.register_buffer("B", torch.randn(input_dim, num_frequencies) * sigma)
        encoded_dim = num_frequencies * 2
        self.net = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        projection = 2.0 * torch.pi * (x @ self.B)
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.encode(x))
