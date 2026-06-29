import torch
import torch.nn as nn
import torch.nn.functional as F



class SplatBiologyDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_genes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gene_head = nn.Linear(hidden_dim, num_genes)
        self.intensity_head = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(z)
        return self.gene_head(h), self.intensity_head(h)


class GaussianSplatField(nn.Module):
    def __init__(
        self,
        num_splats: int,
        latent_dim: int,
        hidden_dim: int,
        num_genes: int,
        *,
        init_log_scale: float = -4.5,
        kernel_sharpness: float = 0.5,
        top_k_splats: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_genes = num_genes
        self.init_log_scale = init_log_scale
        self.kernel_sharpness = kernel_sharpness
        self.top_k_splats = top_k_splats
        self.num_splats = num_splats
        self.means = nn.Parameter(torch.rand(num_splats, 2))
        self.log_scales = nn.Parameter(torch.full((num_splats, 2), init_log_scale))
        self.rotation = nn.Parameter(torch.zeros(num_splats, 1))
        self.opacity_logits = nn.Parameter(torch.zeros(num_splats, 1))
        self.latents = nn.Parameter(torch.randn(num_splats, latent_dim) * 0.01)
        self.log_theta = nn.Parameter(torch.zeros(num_genes))
        self.decoder = SplatBiologyDecoder(latent_dim, hidden_dim, num_genes)

    def gene_rates(self) -> torch.Tensor:
        gene_logits, log_intensity = self.decoder(self.latents)
        gene_composition = torch.softmax(gene_logits, dim=-1)
        total_intensity = F.softplus(log_intensity)
        return total_intensity * gene_composition

    def get_log_theta(self) -> torch.Tensor:
        return self.log_theta

    def compute_weights(self, query_xy: torch.Tensor) -> torch.Tensor:
        scales = torch.exp(self.log_scales).clamp_min(1e-4)
        deltas = query_xy[:, None, :] - self.means[None, :, :]

        cos_t = torch.cos(self.rotation).view(1, -1)
        sin_t = torch.sin(self.rotation).view(1, -1)
        dx = deltas[..., 0]
        dy = deltas[..., 1]

        local_x = cos_t * dx + sin_t * dy
        local_y = -sin_t * dx + cos_t * dy

        scaled_x = local_x / scales[:, 0].view(1, -1)
        scaled_y = local_y / scales[:, 1].view(1, -1)
        dist2 = scaled_x.square() + scaled_y.square()

        weights = torch.exp(-self.kernel_sharpness * dist2) * torch.sigmoid(self.opacity_logits).squeeze(-1)
        if self.top_k_splats is not None and 0 < self.top_k_splats < weights.shape[1]:
            top_weights, top_idx = torch.topk(weights, k=self.top_k_splats, dim=1, largest=True, sorted=False)
            sparse_weights = torch.zeros_like(weights)
            sparse_weights.scatter_(1, top_idx, top_weights)
            weights = sparse_weights
        return weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

    def mixed_rates(self, query_xy: torch.Tensor) -> torch.Tensor:
        return self.compute_weights(query_xy) @ self.gene_rates()

    def total_intensity(self, query_xy: torch.Tensor) -> torch.Tensor:
        return self.mixed_rates(query_xy).sum(dim=-1)

    def forward(self, query_xy: torch.Tensor) -> torch.Tensor:
        return (self.mixed_rates(query_xy) + 1e-6).log()


class AmplitudeSplatField(nn.Module):
    def __init__(
        self,
        num_splats: int,
        latent_dim: int,
        hidden_dim: int,
        num_genes: int,
        *,
        init_log_scale: float = -4.5,
        kernel_sharpness: float = 0.5,
        top_k_splats: int | None = None,
        init_log_amplitude: float = 0.0,
    ):
        super().__init__()
        self.splat_field = GaussianSplatField(
            num_splats=num_splats,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_genes=num_genes,
            init_log_scale=init_log_scale,
            kernel_sharpness=kernel_sharpness,
            top_k_splats=top_k_splats,
        )
        self.log_amplitudes = nn.Parameter(torch.full((num_splats, 1), init_log_amplitude))

    @property
    def latent_dim(self) -> int:
        return self.splat_field.latent_dim

    @property
    def hidden_dim(self) -> int:
        return self.splat_field.hidden_dim

    @property
    def num_genes(self) -> int:
        return self.splat_field.num_genes

    @property
    def init_log_scale(self) -> float:
        return self.splat_field.init_log_scale

    @property
    def top_k_splats(self) -> int | None:
        return self.splat_field.top_k_splats

    @property
    def kernel_sharpness(self) -> float:
        return self.splat_field.kernel_sharpness

    @property
    def num_splats(self) -> int:
        return self.splat_field.num_splats

    @property
    def means(self) -> nn.Parameter:
        return self.splat_field.means

    @property
    def log_scales(self) -> nn.Parameter:
        return self.splat_field.log_scales

    @property
    def rotation(self) -> nn.Parameter:
        return self.splat_field.rotation

    @property
    def opacity_logits(self) -> nn.Parameter:
        return self.splat_field.opacity_logits

    @property
    def latents(self) -> nn.Parameter:
        return self.splat_field.latents

    @property
    def log_theta(self) -> nn.Parameter:
        return self.splat_field.log_theta

    @property
    def decoder(self) -> SplatBiologyDecoder:
        return self.splat_field.decoder

    def get_log_theta(self) -> torch.Tensor:
        return self.splat_field.get_log_theta()

    def compute_weights(self, query_xy: torch.Tensor) -> torch.Tensor:
        return self.splat_field.compute_weights(query_xy)

    def splat_composition(self) -> torch.Tensor:
        gene_logits, _ = self.splat_field.decoder(self.splat_field.latents)
        return torch.softmax(gene_logits, dim=-1)

    def splat_amplitudes(self) -> torch.Tensor:
        return torch.exp(self.log_amplitudes).clamp_min(1e-6)

    def splat_rates(self) -> torch.Tensor:
        return self.splat_amplitudes() * self.splat_composition()

    def mixed_composition(self, query_xy: torch.Tensor) -> torch.Tensor:
        return self.compute_weights(query_xy) @ self.splat_composition()

    def total_intensity(self, query_xy: torch.Tensor) -> torch.Tensor:
        return (self.compute_weights(query_xy) @ self.splat_amplitudes()).squeeze(-1)

    def mixed_rates(self, query_xy: torch.Tensor) -> torch.Tensor:
        return self.compute_weights(query_xy) @ self.splat_rates()

    def forward(self, query_xy: torch.Tensor) -> torch.Tensor:
        return (self.mixed_rates(query_xy) + 1e-6).log()
