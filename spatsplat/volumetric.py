import math

import torch
import torch.nn as nn

from .models import AmplitudeSplatField, GaussianSplatField, SplatBiologyDecoder


class VolumetricGaussianSplatField(GaussianSplatField):
    """Gaussian splats with local z support and optional bounded x/y tilt.

    Nonzero tilt requires coordinates normalized with a single shared axis scale
    so angles retain their meaning. Zero tilt preserves the legacy state schema.
    """

    def __init__(
        self,
        num_splats: int,
        latent_dim: int,
        hidden_dim: int,
        num_genes: int,
        *,
        init_log_scale_xy: float = -3.2,
        init_log_scale_z: float = -1.9,
        kernel_sharpness: float = 0.5,
        top_k_splats: int | None = None,
        max_tilt_degrees: float = 0.0,
    ):
        nn.Module.__init__(self)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_genes = num_genes
        self.init_log_scale = init_log_scale_xy
        self.init_log_scale_xy = init_log_scale_xy
        self.init_log_scale_z = init_log_scale_z
        self.kernel_sharpness = kernel_sharpness
        self.top_k_splats = top_k_splats
        self.num_splats = num_splats
        if not 0 <= max_tilt_degrees < 90:
            raise ValueError('max_tilt_degrees must be in [0, 90)')
        self.max_tilt_degrees = float(max_tilt_degrees)
        if max_tilt_degrees:
            self.tilt_raw = nn.Parameter(torch.zeros(num_splats, 2))


        self.means = nn.Parameter(torch.rand(num_splats, 3))
        initial_scales = torch.full((num_splats, 3), init_log_scale_xy)
        initial_scales[:, 2] = init_log_scale_z
        self.log_scales = nn.Parameter(initial_scales)
        self.rotation = nn.Parameter(torch.zeros(num_splats, 1))
        self.opacity_logits = nn.Parameter(torch.zeros(num_splats, 1))
        self.latents = nn.Parameter(torch.randn(num_splats, latent_dim) * 0.01)
        self.decoder = SplatBiologyDecoder(latent_dim, hidden_dim, num_genes)

    def tilt_vectors(self) -> torch.Tensor:
        """Bound the combined x/y tilt; coordinates must use equal axis units."""
        if not self.max_tilt_degrees:
            return self.means.new_zeros((self.num_splats, 2))
        return (math.radians(self.max_tilt_degrees) * self.tilt_raw
                / torch.sqrt(1 + self.tilt_raw.square().sum(dim=1, keepdim=True)))

    def tilt_matrices(self) -> torch.Tensor:
        """Rodrigues rotation, with finite derivatives at the zero-tilt initialization."""
        v = self.tilt_vectors()
        x, y = v.unbind(dim=1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero, zero, y, zero, zero, -x, -y, x, zero), dim=1).reshape(-1, 3, 3)
        theta = torch.linalg.vector_norm(v, dim=1)
        a = torch.sinc(theta / math.pi)[:, None, None]
        b = (0.5 * torch.sinc(theta / (2 * math.pi)).square())[:, None, None]
        return torch.eye(3, device=v.device, dtype=v.dtype)[None] + a * skew + b * (skew @ skew)

    def compute_weights(self, query_xyz: torch.Tensor) -> torch.Tensor:
        if query_xyz.ndim != 2 or query_xyz.shape[1] != 3:
            raise RuntimeError(
                f"Expected query_xyz with shape [N, 3], got {tuple(query_xyz.shape)}"
            )

        scales = torch.exp(self.log_scales).clamp_min(1e-4)
        deltas = query_xyz[:, None, :] - self.means[None, :, :]
        if self.max_tilt_degrees:
            # World -> untilted frame, followed by the existing in-plane rotation.
            deltas = torch.einsum('nsi,sij->nsj', deltas, self.tilt_matrices())

        cos_t = torch.cos(self.rotation).view(1, -1)
        sin_t = torch.sin(self.rotation).view(1, -1)
        dx = deltas[..., 0]
        dy = deltas[..., 1]

        local_x = cos_t * dx + sin_t * dy
        local_y = -sin_t * dx + cos_t * dy
        local_z = deltas[..., 2]

        scaled_x = local_x / scales[:, 0].view(1, -1)
        scaled_y = local_y / scales[:, 1].view(1, -1)
        scaled_z = local_z / scales[:, 2].view(1, -1)
        dist2 = scaled_x.square() + scaled_y.square() + scaled_z.square()

        log_opacity = torch.log(torch.sigmoid(self.opacity_logits).squeeze(-1).clamp_min(1e-8))
        logits = -self.kernel_sharpness * dist2 + log_opacity.view(1, -1)
        if self.top_k_splats is not None and 0 < self.top_k_splats < logits.shape[1]:
            top_logits, top_idx = torch.topk(
                logits, k=self.top_k_splats, dim=1, largest=True, sorted=False
            )
            sparse_logits = torch.full_like(logits, -torch.inf)
            sparse_logits.scatter_(1, top_idx, top_logits)
            logits = sparse_logits
        return torch.softmax(logits, dim=1)

    def project_geometry_(
        self,
        *,
        min_scale_xy: float = 0.005,
        max_scale_xy: float = 0.30,
        min_scale_z: float = 0.08,
        max_scale_z: float = 0.30,
    ) -> None:
        if not (0 < min_scale_xy <= max_scale_xy):
            raise ValueError("Expected 0 < min_scale_xy <= max_scale_xy")
        if not (0 < min_scale_z <= max_scale_z):
            raise ValueError("Expected 0 < min_scale_z <= max_scale_z")
        with torch.no_grad():
            self.means.clamp_(0.0, 1.0)
            self.log_scales[:, :2].clamp_(math.log(min_scale_xy), math.log(max_scale_xy))
            self.log_scales[:, 2].clamp_(math.log(min_scale_z), math.log(max_scale_z))


class VolumetricAmplitudeSplatField(AmplitudeSplatField):
    """Amplitude splat field for registered serial-section volumes."""

    def __init__(
        self,
        num_splats: int,
        latent_dim: int,
        hidden_dim: int,
        num_genes: int,
        *,
        init_log_scale_xy: float = -3.2,
        init_log_scale_z: float = -1.9,
        kernel_sharpness: float = 0.5,
        top_k_splats: int | None = None,
        init_log_amplitude: float = 0.0,
        max_tilt_degrees: float = 0.0,
    ):
        nn.Module.__init__(self)
        self.splat_field = VolumetricGaussianSplatField(
            num_splats=num_splats,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_genes=num_genes,
            init_log_scale_xy=init_log_scale_xy,
            init_log_scale_z=init_log_scale_z,
            kernel_sharpness=kernel_sharpness,
            top_k_splats=top_k_splats,
            max_tilt_degrees=max_tilt_degrees,
        )
        self.log_amplitudes = nn.Parameter(
            torch.full((num_splats, 1), init_log_amplitude)
        )

    @property
    def init_log_scale_xy(self) -> float:
        return self.splat_field.init_log_scale_xy

    @property
    def init_log_scale_z(self) -> float:
        return self.splat_field.init_log_scale_z

    def project_geometry_(
        self,
        *,
        min_scale_xy: float = 0.005,
        max_scale_xy: float = 0.30,
        min_scale_z: float = 0.08,
        max_scale_z: float = 0.30,
    ) -> None:
        self.splat_field.project_geometry_(
            min_scale_xy=min_scale_xy,
            max_scale_xy=max_scale_xy,
            min_scale_z=min_scale_z,
            max_scale_z=max_scale_z,
        )
