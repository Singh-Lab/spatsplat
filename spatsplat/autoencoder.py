import torch
import torch.nn as nn

from .models import SplatBiologyDecoder


class GeneProfileEncoder(nn.Module):
    def __init__(self, num_genes: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeneProfileAutoencoder(nn.Module):
    def __init__(self, num_genes: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = GeneProfileEncoder(num_genes=num_genes, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = SplatBiologyDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, num_genes=num_genes)

    def forward(self, profiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(profiles)
        gene_logits, _ = self.decoder(z)
        recon_profiles = torch.softmax(gene_logits, dim=1)
        return z, gene_logits, recon_profiles


def profile_reconstruction_loss(target_profiles: torch.Tensor, pred_profiles: torch.Tensor) -> torch.Tensor:
    return -(target_profiles * pred_profiles.clamp_min(1e-8).log()).sum(dim=1).mean()


def evaluate_profile_autoencoder(
    model: GeneProfileAutoencoder,
    profiles: torch.Tensor,
    idx: torch.Tensor,
    *,
    batch_size: int = 512,
) -> dict[str, float | int]:
    profile_device = profiles.device
    model_device = next(model.parameters()).device
    idx = idx.to(device=profile_device)
    was_training = model.training
    model.eval()
    losses = []
    maes = []
    cosines = []

    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start : start + batch_size]
            batch_profiles = profiles[batch_idx].to(device=model_device)
            _, _, recon_profiles = model(batch_profiles)
            losses.append(profile_reconstruction_loss(batch_profiles, recon_profiles).item())
            maes.append((recon_profiles - batch_profiles).abs().mean(dim=1))
            recon_norm = recon_profiles / recon_profiles.norm(dim=1, keepdim=True).clamp_min(1e-8)
            target_norm = batch_profiles / batch_profiles.norm(dim=1, keepdim=True).clamp_min(1e-8)
            cosines.append((recon_norm * target_norm).sum(dim=1))

    if was_training:
        model.train()

    mae_tensor = torch.cat(maes) if maes else torch.empty(0, device=model_device)
    cosine_tensor = torch.cat(cosines) if cosines else torch.empty(0, device=model_device)
    return {
        "loss": float(sum(losses) / len(losses)) if losses else float("nan"),
        "mae": float(mae_tensor.mean().item()) if len(mae_tensor) else float("nan"),
        "cosine": float(cosine_tensor.mean().item()) if len(cosine_tensor) else float("nan"),
        "n": int(len(idx)),
    }


def train_profile_autoencoder(
    model: GeneProfileAutoencoder,
    profiles: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    *,
    epochs: int,
    batch_size: int = 512,
    lr: float | None = 1e-3,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float | None = 5.0,
    on_epoch_end=None,
) -> dict[str, list[float]]:
    if optimizer is None:
        if lr is None:
            raise RuntimeError("Either optimizer or lr must be provided")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_cosine": []}
    profile_device = profiles.device
    model_device = next(model.parameters()).device
    train_idx = train_idx.to(device=profile_device)
    val_idx = val_idx.to(device=profile_device)

    for epoch in range(epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), device=profile_device)]
        running_loss = 0.0
        num_batches = 0

        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            batch_profiles = profiles[batch_idx].to(device=model_device)
            _, _, recon_profiles = model(batch_profiles)
            loss = profile_reconstruction_loss(batch_profiles, recon_profiles)

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        train_loss = running_loss / max(1, num_batches)
        val_metrics = evaluate_profile_autoencoder(model, profiles, val_idx, batch_size=batch_size)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_mae"].append(float(val_metrics["mae"]))
        history["val_cosine"].append(float(val_metrics["cosine"]))

        if on_epoch_end is not None:
            on_epoch_end(epoch, train_loss, val_metrics)

    return history
