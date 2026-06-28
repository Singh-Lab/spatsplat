from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from .models import GaussianSplatField


def _maybe_asdict(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    return value


def _model_init_kwargs(model: GaussianSplatField) -> dict[str, Any]:
    return {
        "num_splats": int(model.num_splats),
        "latent_dim": int(model.latent_dim),
        "hidden_dim": int(model.hidden_dim),
        "num_genes": int(model.num_genes),
        "init_log_scale": float(model.init_log_scale),
        "kernel_sharpness": float(model.kernel_sharpness),
        "top_k_splats": None if model.top_k_splats is None else int(model.top_k_splats),
    }


def save_splat_checkpoint(
    path: str | Path,
    model: GaussianSplatField,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    data_config: Any = None,
    splat_config: Any = None,
    training_config: Any = None,
    epoch: int | None = None,
    metrics: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    checkpoint = {
        "checkpoint_type": "gaussian_splat_field",
        "model_class": "GaussianSplatField",
        "model_init": _model_init_kwargs(model),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "data_config": _maybe_asdict(data_config),
        "splat_config": _maybe_asdict(splat_config),
        "training_config": _maybe_asdict(training_config),
        "epoch": epoch,
        "metrics": metrics or {},
        "extra_state": extra_state or {},
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def load_splat_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[GaussianSplatField, dict[str, Any]]:
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location or "cpu")

    if checkpoint.get("model_class") != "GaussianSplatField":
        raise RuntimeError(f"Unsupported checkpoint model class: {checkpoint.get('model_class')}")

    model = GaussianSplatField(**checkpoint["model_init"])
    model.load_state_dict(checkpoint["model_state"])

    if map_location is not None:
        model = model.to(map_location)

    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return model, checkpoint
