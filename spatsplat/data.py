from dataclasses import dataclass
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse as scipy_sparse

from .config import DataConfig


@dataclass
class LoadedData:
    adata: sc.AnnData
    trans_df: pd.DataFrame
    gene_list: list[str]
    gene_to_id: dict[str, int]
    device: str
    num_genes: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


@dataclass
class BuiltDataset:
    bin_features: torch.Tensor
    bin_targets: torch.Tensor
    cell_features: torch.Tensor
    cell_spans: torch.Tensor
    cell_targets: torch.Tensor
    off_tissue_points: torch.Tensor
    tissue_mask_tensor: torch.Tensor
    cell_support_mask: np.ndarray
    mask_grid_size: int
    num_bins: int
    transcript_active_bin_mask: np.ndarray
    active_bin_mask: np.ndarray


def _build_cell_support_mask(
    cell_coords: np.ndarray,
    cell_spans_np: np.ndarray,
    mask_grid_size: int,
    min_radius: float,
) -> np.ndarray:
    cell_support_mask = np.zeros((mask_grid_size, mask_grid_size), dtype=bool)
    for (cx, cy), (sx, sy) in zip(cell_coords, cell_spans_np):
        gx = cx * mask_grid_size
        gy = cy * mask_grid_size
        rx = max(min_radius, sx * mask_grid_size)
        ry = max(min_radius, sy * mask_grid_size)

        x0 = max(0, int(np.floor(gx - rx)))
        x1 = min(mask_grid_size - 1, int(np.ceil(gx + rx)))
        y0 = max(0, int(np.floor(gy - ry)))
        y1 = min(mask_grid_size - 1, int(np.ceil(gy + ry)))

        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        ellipse = ((xx + 0.5 - gx) / rx) ** 2 + ((yy + 0.5 - gy) / ry) ** 2 <= 1.0
        cell_support_mask[y0 : y1 + 1, x0 : x1 + 1] |= ellipse

    return cell_support_mask


def build_spatial_targets(loaded: LoadedData, config: DataConfig | None = None) -> BuiltDataset:
    config = config or DataConfig()
    trans_df = loaded.trans_df
    device = loaded.device

    x_bin = np.clip((trans_df["x_norm"].to_numpy() * config.bin_size).astype(np.int32), 0, config.bin_size - 1)
    y_bin = np.clip((trans_df["y_norm"].to_numpy() * config.bin_size).astype(np.int32), 0, config.bin_size - 1)
    bin_id = y_bin * config.bin_size + x_bin
    num_bins = config.bin_size * config.bin_size

    bin_counts = np.zeros((num_bins, loaded.num_genes), dtype=np.float32)
    np.add.at(bin_counts, (bin_id, trans_df["gene_id"].to_numpy()), 1.0)

    bin_centers = np.stack(
        [
            (np.arange(num_bins) % config.bin_size + 0.5) / config.bin_size,
            (np.arange(num_bins) // config.bin_size + 0.5) / config.bin_size,
        ],
        axis=1,
    ).astype(np.float32)
    bin_totals = bin_counts.sum(axis=1)
    transcript_active_bin_mask = bin_totals > 0

    cell_gene_df = trans_df.groupby(["cell_id", "gene_id"]).size().unstack(fill_value=0)
    cell_gene_df = cell_gene_df.reindex(columns=np.arange(loaded.num_genes), fill_value=0)
    shared_cells = loaded.adata.obs.index.intersection(cell_gene_df.index)

    cell_coords = loaded.adata.obs.loc[shared_cells, ["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    cell_coords[:, 0] = (cell_coords[:, 0] - loaded.x_min) / (loaded.x_max - loaded.x_min)
    cell_coords[:, 1] = (cell_coords[:, 1] - loaded.y_min) / (loaded.y_max - loaded.y_min)

    if "cell_area" in loaded.adata.obs.columns:
        cell_area = loaded.adata.obs.loc[shared_cells, "cell_area"].to_numpy(dtype=np.float32)
        cell_radius_um = np.sqrt(np.clip(cell_area, 1.0, None) / np.pi)
    else:
        cell_radius_um = np.full(
            len(shared_cells),
            fill_value=0.5 * min(
                (loaded.x_max - loaded.x_min) / config.bin_size,
                (loaded.y_max - loaded.y_min) / config.bin_size,
            ),
            dtype=np.float32,
        )

    cell_spans_np = np.stack(
        [
            cell_radius_um / (loaded.x_max - loaded.x_min),
            cell_radius_um / (loaded.y_max - loaded.y_min),
        ],
        axis=1,
    ).astype(np.float32)
    cell_counts = cell_gene_df.loc[shared_cells].to_numpy(dtype=np.float32)

    cell_support_mask = _build_cell_support_mask(
        cell_coords=cell_coords,
        cell_spans_np=cell_spans_np,
        mask_grid_size=config.mask_grid_size,
        min_radius=config.cell_mask_min_radius,
    )
    grid_centers = np.stack(
        [
            (np.arange(config.mask_grid_size * config.mask_grid_size) % config.mask_grid_size + 0.5)
            / config.mask_grid_size,
            (np.arange(config.mask_grid_size * config.mask_grid_size) // config.mask_grid_size + 0.5)
            / config.mask_grid_size,
        ],
        axis=1,
    ).astype(np.float32)
    off_tissue_mask = ~cell_support_mask.reshape(-1)
    off_tissue_points_np = grid_centers[off_tissue_mask]
    tissue_mask_tensor = torch.from_numpy(cell_support_mask.astype(np.float32)).to(device)[None, None, :, :]

    bin_mask_x = np.clip((bin_centers[:, 0] * config.mask_grid_size).astype(np.int32), 0, config.mask_grid_size - 1)
    bin_mask_y = np.clip((bin_centers[:, 1] * config.mask_grid_size).astype(np.int32), 0, config.mask_grid_size - 1)
    bin_cell_support_mask = cell_support_mask[bin_mask_y, bin_mask_x]
    active_bin_mask = transcript_active_bin_mask & bin_cell_support_mask

    return BuiltDataset(
        bin_features=torch.from_numpy(bin_centers[active_bin_mask]).to(device),
        bin_targets=torch.from_numpy(bin_counts[active_bin_mask]).to(device),
        cell_features=torch.from_numpy(cell_coords).to(device),
        cell_spans=torch.from_numpy(cell_spans_np).to(device),
        cell_targets=torch.from_numpy(cell_counts).to(device),
        off_tissue_points=torch.from_numpy(off_tissue_points_np).to(device),
        tissue_mask_tensor=tissue_mask_tensor,
        cell_support_mask=cell_support_mask,
        mask_grid_size=config.mask_grid_size,
        num_bins=num_bins,
        transcript_active_bin_mask=transcript_active_bin_mask,
        active_bin_mask=active_bin_mask,
    )


def build_cell_only_targets(loaded: LoadedData, config: DataConfig | None = None) -> BuiltDataset:
    config = config or DataConfig()
    device = loaded.device

    cell_coords = loaded.adata.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    cell_coords[:, 0] = (cell_coords[:, 0] - loaded.x_min) / (loaded.x_max - loaded.x_min + 1e-8)
    cell_coords[:, 1] = (cell_coords[:, 1] - loaded.y_min) / (loaded.y_max - loaded.y_min + 1e-8)

    if "cell_area" in loaded.adata.obs.columns:
        cell_area = loaded.adata.obs["cell_area"].to_numpy(dtype=np.float32)
        cell_radius_units = np.sqrt(np.clip(cell_area, 1.0, None) / np.pi)
    else:
        cell_radius_units = np.full(len(cell_coords), fill_value=1.0, dtype=np.float32)

    cell_spans_np = np.stack(
        [
            cell_radius_units / (loaded.x_max - loaded.x_min + 1e-8),
            cell_radius_units / (loaded.y_max - loaded.y_min + 1e-8),
        ],
        axis=1,
    ).astype(np.float32)
    if scipy_sparse.issparse(loaded.adata.X):
        cell_counts = loaded.adata.X.toarray().astype(np.float32)
    else:
        cell_counts = np.asarray(loaded.adata.X, dtype=np.float32)

    cell_support_mask = _build_cell_support_mask(
        cell_coords=cell_coords,
        cell_spans_np=cell_spans_np,
        mask_grid_size=config.mask_grid_size,
        min_radius=config.cell_mask_min_radius,
    )
    grid_centers = np.stack(
        [
            (np.arange(config.mask_grid_size * config.mask_grid_size) % config.mask_grid_size + 0.5)
            / config.mask_grid_size,
            (np.arange(config.mask_grid_size * config.mask_grid_size) // config.mask_grid_size + 0.5)
            / config.mask_grid_size,
        ],
        axis=1,
    ).astype(np.float32)
    off_tissue_mask = ~cell_support_mask.reshape(-1)
    off_tissue_points_np = grid_centers[off_tissue_mask]
    tissue_mask_tensor = torch.from_numpy(cell_support_mask.astype(np.float32)).to(device)[None, None, :, :]

    num_bins = config.bin_size * config.bin_size
    empty_bin_features = torch.empty((0, 2), dtype=torch.float32, device=device)
    empty_bin_targets = torch.empty((0, loaded.num_genes), dtype=torch.float32, device=device)
    empty_bin_mask = np.zeros(num_bins, dtype=bool)

    return BuiltDataset(
        bin_features=empty_bin_features,
        bin_targets=empty_bin_targets,
        cell_features=torch.from_numpy(cell_coords).to(device),
        cell_spans=torch.from_numpy(cell_spans_np).to(device),
        cell_targets=torch.from_numpy(cell_counts).to(device),
        off_tissue_points=torch.from_numpy(off_tissue_points_np).to(device),
        tissue_mask_tensor=tissue_mask_tensor,
        cell_support_mask=cell_support_mask,
        mask_grid_size=config.mask_grid_size,
        num_bins=num_bins,
        transcript_active_bin_mask=empty_bin_mask,
        active_bin_mask=empty_bin_mask.copy(),
    )


def make_split_indices(
    num_items: int,
    *,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    order = torch.randperm(num_items, generator=g, device=device)
    train_end = int(num_items * train_frac)
    val_end = train_end + int(num_items * val_frac)
    return order[:train_end], order[train_end:val_end], order[val_end:]
