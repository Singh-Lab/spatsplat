import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from dotenv import load_dotenv

from spatsplat.config import DataConfig
from spatsplat.data import LoadedData


def _extract_spatial_xy_from_adata(adata: sc.AnnData) -> np.ndarray:
    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        if xy.ndim == 2 and xy.shape[1] >= 2:
            return xy[:, :2]

    obs_candidates = [
        ("x_centroid", "y_centroid"),
        ("x", "y"),
        ("X", "Y"),
        ("center_x", "center_y"),
        ("Centroid X µm", "Centroid Y µm"),
        ("Centroid X", "Centroid Y"),
        ("spatial_x", "spatial_y"),
        ("coord_x", "coord_y"),
    ]
    for x_col, y_col in obs_candidates:
        if x_col in adata.obs.columns and y_col in adata.obs.columns:
            return adata.obs[[x_col, y_col]].to_numpy(dtype=np.float32)

    raise RuntimeError(
        "Could not find spatial coordinates in the h5ad file. "
        "Expected adata.obsm['spatial'] or a recognized pair of obs columns."
    )


def load_embryonic_mouse_brain_atlas_dataset(
    control_name: str = "Control02",
    *,
    dataset_name: str = "embryonic_mouse_brain_atlas",
    device: str | None = None,
    config: DataConfig | None = None,
) -> LoadedData:
    config = config or DataConfig()
    load_dotenv()
    data_dir = os.getenv("SPATSPLAT_DATA_DIR")
    if not data_dir:
        raise RuntimeError("SPATSPLAT_DATA_DIR is not set")

    filename = f"{control_name}.h5ad" if not control_name.endswith(".h5ad") else control_name
    h5ad_path = Path(data_dir) / "raw" / dataset_name / filename
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Could not find embryonic mouse brain atlas file: {h5ad_path}")

    adata = sc.read_h5ad(h5ad_path)
    adata.obs_names = adata.obs_names.astype(str)

    xy = _extract_spatial_xy_from_adata(adata)
    adata.obs["x_centroid"] = xy[:, 0].astype(np.float32)
    adata.obs["y_centroid"] = xy[:, 1].astype(np.float32)
    adata.obsm["spatial"] = xy.astype(np.float32)
    adata.uns["spatial"] = {control_name: {"metadata": {}}}

    x_min, x_max = float(xy[:, 0].min()), float(xy[:, 0].max())
    y_min, y_max = float(xy[:, 1].min()), float(xy[:, 1].max())

    if hasattr(adata.X, "toarray"):
        counts = adata.X.toarray()
    else:
        counts = np.asarray(adata.X)
    adata.X = np.asarray(counts, dtype=np.float32)

    gene_list = list(adata.var_names.astype(str))
    gene_to_id = {gene: i for i, gene in enumerate(gene_list)}

    trans_df = pd.DataFrame(
        {
            "cell_id": pd.Series(dtype=str),
            "x": pd.Series(dtype=np.float32),
            "y": pd.Series(dtype=np.float32),
            "z": pd.Series(dtype=np.float32),
            "feature_name": pd.Series(dtype=str),
            "gene_id": pd.Series(dtype=np.int32),
            "x_norm": pd.Series(dtype=np.float32),
            "y_norm": pd.Series(dtype=np.float32),
            "z_norm": pd.Series(dtype=np.float32),
        }
    )

    return LoadedData(
        adata=adata,
        trans_df=trans_df,
        gene_list=gene_list,
        gene_to_id=gene_to_id,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        num_genes=len(gene_list),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=0.0,
        z_max=1.0,
    )
