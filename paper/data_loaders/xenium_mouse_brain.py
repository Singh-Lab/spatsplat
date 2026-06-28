import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from dotenv import load_dotenv

from spatsplat.config import DataConfig
from spatsplat.data import LoadedData


def load_tiny_xenium_dataset(device: str | None = None, config: DataConfig | None = None) -> LoadedData:
    """
    Load the tiny Xenium mouse brain dataset.
    
    Requires SPATSPLAT_DATA_DIR environment variable to be set.
    
    Args:
        device: Device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        config: DataConfig for quality filtering. Defaults to DataConfig().
    
    Returns:
        LoadedData: Container with adata, transcripts, genes, spatial coordinates, etc.
    """
    config = config or DataConfig()
    load_dotenv()
    data_dir = os.getenv("SPATSPLAT_DATA_DIR")
    if not data_dir:
        raise RuntimeError("SPATSPLAT_DATA_DIR is not set")

    tiny_data_path = Path(data_dir) / "raw" / "tiny_dataset"
    adata = sc.read_10x_h5(tiny_data_path / "cell_feature_matrix.h5")
    adata.uns["spatial"] = {"dataset_1": {"metadata": {}}}

    cells_path = tiny_data_path / "cells.parquet"
    if not cells_path.exists():
        cells_path = tiny_data_path / "cells.csv.gz"

    cells_df = pd.read_parquet(cells_path) if cells_path.suffix == ".parquet" else pd.read_csv(cells_path)
    cells_df = cells_df.set_index("cell_id")
    cells_df.index = cells_df.index.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs = adata.obs.join(cells_df, how="inner")
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].values.astype(np.float32)

    trans_df = pd.read_parquet(
        tiny_data_path / "transcripts.parquet",
        columns=["cell_id", "x_location", "y_location", "z_location", "feature_name", "qv"],
    )
    trans_df = trans_df.rename(columns={"x_location": "x", "y_location": "y", "z_location": "z"})
    trans_df = trans_df[trans_df["qv"] >= config.qv_threshold].copy()
    trans_df["feature_name"] = trans_df["feature_name"].astype(str)
    trans_df["cell_id"] = trans_df["cell_id"].astype(str)
    trans_df = trans_df[
        ~trans_df["feature_name"].str.contains("Control|Unassigned|BLANK", case=False, na=False)
    ].copy()

    x_min, x_max = trans_df["x"].min(), trans_df["x"].max()
    y_min, y_max = trans_df["y"].min(), trans_df["y"].max()
    z_min, z_max = trans_df["z"].min(), trans_df["z"].max()

    trans_df["x_norm"] = (trans_df["x"] - x_min) / (x_max - x_min)
    trans_df["y_norm"] = (trans_df["y"] - y_min) / (y_max - y_min)
    trans_df["z_norm"] = (trans_df["z"] - z_min) / (z_max - z_min + 1e-8)

    gene_list = sorted(trans_df["feature_name"].unique())
    gene_to_id = {gene: i for i, gene in enumerate(gene_list)}
    trans_df["gene_id"] = trans_df["feature_name"].map(gene_to_id).astype(np.int32)

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
        z_min=z_min,
        z_max=z_max,
    )
