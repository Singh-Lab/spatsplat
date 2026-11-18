import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from dotenv import load_dotenv

from spatsplat.config import DataConfig
from spatsplat.data import LoadedData


def load_merfish_mouse_brain_dataset(
    sample_name: str = "S1R1",
    *,
    dataset_name: str = "merfish-mousebraindata",
    include_blank_probes: bool = False,
    device: str | None = None,
    config: DataConfig | None = None,
) -> LoadedData:
    """
    Load the MERFISH mouse brain dataset.
    
    Requires SPATSPLAT_DATA_DIR environment variable to be set.
    
    Args:
        sample_name: Sample identifier (e.g., "S1R1").
        dataset_name: Name of dataset directory within SPATSPLAT_DATA_DIR/raw/.
        include_blank_probes: Whether to include Blank probe measurements.
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

    base_path = Path(data_dir) / "raw" / dataset_name
    expr_path = base_path / f"cell_by_gene_{sample_name}.csv"
    meta_path = base_path / f"cell_metadata_{sample_name}.csv"
    transcript_path = base_path / f"detected_transcripts_{sample_name}.csv"

    for path in [expr_path, meta_path]:
        if not path.exists():
            raise FileNotFoundError(f"Could not find required MERFISH file: {path}")

    expr_df = pd.read_csv(expr_path)
    meta_df = pd.read_csv(meta_path)

    id_col = "Unnamed: 0"
    if id_col not in expr_df.columns or id_col not in meta_df.columns:
        raise RuntimeError("Expected MERFISH cell ID column 'Unnamed: 0' in expression and metadata tables")

    expr_df = expr_df.rename(columns={id_col: "cell_id"}).copy()
    meta_df = meta_df.rename(columns={id_col: "cell_id"}).copy()
    expr_df["cell_id"] = expr_df["cell_id"].astype(str)
    meta_df["cell_id"] = meta_df["cell_id"].astype(str)

    merged = meta_df.merge(expr_df, on="cell_id", how="inner", suffixes=("_meta", ""))
    if merged.empty:
        raise RuntimeError(f"No shared cells found between {expr_path.name} and {meta_path.name}")

    gene_cols = [c for c in expr_df.columns if c != "cell_id"]
    if not include_blank_probes:
        gene_cols = [c for c in gene_cols if not c.startswith("Blank-")]

    counts = merged[gene_cols].to_numpy(dtype=np.float32)
    adata = sc.AnnData(X=counts)
    adata.obs_names = merged["cell_id"].astype(str).to_numpy()
    adata.var_names = pd.Index(gene_cols, dtype=str)
    adata.obs = merged.set_index("cell_id")
    adata.obs_names = adata.obs.index.astype(str)
    adata.obs["x_centroid"] = adata.obs["center_x"].astype(np.float32)
    adata.obs["y_centroid"] = adata.obs["center_y"].astype(np.float32)
    adata.obs["cell_area"] = adata.obs["volume"].astype(np.float32)
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    adata.uns["spatial"] = {sample_name: {"metadata": {}}}

    x = adata.obs["x_centroid"].to_numpy(dtype=np.float32)
    y = adata.obs["y_centroid"].to_numpy(dtype=np.float32)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    gene_to_id = {gene: i for i, gene in enumerate(gene_cols)}
    if transcript_path.exists():
        raw_trans_df = pd.read_csv(transcript_path)
        raw_trans_df["gene"] = raw_trans_df["gene"].astype(str)
        if not include_blank_probes:
            raw_trans_df = raw_trans_df[~raw_trans_df["gene"].str.startswith("Blank-")].copy()
        raw_trans_df = raw_trans_df[raw_trans_df["gene"].isin(gene_to_id)].copy()

        trans_df = pd.DataFrame(
            {
                "cell_id": pd.Series(dtype=str, index=raw_trans_df.index),
                "x": raw_trans_df["global_x"].astype(np.float32),
                "y": raw_trans_df["global_y"].astype(np.float32),
                "z": raw_trans_df["global_z"].astype(np.float32),
                "feature_name": raw_trans_df["gene"],
            }
        )
        trans_df["x_norm"] = (trans_df["x"] - x_min) / (x_max - x_min + 1e-8)
        trans_df["y_norm"] = (trans_df["y"] - y_min) / (y_max - y_min + 1e-8)
        z_min = float(trans_df["z"].min()) if len(trans_df) else 0.0
        z_max = float(trans_df["z"].max()) if len(trans_df) else 1.0
        trans_df["z_norm"] = (trans_df["z"] - z_min) / (z_max - z_min + 1e-8)
        trans_df["gene_id"] = trans_df["feature_name"].map(gene_to_id).astype(np.int32)
        trans_df = trans_df[
            ["cell_id", "x", "y", "z", "feature_name", "gene_id", "x_norm", "y_norm", "z_norm"]
        ].copy()
    else:
        z_min, z_max = 0.0, 1.0
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
        gene_list=list(gene_cols),
        gene_to_id=gene_to_id,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        num_genes=len(gene_cols),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )
