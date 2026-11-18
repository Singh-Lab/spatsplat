import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from dotenv import load_dotenv

from spatsplat.config import DataConfig
from spatsplat.data import LoadedData


def load_cosmx_nsclc_dataset(
    sample_name: str = "Lung5_Rep1",
    *,
    dataset_name: str = "cosmx_nsclc",
    include_negative_probes: bool = False,
    device: str | None = None,
    config: DataConfig | None = None,
) -> LoadedData:
    config = config or DataConfig()
    load_dotenv()
    data_dir = os.getenv("SPATSPLAT_DATA_DIR")
    if not data_dir:
        raise RuntimeError("SPATSPLAT_DATA_DIR is not set")

    sample_dir = Path(data_dir) / "raw" / dataset_name / "nanostring_nsclc" / f"{sample_name}-Flat_files_and_images"
    if not sample_dir.exists():
        raise FileNotFoundError(f"Could not find CosMx sample directory: {sample_dir}")

    expr_candidates = sorted(sample_dir.glob("*_exprMat_file_SUBSET.csv"))
    meta_candidates = sorted(sample_dir.glob("*_metadata_file_SUBSET.csv"))
    if not expr_candidates or not meta_candidates:
        raise FileNotFoundError(f"Missing expression or metadata CSVs in {sample_dir}")

    expr_df = pd.read_csv(expr_candidates[0])
    meta_df = pd.read_csv(meta_candidates[0])

    id_cols = ["fov", "cell_ID"]
    expr_df = expr_df.copy()
    meta_df = meta_df.copy()
    expr_df["cell_key"] = expr_df["fov"].astype(str) + "_" + expr_df["cell_ID"].astype(str)
    meta_df["cell_key"] = meta_df["fov"].astype(str) + "_" + meta_df["cell_ID"].astype(str)

    merged = meta_df.merge(expr_df, on=["fov", "cell_ID", "cell_key"], how="inner", suffixes=("_meta", ""))
    if merged.empty:
        raise RuntimeError(f"No shared cells found between metadata and expression tables in {sample_dir}")

    gene_cols = [c for c in expr_df.columns if c not in id_cols + ["cell_key"]]
    if not include_negative_probes:
        gene_cols = [c for c in gene_cols if not c.startswith("NegPrb")]

    counts = merged[gene_cols].to_numpy(dtype=np.float32)
    adata = sc.AnnData(X=counts)
    adata.obs_names = merged["cell_key"].astype(str).to_numpy()
    adata.var_names = pd.Index(gene_cols, dtype=str)
    adata.obs = merged.set_index("cell_key")
    adata.obs_names = adata.obs.index.astype(str)
    adata.obs["x_centroid"] = adata.obs["CenterX_global_px"].astype(np.float32)
    adata.obs["y_centroid"] = adata.obs["CenterY_global_px"].astype(np.float32)
    adata.obs["cell_area"] = adata.obs["Area"].astype(np.float32)
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    adata.uns["spatial"] = {sample_name: {"metadata": {}}}

    x = adata.obs["x_centroid"].to_numpy(dtype=np.float32)
    y = adata.obs["y_centroid"].to_numpy(dtype=np.float32)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

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
        gene_to_id={gene: i for i, gene in enumerate(gene_cols)},
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        num_genes=len(gene_cols),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=0.0,
        z_max=1.0,
    )
