import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from dotenv import load_dotenv
from scipy import io as scipy_io
from scipy import sparse as scipy_sparse

from spatsplat.config import DataConfig
from spatsplat.data import LoadedData


def load_pdac_mets_visium_dataset(
    sample_name: str | None = None,
    *,
    dataset_name: str = "pdac_mets_visium_gse272362",
    origin: str | None = None,
    sample_id: str | None = None,
    only_in_tissue: bool = True,
    allow_full_dataset: bool = False,
    device: str | None = None,
    config: DataConfig | None = None,
) -> LoadedData:
    config = config or DataConfig()
    load_dotenv()
    data_dir = os.getenv("SPATSPLAT_DATA_DIR")
    if not data_dir:
        raise RuntimeError("SPATSPLAT_DATA_DIR is not set")

    base_path = Path(data_dir) / "processed" / dataset_name
    counts_path = base_path / "spatial_counts.mtx"
    genes_path = base_path / "spatial_counts_genes.tsv.gz"
    barcodes_path = base_path / "spatial_counts_barcodes.tsv.gz"
    metadata_path = base_path / "spot_metadata.csv.gz"

    for path in [counts_path, genes_path, barcodes_path, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(f"Could not find required PDAC Visium file: {path}")

    meta_df = pd.read_csv(metadata_path, low_memory=False)
    meta_df["cell_id"] = meta_df["cell_id"].astype(str)

    mask = pd.Series(True, index=meta_df.index)
    if sample_name is not None:
        mask &= meta_df["image_name"].astype(str) == str(sample_name)
    if origin is not None:
        mask &= meta_df["Origin"].astype(str) == str(origin)
    if sample_id is not None:
        mask &= meta_df["Sample_ID2"].astype(str) == str(sample_id)
    if only_in_tissue and "tissue" in meta_df.columns:
        mask &= meta_df["tissue"].fillna(0).astype(int) == 1

    selected_meta = meta_df.loc[mask].copy()
    if selected_meta.empty:
        raise RuntimeError(
            "No PDAC Visium spots matched the requested filters "
            f"(sample_name={sample_name}, origin={origin}, sample_id={sample_id}, only_in_tissue={only_in_tissue})"
        )
    if sample_name is None and origin is None and sample_id is None and not allow_full_dataset:
        raise RuntimeError(
            "Refusing to load the full PDAC Visium dataset without an explicit filter. "
            "Pass sample_name, origin, or sample_id, or set allow_full_dataset=True."
        )

    genes_df = pd.read_csv(genes_path, header=None, compression="gzip")
    gene_list = genes_df.iloc[:, 0].astype(str).tolist()
    barcodes_df = pd.read_csv(barcodes_path, header=None, compression="gzip")
    all_barcodes = barcodes_df.iloc[:, 0].astype(str).tolist()

    barcode_to_idx = {barcode: i for i, barcode in enumerate(all_barcodes)}
    selected_indices = selected_meta["cell_id"].map(barcode_to_idx)
    if selected_indices.isna().any():
        missing = selected_meta.loc[selected_indices.isna(), "cell_id"].tolist()[:10]
        raise RuntimeError(f"Failed to align selected PDAC spots to count-matrix barcodes. Examples: {missing}")
    selected_indices_np = selected_indices.to_numpy(dtype=np.int64)

    counts_sparse = scipy_io.mmread(counts_path)
    if not scipy_sparse.issparse(counts_sparse):
        counts_sparse = scipy_sparse.csr_matrix(counts_sparse)
    else:
        counts_sparse = counts_sparse.tocsr()
    counts_sparse = counts_sparse[:, selected_indices_np].transpose().tocsr()

    x = selected_meta["imagecol"].to_numpy(dtype=np.float32)
    y = selected_meta["imagerow"].to_numpy(dtype=np.float32)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    adata = sc.AnnData(X=counts_sparse)
    adata.obs_names = selected_meta["cell_id"].astype(str).to_numpy()
    adata.var_names = pd.Index(gene_list, dtype=str)
    adata.obs = selected_meta.set_index("cell_id")
    adata.obs_names = adata.obs.index.astype(str)
    adata.obs["x_centroid"] = adata.obs["imagecol"].astype(np.float32)
    adata.obs["y_centroid"] = adata.obs["imagerow"].astype(np.float32)
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)

    sample_key = sample_name or origin or sample_id or dataset_name
    adata.uns["spatial"] = {sample_key: {"metadata": {}}}

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
        gene_to_id={gene: i for i, gene in enumerate(gene_list)},
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        num_genes=len(gene_list),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=0.0,
        z_max=1.0,
    )
