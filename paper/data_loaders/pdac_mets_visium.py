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


def select_pdac_patient_triads(
    *,
    dataset_name: str = "pdac_mets_visium_gse272362",
    min_tumor_fraction: float = 0.20,
    min_median_reads: float = 1000.0,
    tumor_label: str = "Tumor Epithelial cells",
) -> dict[str, dict[str, str]]:
    """Select patients with a full primary/lymph-node-met/liver-met triad whose
    primary (Pancreas) tumor slice clears a cancer-content bar.

    Returns a mapping ``{patient_id: {"primary": image_name, "lymph_node_met":
    image_name, "liver_sample": image_name}}`` with the same role keys the
    single-patient PDAC notebook already uses, so it can be dropped in as a
    replacement for a hardcoded per-patient sample dict.
    """
    load_dotenv()
    data_dir = os.getenv("SPATSPLAT_DATA_DIR")
    if not data_dir:
        raise RuntimeError("SPATSPLAT_DATA_DIR is not set")

    metadata_path = Path(data_dir) / "processed" / dataset_name / "spot_metadata.csv.gz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find PDAC Visium spot metadata: {metadata_path}")

    meta_df = pd.read_csv(metadata_path, low_memory=False)
    if "tissue" in meta_df.columns:
        meta_df = meta_df.loc[meta_df["tissue"].fillna(0).astype(int) == 1]

    origin_to_role = {
        "Pancreas": "primary",
        "Lymph node": "lymph_node_met",
        "Liver": "liver_sample",
    }
    meta_df = meta_df.loc[meta_df["Origin"].astype(str).isin(origin_to_role)].copy()
    meta_df["is_tumor"] = meta_df["first_type"].astype(str).eq(tumor_label)
    meta_df["is_reject"] = meta_df["spot_class"].astype(str).eq("reject")

    triads: dict[str, dict[str, str]] = {}
    for patient, patient_rows in meta_df.groupby("patient"):
        role_to_image: dict[str, str] = {}
        for origin, role in origin_to_role.items():
            origin_rows = patient_rows.loc[patient_rows["Origin"].astype(str) == origin]
            image_names = origin_rows["image_name"].astype(str).unique()
            if len(image_names) != 1:
                continue
            role_to_image[role] = image_names[0]

        if set(role_to_image) != {"primary", "lymph_node_met", "liver_sample"}:
            continue

        primary_rows = patient_rows.loc[patient_rows["image_name"].astype(str) == role_to_image["primary"]]
        primary_non_reject = primary_rows.loc[~primary_rows["is_reject"]]
        if primary_non_reject.empty:
            continue
        primary_tumor_fraction = float(primary_non_reject["is_tumor"].mean())
        primary_median_reads = float(primary_rows["nCount_Spatial"].median())

        if primary_tumor_fraction >= min_tumor_fraction and primary_median_reads >= min_median_reads:
            triads[str(patient)] = role_to_image

    return dict(sorted(triads.items(), key=lambda item: item[0]))


def select_pdac_patients(
    *,
    dataset_name: str = "pdac_mets_visium_gse272362",
    min_tumor_fraction: float = 0.20,
    min_median_reads: float = 1000.0,
    tumor_label: str = "Tumor Epithelial cells",
) -> dict[str, dict[str, str]]:
    """Select patients with a primary (Pancreas) tumor slice clearing a cancer-content
    bar, including whichever of the lymph-node-met / liver-met samples also exist for
    that patient (zero, one, or both).

    Unlike ``select_pdac_patient_triads``, this does not require the full
    primary/lymph-node-met/liver-met triad -- only "primary" is guaranteed to be a key
    in the returned per-patient dict; "lymph_node_met" and/or "liver_sample" are present
    only when that tissue type actually exists for the patient. Callers must branch on
    which keys are present rather than assuming all three.
    """
    load_dotenv()
    data_dir = os.getenv("SPATSPLAT_DATA_DIR")
    if not data_dir:
        raise RuntimeError("SPATSPLAT_DATA_DIR is not set")

    metadata_path = Path(data_dir) / "processed" / dataset_name / "spot_metadata.csv.gz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find PDAC Visium spot metadata: {metadata_path}")

    meta_df = pd.read_csv(metadata_path, low_memory=False)
    if "tissue" in meta_df.columns:
        meta_df = meta_df.loc[meta_df["tissue"].fillna(0).astype(int) == 1]

    origin_to_role = {
        "Pancreas": "primary",
        "Lymph node": "lymph_node_met",
        "Liver": "liver_sample",
    }
    meta_df = meta_df.loc[meta_df["Origin"].astype(str).isin(origin_to_role)].copy()
    meta_df["is_tumor"] = meta_df["first_type"].astype(str).eq(tumor_label)
    meta_df["is_reject"] = meta_df["spot_class"].astype(str).eq("reject")

    selected: dict[str, dict[str, str]] = {}
    for patient, patient_rows in meta_df.groupby("patient"):
        role_to_image: dict[str, str] = {}
        for origin, role in origin_to_role.items():
            origin_rows = patient_rows.loc[patient_rows["Origin"].astype(str) == origin]
            image_names = origin_rows["image_name"].astype(str).unique()
            if len(image_names) != 1:
                continue
            role_to_image[role] = image_names[0]

        if "primary" not in role_to_image:
            continue

        primary_rows = patient_rows.loc[patient_rows["image_name"].astype(str) == role_to_image["primary"]]
        primary_non_reject = primary_rows.loc[~primary_rows["is_reject"]]
        if primary_non_reject.empty:
            continue
        primary_tumor_fraction = float(primary_non_reject["is_tumor"].mean())
        primary_median_reads = float(primary_rows["nCount_Spatial"].median())

        if primary_tumor_fraction >= min_tumor_fraction and primary_median_reads >= min_median_reads:
            selected[str(patient)] = role_to_image

    return dict(sorted(selected.items(), key=lambda item: item[0]))
