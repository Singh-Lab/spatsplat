import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from dotenv import load_dotenv

from spatsplat.config import DataConfig
from spatsplat.data import LoadedData

DATASET_NAME = "pdac_mets_visium_gse274557"
RAW_SUBDIR = "pdac_mets_visium_gse_274557"


def _data_dir() -> Path:
    load_dotenv()
    data_dir = os.getenv("SPATSPLAT_DATA_DIR")
    if not data_dir:
        raise RuntimeError("SPATSPLAT_DATA_DIR is not set")
    return Path(data_dir)


def _processed_dir() -> Path:
    return _data_dir() / "processed" / DATASET_NAME


def _raw_dir() -> Path:
    return _data_dir() / "raw" / RAW_SUBDIR


def load_discovery_cohort_table() -> pd.DataFrame:
    """Sample-level table for the 49-sample discovery cohort: patient, tissue origin, GSM
    id, and the paper's published per-sample QC/neoplastic-spot counts (Table S5), joined
    against GEO's own sample metadata for the GSM<->title mapping.

    Restricted to the discovery cohort deliberately -- Table S6 (published lineage-state
    percentages, used to validate scoring in
    notebooks/lineage_scoring_validation-pdac_visium_pei-pancreatic_cancer.ipynb) only covers
    this set. The 6 extended-cohort replicate-primary sections and 2 PDX sections use a
    different internal sample-ID scheme in the supplement than in GEO (e.g. Table S5's
    "Pt-3D" vs. GEO's "Pt-3_Pri2") and are not reconciled here.
    """
    processed_dir = _processed_dir()
    sample_meta = pd.read_csv(processed_dir / "sample_metadata.tsv", sep="\t")

    raw5 = pd.read_csv(processed_dir / "table_s5_sample_qc_neoplastic_counts.csv", header=None, skiprows=1)
    qc_cols = [
        "patient", "sample_title", "tissue_origin", "abbrev",
        "ncount_median", "nfeature_median", "n_spots_raw", "n_spots_pass", "n_spots_neoplastic",
    ]
    discovery = raw5.iloc[:, 0:9].copy()
    discovery.columns = qc_cols
    discovery["patient"] = discovery["patient"].ffill()
    discovery = discovery.dropna(subset=["sample_title"])

    cohort = discovery.merge(sample_meta[["gsm", "title"]], left_on="sample_title", right_on="title", how="left")
    missing = cohort.loc[cohort["gsm"].isna(), "sample_title"].tolist()
    if missing:
        raise RuntimeError(f"Could not match GSM id for discovery-cohort samples: {missing}")
    return cohort.drop(columns=["title"]).reset_index(drop=True)


_ORIGIN_TO_ROLE_PREFIX = {
    "Primary PDAC": "primary",
    "Liver metastasis": "liver_met",
    "Lung metastasis": "lung_met",
    "Peritoneal metastasis": "peritoneal_met",
}


def select_pei_patients(*, min_primary_tumor_fraction: float = 0.0) -> dict[str, dict[str, str]]:
    """Group each discovery-cohort patient's sections into ``{role: sample_title}``.

    Every one of the 13 discovery-cohort patients has exactly one primary section plus
    2-4 metastatic sections (liver/lung/peritoneal, sometimes more than one of a given
    site -- e.g. Pt-8 has two liver mets, Pt-13 has two peritoneal mets). Roles for a
    repeated tissue type are numbered in sample-title order (e.g. two liver mets become
    ``liver_met_1``/``liver_met_2``), always numbered even when there is only one
    occurrence, for a uniform key scheme across patients. ``primary`` is never numbered --
    multiple-primary patients (extended-cohort replicate sections) are out of scope, see
    ``load_discovery_cohort_table()``'s docstring.

    ``min_primary_tumor_fraction`` screens patients by their primary section's neoplastic
    fraction (Table S5's ``n_spots_neoplastic / n_spots_pass``) -- default 0.0 keeps all 13
    patients; raise it to exclude low-tumor-content primaries (e.g. Pt-12's primary has only
    17/2852 neoplastic spots, ~0.6%, a natural low-content stress-test case akin to PT_12 in
    the Khaliq cohort).
    """
    cohort = load_discovery_cohort_table()
    patients: dict[str, dict[str, str]] = {}

    for patient, rows in cohort.groupby("patient"):
        rows = rows.sort_values("sample_title")
        primary_rows = rows.loc[rows["tissue_origin"] == "Primary PDAC"]
        if len(primary_rows) != 1:
            continue  # every discovery-cohort patient has exactly one; defensive only
        primary_row = primary_rows.iloc[0]
        primary_fraction = float(primary_row["n_spots_neoplastic"]) / float(primary_row["n_spots_pass"])
        if primary_fraction < min_primary_tumor_fraction:
            continue

        role_to_sample: dict[str, str] = {"primary": str(primary_row["sample_title"])}
        role_counts: dict[str, int] = {}
        met_rows = rows.loc[rows["tissue_origin"] != "Primary PDAC"]
        for _, row in met_rows.iterrows():
            prefix = _ORIGIN_TO_ROLE_PREFIX.get(str(row["tissue_origin"]))
            if prefix is None:
                continue
            role_counts[prefix] = role_counts.get(prefix, 0) + 1
            role_to_sample[f"{prefix}_{role_counts[prefix]}"] = str(row["sample_title"])

        patients[str(patient)] = role_to_sample

    return dict(sorted(patients.items(), key=lambda item: item[0]))


def load_classical_basal_markers() -> tuple[list[str], list[str]]:
    """Table S4 marker genes (classical, basal), adapted from PMID 34890551."""
    processed_dir = _processed_dir()
    s4 = pd.read_csv(processed_dir / "table_s4_classical_basal_markers.csv", header=0)

    def _clean(series: pd.Series) -> list[str]:
        vals = series.dropna().astype(str).tolist()
        return [v for v in vals if v.isupper() and v.replace("-", "").isalnum() and ":" not in v and " " not in v]

    return _clean(s4["Classical"]), _clean(s4["Basal"])


def load_emt_dissemination_markers() -> list[str]:
    """EMT/metastatic-dissemination marker genes, curated from an independent PDAC primary/liver-
    metastasis study (matched snRNA-seq + spatial transcriptomics, Monocle pseudotime identified a
    "disseminating" ductal subpopulation localized at the tumor boundary with significant EMT-gene
    upregulation; PMID 40269163 preprint / PMC12036881, Theranostics 2025). Distinct dataset and
    method from the Pei et al. classical/basal markers above -- kept separate rather than merged.
    Core EMT/mesenchymal genes plus the signaling genes the paper specifically implicated
    (SPP1-CD44 and MIF-(CD74+CXCR4) as key ligand-receptor pairs at the dissemination front)."""
    return [
        "LUM", "FSTL1", "MMP2", "TAGLN", "ACTA2", "VIM", "MMP1", "VEGFA", "ANPEP", "CDH2",
        "SPP1", "CD44", "MIF", "CD74", "CXCR4",
    ]


def load_malignant_epithelial_stromal_markers() -> tuple[list[str], list[str]]:
    """Canonical ductal/malignant-epithelial vs. stromal/immune marker panels for PDAC, curated
    independently of the classical/basal axis above (Moffitt et al. framework, central to the Pei
    et al. paper but not discussed or validated in the Khaliq et al. paper) and independently of
    the EMT/dissemination panel above -- used only to call each splat's compartment (tumor-like
    vs. non-tumor-like) for a purely geometric distance-to-boundary ("zonation") analysis, without
    presupposing which transcriptional-subtype axis matters. Malignant/ductal identity markers
    are standard across the PDAC snRNA-seq/spatial literature (e.g. Peng et al. 2019 Cell Res.;
    Hwang et al. 2022 Nat. Genet.); stromal/immune markers are standard fibroblast/CAF and
    leukocyte lineage genes.
    """
    malignant_epithelial = [
        "EPCAM", "KRT19", "KRT8", "KRT18", "KRT7", "MUC1", "SOX9",
        "CEACAM5", "CEACAM6", "S100P", "TFF1", "TFF2",
    ]
    stromal_immune = [
        "COL1A1", "COL1A2", "COL3A1", "DCN", "ACTA2", "PDGFRB",
        "PTPRC", "CD3D", "CD3E", "CD68", "CD14", "CD79A",
    ]
    return malignant_epithelial, stromal_immune


def load_pdac_visium_pei_dataset(
    sample_title: str,
    *,
    device: str | None = None,
    config: DataConfig | None = None,
) -> LoadedData:
    """Load one discovery-cohort Visium sample from Pei et al. 2025 (GSE274557 / Nature
    642:212-221), the MD Anderson/UCSD/Nebraska rapid-autopsy PDAC cohort -- a separate,
    independent dataset from the Indiana University / Khaliq et al. 2024 cohort loaded by
    ``pdac_mets_visium.py`` (no shared patients, no shared authors).

    ``sample_title`` is the GEO sample title / Table S5 sample id, e.g. "Pt-8A" (primary),
    "Pt-8B"/"Pt-8C" (liver mets), "Pt-8D" (peritoneal met) -- see
    ``load_discovery_cohort_table()`` for the full 49-sample discovery cohort and each
    sample's published QC/neoplastic-spot count.

    Raw Space Ranger output (old-format ``tissue_positions_list.csv``, no header) is parsed
    directly -- no R/Seurat extraction step needed, unlike the Khaliq cohort's RDS export.
    """
    config = config or DataConfig()
    raw_dir = _raw_dir()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Could not find raw Pei et al. PDAC Visium directory: {raw_dir}")

    cohort = load_discovery_cohort_table()
    row_match = cohort.loc[cohort["sample_title"] == sample_title]
    if row_match.empty:
        raise RuntimeError(
            f"sample_title={sample_title!r} is not in the discovery cohort. "
            f"Available: {sorted(cohort['sample_title'])}"
        )
    row = row_match.iloc[0]
    gsm = str(row["gsm"])

    h5_path = raw_dir / f"{gsm}_{sample_title}_filtered_feature_bc_matrix.h5"
    tar_path = raw_dir / f"{gsm}_{sample_title}_spatial.tar.gz"
    if not h5_path.exists() or not tar_path.exists():
        raise FileNotFoundError(f"Could not find raw Space Ranger output for {sample_title} ({gsm})")

    extract_target = raw_dir / "extracted" / gsm
    if not extract_target.exists():
        extract_target.mkdir(parents=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(extract_target)
    spatial_dirs = list(extract_target.glob("*spatial*"))
    if len(spatial_dirs) != 1:
        raise RuntimeError(f"Expected exactly one spatial subdir for {gsm}, found {spatial_dirs}")
    spatial_dir = spatial_dirs[0]

    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    pos = pd.read_csv(
        spatial_dir / "tissue_positions_list.csv",
        header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"],
    ).set_index("barcode")
    pos = pos.reindex(adata.obs_names)
    adata.obs["in_tissue"] = pos["in_tissue"].to_numpy()
    adata.obs["x_centroid"] = pos["pxl_col_in_fullres"].to_numpy(dtype=np.float32)
    adata.obs["y_centroid"] = pos["pxl_row_in_fullres"].to_numpy(dtype=np.float32)
    adata = adata[adata.obs["in_tissue"] == 1].copy()

    adata.obs["patient"] = str(row["patient"])
    adata.obs["tissue_origin"] = str(row["tissue_origin"])
    adata.obs["sample_title"] = sample_title
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    adata.uns["spatial"] = {sample_title: {"metadata": {}}}

    x = adata.obs["x_centroid"].to_numpy(dtype=np.float32)
    y = adata.obs["y_centroid"].to_numpy(dtype=np.float32)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    gene_list = list(map(str, adata.var_names))

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
