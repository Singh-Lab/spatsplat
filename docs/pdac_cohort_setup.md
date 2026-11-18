# PDAC Cohort Analyses

The five additional PDAC notebooks use the Khaliq (GSE272362) and Pei
(GSE274557) cohorts. They are analyses of measured tissues and transfer/zonation
scores. Their scientific code is retained without changing the analysis design.

## Environment

Use the main environment and editable library installation. The batch runners
also require `papermill`, now included in `environment.yml`. Activate the
environment and launch commands from the repository root. Set
`SPATSPLAT_DATA_DIR` in the local `.env` before loading either cohort.

## Khaliq Data

Follow the README's PDAC download and R extraction instructions. The loader
expects `processed/pdac_mets_visium_gse272362/` under the data directory, including
the exported count matrix, gene/barcode lists, and `spot_metadata.csv.gz`.
Patient selection additionally needs the original metadata columns `Origin`,
`patient`, `image_name`, `first_type`, `spot_class`, and `nCount_Spatial`.

## Pei Data: Preparation Required

The provided loader is not a downloader. It expects these external files:

```text
$SPATSPLAT_DATA_DIR/
  raw/pdac_mets_visium_gse_274557/
    <GSM>_<sample_title>_filtered_feature_bc_matrix.h5
    <GSM>_<sample_title>_spatial.tar.gz
  processed/pdac_mets_visium_gse274557/
    sample_metadata.tsv
    table_s4_classical_basal_markers.csv
    table_s5_sample_qc_neoplastic_counts.csv
    table_s6_lineage_state_percentages.csv
```

`sample_metadata.tsv` supplies `gsm` and `title`. Supplementary tables S4--S6
must be exported in the layout expected by the loader/notebooks (including their
header rows). S6 is used by the lineage comparison cells. The release currently
does not include those source tables or a script recording how the CSV exports
were prepared. Obtain the preparation procedure from the analysis authors
before claiming a complete clean-data reproduction of these notebooks. Do not
guess the table layouts or substitute another PDAC cohort.

## Execution Order

- `family_transfer_multi_patient-pdac_visium_pei-pancreatic_cancer.ipynb`:
  family/transfer analysis on the Pei cohort.
- `gradient_metastasis-pdac_mets_visium-pancreatic_cancer.ipynb`:
  Khaliq gradient analysis.
- `zonation_metastasis-pdac_mets_visium-pancreatic_cancer.ipynb`:
  Khaliq zonation analysis.
- `zonation_metastasis-pdac_visium_pei-pancreatic_cancer.ipynb`:
  Pei zonation analysis, executed for each selected patient.
- `zonation_metastasis_comparison-pdac_visium_pei-pancreatic_cancer.ipynb`:
  run after the Pei zonation analyses; it reads
  `figures/zonation_metastasis_pei/<patient_id>/patient_summary.json`.

Some prose mentions earlier reconstruction, multi-patient Khaliq, and lineage
validation notebooks that are not distributed here. Those references describe
the development of the analyses, not included runnable entry points.

The three `scripts/run_*_notebook.py` wrappers execute the corresponding
notebooks with Papermill. Use `--help` to inspect patient selection arguments.
They train fields; they are not plot-only utilities. The `.slurm` files are
optional templates. Submit from the repository root after activating the
environment, specify your site's account/partition/QOS via `sbatch` options,
and review the resource limits. `SPATSPLAT_PROJECT_ROOT` and `SPATSPLAT_PYTHON`
can override the submission directory and active Python executable.

## Reproduction Status

Notebook syntax and local module dependencies were checked during packaging.
The analyses were not executed or scientifically revalidated as part of release
assembly. Existing notebook outputs are retained as provided.
