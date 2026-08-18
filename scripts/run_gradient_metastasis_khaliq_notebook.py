"""Execute notebooks/gradient_metastasis-pdac_mets_visium-pancreatic_cancer.ipynb once per
requested Khaliq et al. 2024 PDAC Visium patient (pdac_mets_visium_gse272362) via papermill.

Each run writes an executed notebook copy to
notebooks/khaliq_gradient_metastasis_runs/<patient_id>.ipynb and (from within the notebook
itself) a summary JSON to
figures/gradient_metastasis_khaliq/<patient_id>/patient_summary.json.

This trains splat fields -- run it as a batch job (see
scripts/run_gradient_metastasis_khaliq_notebook.slurm), not on a login node.
"""

import argparse

import papermill as pm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "notebooks" / "gradient_metastasis-pdac_mets_visium-pancreatic_cancer.ipynb"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "khaliq_gradient_metastasis_runs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patients",
        type=str,
        required=True,
        help="Comma-separated patient ids to run, e.g. PT_4,PT_6,PT_8,PT_10.",
    )
    args = parser.parse_args()
    patient_ids = [p.strip() for p in args.patients.split(",") if p.strip()]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running {TEMPLATE_PATH.name} for patients: {patient_ids}")

    failed = []
    for patient_id in patient_ids:
        output_path = OUTPUT_DIR / f"{patient_id}.ipynb"
        print(f"\n=== {patient_id} -> {output_path} ===")
        try:
            pm.execute_notebook(
                str(TEMPLATE_PATH),
                str(output_path),
                parameters={"patient_id": patient_id},
                progress_bar=False,
            )
            print(f"=== {patient_id} done (no errors) ===")
        except pm.exceptions.PapermillExecutionError as exc:
            failed.append(patient_id)
            print(f"=== {patient_id} FAILED: {exc} ===")

    print(f"\nAll {len(patient_ids)} patient run(s) attempted: {patient_ids}")
    if failed:
        print(f"Patients with an error (see their executed notebook for details): {failed}")


if __name__ == "__main__":
    main()
