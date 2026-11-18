"""Execute notebooks/zonation_metastasis-pdac_visium_pei-pancreatic_cancer.ipynb once per
qualifying Pei et al. 2025 PDAC Visium patient (GSE274557 discovery cohort) via papermill.

Each run writes an executed notebook copy to
notebooks/zonation_metastasis_pei_runs/<patient_id>.ipynb and (from within the notebook
itself) a summary JSON to figures/zonation_metastasis_pei/<patient_id>/patient_summary.json.

This trains splat fields -- run it as a batch job (see
scripts/run_zonation_pei_notebook.slurm), not on a login node.
"""

import argparse
import sys
from pathlib import Path

import papermill as pm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "notebooks" / "zonation_metastasis-pdac_visium_pei-pancreatic_cancer.ipynb"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "zonation_metastasis_pei_runs"

sys.path.insert(0, str(PROJECT_ROOT))
from paper.data_loaders.pdac_mets_visium_pei import select_pei_patients  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patients",
        type=str,
        default=None,
        help="Comma-separated patient ids to run (default: all qualifying patients from select_pei_patients).",
    )
    parser.add_argument(
        "--min-primary-tumor-fraction",
        type=float,
        default=0.20,
        help="Minimum primary-section neoplastic-spot fraction (Table S5) required to qualify (default 0.20).",
    )
    args = parser.parse_args()

    qualifying = select_pei_patients(min_primary_tumor_fraction=args.min_primary_tumor_fraction)
    if not qualifying:
        raise RuntimeError("select_pei_patients() returned no qualifying patients")

    if args.patients:
        patient_ids = [p.strip() for p in args.patients.split(",") if p.strip()]
        unknown = sorted(set(patient_ids) - set(qualifying))
        if unknown:
            raise RuntimeError(f"Requested patients not in qualifying set {sorted(qualifying)}: {unknown}")
    else:
        patient_ids = sorted(qualifying)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running {TEMPLATE_PATH.name} for patients: {patient_ids}")

    failed_patients = []
    for patient_id in patient_ids:
        output_path = OUTPUT_DIR / f"{patient_id}.ipynb"
        print(f"\n=== {patient_id} -> {output_path} ===")
        try:
            pm.execute_notebook(
                str(TEMPLATE_PATH),
                str(output_path),
                parameters={"patient_id": patient_id, "min_primary_tumor_fraction": args.min_primary_tumor_fraction},
                progress_bar=False,
            )
            print(f"=== {patient_id} done (no errors) ===")
        except pm.exceptions.PapermillExecutionError as exc:
            failed_patients.append(patient_id)
            print(f"=== {patient_id} FAILED: {exc} ===")

    print(f"\nAll {len(patient_ids)} patient run(s) attempted: {patient_ids}")
    if failed_patients:
        print(f"Patients with an error (see their executed notebook for details): {failed_patients}")


if __name__ == "__main__":
    main()
