#!/bin/bash
#SBATCH --job-name=zonation-khaliq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

set -euo pipefail

# Submit from the repository root after activating the environment.
cd "${SPATSPLAT_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit this script with sbatch}}"
PYTHON="${SPATSPLAT_PYTHON:-python}"
# Patient list is a positional sbatch argument (sbatch --export splits on commas and would
# silently truncate a comma-separated list -- see run_gradient_metastasis_khaliq_notebook.slurm).
"$PYTHON" scripts/run_zonation_khaliq_notebook.py --patients "${1:?patient list required}"
