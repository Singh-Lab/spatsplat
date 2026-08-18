#!/bin/bash
#SBATCH --job-name=gradient-metastasis-khaliq
#SBATCH --account=mona
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/MONA/ns5404/singh_research/spatsplat_project/data/tmp_slurm_jobs/gradient_metastasis_khaliq_%j.out

cd /home/ns5404/singh_research/spatsplat_project/spatsplat
# Patient list is a positional sbatch argument (sbatch --export splits on commas and would
# silently truncate a comma-separated list -- see run_pdac_visium_pei_multi_patient.slurm).
/home/ns5404/.conda/envs/spatsplat/bin/python scripts/run_gradient_metastasis_khaliq_notebook.py --patients "${1:?patient list required}"
