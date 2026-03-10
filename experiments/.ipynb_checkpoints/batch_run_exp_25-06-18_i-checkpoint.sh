#!/bin/sh
#
#SBATCH --job-name="exp_25-06-18_vary_aging"
#SBATCH --partition=compute
#SBATCH --time=03:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-TPM-MAS
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load gcc
module load 2024r1
module load python

cd /home/thoridwagenbla/tipping_model/model
python batch_run_25-06-18_i.py