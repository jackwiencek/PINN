#!/bin/bash
#SBATCH --job-name=pinn_cpp
#SBATCH --account=PAS2137
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=c_logs/slurm_%j.out
#SBATCH --error=c_logs/slurm_%j.err

set -e

REPO=/fs/scratch/PAS2137/PINN        # adjust if repo is elsewhere
LIBTORCH=/fs/scratch/PAS2137/libtorch

module load gnu/12.3.0                # or: module load gcc

cd "$REPO/cpp"
mkdir -p ../c_logs ../c_runs

# Build (skipped if already built)
make -f make_experiments LIBTORCH=$LIBTORCH

cd "$REPO"
cpp/build/bin/pinn_experiments
