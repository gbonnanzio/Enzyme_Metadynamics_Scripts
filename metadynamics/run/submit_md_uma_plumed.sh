#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=p30041
#SBATCH --partition=gengpu
#SBATCH --constraint=sxm
#SBATCH --gres=gpu:1
#SBATCH --exclude=qgpu2005,qgpu2006
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10G
#SBATCH --job-name=6-S1
#SBATCH --output=%x_%j.out

# ASE + PLUMED metadynamics (heat -> equilibrate -> well-tempered MtD) driving
# UMA in-process on the GPU. Run from this directory:  sbatch submit_md_uma_plumed.sh
#
# One-time setup (PLUMED python backend, not installed by default):
#   conda install -p /projects/p30041/gbf4422/envs/uma-env -c conda-forge py-plumed

source ~/.bashrc
ulimit -s unlimited

export OMP_NUM_THREADS=${SLURM_NTASKS}
export MKL_NUM_THREADS=${SLURM_NTASKS}

UMA_PYTHON=/projects/p30041/gbf4422/envs/uma-env/bin/python3

"$UMA_PYTHON" md_uma_plumed.py
