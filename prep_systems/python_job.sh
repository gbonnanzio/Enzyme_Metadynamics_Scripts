#!/bin/bash
#SBATCH -A p30041  
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 01:00:00
#SBATCH --job-name="py"
#SBATCH --output=prep.txt

source ~/.bashrc
conda activate /home/gbf4422/Downloads/miniconda3/envs/jupyter-md-env

# DMC stage (first half of the reaction)
#python 01_generate_DMC_for_MD.py
python 02_get_active_DMC_residues.py
#python 03_prepare_DMC_from_MD.py

# AMC stage (second half of the reaction)
#python 04_generate_hetero_AMC_for_MD.py
#python 05_get_active_AMC_residues.py
#python 06_prepare_AMC_from_MD.py
