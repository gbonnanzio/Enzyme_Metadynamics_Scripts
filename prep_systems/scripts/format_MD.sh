#!/bin/bash

source ~/.bashrc
conda activate /home/gbf4422/Downloads/miniconda3/envs/jupyter-md-env

exec_dir=$1
cd $exec_dir

cpptraj -i cpptraj.in 
