#!/bin/bash

# Assemble the solvated complex with tleap. The ACC and INO forcefields
# (ACC_ff.mol2/INO_ff.mol2 -> ACC.mol2/INO.mol2 with placed coordinates, and
# ACC.frcmod/INO.frcmod) are now generated separately from clean/merged geometry by
# the pipeline script + write_placed_mol2, so this script only runs tleap and stages
# the run directory.
#   $1 = prep directory containing ACC.mol2, ACC.frcmod, INO.mol2, INO.frcmod,
#        receptor.pdb, TPP.mol2, TPP.frcmod

source ~/.bashrc
conda activate /home/gbf4422/Downloads/miniconda3/envs/jupyter-md-env

# Directory containing this script (and the bundled tleap input + run_templates).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec_dir=$1

cd "$exec_dir" || exit 1
pwd

tleap -f "$SCRIPT_DIR/tleap_AMC_md_script_updated.in"

mkdir ../run/
cp complex_solvated* ../run/
cp "$SCRIPT_DIR/../run_templates/AMC/"* ../run/
