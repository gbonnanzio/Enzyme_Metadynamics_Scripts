#!/bin/bash

# Assemble the solvated complex with tleap. The DON forcefield (DON_ff.mol2 -> DON.mol2
# with placed coordinates, and DON.frcmod) is now generated separately from the clean
# substrate geometry by generate_don_ff.sh + write_placed_mol2, so this script only runs
# tleap and stages the run directory.
#   $1 = prep directory containing DON.mol2, DON.frcmod, receptor.pdb, TPP.mol2, TPP.frcmod

source ~/.bashrc
conda activate /home/gbf4422/Downloads/miniconda3/envs/jupyter-md-env

# Directory containing this script (and the bundled tleap input + run_templates).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec_dir=$1

cd "$exec_dir" || exit 1
pwd

tleap -f "$SCRIPT_DIR/tleap_DMC_md_script_updated.in"

mkdir ../run/
cp complex_solvated* ../run/
cp "$SCRIPT_DIR/../run_templates/DMC/"* ../run/
