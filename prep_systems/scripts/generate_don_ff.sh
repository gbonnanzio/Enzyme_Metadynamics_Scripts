#!/bin/bash

# Generate the DON GAFF2 forcefield (mol2 + frcmod) from a CLEAN substrate geometry,
# independent of how the substrate is later placed/rotated into the active site.
#   $1 = absolute path to the clean initial substrate pdb (../substrates_initial/N.pdb)
#   $2 = net charge of the substrate
#   $3 = prep directory to write <resname>_ff.mol2 / <resname>.frcmod into
#   $4 = residue name (optional, default DON) -- e.g. ACC, INO
#
# No pdb4amber: the initial pdb carries CONECT records (antechamber uses them for
# connectivity) and an undistorted geometry, so bond-order perception, atom typing and
# AM1-BCC charges are all derived from clean chemistry rather than the docked pose.
# Skipping pdb4amber also preserves the input atom order so the placed coordinates can
# later be transferred back into DON_ff.mol2 by index (see write_placed_mol2).

source ~/.bashrc
conda activate /home/gbf4422/Downloads/miniconda3/envs/jupyter-md-env

initial_pdb=$1
charge=$2
exec_dir=$3
resname=${4:-DON}

cd "$exec_dir" || exit 1
pwd

antechamber -i "$initial_pdb" -fi pdb -o "${resname}_ff.mol2" -fo mol2 -c bcc -pf yes -nc "$charge" -at gaff2 -j 5 -rn "$resname"
parmchk2 -i "${resname}_ff.mol2" -f mol2 -o "${resname}.frcmod" -s 2
