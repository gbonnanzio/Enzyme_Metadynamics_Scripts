#!/bin/bash

module purge
module use /software/spack_v20d1/spack/share/spack/modules/linux-rhel7-x86_64/
module load amber/24

pmemd.cuda -O -i min.in -o min.out -p complex_solvated.prmtop -c complex_solvated.inpcrd -r min.rst -ref complex_solvated.inpcrd

