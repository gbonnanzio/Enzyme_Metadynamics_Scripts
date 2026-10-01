#!/bin/bash

module purge
module use /software/spack_v20d1/spack/share/spack/modules/linux-rhel7-x86_64/
module load amber/24

set -euo pipefail

pmemd.cuda -O -i equil2.in -o equil2.out -p complex_solvated.prmtop -c equil.rst -r equil2.rst -inf equil2.info -x mdcrd_equil2.nc -e equil2.mden -ref min.rst
