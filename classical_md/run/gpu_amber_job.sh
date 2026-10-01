#!/bin/bash

module purge
module use /software/spack_v20d1/spack/share/spack/modules/linux-rhel7-x86_64/
module load amber/24

set -euo pipefail

#pmemd.cuda -O -i heat.in       -o heat.out       -p complex_solvated.prmtop -c min.rst        -r heat.rst       -inf heat.info       -x mdcrd_heat.nc       -e heat.mden       -ref min.rst
#pmemd.cuda -O -i density_10.in -o density_10.out -p complex_solvated.prmtop -c heat.rst       -r density_10.rst -inf density_10.info -x mdcrd_density_10.nc -e density_10.mden -ref min.rst
#pmemd.cuda -O -i density_05.in -o density_05.out -p complex_solvated.prmtop -c density_10.rst -r density_05.rst -inf density_05.info -x mdcrd_density_05.nc -e density_05.mden -ref min.rst
pmemd.cuda -O -i equil.in      -o equil.out      -p complex_solvated.prmtop -c density_05.rst -r equil.rst      -inf equil.info      -x mdcrd_equil.nc      -e equil.mden      -ref min.rst
