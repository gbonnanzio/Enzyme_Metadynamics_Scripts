#!/bin/bash

module purge
module use /software/spack_v20d1/spack/share/spack/modules/linux-rhel7-x86_64/
module load amber/24

mpirun -np 10 MMPBSA.py.MPI -O -i mmgbsa.in -o mmgbsa_results.dat -sp complex_solvated.prmtop -cp ../prep/dry_complex.prmtop -rp ../prep/receptor.prmtop -lp ../prep/DON.prmtop -y mdcrd_equil.nc

rm _MMPBSA*

