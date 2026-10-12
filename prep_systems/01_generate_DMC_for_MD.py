import MDAnalysis as mda
import numpy as np
import os
import shutil
import subprocess
from utils import *
import warnings

# Suppress warnings specific to MDAnalysis
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")

# path specific to this cluster
SYSTEM_PATH = "/projects/p30041/gbf4422/"

# Directory holding this script and its bundled inputs (scripts/, 5EJ5/, substrates_initial/).
PREP_DIR = os.path.dirname(os.path.abspath(__file__))

# Defined in the array below are the distances away from ThDP atoms to substrate atoms 
# determine by the optimized structure of substrate 6 and are used to place the aka head atoms
# radii are in order of (columns) C1, N1, N2, S1 and then rows (C2, C3, O1) 
# so C1 is (0,0) Angstroms away from C2. All distances in Angstroms  
radii = [                         # DON
    [3.305, 3.836, 3.868, 4.339], # C2
    [4.089, 4.487, 5.100, 5.083], # C3
    [3.407, 3.650, 2.898, 4.685]] # O1
#TPP #C1    #N1    #N2    #S1

# target angles represents the optimized angles of C1-C2-R ,O1-C2-R ,C3-C2-R
# we will use these to help us reposition the R group first atom 
target_angles = [79.9,117.6,117.2]

# load receptor universe and extract the different parts of the protein int1 receptor 
receptor = mda.Universe(os.path.join(PREP_DIR, '5EJ5', 'DMC_receptor.pdb'))

head_dir = f"{SYSTEM_PATH}5EJ5/MD_200/DMC/"
write_universe(f"{head_dir}receptor/receptor.pdb",receptor)
edit_protein_files(f"{head_dir}receptor/receptor.pdb",extra_resnames=["TPP","MG"])

# output the thiamine diphosphate cofactors (TPP)
TPP_residues = receptor.select_atoms("resname TPP").residues
primary_TPP_resnum = TPP_residues[0].resnum
primary_TPP = receptor.select_atoms(f"resnum {primary_TPP_resnum}")

write_universe(F"{head_dir}receptor/TPP_all.pdb",TPP_residues)
write_universe(F"{head_dir}receptor/TPP_primary.pdb",primary_TPP)

# output just the protein and edit the file so it is Amber readable 
protein = receptor.select_atoms(f"protein or resname MG")
write_universe(f"{head_dir}receptor/protein_and_MG.pdb",protein)
edit_protein_files(f"{head_dir}receptor/protein_and_MG.pdb",extra_resnames=["MG"])

# get the important indexes of the ThDP atoms 
ThDP_important_indexes = get_ThDP_indexes(primary_TPP)

# get the coordinates of important atoms in ThDP
C1_coords = get_atom_position(primary_TPP,ThDP_important_indexes['C1'])
N1_coords = get_atom_position(primary_TPP,ThDP_important_indexes['N1'])
N2_coords = get_atom_position(primary_TPP,ThDP_important_indexes['N2'])
S1_coords = get_atom_position(primary_TPP,ThDP_important_indexes['S1'])

# ThDP C1, N1, N2, S1 atom coords that will be treated as sphere centers 
centers = np.array([C1_coords,N1_coords,N2_coords,S1_coords])

# we will use the average vector of S1-C1 and N1-C1 to get a guess of where C2 should be located
vector_S1_to_C1 = C1_coords - S1_coords
vector_N1_to_C1 = C1_coords - N1_coords
avg_vector = (vector_S1_to_C1 + vector_N1_to_C1)/2
unit_vector = avg_vector / np.linalg.norm(avg_vector)
guess_C2 = C1_coords + unit_vector * radii[0][0] # C2-C1 distance based on DMC template

# iterate through all substrates to align to the int1 geometry
# read in each of the substrate files
directory = os.path.join(PREP_DIR, 'substrates_initial') + '/'
#file_names = [f for f in os.listdir(directory)]
file_names = [str(i)+'.pdb' for i in [4,6,11,16]]
clash_threshold = 1.5

for curr_file_name in file_names:
    print(curr_file_name)
    
    curr_substrate = curr_file_name.split('.')[0]
    # load substrate universe
    file_start = curr_file_name.split('.')[0]
    prep_dir = head_dir + file_start + "/MD/prep/"


    substrate = mda.Universe(directory+curr_file_name)

    # substrate net charge (alpha-keto acid = -1; substrates 4/6/11/16 carry an extra -1)
    don_charge = -1
    if curr_substrate in ['4','6','11','16']:
        don_charge += -1

    # generate the GAFF2 forcefield from the CLEAN initial geometry (pose-independent):
    # atom types + AM1-BCC charges + frcmod come from ../substrates_initial/, not the
    # rotated pose. Runs before placement so its connectivity is available for the scan.
    os.makedirs(prep_dir, exist_ok=True)
    initial_pdb_abs = os.path.abspath(directory + curr_file_name)
    ff_result = subprocess.run(
        ["bash", os.path.join(PREP_DIR, "scripts", "generate_don_ff.sh"),
         initial_pdb_abs, str(don_charge), prep_dir],
        capture_output=True,
        text=True
    )
    if ff_result.returncode != 0:
        print('ERROR generating DON forcefield (antechamber/parmchk2)')
        print(ff_result.stderr)
        continue
    print('DON forcefield generation successful')

    # identify the atoms that comprise the aka substrates
    substrate_important_indexes = get_substrate_aka_indexes(substrate.atoms)

    # initial coords of C2, C3, and O1 
    initial_positions = [get_atom_position(substrate,substrate_important_indexes['C2']),
                        get_atom_position(substrate,substrate_important_indexes['C3']),
                        get_atom_position(substrate,substrate_important_indexes['O1'])]
    # we go through two rounds of optimization, first using the guess location of 
    # C2 as the starting position for each atom we are trying to place (C2,C3,O1) 
    initial_guess = np.hstack([guess_C2 for i in range(3)])
    C2_optimized, C3_optimized, O1_optimized = optimize_coordinates(initial_guess, centers, radii)
    all_optimized = [C2_optimized, C3_optimized, O1_optimized]
    # get the final error for each atom's position 
    C2_err = atom_objective(C2_optimized, centers, radii[0])
    C3_err = atom_objective(C3_optimized, centers, radii[1])
    O1_err = atom_objective(O1_optimized, centers, radii[2])
    # use the atom with the minimum error for the next round of optimization
    all_errors = [C2_err,C3_err,O1_err]
    min_error_index = all_errors.index(min(all_errors))
    redo_initial_guess = np.hstack([all_optimized[min_error_index] for i in range(3)])
    C2_reoptimized, C3_reoptimized, O1_reoptimized = optimize_coordinates(redo_initial_guess,centers, radii)
    final_positions = [C2_reoptimized, C3_reoptimized, O1_reoptimized]
    # Get the rotation and translation matrix from our initial substrate to our int1 geoemtry
    R_int1, t_int1 = kabsch_algorithm(np.array(initial_positions),np.array(final_positions))
    
    # make a copy of the substrate object and update atom positions by aligning aka head atoms
    substrate_aka_aligned = substrate.copy()
    for i in range(0,len(substrate_aka_aligned.atoms.positions)):
        atom_coords = substrate_aka_aligned.atoms[i].position
        new_coords = np.dot(R_int1, atom_coords) + t_int1
        substrate_aka_aligned.atoms[i].position = new_coords
        substrate_aka_aligned.atoms[i].residue.resid = 1
        substrate_aka_aligned.atoms[i].residue.resname ='DON'
        substrate_aka_aligned.atoms[i].record_type = 'HETATM'
        
    # get the updated coords for important atoms 
    C2_coords = substrate_aka_aligned.atoms.positions[substrate_important_indexes['C2']]
    O1_coords = substrate_aka_aligned.atoms.positions[substrate_important_indexes['O1']]
    C3_coords = substrate_aka_aligned.atoms.positions[substrate_important_indexes['C3']]
    R_coords =  substrate_aka_aligned.atoms.positions[substrate_important_indexes['R']]
    # we have placed an SP2 hydbridized substrate (around C2), move R to make it SP3
    R_coords_opt = optimize_angles(R_coords,C1_coords,C2_coords,O1_coords,C3_coords,target_angles)
    # get a rotation and translation matrix from the original to the optimized R location 
    R_tail, t_tail = kabsch_algorithm([R_coords,C2_coords],[R_coords_opt,C2_coords])
    
    # apply rotation and translation transformation to only the tail atoms 
    substrate_tail_atom_indexes = [i for i in range(0,len(substrate.atoms)) if i not in substrate_important_indexes.values()]
    substrate_tail_atom_indexes.append(substrate_important_indexes['R'])
    
    # make a copy of the substrate object and update tail atom positions
    substrate_aligned = substrate_aka_aligned.copy()
    for i in substrate_tail_atom_indexes:
        atom_coords = substrate_aligned.atoms[i].position
        new_coords = np.dot(R_tail, atom_coords) + t_tail
        substrate_aligned.atoms[i].position = new_coords
        
    # add the substrate to the receptor
    complex_universe = mda.Merge(receptor.atoms,substrate_aligned.atoms)
    don = complex_universe.select_atoms("resname DON")

    # --- soft multi-torsion clash reduction (cyclic coordinate descent) ---
    # rotatable tail bonds come from the clean-geometry mol2 connectivity, so bond orders
    # and rings are known exactly; the analytic head (C2/O1/C3/O2/O3) never moves.
    mol2_bonds = get_mol2_bonds(prep_dir + 'DON_ff.mol2')
    gaff_types = get_mol2_atom_types(prep_dir + 'DON_ff.mol2')
    head_local = [substrate_important_indexes[k] for k in ['C2','O1','C3','O2','O3']]
    # C2-R is single by definition in an alpha-keto acid; always allow it to rotate
    force_bonds = [(substrate_important_indexes['C2'], substrate_important_indexes['R'])]
    rot_local = find_rotatable_bonds(mol2_bonds, len(substrate.atoms), gaff_types, head_local,
                                     force_bonds=force_bonds)

    if len(rot_local) == 0:
        print(f"  No rotatable tail bonds for substrate {curr_substrate}; skipping torsion scan")
        min_rotated_complex = complex_universe
    else:
        # map bonds / subtrees / connectivity into complex-index space
        rot_complex = [(don.atoms[p].index, don.atoms[d].index,
                        [don.atoms[i].index for i in sub]) for (p, d, sub) in rot_local]
        bonds_complex = [(don.atoms[a].index, don.atoms[b].index, o) for (a, b, o) in mol2_bonds]
        tail_indices = rot_complex[0][2]  # largest subtree = whole movable tail
        elements_all = list(complex_universe.atoms.types)

        scorer = make_soft_clash_scorer(complex_universe, tail_indices, bonds_complex, elements_all)
        print(f"  Substrate {curr_substrate}: {len(rot_complex)} rotatable tail bond(s); "
              f"start soft {scorer(complex_universe):.3f}, "
              f"hard {get_atom_clashes(complex_universe, tail_indices, clash_threshold)}")
        min_rotated_complex = minimize_clashes_ccd(complex_universe, rot_complex, scorer)
        print(f"  After CCD: soft {scorer(min_rotated_complex):.3f}, "
              f"hard {get_atom_clashes(min_rotated_complex, tail_indices, clash_threshold)}")

    # output all necessary files
    don_final = min_rotated_complex.select_atoms("resname DON")
    write_universe(f"{prep_dir}don.pdb", don_final)

    # transfer only the placed coordinates into the clean-geometry mol2 (types/charges
    # from DON_ff.mol2 are kept); tleap then loads placed coords + pose-independent FF
    write_placed_mol2(prep_dir + 'DON_ff.mol2', don_final.atoms.positions,
                      prep_dir + 'DON.mol2', placed_elements=list(don_final.atoms.types))

    shutil.copy(head_dir + 'receptor/receptor.pdb', prep_dir + 'receptor.pdb')
    shutil.copy(head_dir + 'receptor/TPP.mol2', prep_dir + 'TPP.mol2')
    shutil.copy(head_dir + 'receptor/TPP.frcmod', prep_dir + 'TPP.frcmod')

    ## Run tleap assembly from the specified directory
    result = subprocess.run(
        ["bash", os.path.join(PREP_DIR, "scripts", "assemble_DMC_for_MD.sh"), prep_dir],
        cwd=head_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print('ERROR WITH tleap')
        print(result.stderr)  # Print the error output
    else:
        print('tleap was successful')

    # read back in the unsolvated complex and get atom ids for restraints
    unsolvated_complex = mda.Universe(prep_dir + 'unsolvated_complex.pdb')
    don = unsolvated_complex.select_atoms("resname DON")
    important_indexes = get_substrate_aka_indexes(don)
    print(f"Substrate {curr_substrate} important indexes:", [don.atoms[important_indexes[i]].name for i in ["C2","C3","O1"]])


