import MDAnalysis as mda
from MDAnalysis.core.universe import Merge
import numpy as np
import os
from utils import *
import warnings
import subprocess
import shutil
from pathlib import Path

# Suppress warnings specific to MDAnalysis
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")

# Directory holding this script and its bundled inputs (scripts/, 5EJ5/, substrates_initial/).
PREP_DIR = os.path.dirname(os.path.abspath(__file__))

# we want to take the enzyme with ThDP and turn ThDP into INO and add ACC 

# Defined in the array below are the distances away from ThDP atoms to 
# covalently bound INO atoms to be placed 
# All distances in Angstroms  
radii_ino = [                     # DON/INO
    [1.349,  2.483, 3.255, 2.703], # C2
    [2.406, 3.009, 2.685, 4.024], # O1
    [2.478, 3.813, 4.160, 3.014]  # R
]#TPP #C1    #N1    #N2    #S1

# target angles represents the optimized angles of C1-C2-O1, O1-C2-R, C1-C2-R
# we will use these to help us reposition the R group first atom 
target_angles_ino = [122.6,115.0,121.1]
# load receptor universe and extract the different parts of the protein int1 receptor 

receptor_and_TPP = mda.Universe(os.path.join(PREP_DIR, '5EJ5', 'AMC_receptor_template.pdb'))

head_dir = '/projects/p30041/gbf4422/5EJ5/MD_Final/AMC/'

# output the thiamine diphosphate cofactors (TPP)
TPP_residues = receptor_and_TPP.select_atoms("resname TPP and resnum 1116").residues
primary_TPP_resnum = TPP_residues[0].resnum
primary_TPP = receptor_and_TPP.select_atoms(f"resnum {primary_TPP_resnum}")

receptor = receptor_and_TPP.select_atoms(f"not resnum {primary_TPP_resnum}")

# write receptor to output
write_universe(f"{head_dir}receptor/receptor.pdb", receptor)
edit_protein_files(f"{head_dir}receptor/receptor.pdb", extra_resnames=["TPP","MG"])

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
guess_C2 = C1_coords + unit_vector * radii_ino[0][0] # C2-C1 distance based on DMC template

# acceptor information
radii_acc = [                     # ACC
    [3.976, 4.693, 5.613, 5.046], # C2A
    [3.947, 4.195, 5.440, 5.249], # C3A
    [4.205, 5.049, 5.184, 5.366]  # O1A
]#TPP #C1    #N1    #N2    #S1


def reflect_point(P, normal, pt):
    '''Reflect point P across the plane defined by `normal` through `pt`.'''
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    P = np.asarray(P, dtype=float)
    return P - 2.0 * np.dot(P - np.asarray(pt, dtype=float), n) * n


def place_donor_from_seed(donor_substrate, seed_guess, centers, radii, initial_positions):
    '''
    Sphere-fit the donor's C2/O1/R to the ThDP sphere constraints starting from
    `seed_guess` (two-round min-error refinement, as before), Kabsch-fit the donor onto
    those targets, and return (placed_donor_universe, [C2*, O1*, R*]). Seeding from
    opposite sides of the ThDP plane converges to the two mirror sphere-fit solutions.
    '''
    guess = np.hstack([seed_guess for _ in range(3)])
    C2_opt, O1_opt, R_opt = optimize_coordinates(guess, centers, radii)
    opts = [C2_opt, O1_opt, R_opt]
    errs = [atom_objective(C2_opt, centers, radii[0]),
            atom_objective(O1_opt, centers, radii[1]),
            atom_objective(R_opt, centers, radii[2])]
    redo = np.hstack([opts[errs.index(min(errs))] for _ in range(3)])
    targets = list(optimize_coordinates(redo, centers, radii))
    R_mat, t_vec = kabsch_algorithm(np.array(initial_positions), np.array(targets))
    placed = donor_substrate.copy()
    placed.atoms.positions = (placed.atoms.positions @ R_mat.T) + t_vec
    return placed, targets


def count_tpp_clashes(placed, tpp_positions, exclude_idx, thresh=2.0):
    '''Count placed-donor atoms (excluding the C2 anchor, whose ~1.5 A bond to ThDP is
    legitimate) that sit within `thresh` A of any TPP atom.'''
    donor_pos = np.delete(placed.atoms.positions, exclude_idx, axis=0)
    dmin = np.linalg.norm(donor_pos[:, None, :] - tpp_positions[None, :, :], axis=2).min(axis=1)
    return int(np.sum(dmin < thresh))


# read in each of the substrate files
directory = os.path.join(PREP_DIR, 'substrates_initial') + '/'
donor_file_names = [f'{i}.pdb' for i in [1,2,4]]
acceptor_file_names = [f'{i}.pdb' for i in [1,2,4]]
clash_threshold = 2.0
ino_base_charge = -3 # with O1 protonated
acc_base_charge = -1

for donor_file_name in donor_file_names:
    for acceptor_file_name in acceptor_file_names:
        if donor_file_name != acceptor_file_name:
            continue
        
        print('Donor, Acceptor',donor_file_name,acceptor_file_name)
        donor_index = donor_file_name.split('.')[0]
        acc_index = acceptor_file_name.split('.')[0]
        output_dir = head_dir + donor_index + '/' + donor_index + '_' + acc_index + '/MD/'
        check_existing = len(list((Path(output_dir) / 'run').glob("*.dat"))) > 0
        
        if check_existing == True:
            print("Already Generated, Skipping...")
            continue

        prep_dir = output_dir + 'prep/'
        os.makedirs(prep_dir, exist_ok=True)

        # Get the charge of INO (needed before INO forcefield generation)
        if donor_index in ['4','6','11','16']:
            ino_charge = ino_base_charge - 1
        else:
            ino_charge = ino_base_charge

        # get the charge of ACC (needed before ACC forcefield generation)
        if acc_index in ['4','6','11','16']:
            acc_charge = acc_base_charge - 1
        else:
            acc_charge = acc_base_charge

        # load the donor substrate universe and put it in the universe
        donor_substrate = mda.Universe(directory+donor_file_name)
        # identify the important atoms for alignment (alpha keto acid atoms) 
        donor_substrate_important_indexes = get_substrate_aka_indexes(donor_substrate.atoms)
        
        # First align the donor substrate and remove CO2 group
        C2_coords = get_atom_position(donor_substrate,donor_substrate_important_indexes['C2'])
        O1_coords = get_atom_position(donor_substrate,donor_substrate_important_indexes['O1'])
        C3_coords = get_atom_position(donor_substrate,donor_substrate_important_indexes['C3'])
        R_coords  = get_atom_position(donor_substrate,donor_substrate_important_indexes['R'])
        O2_coords = get_atom_position(donor_substrate,donor_substrate_important_indexes['O2'])
        O3_coords = get_atom_position(donor_substrate,donor_substrate_important_indexes['O3'])

        donor_indices_to_remove = [donor_substrate_important_indexes['C3'],
                            donor_substrate_important_indexes['O2'],
                            donor_substrate_important_indexes['O3']]

        initial_positions = [C2_coords,O1_coords,R_coords]

        # --- rotamer guard ---
        # The per-atom sphere-fit has two mirror solutions across the ThDP ring plane; a
        # template change can tip the donor into the strained mirror where the hydroxyl O1
        # points into the ring (clashing, and mis-perceived by antechamber). Solve both
        # mirrors (seed from each side of the ThDP plane) and keep the one that seats the
        # donor clear of TPP.
        thdp_normal, thdp_centroid = calculate_average_plane(centers)
        cand_A_u, targets_A = place_donor_from_seed(donor_substrate, guess_C2, centers, radii_ino, initial_positions)
        guess_C2_mirror = reflect_point(guess_C2, thdp_normal, thdp_centroid)
        cand_B_u, targets_B = place_donor_from_seed(donor_substrate, guess_C2_mirror, centers, radii_ino, initial_positions)

        tpp_positions = primary_TPP.atoms.positions
        c2_local = donor_substrate_important_indexes['C2']
        o1_local = donor_substrate_important_indexes['O1']
        clash_A = count_tpp_clashes(cand_A_u, tpp_positions, c2_local)
        clash_B = count_tpp_clashes(cand_B_u, tpp_positions, c2_local)
        o1_clear_A = np.linalg.norm(cand_A_u.atoms.positions[o1_local] - tpp_positions, axis=1).min()
        o1_clear_B = np.linalg.norm(cand_B_u.atoms.positions[o1_local] - tpp_positions, axis=1).min()
        # prefer fewer TPP clashes; tie-break by larger O1 clearance; fall back to A on a tie
        if (clash_B, -o1_clear_B) < (clash_A, -o1_clear_A):
            donor_aligned, (C2_reoptimized, O1_reoptimized, R_reoptimized) = cand_B_u, targets_B
            chosen = 'B (mirror)'
        else:
            donor_aligned, (C2_reoptimized, O1_reoptimized, R_reoptimized) = cand_A_u, targets_A
            chosen = 'A'
        print(f"  INO rotamer: A(clashes={clash_A}, O1clr={o1_clear_A:.2f}) "
              f"B(clashes={clash_B}, O1clr={o1_clear_B:.2f}) -> chose {chosen}")

        # label the chosen placement as INO
        for i in range(0,len(donor_aligned.atoms.positions)):
            donor_aligned.atoms[i].residue.resid = 1
            donor_aligned.atoms[i].residue.resname ='INO'
            donor_aligned.atoms[i].record_type = 'HETATM'

        # remove atoms C3, O2, and O3 from substrate_aka_aligned
        mask = ~np.isin(donor_aligned.atoms.indices, donor_indices_to_remove)
        atoms_to_keep = donor_aligned.atoms[mask]
        modified_donor = Merge(atoms_to_keep) 

        # add a hydroxyl H on the ACTUAL placed O1 atom, bent to a realistic ~109.5 deg
        # C2-O1-H angle (not linear) in the C2-O1-R plane, on whichever side sits farther
        # from the ThDP ring. This keeps it unambiguously an O1-H hydroxyl (antechamber
        # cannot bond it to C2) while giving a physically sensible hydroxyl geometry.
        C2_placed = donor_aligned.atoms[c2_local].position
        O1_placed = donor_aligned.atoms[o1_local].position
        R_placed = donor_aligned.atoms[donor_substrate_important_indexes['R']].position
        oc = C2_placed - O1_placed
        oc /= np.linalg.norm(oc)                       # O1 -> C2 (the O-C bond)
        ref = R_placed - C2_placed                     # in-plane reference (C2 -> R)
        perp = ref - np.dot(ref, oc) * oc
        perp /= np.linalg.norm(perp)                   # unit vector perpendicular to O-C
        theta = np.deg2rad(109.5)
        h_a = O1_placed + (np.cos(theta) * oc + np.sin(theta) * perp) * bond_dists['O-H']
        h_b = O1_placed + (np.cos(theta) * oc - np.sin(theta) * perp) * bond_dists['O-H']
        d_a = np.linalg.norm(h_a - tpp_positions, axis=1).min()
        d_b = np.linalg.norm(h_b - tpp_positions, axis=1).min()
        guess_H1_coords = h_a if d_a >= d_b else h_b
        # unit_vector (ThDP-C1 -> C2) is reused below to seed the ACC C2A guess
        vector_C1_to_C2 = C2_reoptimized - C1_coords # should be in same direction
        unit_vector = vector_C1_to_C2 / np.linalg.norm(vector_C1_to_C2)
        H_u = mda.Universe.empty(1,
                                 n_residues=1,
                                 trajectory=True) # necessary for adding coordinates
        H_count = np.count_nonzero(donor_aligned.atoms.types == 'H')
        H_u.add_TopologyAttr('name',['H'+str(H_count)])
        H_u.add_TopologyAttr('type', ['H'])
        H_u.add_TopologyAttr('resname', ['INO'])
        H_u.add_TopologyAttr('resid', [1])
        H_u.atoms.positions = [guess_H1_coords]

        # rename TPP atoms to INO
        for i in range(0,len(primary_TPP.atoms)):
            primary_TPP.atoms[i].residue.resid = 1
            primary_TPP.atoms[i].residue.resname ='INO'

        # merge TPP, donor atoms, and the O1 hydrogen
        final_ino_universe = mda.Merge(primary_TPP.atoms,modified_donor.atoms,H_u.atoms)
        write_universe(f"{prep_dir}ino.pdb", final_ino_universe)

        # generate INO's GAFF2 forcefield from the merged (TPP+donor) geometry: this is
        # the earliest point INO exists as a chemical entity, since antechamber needs the
        # TPP/donor atoms in bonding proximity to perceive the new C1-C2 covalent bond.
        # No pdb4amber -- preserves ino.pdb's atom order so INO_ff.mol2 indices map 1:1
        # to final_ino_universe.atoms for the CCD rotation below.
        ino_ff_result = subprocess.run(
            ["bash", os.path.join(PREP_DIR, "scripts", "generate_don_ff.sh"),
             os.path.abspath(f"{prep_dir}ino.pdb"), str(ino_charge), prep_dir, "INO"],
            capture_output=True,
            text=True
        )
        if ino_ff_result.returncode != 0:
            print('ERROR generating INO forcefield (antechamber/parmchk2)')
            print(ino_ff_result.stderr)
            continue
        print('INO forcefield generation successful')

######################################################################################################################
        # Now we are going to place the acceptor substrate
        # we first need a guess of where C2A is 
        # (it is not in the same direction as C1 to C2 but it will be reoptimized anyways)
        guess_C2A = C1_coords + unit_vector * radii_acc[0][0]

        # generate ACC's GAFF2 forcefield from the CLEAN, unplaced acceptor geometry
        # (pose-independent), mirroring generate_don_ff.sh's use in 01_generate_DMC_for_MD.py
        acc_initial_pdb_abs = os.path.abspath(directory + acceptor_file_name)
        acc_ff_result = subprocess.run(
            ["bash", os.path.join(PREP_DIR, "scripts", "generate_don_ff.sh"),
             acc_initial_pdb_abs, str(acc_charge), prep_dir, "ACC"],
            capture_output=True,
            text=True
        )
        if acc_ff_result.returncode != 0:
            print('ERROR generating ACC forcefield (antechamber/parmchk2)')
            print(acc_ff_result.stderr)
            continue
        print('ACC forcefield generation successful')

        # identify the important atoms for alignment (alpha keto acid atoms)
        acceptor_substrate = mda.Universe(directory+acceptor_file_name)
        acceptor_substrate_important_indexes = get_substrate_aka_indexes(acceptor_substrate.atoms)
        C2A_coords = get_atom_position(acceptor_substrate,acceptor_substrate_important_indexes['C2'])
        O1A_coords = get_atom_position(acceptor_substrate,acceptor_substrate_important_indexes['O1'])
        C3A_coords = get_atom_position(acceptor_substrate,acceptor_substrate_important_indexes['C3'])
        RA_coords  = get_atom_position(acceptor_substrate,acceptor_substrate_important_indexes['R'])
        O2A_coords = get_atom_position(acceptor_substrate,acceptor_substrate_important_indexes['O2'])
        O3A_coords = get_atom_position(acceptor_substrate,acceptor_substrate_important_indexes['O3'])

        acceptor_initial_positions = [C2A_coords,C3A_coords,O1A_coords]
        acceptor_initial_guess = np.hstack([guess_C2A for i in range(3)])

        C2A_optimized, C3A_optimized, O1A_optimized = optimize_coordinates(acceptor_initial_guess, centers, radii_acc)
        acceptor_all_optimized = [C2A_optimized, C3A_optimized, O1A_optimized]
        # get the final error for each atom's position 
        C2A_err = atom_objective(C2A_optimized, centers, radii_acc[0])
        C3A_err = atom_objective(C3A_optimized, centers, radii_acc[1])
        O1A_err = atom_objective(O1A_optimized, centers, radii_acc[2])
        # use the atom with the minimum error for the next round of optimization
        acceptor_all_errors = [C2A_err,C3A_err,O1A_err]
        min_error_index = acceptor_all_errors.index(min(acceptor_all_errors))
        redo_initial_guess = np.hstack([acceptor_all_optimized[min_error_index] for i in range(3)])
        C2A_reoptimized, C3A_reoptimized, O1A_reoptimized = optimize_coordinates(redo_initial_guess,centers, radii_acc)
        acceptor_final_positions = [C2A_reoptimized, C3A_reoptimized, O1A_reoptimized]
        # Get the rotation and translation matrix from our initial substrate to our int1 geoemtry
        R_acc, t_acc = kabsch_algorithm(np.array(acceptor_initial_positions),np.array(acceptor_final_positions))

        # make a copy of the substrate object and update atom positions by aligning aka head atoms
        acceptor_aligned = acceptor_substrate.copy()
        for i in range(0,len(acceptor_aligned.atoms.positions)):
            atom_coords = acceptor_aligned.atoms[i].position
            new_coords = np.dot(R_acc, atom_coords) + t_acc
            acceptor_aligned.atoms[i].position = new_coords
            acceptor_aligned.atoms[i].residue.resid = 1
            acceptor_aligned.atoms[i].residue.resname ='ACC'
            acceptor_aligned.atoms[i].record_type = 'HETATM'
        
        # merge receptor, ino, and acceptor
        complex_universe = mda.Merge(receptor.atoms,final_ino_universe.atoms,acceptor_aligned.atoms)

        # --- INO tail rotation (CCD), mirroring 01_generate_DMC_for_MD.py's DON approach ---
        # rotation candidates are restricted to the donor-derived region of INO (never TPP):
        # filter the mol2 bonds to those with both endpoints in the donor-derived atom range
        # before calling find_rotatable_bonds, so TPP is never part of the search graph.
        ino_mol2_bonds = get_mol2_bonds(prep_dir + 'INO_ff.mol2')
        ino_gaff_types = get_mol2_atom_types(prep_dir + 'INO_ff.mol2')

        n_tpp_atoms = len(primary_TPP.atoms)
        kept_donor_indices = list(atoms_to_keep.indices)
        local_C2_in_ino = n_tpp_atoms + kept_donor_indices.index(donor_substrate_important_indexes['C2'])
        local_O1_in_ino = n_tpp_atoms + kept_donor_indices.index(donor_substrate_important_indexes['O1'])
        local_R_in_ino  = n_tpp_atoms + kept_donor_indices.index(donor_substrate_important_indexes['R'])

        ino_donor_bonds = [(a, b, o) for (a, b, o) in ino_mol2_bonds if a >= n_tpp_atoms and b >= n_tpp_atoms]
        ino_head_local = [local_C2_in_ino, local_O1_in_ino]
        ino_force_bonds = [(local_C2_in_ino, local_R_in_ino)]
        ino_rot_local = find_rotatable_bonds(ino_donor_bonds, len(final_ino_universe.atoms),
                                             ino_gaff_types, ino_head_local, force_bonds=ino_force_bonds)

        ino_sel = complex_universe.select_atoms("resname INO")
        if len(ino_rot_local) == 0:
            print(f"  No rotatable tail bonds for INO (donor {donor_index}); skipping torsion scan")
            ino_rotated_complex = complex_universe
        else:
            ino_rot_complex = [(ino_sel.atoms[p].index, ino_sel.atoms[d].index,
                                [ino_sel.atoms[i].index for i in sub]) for (p, d, sub) in ino_rot_local]
            ino_bonds_complex = [(ino_sel.atoms[a].index, ino_sel.atoms[b].index, o) for (a, b, o) in ino_mol2_bonds]
            ino_tail_indices = ino_rot_complex[0][2]
            elements_all = list(complex_universe.atoms.types)

            ino_scorer = make_soft_clash_scorer(complex_universe, ino_tail_indices, ino_bonds_complex, elements_all)
            print(f"  INO: {len(ino_rot_complex)} rotatable tail bond(s); "
                  f"start soft {ino_scorer(complex_universe):.3f}, "
                  f"hard {get_atom_clashes(complex_universe, ino_tail_indices, clash_threshold)}")
            ino_rotated_complex = minimize_clashes_ccd(complex_universe, ino_rot_complex, ino_scorer)
            print(f"  After CCD: soft {ino_scorer(ino_rotated_complex):.3f}, "
                  f"hard {get_atom_clashes(ino_rotated_complex, ino_tail_indices, clash_threshold)}")

        # --- ACC tail rotation (CCD), against the complex with INO's final tail conformation ---
        acc_mol2_bonds = get_mol2_bonds(prep_dir + 'ACC_ff.mol2')
        acc_gaff_types = get_mol2_atom_types(prep_dir + 'ACC_ff.mol2')
        acc_head_local = [acceptor_substrate_important_indexes[k] for k in ['C2','O1','C3','O2','O3']]
        acc_force_bonds = [(acceptor_substrate_important_indexes['C2'], acceptor_substrate_important_indexes['R'])]
        acc_rot_local = find_rotatable_bonds(acc_mol2_bonds, len(acceptor_substrate.atoms),
                                             acc_gaff_types, acc_head_local, force_bonds=acc_force_bonds)

        acc_sel = ino_rotated_complex.select_atoms("resname ACC")
        if len(acc_rot_local) == 0:
            print(f"  No rotatable tail bonds for ACC (acceptor {acc_index}); skipping torsion scan")
            min_rotated_complex = ino_rotated_complex
        else:
            acc_rot_complex = [(acc_sel.atoms[p].index, acc_sel.atoms[d].index,
                                [acc_sel.atoms[i].index for i in sub]) for (p, d, sub) in acc_rot_local]
            acc_bonds_complex = [(acc_sel.atoms[a].index, acc_sel.atoms[b].index, o) for (a, b, o) in acc_mol2_bonds]
            acc_tail_indices = acc_rot_complex[0][2]
            elements_all = list(ino_rotated_complex.atoms.types)

            acc_scorer = make_soft_clash_scorer(ino_rotated_complex, acc_tail_indices, acc_bonds_complex, elements_all)
            print(f"  ACC: {len(acc_rot_complex)} rotatable tail bond(s); "
                  f"start soft {acc_scorer(ino_rotated_complex):.3f}, "
                  f"hard {get_atom_clashes(ino_rotated_complex, acc_tail_indices, clash_threshold)}")
            min_rotated_complex = minimize_clashes_ccd(ino_rotated_complex, acc_rot_complex, acc_scorer)
            print(f"  After CCD: soft {acc_scorer(min_rotated_complex):.3f}, "
                  f"hard {get_atom_clashes(min_rotated_complex, acc_tail_indices, clash_threshold)}")

        # output final placed/rotated geometries and bake coordinates into the clean-geometry mol2s
        ino_final = min_rotated_complex.select_atoms("resname INO")
        acc_final = min_rotated_complex.select_atoms("resname ACC")
        write_universe(f"{prep_dir}ino.pdb", ino_final)
        write_universe(f"{prep_dir}acc.pdb", acc_final)

        write_placed_mol2(prep_dir + 'INO_ff.mol2', ino_final.atoms.positions,
                          prep_dir + 'INO.mol2', placed_elements=list(ino_final.atoms.types))
        write_placed_mol2(prep_dir + 'ACC_ff.mol2', acc_final.atoms.positions,
                          prep_dir + 'ACC.mol2', placed_elements=list(acc_final.atoms.types))

        # Final assembly
        shutil.copy(head_dir + 'receptor/receptor.pdb', prep_dir + 'receptor.pdb')
        shutil.copy(head_dir + 'receptor/TPP.mol2', prep_dir + 'TPP.mol2')
        shutil.copy(head_dir + 'receptor/TPP.frcmod', prep_dir + 'TPP.frcmod')

        ## Run the bash script from the specified directory

        result = subprocess.run(
            ["bash", os.path.join(PREP_DIR, "scripts", "assemble_AMC_for_MD.sh"), prep_dir],
            cwd=head_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print('ERROR WITH tleap')
            print(result.stderr)  # Print the error output
                    #print(result.stdout)  # Print the normal output
        else:
            print('tleap was successful')

        # read back in the unsolvated complex and get atom ids for restraints
        unsolvated_complex = mda.Universe(prep_dir + 'unsolvated_complex.pdb')
        acc = unsolvated_complex.select_atoms("resname ACC")
        important_indexes = get_substrate_aka_indexes(acc)
        print(f"Substrate {acc_index} important indexes:", [acc.atoms[important_indexes[i]].name for i in ["C2","C3","O1"]])



