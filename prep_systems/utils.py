import MDAnalysis as mda
import numpy as np
from collections import Counter
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import os
import csv
import re 
import subprocess
from typing import Union, List

# Directory holding this module and its bundled helper scripts (scripts/).
PREP_DIR = os.path.dirname(os.path.abspath(__file__))


bond_dists ={'C-C':1.53,'C=C':1.35,'long C-C':1.64,'CO2 C-O':1.27,'C=O':1.23,'C-O':1.35,'C=N':1.35,'C-S':1.71,'S--N':2.45,'N-H':1.05,'C-N':1.382,'O-H':1.05}
threshold = 0.25

def get_atom_clashes(universe,atom_indices,threshold):
    # Create an AtomGroup for the specified atom indices
    nearby_atoms = universe.select_atoms(f"(around " + str(threshold) + f" index {' '.join(map(str, atom_indices))})")
    return len(nearby_atoms)

def get_atom_position(mol_universe:mda.core.universe.Universe,index:int):
    position = mol_universe.atoms[index].position
    return position

def get_dist(coord1,coord2):
    d=np.sqrt((coord1[0]-coord2[0])**2+(coord1[1]-coord2[1])**2+(coord1[2]-coord2[2])**2)
    return d

def get_angle(coord1,coord2,coord3):
        # Calculate vectors BA and BC
        BA = coord1 - coord2
        BC = coord3 - coord2

        # Compute the dot product and magnitudes of BA and BC
        dot_product = np.dot(BA, BC)
        magnitude_BA = np.linalg.norm(BA)
        magnitude_BC = np.linalg.norm(BC)

        # Calculate the cosine of the angle
        cos_theta = dot_product / (magnitude_BA * magnitude_BC)

        # Handle potential floating-point errors
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Calculate the angle in radians and then convert to degrees
        angle_radians = np.arccos(cos_theta)
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees
    
def calculate_average_plane(points):
    """
    Calculate the average plane through a group of points.
    Parameters:
        points (numpy.ndarray): A (N, 3) array where each row represents a 3D point.
    Returns:
        plane_normal (numpy.ndarray): The normal vector of the plane.
        plane_point (numpy.ndarray): A point on the plane (centroid of the points).
    """
    if points.shape[1] != 3:
        raise ValueError("Input points array must have shape (N, 3).")
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    # Subtract the centroid to center the points
    centered_points = points - centroid
    # Perform SVD on the centered points
    _, _, vh = np.linalg.svd(centered_points)
    # The normal vector of the plane is the last column of vh
    plane_normal = vh[-1]

    return plane_normal, centroid

def atom_objective(point, centers, radii):
    return sum((np.linalg.norm(point - center) - radius)**2 for center, radius in zip(centers, radii))

def combined_objective(all_points, centers, radii):
    # Split all_points into the three key points
    C2_coords, C3_coords, O1_coords = np.split(all_points, len(all_points)/len(radii))
    
    # Calculate atom-level errors for each point to fit them to sphere constraints
    atom_err_C2 = atom_objective(C2_coords, centers, radii[0])
    atom_err_C3 = atom_objective(C3_coords, centers, radii[1])
    atom_err_O1 = atom_objective(O1_coords, centers, radii[2])
    
    total_atom_err = atom_err_C2 + atom_err_C3 + atom_err_O1
    
    return  total_atom_err 

def optimize_coordinates(initial_guess:np.ndarray, centers:np.ndarray, radii:list):
    '''
    OBJECTIVE:
        Find the coordinates of three points in space given the centers of four spheres
        centered around reference atoms: C1, N1, N2, S1. 
        The radii were picked such that the intersection of these spheres will be the 
        coordinates of each atom C2, C3, O1
        initial_guess = [C2_x, C2_y, C2_z, C3_x, C3_y...] (needs to be 1D for optimizer)
        centers = [[C1_x,C1_y,C1_z], 
                   [N1_x,N1_y,N1_z],
                   [N2_x,N2_y,N2_z],
                   [S1_x,S1_y,S1_z]]
        radii = [[r_C2_C1, r_C2_N1, r_C2_N2, r_C2_S1],
                 [r_C3_C1, r_C3_N1, r_C3_N2, r_C3_S1]]
                 [r_O1_C1, r_O1_N1, r_O1_N2, r_O1_S1]]
    '''
    # Set up optimization
    tolerance = 1e-6
    result = minimize(
        combined_objective,
        initial_guess,
        args=(centers, radii),
        tol=tolerance
    )
    C2_coords, C3_coords, O1_coords = np.split(result.x, 3)
    # Check for successful optimization
    if result.success or result.fun < tolerance:
        print('CONVERGED')
    else:
        print('NOT CONVERGED')
    return C2_coords, C3_coords, O1_coords

def angle_objective(atom1,atom2,atom3,angle):
    angle_err = abs(get_angle(atom1,atom2,atom3) - angle)
    return angle_err

def combined_angle_objective(tail_coord, C1_coord, C2_coord, O1_coord, C3_coord, angles):
    # Calculate atom-level errors for each point to fit them to sphere constraints
    angle_err_C1_C2_R = angle_objective(C1_coord, C2_coord, tail_coord, angles[0])
    angle_err_O1_C2_R = angle_objective(O1_coord, C2_coord, tail_coord, angles[1])
    angle_err_C3_C2_R = angle_objective(C3_coord, C2_coord, tail_coord, angles[2])
    
    total_angle_err = angle_err_C1_C2_R + angle_err_O1_C2_R + angle_err_C3_C2_R
    return  total_angle_err 

def optimize_angles(initial_guess,C1_coords,C2_coords,O1_coords,C3_coords,angles):
    '''
    OBJECTIVE: 
    Identify the coordinate of atom R (initial_guess) that lead to the angles
    C1-C2-R, O1-C2-R, C3-C2-R given in angles
    '''
    # Flatten initial guess as midpoint of each set of centers
    # Set up optimization
    tolerance = 1e-6
    max_dist = 1.382
    bounds = [(C2_coords[0]-max_dist,C2_coords[0]+max_dist),(C2_coords[1]-max_dist,C2_coords[1]+max_dist),(C2_coords[2]-max_dist,C2_coords[2]+max_dist)]
    result = minimize(
        combined_angle_objective,
        initial_guess,
        args=(C1_coords,C2_coords,O1_coords,C3_coords,angles),
        tol=tolerance,
        bounds=bounds
    )
    
    # Check for successful optimization
    if result.success or result.fun < tolerance:
        print('CONVERGED')
    else:
        print('NOT CONVERGED')
    
    return result.x

def optimize_angles_updated(initial_guess,angle_groups,angles):
    '''
    OBJECTIVE: 
    Identify the coordinate of atom R (initial_guess) that lead to the angles
    C1-C2-R, O1-C2-R, C3-C2-R given in angles
    '''
    # Flatten initial guess as midpoint of each set of centers
    # Set up optimization
    tolerance = 1e-6
    max_dist = 1.382
    bounds = [(C2_coords[0]-max_dist,C2_coords[0]+max_dist),(C2_coords[1]-max_dist,C2_coords[1]+max_dist),(C2_coords[2]-max_dist,C2_coords[2]+max_dist)]
    result = minimize(
        combined_angle_objective,
        initial_guess,
        args=(C1_coords,C2_coords,O1_coords,C3_coords,angles),
        tol=tolerance,
        bounds=bounds
    )
    
    # Check for successful optimization
    if result.success or result.fun < tolerance:
        print('CONVERGED')
    else:
        print('NOT CONVERGED')
    
    return result.x

def write_universe(filepath:str, universe:mda.core.universe.Universe):
    '''
    OBJECTIVE: 
    Write an MDanalysis atom universe to directory+filename. If none exist,
    create that directory  
    '''
    # Check if the directory exists
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist. Creating it...")
        os.makedirs(directory)

    # Write the file
    universe.atoms.write(filepath)
    print(f"File '{filepath}' has been written.")

def edit_protein_files(file_path, extra_resnames=None):
    """
    Add 'TER' after the last atom of selected residue names (default: OXT-containing residues).
    extra_resnames: list of residue names (e.g. ['ACE', 'NME']) for which
                    a TER should be added after their last atom.
    """
    if extra_resnames is None:
        extra_resnames = []

    with open(file_path, 'r') as infile:
        lines = infile.readlines()

    with open(file_path, 'w') as outfile:
        prev_key = None  # (resname, chainID, resnum)

        for i, line in enumerate(lines):
            outfile.write(line)

            if not line.startswith(("ATOM", "HETATM")):
                prev_key = None
                continue

            # Parse residue info
            resname = line[17:20].strip()
            chain   = line[21].strip()
            resnum  = line[22:26].strip()

            current_key = (resname, chain, resnum)

            # Look ahead: does the next ATOM/HETATM belong to a different residue?
            next_is_new_residue = False
            if i + 1 < len(lines):
                nxt = lines[i+1]
                if nxt.startswith(("ATOM", "HETATM")):
                    next_key = (nxt[17:20].strip(), nxt[21].strip(), nxt[22:26].strip())
                    next_is_new_residue = (next_key != current_key)
                else:
                    # Next line is not atom → current residue definitely ends here
                    next_is_new_residue = True
            else:
                # End of file → residue ends
                next_is_new_residue = True

            # Condition 1: original OXT behavior
            if " OXT " in line:
                outfile.write("TER\n")

            # Condition 2: user-specified resnames
            if resname in extra_resnames and next_is_new_residue:
                outfile.write("TER\n")

    print("Edited", file_path, "for Amber")

def _guess_elements(atoms):
    """Per-atom element symbols for an AtomGroup. Uses the universe's ``elements``
    attribute when it is present and fully populated, otherwise guesses from the
    atom names (handles two-letter elements like MG / CA-the-metal correctly via
    MDAnalysis' guesser). Returned so the merged cluster always carries an explicit
    element column (a blank element col crashes UMA)."""
    try:
        els = [str(e).strip() for e in atoms.elements]
        if all(els):
            return els
    except (AttributeError, mda.exceptions.NoDataError):
        pass
    from MDAnalysis.topology.guessers import guess_atom_element
    return [guess_atom_element(n) for n in atoms.names]

def solvate_droplet(solute_universe, center=None, radius=None, mode="droplet",
                    surface_dist=None, clash=2.4, density=0.0333,
                    start_resid=None, seed=0):
    """Wrap ``solute_universe`` in explicit TIP3P water, either a full droplet or a
    thin surface shell.

    A cubic water lattice (spacing = density**(-1/3), ~3.1 A at bulk density) is
    laid over the relevant region; a candidate water O is kept only if it is at
    least ``clash`` A from every solute atom AND, depending on ``mode``:
      "droplet" — within ``radius`` of ``center`` (fills the whole enclosing
                  sphere, corners included).
      "shell"   — within ``surface_dist`` of ANY solute atom (a skin hugging the
                  molecular surface; far fewer waters).

    Each kept water gets a randomly-oriented (seeded) rigid TIP3P geometry. The
    waters are appended AFTER all solute atoms so existing 0-based indices are
    preserved, and an explicit ``elements`` array is set on the merged universe
    (guards the blank-element / UMA gather-oob crash).

    Returns (solvated_universe, added_water_resids, added_water_O_indices), the
    last two 1:1 and the O indices 0-based into the returned universe.
    """
    solute_pos = solute_universe.atoms.positions
    spacing = density ** (-1.0 / 3.0)

    if mode == "droplet":
        if center is None or radius is None:
            raise ValueError("mode='droplet' requires center and radius.")
        center = np.asarray(center, dtype=float)
        n = int(np.ceil(2.0 * radius / spacing))
        offs = (np.arange(n + 1) - n / 2.0) * spacing
        gx, gy, gz = np.meshgrid(offs, offs, offs, indexing="ij")
        grid = center + np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        grid = grid[np.linalg.norm(grid - center, axis=1) <= radius]
    elif mode == "shell":
        if surface_dist is None:
            raise ValueError("mode='shell' requires surface_dist.")
        # lattice over the solute bounding box padded by the shell thickness
        lo = solute_pos.min(axis=0) - surface_dist
        hi = solute_pos.max(axis=0) + surface_dist
        axes = [np.arange(lo[d], hi[d] + spacing, spacing) for d in range(3)]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    else:
        raise ValueError(f"unknown solvation mode {mode!r} — expected 'droplet' or 'shell'")

    if len(grid):
        # min distance from each candidate O to any solute atom
        dmin = np.sqrt(
            ((grid[:, None, :] - solute_pos[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        keep = dmin >= clash
        if mode == "shell":
            keep &= dmin <= surface_dist   # skin: drop corners far from the solute
        grid = grid[keep]
    n_wat = len(grid)
    if mode == "droplet":
        where = f"droplet r={radius:.1f} A about {np.asarray(center).round(2)}"
    else:
        where = f"shell <= {surface_dist} A of surface"
    print(f"[solvate] {where}: placed {n_wat} waters "
          f"(spacing {spacing:.2f} A, clash {clash} A)")

    # rigid TIP3P geometry (O at origin), one random orientation per water
    r_oh, half = 0.9572, np.radians(104.52 / 2.0)
    local_h = np.array([[r_oh * np.sin(half),  r_oh * np.cos(half), 0.0],
                        [-r_oh * np.sin(half), r_oh * np.cos(half), 0.0]])
    rots = R.random(n_wat, random_state=seed) if n_wat else None

    coords = np.empty((3 * n_wat, 3), dtype=float)
    for i, o_pos in enumerate(grid):
        coords[3 * i] = o_pos
        coords[3 * i + 1:3 * i + 3] = o_pos + rots[i].apply(local_h)

    if start_resid is None:
        start_resid = int(solute_universe.atoms.resids.max()) + 1
    added_resids = list(range(start_resid, start_resid + n_wat))

    wat = mda.Universe.empty(
        n_atoms=3 * n_wat, n_residues=n_wat,
        atom_resindex=np.repeat(np.arange(n_wat), 3),
        residue_segindex=np.zeros(n_wat, dtype=int), trajectory=True)
    wat.add_TopologyAttr("names", ["O", "H1", "H2"] * n_wat)
    wat.add_TopologyAttr("types", ["O", "H", "H"] * n_wat)
    wat.add_TopologyAttr("elements", ["O", "H", "H"] * n_wat)
    wat.add_TopologyAttr("resnames", ["WAT"] * n_wat)
    wat.add_TopologyAttr("resids", added_resids)
    if n_wat:
        wat.atoms.positions = coords

    solute_elems = _guess_elements(solute_universe.atoms)
    merged = mda.Merge(solute_universe.atoms, wat.atoms)
    # Merge can drop the element column; set it back explicitly (solute then water).
    merged.add_TopologyAttr("elements", solute_elems + ["O", "H", "H"] * n_wat)

    n_solute = len(solute_universe.atoms)
    added_o_indices = [n_solute + 3 * i for i in range(n_wat)]
    return merged, added_resids, added_o_indices

def determine_index_shift(original_u,merged_u,original_indexes):
    num_atom_diff = len(merged_u.atoms)-len(original_u.atoms)
    # if original atoms were added first
    if np.all(merged_u.atoms[0].position == original_u.atoms[0].position):
        return 0
    elif np.all(merged_u.atoms[num_atom_diff].position == original_u.atoms[0].position):
        return num_atom_diff
    else:
        return None

def kabsch_algorithm(P:np.ndarray, Q:np.ndarray):
    '''
    OBJECTIVE:
        Given two sets of multiple coordinates (at least three), find the 
        rotational and translational matrix you would apply to P to get to Q
    '''
    # Step 1: Compute the centroids of P and Q
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Step 2: Center the points
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Step 3: Compute the covariance matrix
    H = np.dot(P_centered.T, Q_centered)

    # Step 4: Compute the optimal rotation matrix using SVD
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Special reflection case handling
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Step 5: Compute the translation vector
    t = centroid_Q - np.dot(R, centroid_P)

    return R, t

def get_ThDP_indexes(tpp_residue:mda.core.universe.Universe):
    '''
    OBJECTIVE:
    Get the important atom indexes of ThDP
        C1 = Carbanion
        S1 = Sulfur
        N1 = Ring Nitrogen
        N2 = Amino group 
    '''
    # Get coordinates of S1 (only sulfur in ThDP)
    S1_index = list(tpp_residue.types).index('S')
    S1_coords = tpp_residue.positions[S1_index]

    # Identify the carbanion of ThDP by distance away from S and N


    # find N1 by distance away from S
    ThDP_N_indexes = [i for i, x in enumerate(list(tpp_residue.types)) if x == 'N']
    min_error = 1000
    found_N1 = False
    for i in ThDP_N_indexes:
        curr_dist = get_dist(S1_coords,tpp_residue.positions[i])
        curr_error = abs(curr_dist - bond_dists['S--N'])
        if curr_error < min_error:
            min_error = curr_error
            min_index = i
        if curr_error < threshold:
            found_N1 = True
            N1_index = i
    
    if found_N1 == False:
        print('Taking best guess at ring N')
        N1_index = min_index
    
    N1_coords = tpp_residue.positions[N1_index]

    # find N2, should be only N with two hydrogens on it  
    ThDP_H_indexes = [i for i, x in enumerate(list(tpp_residue.types)) if x == 'H']
    potential_N2_indexes = []
    for i in ThDP_N_indexes:
        for j in ThDP_H_indexes:
            curr_dist = get_dist(tpp_residue.positions[i],tpp_residue.positions[j])
            if (abs(curr_dist - bond_dists['N-H']) < threshold):
                potential_N2_indexes.append(i)
    # Count occurrences
    counter = Counter(potential_N2_indexes)
    # Filter the list to keep only elements that occur exactly twice
    N2_trimmed = list(set([item for item in potential_N2_indexes if counter[item] == 2]))
    #N2_trimmed = list(set(potential_N2_indexes))
    if len(N2_trimmed) > 1:
        print("UH OH! Multiple amino groups found")
    else:
        N2_index = N2_trimmed[0]

    # We have now found, S1, N1, and N2... use S1 and N1 to find C1
    min_error = 1000
    found_C1 = False
    # iterate through carbon atoms
    ThDP_C_indexes = [i for i, x in enumerate(list(tpp_residue.types)) if x == 'C']
    for i in ThDP_C_indexes:
        S_dist = get_dist(S1_coords,tpp_residue.positions[i])
        S_err = abs(S_dist - bond_dists['C-S'])
        N_dist = get_dist(N1_coords,tpp_residue.positions[i])
        N_err = abs(N_dist - bond_dists['C=N'])
        
        curr_error = S_err + N_err
        if curr_error < min_error:
            min_error = curr_error
            min_index = i

        if ((S_err < threshold) and (N_err < threshold)):
            found_C1 = True
            C1_index = i

    if found_C1 == False:
        print('Taking best guess at carbanion')
        C1_index = min_index
    
    ThDP_important_indexes = {'C1':C1_index,'N1':N1_index,'S1':S1_index,'N2':N2_index}
    return ThDP_important_indexes

def get_substrate_aka_indexes(substrate_atoms:mda.core.universe.Universe):
    '''
    OBJECTIVE: 
        Get the important indexes (alpha-keto acid atoms) of the substrate molecule 
        C2 = Carbonyl carbon
        C3 = Carboxylic acid carbon
        O1 = Carbonyl oxygen
        O2/O3 = Oxygens of carboxylic acid 
    '''
    # get all C's and O's in the substrate 
    substrate_atom_types = list(substrate_atoms.types)
    substrate_C_indexes = [i for i, x in enumerate(substrate_atom_types) if x == 'C']
    substrate_O_indexes = [i for i, x in enumerate(substrate_atom_types) if x == 'O']

    # identify carbon oxygen pairs 
    potential_C2_O1_pairs = {} # {carbon index:[oxygen indexes]}
    potential_C3_O2_O3_pairs = {} # {carbon index:[oxygen indexes]}
   
    for i in substrate_C_indexes:
        curr_C_coords = substrate_atoms.positions[i]
        for j in substrate_O_indexes:
            curr_O_coords = substrate_atoms.positions[j]
            curr_dist = get_dist(curr_C_coords,curr_O_coords)
            C_d_O_err = abs(curr_dist - bond_dists['C=O']) # double bond

            if C_d_O_err < threshold: # we have found a C=O bond
                true_carbonyl = True
                for k in substrate_O_indexes: # check if this is a carboxylic acid
                    if k != j:
                        curr_O_coords = substrate_atoms.positions[k]
                        curr_dist = get_dist(curr_C_coords,curr_O_coords)
                        C_s_O_err = abs(curr_dist - bond_dists['CO2 C-O']) # single bond

                        if C_s_O_err < threshold: # this is a carboxylic acid not a carbonyl
                            potential_C3_O2_O3_pairs[i] = [j,k]
                            true_carbonyl = False
                # if we found a carbonyl carbon
                if true_carbonyl:
                    potential_C2_O1_pairs[i] = [j]

    # identify the carbonyl carbon
    potential_C2_carbons = []
    for curr_C_index in potential_C2_O1_pairs:
        if curr_C_index in potential_C3_O2_O3_pairs:
            continue
        else:
            potential_oxygens = potential_C2_O1_pairs[curr_C_index]
            if len(potential_oxygens) == 1:
                potential_C2_carbons.append(curr_C_index)
    
    # identify the carbonyl carbon (C2) among the candidate(s)
    C2_index = None
    if len(potential_C2_carbons) == 1:
        C2_index = potential_C2_carbons[0]

    # If there are multiple carbonyl candidates (e.g. an ester/ketone elsewhere in the
    # molecule that the loose distance threshold also flags), the alpha-keto acid C2 is
    # the carbonyl carbon that is C-C bonded (~1.53 A, or an elongated 1.64 A) to a
    # carboxyl carbon (C3 candidate). This disambiguates for any number of C3 candidates.
    elif len(potential_C2_carbons) > 1 and len(potential_C3_O2_O3_pairs) >= 1:
        min_index = None
        min_err = 100
        for curr_C2_index in potential_C2_carbons:
            curr_C2_coords = substrate_atoms.positions[curr_C2_index]
            for curr_C3_index in potential_C3_O2_O3_pairs:
                curr_dist = get_dist(curr_C2_coords, substrate_atoms.positions[curr_C3_index])
                C_C_err = min(abs(curr_dist - bond_dists['C-C']),
                              abs(curr_dist - bond_dists['long C-C']))
                if C_C_err < min_err:
                    min_err = C_C_err
                    min_index = curr_C2_index
        if min_index is not None and min_err < threshold:
            C2_index = min_index

    # Fallback: no carbonyl is bonded to a carboxyl carbon -> use the alcohol heuristic to
    # discard candidates whose "carbonyl" oxygen actually carries a hydrogen (an alcohol).
    if C2_index is None:
        print('UH OH! More than one carbonyl carbon was found! Indexes: ',potential_C2_carbons)
        print('Coords: ',[substrate_atoms.positions[i] for i in potential_C2_carbons])
        print('Checking if carbonyl oxygen is an alcohol')

        # check if there are any hydrogens on the oxygen (this is an alcohol not a ketone)
        is_not_alcohol = []
        substrate_H_indexes = [i for i, x in enumerate(substrate_atom_types) if x == 'H']
        for curr_C2_index in potential_C2_carbons:
            curr_O1_index = potential_C2_O1_pairs[curr_C2_index][0]
            curr_O1_coords = substrate_atoms.positions[curr_O1_index]
            potential_alcohol = False
            for curr_H_index in substrate_H_indexes:
                curr_H_coords = substrate_atoms.positions[curr_H_index]
                curr_dist = get_dist(curr_O1_coords,curr_H_coords)
                O_H_err = abs(curr_dist - bond_dists['O-H'])
                if  O_H_err < threshold: # this is a carboxylic acid not a carbonyl
                    potential_alcohol = True
            is_not_alcohol.append(not potential_alcohol)

        if sum(is_not_alcohol) != 1:
            print('Could not find carbonyl carbon: exiting')
            return None

        print('Found potential carbonyl carbon, check results carefully!')
        C2_index = potential_C2_carbons[is_not_alcohol.index(True)]

    O1_index = potential_C2_O1_pairs[C2_index][0]

    C2_coords = substrate_atoms.positions[C2_index]
    
    # Find the carboxylic acid carbon (C3)
    if len(potential_C3_O2_O3_pairs) == 1: # this is our only guess
        C3_index = list(potential_C3_O2_O3_pairs.keys())[0]
        O2_index = potential_C3_O2_O3_pairs[C3_index][0]  
        O3_index = potential_C3_O2_O3_pairs[C3_index][1] 
    else:
        print('C3 Not yet found checking C2-C3 distance')
        found_C3 = False
        for i in list(potential_C3_O2_O3_pairs.keys()):
            curr_dist = get_dist(C2_coords,substrate_atoms.positions[i]) # C2 and C3 should be 1.53 A apart
            C_C_err = abs(curr_dist - bond_dists['C-C'])
            if C_C_err < threshold: 
                if found_C3 == True:
                    print('UH OH! More than one carboxylic acid carbon was found')
                    return None
                else:
                    C3_index = i    
                    O2_index = potential_C3_O2_O3_pairs[C3_index][0]  
                    O3_index = potential_C3_O2_O3_pairs[C3_index][1]  
                    found_C3 = True
            else: # try an elongated C-C bond 
                C_C_err = abs(curr_dist - bond_dists['long C-C'])
                if C_C_err < threshold: 
                    if found_C3 == True:
                        print('UH OH! More than one carboxylic acid carbon was found')
                        return None
                    else:
                        C3_index = i    
                        O2_index = potential_C3_O2_O3_pairs[C3_index][0]  
                        O3_index = potential_C3_O2_O3_pairs[C3_index][1]  
                        found_C3 = True
    
    substrate_important_indexes = {'C2':C2_index,
                                   'O1':O1_index,
                                   'C3':C3_index,
                                   'O2':O2_index,
                                   'O3':O3_index}

    # now find the first atom of the R group of the alpha keto acid 
    for i in range(0,len(substrate_atoms)):
        if i not in substrate_important_indexes.values():
            curr_atom_type = substrate_atoms.types[i]
            curr_atom_coords = substrate_atoms.positions[i]
            # given the original 20 substrates this connecting atom could either be a carbon or a nitrogen 
            if curr_atom_type == 'N':
                set_dist = bond_dists['C-N']
            elif curr_atom_type == 'C':
                set_dist = bond_dists['C-C']
            else: # if it is neither N or C it is not the connecting atom, skip it 
                continue

            curr_dist = get_dist(curr_atom_coords,substrate_atoms.positions[C2_index])
            dist_err = abs(curr_dist - set_dist) 
            if dist_err < threshold: 
                R_index = i 
    
    substrate_important_indexes['R'] = R_index
    
    return substrate_important_indexes

def rotate_atoms(universe, atom1_idx, atom2_idx, atom_list, angle_deg):
    """
    Rotates atoms in `atom_list` around the axis defined by `atom1_idx` and `atom2_idx` by `angle_deg`.
    
    Parameters:
    - universe: MDAnalysis Universe object
    - atom1_idx: Index of the first fixed atom
    - atom2_idx: Index of the second fixed atom
    - atom_list: List of atom indices to be rotated
    - angle_deg: Angle of rotation in degrees
    """
    # Get coordinates of fixed atoms
    atom1 = universe.atoms[atom1_idx]
    atom2 = universe.atoms[atom2_idx]
    coord1 = atom1.position
    coord2 = atom2.position
    
    # Define the rotation axis (normalized vector between atom1 and atom2)
    axis = coord2 - coord1
    axis = axis / np.linalg.norm(axis)

    # Define rotation angle in radians
    angle_rad = np.deg2rad(angle_deg)
    
    # Create the rotation object
    rotation = R.from_rotvec(angle_rad * axis)

    rotated_universe = universe.copy()

    # Rotate each atom in the atom_list
    for atom_idx in atom_list:
        atom = rotated_universe.atoms[atom_idx]
        # Translate the atom to the origin (relative to atom1)
        atom_position = atom.position - coord1
        # Apply rotation
        rotated_position = rotation.apply(atom_position)
        # Translate back to original frame
        atom.position = rotated_position + coord1

    return rotated_universe

def optimize_tail_position(initial_guess,C1_coords,C2_coords,O1_coords,C3_coords,angles):

    # Flatten initial guess as midpoint of each set of centers
    # Set up optimization
    tolerance = 1e-6
    max_dist = 1.382
    bounds = [(C2_coords[0]-max_dist,C2_coords[0]+max_dist),(C2_coords[1]-max_dist,C2_coords[1]+max_dist),(C2_coords[2]-max_dist,C2_coords[2]+max_dist)]
    result = minimize(
        combined_angle_objective,
        initial_guess,
        args=(C1_coords,C2_coords,O1_coords,C3_coords,angles),
        tol=tolerance,
        bounds=bounds
    )
    
    # Check for successful optimization
    if result.success or result.fun < tolerance:
        print('CONVERGED')
    else:
        print('NOT CONVERGED')
    
    return result.x

def simplify_integer_list(int_list):
    if not int_list:
        return ""

    # Sort the list to make sure the integers are in ascending order
    sorted_list = sorted(int_list)
    ranges = []
    start = sorted_list[0]
    end = start

    for number in sorted_list[1:]:
        if number == end + 1:
            end = number
        else:
            if start == end:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}:{end}")
            start = number
            end = start

    # Add the last range or number
    if start == end:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}:{end}")

    return ' '.join(ranges)

def get_atoms_by_distance(mol_universe:mda.Universe, distance:float, atom_id:int, additional_resids=None, solvate_residue:bool=False, water_distance:float=3.0, solvated_residue:float=None, exclude_water=False):
    # if we want additional residues included in the selection 
    if additional_resids is not None:
        added_resid_str = f"or resid {' or resid '.join(map(str, additional_resids))}"
    else:
        added_resid_str = ''
    # if we are going to add water
    if solvate_residue == True and solvated_residue is not None:
        water_str = f"or ((around {water_distance} resname {solvated_residue}) and resname WAT)"
    else:
        water_str = ''
    # if we are going to exclude water from the selection
    if exclude_water == True:
        water_str = f"{water_str} and not resname WAT"
    else:
        water_str = f"{water_str}"
    added_selection_str = added_resid_str + ' ' + water_str

    # get atoms a certain distance away from the atom with atom_id 
    selection_shell = mol_universe.select_atoms(f"(around {distance} index {str(atom_id)}) {added_selection_str}")
    
    # get the residues those atoms belong to
    selected_residues = set([atom.residue for atom in selection_shell])
    
    # get all the atoms that belong to the residues for each of these groups 
    selected_atoms = mol_universe.select_atoms(" or ".join([f"resid {residue.resid}" for residue in selected_residues]))
    selected_atoms_indexes = [curr_atom.index for curr_atom in selected_atoms]
    
    return selected_residues, selected_atoms_indexes

def get_atoms_by_reslist(mol_universe:mda.Universe,residue_list:list):

    # get all the atoms that belong to the residues for each of these groups 
    selected_atoms = mol_universe.select_atoms(" or ".join([f"resid {residue}" for residue in residue_list]))
    selected_atoms_indexes = [curr_atom.index for curr_atom in selected_atoms]

    return selected_atoms_indexes

def find_adjacent_oxygens(atom_universe,carbon_index, bond_dists, threshold):
    
    carbon_coords = atom_universe.atoms[carbon_index].position

    atom_types = list(atom_universe.types)
    O_indexes = [i for i, x in enumerate(atom_types) if x == 'O']

    adjacent_oxygens = []
    for curr_O_index in O_indexes:
        curr_O_coords = atom_universe.atoms[curr_O_index].position
        curr_dist = get_dist(carbon_coords,curr_O_coords)
        dist_err = abs(curr_dist-bond_dists['CO2 C-O'])
        if dist_err < threshold:
            adjacent_oxygens.append(curr_O_index)

    return adjacent_oxygens

def get_ini_indexes(ini_atoms):
    # ini_atoms is the universe of just the INI residue  

    # Identify the important atom indexes of the ThDP cofactor
    # {'Carbonyl_Carbon (C1)':index...}
    ThDP_indexes =  get_ThDP_indexes(ini_atoms)
    C1_index = ThDP_indexes['C1']
    C1_coords = ini_atoms.atoms[ThDP_indexes['C1']].position

    # get all C's in list 
    ini_atom_types = list(ini_atoms.types)
    ini_C_indexes = [i for i, x in enumerate(ini_atom_types) if x == 'C']

    # search for C2, should be C-C distance away from C1 
    potential_C2_indexes = []
    for i in ini_C_indexes:
        if i == C1_index:
            continue
        else:
            curr_C_coords = ini_atoms.positions[i]
            curr_dist = get_dist(C1_coords,curr_C_coords)
            curr_err = abs(curr_dist-bond_dists['C-C'])
            if curr_err < threshold:
                potential_C2_indexes.append(i)

    if len(potential_C2_indexes) != 1:
        print('ERROR FINDING C2 ATOM')
        return None
    else:
        C2_index = potential_C2_indexes[0]
        C2_coords = ini_atoms.atoms[C2_index].position
    
    print('C2',C2_coords)
    # search for O1
    ini_O_indexes = [i for i, x in enumerate(ini_atom_types) if x == 'O']
    potential_O1_indexes = []
    for i in ini_O_indexes:
        curr_O_coords = ini_atoms.positions[i]
        curr_dist = get_dist(C2_coords,curr_O_coords)
        curr_err = abs(curr_dist-bond_dists['C-O'])
        if curr_err < threshold:
            potential_O1_indexes.append(i)
    
    if len(potential_O1_indexes) != 1:
        print(potential_O1_indexes)
        print('ERROR FINDING O1 ATOM')
        return None
    else:
        O1_index = potential_O1_indexes[0]
    
    print('O1', ini_atoms.atoms[O1_index].position)  
    # search for C3
    potential_C3_indexes = []
    for i in ini_C_indexes:
        if i == C1_index or i == C2_index:
            continue
        else:
            curr_C_coords = ini_atoms.positions[i]
            curr_dist = get_dist(C2_coords,curr_C_coords)
            curr_err = abs(curr_dist-bond_dists['long C-C'])
            if curr_err < threshold:
                potential_C3_indexes.append(i)

    if len(potential_C3_indexes) == 0:
        print('ERROR FINDING C3 ATOM, REDUCING THRESHOLD')
        for i in ini_C_indexes:
            if i == C1_index or i == C2_index:
                continue
            else:
                curr_C_coords = ini_atoms.positions[i]
                curr_dist = get_dist(C2_coords,curr_C_coords)
                curr_err = abs(curr_dist-bond_dists['C-C'])
                if curr_err < threshold:
                    potential_C3_indexes.append(i)
        if len(potential_C3_indexes) == 0:
            print('COULD NOT FIND C3, EXITING')
            return None
        
    if len(potential_C3_indexes) == 1:
        C3_index = potential_C3_indexes[0] 
        CO2_oxygens = find_adjacent_oxygens(ini_atoms,C3_index)   
    else:
        refined_C3_indexes = []
        refined_CO2_oxygens = []
        for curr_C3_index in potential_C3_indexes:
            curr_CO2_oxygens = find_adjacent_oxygens(ini_atoms,curr_C3_index) 
            if len(curr_CO2_oxygens) != 2:
                continue
            else:
                refined_C3_indexes.append(curr_C3_index)
                refined_CO2_oxygens.append(curr_CO2_oxygens)

        if len(refined_C3_indexes) != 1:
            print('COULD NOT FIND C3, EXITING')
            return None
        else:
            C3_index = refined_C3_indexes[0]
            #print(ini_atoms.atoms[C3_index].position)
            CO2_oxygens = refined_CO2_oxygens[0]
    INI_important_indexes = {'C1':C1_index,'C2':C2_index,'C3':C3_index,'O1':O1_index,'CO2_0':CO2_oxygens[0],'CO2_1':CO2_oxygens[1]}
    return INI_important_indexes

def get_ino_indexes(ini_atoms,with_hydrogen=False):
    # ini_atoms is the universe of just the INI residue  

    # Identify the important atom indexes of the ThDP cofactor
    # {'Carbonyl_Carbon (C1)':index...}
    ThDP_indexes =  get_ThDP_indexes(ini_atoms)
    C1_index = ThDP_indexes['C1']
    C1_coords = ini_atoms.atoms[ThDP_indexes['C1']].position

    # get all C's in list 
    ini_atom_types = list(ini_atoms.types)
    ini_C_indexes = [i for i, x in enumerate(ini_atom_types) if x == 'C']

    # search for C2, should be C=C distance away from C1 
    potential_C2_indexes = []
    for i in ini_C_indexes:
        if i == C1_index:
            continue
        else:
            curr_C_coords = ini_atoms.positions[i]
            curr_dist = get_dist(C1_coords,curr_C_coords)
            curr_err = abs(curr_dist-bond_dists['C=C'])
            if curr_err < threshold:
                potential_C2_indexes.append(i)

    if len(potential_C2_indexes) != 1:
        print('ERROR FINDING C2 ATOM')
        return None
    else:
        C2_index = potential_C2_indexes[0]
        C2_coords = ini_atoms.atoms[C2_index].position

    # search for O1
    ini_O_indexes = [i for i, x in enumerate(ini_atom_types) if x == 'O']
    potential_O1_indexes = []
    for i in ini_O_indexes:
        curr_O_coords = ini_atoms.positions[i]
        curr_dist = get_dist(C2_coords,curr_O_coords)
        curr_err = abs(curr_dist-bond_dists['C-O'])
        if curr_err < threshold:
            potential_O1_indexes.append(i)
    
    if len(potential_O1_indexes) != 1:
        print('ERROR FINDING O1 ATOM')
        return None
    else:
        O1_index = potential_O1_indexes[0]

    # if H on O1 is requested
    if with_hydrogen == True:
        ini_H_indexes = [i for i, x in enumerate(ini_atom_types) if x == 'H']
        potential_H1_indexes = []
        O1_coords = ini_atoms.atoms[O1_index].position
        for i in ini_H_indexes:
            curr_H_coords = ini_atoms.positions[i]
            curr_dist = get_dist(O1_coords,curr_H_coords)
            curr_err = abs(curr_dist-bond_dists['O-H'])
            if curr_err < threshold:
                potential_H1_indexes.append(i)
        
        if len(potential_H1_indexes) != 1:
            print('ERROR FINDING H ON O1 ATOM')
            return None
        else:
            H1_index = potential_H1_indexes[0]
        
        INO_important_indexes = {'C1':C1_index,'C2':C2_index,'O1':O1_index,'H1':H1_index}
    else: 
        INO_important_indexes = {'C1':C1_index,'C2':C2_index,'O1':O1_index}
    return INO_important_indexes   

def get_inp_indexes(inp_atoms):
    # Identify the important atom indexes of the ThDP cofactor
    # {'Carbonyl_Carbon (C1)':index...}
    ThDP_indexes =  get_ThDP_indexes(inp_atoms)
    C1_index = ThDP_indexes['C1']
    C1_coords = inp_atoms.atoms[ThDP_indexes['C1']].position

    # get all C's in list 
    inp_atom_types = list(inp_atoms.types)
    inp_C_indexes = [i for i, x in enumerate(inp_atom_types) if x == 'C']

    # search for C2, should be C=C distance away from C1 
    potential_C2_indexes = []
    for i in inp_C_indexes:
        if i == C1_index:
            continue
        else:
            curr_C_coords = inp_atoms.positions[i]
            curr_dist = get_dist(C1_coords,curr_C_coords)
            curr_err = abs(curr_dist-bond_dists['C-C'])
            if curr_err < threshold:
                potential_C2_indexes.append(i)

    if len(potential_C2_indexes) != 1:
        print('ERROR FINDING C2 ATOM')
        return None
    else:
        C2_index = potential_C2_indexes[0]
        C2_coords = inp_atoms.atoms[C2_index].position
    
    INP_important_indexes = {'C1':C1_index,'C2':C2_index}

    return INP_important_indexes   

def write_orca_mm_input(
    output_path: str,
    file_name: str,
    first_line: str,
    charge: int,
    nprocs: int=64,
    maxcore: int=1000,
    multiplicity: int=1,
    pdb_filename: str = "qm_complex.pdb",
    orcaff_filename: str = "qm_complex.ORCAFF.prms",
):
    """
    Generate an ORCA QMMM input file.

    Parameters:
    - output_path (str): Path to write the ORCA input file.
    - qm_atoms (list): List of integers for QM atom indices (1-based).
    - charge (int): Total system charge.
    - multiplicity (int): Spin multiplicity.
    - nprocs (int): Number of processor cores to use.
    - maxcore (int): Memory per core in MB.
    - pdb_filename (str): Name of the PDB file.
    - orcaff_filename (str): Name of the ORCAFF parameter file.
    """
    os.makedirs(output_path, exist_ok=True)

    content = f"""{first_line}
% PAL NPROCS {nprocs} end
% Maxcore {maxcore}
% MM
    Rigid_MM_water true
    ORCAFFFilename "{orcaff_filename}"
end
* pdbfile {charge} {multiplicity} {pdb_filename}
"""

    full_path = os.path.join(output_path, file_name)
    with open(full_path, "w") as f:
        f.write(content)

    print(f"ORCA input written to {full_path}")
    return None

def write_orca_qmmm_input(
    output_path: str,
    file_name: str,
    first_line: str,
    qm_atoms: Union[List[int], str],
    active_atoms: Union[List[int], str],
    charge: int,
    nprocs: int = 64,
    maxcore: int = 1000,
    multiplicity: int=1,
    pdb_filename: str = "qm_complex.pdb",
    orcaff_filename: str = "qm_complex.ORCAFF.prms",
    fixed_atoms = None,
    solvent_epsilon = None,
    custom_geom_inputs = None
):
    """
    Generate an ORCA QMMM input file.

    Parameters:
    - output_path (str): Path to write the ORCA input file.
    - qm_atoms (list): List of integers for QM atom indices (1-based).
    - active_atoms (list): List of integers for Active atom indices (1-based).
    - charge (int): Total system charge.
    - multiplicity (int): Spin multiplicity.
    - nprocs (int): Number of processor cores to use.
    - maxcore (int): Memory per core in MB.
    - pdb_filename (str): Name of the PDB file.
    - orcaff_filename (str): Name of the ORCAFF parameter file.
    """
    os.makedirs(output_path, exist_ok=True)
    if isinstance(qm_atoms, list):
        qm_atoms = simplify_integer_list(qm_atoms)
    if isinstance(active_atoms, list):
        active_atoms = simplify_integer_list(active_atoms)

    content = f"""{first_line}
% PAL NPROCS {nprocs} end
% Maxcore {maxcore}
% QMMM
    QMAtoms {{{qm_atoms}}} end
    ActiveAtoms {{{active_atoms}}} end
"""
    if fixed_atoms is not None:
        if isinstance(fixed_atoms,list):
            fixed_atoms = simplify_integer_list(fixed_atoms)
        content += f"    OptRegion_FixedAtoms {{{fixed_atoms}}} end\n"
    content += f"""\tRigid_MM_water true
    ORCAFFFilename "{orcaff_filename}"
    PrintFullTrj False
end
"""
    if custom_geom_inputs is not None:
        content += f"%Geom\n{custom_geom_inputs}\nend\n"
    if solvent_epsilon is not None:
        content += f"% CPCM \n\tepsilon = {solvent_epsilon} \nend\n"
    content += f"""* pdbfile {charge} {multiplicity} {pdb_filename}
"""
    full_path = os.path.join(output_path, file_name)
    with open(full_path, "w") as f:
        f.write(content)

    print(f"ORCA input written to {full_path}")
    return None

def write_orca_xyz_input(
    output_path: str,
    file_name: str,
    first_line: str,
    charge: int,
    nprocs: int,
    maxcore: int,
    multiplicity: int=1,
    xyz_filename: str = "initial.xyz",
    constrained_atoms = None,
    add_geom_commands = None,
    solvent_epsilon = None,
):
    """
    Generate an ORCA QMMM input file.

    Parameters:
    - output_path (str): Path to write the ORCA input file.
    - qm_atoms (list): List of integers for QM atom indices (1-based).
    - active_atoms (list): List of integers for Active atom indices (1-based).
    - charge (int): Total system charge.
    - multiplicity (int): Spin multiplicity.
    - nprocs (int): Number of processor cores to use.
    - maxcore (int): Memory per core in MB.
    - pdb_filename (str): Name of the PDB file.
    - orcaff_filename (str): Name of the ORCAFF parameter file.
    """
    os.makedirs(output_path, exist_ok=True)

    content = f"""{first_line}
% PAL NPROCS {nprocs} end
% Maxcore {maxcore}"""
    if constrained_atoms is not None or add_geom_commands is not None:
        content += "\n%Geom"
        if constrained_atoms is not None:
            constrained_list = [f"{{C {i} C}}" for i in constrained_atoms]
            content += f"\n\tConstraints\n\t\t" + "\n\t\t".join(constrained_list) + "\n\tend"
        if add_geom_commands is not None:
            content += f"\n\t{add_geom_commands}"
        content += "\nEND"
    if solvent_epsilon is not None:
        content += f"\n% CPCM \n\tepsilon = {solvent_epsilon} \nend"
    
    content += f"""\n* xyzfile {charge} {multiplicity} {xyz_filename}\n"""
    full_path = os.path.join(output_path, file_name)
    with open(full_path, "w") as f:
        f.write(content)

    print(f"ORCA input written to {full_path}")
    return None

def write_sbatch_script(
    output_path: str,
    inp_files: list, # ["opt.inp","sp.inp"]
    filename="submit_job.sh",
    system="Quest",
    partition="normal",
    account="p30041",
    nodes=1,
    ntasks=64,
    mem_per_cpu="0",
    time="48:00:00",
    job_name="orca_job",
    output="orca_job.txt",
    orca_path="/software/2025/orca/6.1.0/orca_6_1_0_linux_x86-64_shared_openmpi416_avx2/orca",
    scratch_path="SCRATCH=/scratch/gbf4422/$SLURM_JOB_ID",
    modules_to_load="source ~/.bashrc \nulimit -s unlimited \nmodule load orca/6.1.0",
    additional_commands = None
):
    if system == 'Expanse':
        partition = "shared"
        account = "TG-CTS120055"
        orca_path = "/expanse/lustre/projects/nwu118/gbonnanzio/packages/orca_6_1_0/orca_6_1_0/orca"
        scratch_path = "SCRATCH=/expanse/lustre/scratch/gbonnanzio/temp_project/$SLURM_JOB_ID"
        modules_to_load = "module load cpu/0.17.3b \nmodule load gcc/10.2.0 \nmodule load openmpi/4.1.1 \nconda activate ambertools-env\n"
    
    sbatch_lines = [
        "#!/bin/bash",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --account={account}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --mem-per-cpu={mem_per_cpu}",
        f"#SBATCH --time={time}",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output}",
        "",
        "# load modules",
        f"{modules_to_load}",
        "",
        "# create scratch directory for tmp files and copy necessary files to it",
        f"{scratch_path}",
        "echo $SCRATCH",
        "mkdir -p $SCRATCH",
        "",
        "cp $SLURM_SUBMIT_DIR/*.inp $SCRATCH/",
        "cp $SLURM_SUBMIT_DIR/*.pdb $SCRATCH/",
        "cp $SLURM_SUBMIT_DIR/*.prms $SCRATCH/",
        "cp $SLURM_SUBMIT_DIR/*.xyz $SCRATCH/",
        "cp $SLURM_SUBMIT_DIR/*.gbw $SCRATCH/",
        "cp $SLURM_SUBMIT_DIR/*.hess $SCRATCH/",
        "",
        "cd $SCRATCH",
        "",
        "# run the ORCA job",
    ]

    for inp in inp_files:
        out_file = inp.replace(".inp", ".out")
        sbatch_lines.append(f"{orca_path} {inp} > $SLURM_SUBMIT_DIR/{out_file}")

    sbatch_lines += [
        "",
        "# copy what you want back to the submit directory",
        "cp $SCRATCH/*.inp $SLURM_SUBMIT_DIR/",
        "cp $SCRATCH/*.pdb $SLURM_SUBMIT_DIR/",
        "cp $SCRATCH/*.xyz $SLURM_SUBMIT_DIR/",
        "cp $SCRATCH/*.gbw $SLURM_SUBMIT_DIR/",
        "cp $SCRATCH/*.hess $SLURM_SUBMIT_DIR/\n",
    ]

    if additional_commands != None:
        sbatch_lines += additional_commands

    with open(output_path + filename, "w") as f:
        f.write("\n".join(sbatch_lines))
    print(f"SBATCH script written to '{output_path}{filename}'")
    return None

def write_resids_to_csv(output_path,file_name,QM_residue_list,active_residue_list):
    
    output_file = output_path + file_name
    # Write to the CSV
    if not os.path.exists(output_path):
        print(f"Directory '{output_path}' does not exist. Creating it...")
        os.makedirs(output_path)
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['QM RESIDUES'] + QM_residue_list)  # Write first list with label
        writer.writerow(['ACTIVE RESIDUES'] + active_residue_list)  # Write second list with label

    print(f"Data written to {output_file}")

def read_resids_from_csv(file_path):
    unique_lists = []
    
    # Read the CSV file
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        
        # Process each row, omitting the first column and ensuring uniqueness
        for row in reader:
            unique_lists.append(list(set(row[1:])))  # Skip the first column and ensure uniqueness

    QM_resids = unique_lists[0]
    QM_resids = [int(i) for i in QM_resids]
    
    active_resids = unique_lists[1]
    active_resids = [int(i) for i in active_resids]
    return QM_resids, active_resids

def extract_QM_input_info(filename,return_fixed=False):
    with open(filename, 'r') as file:
        lines = file.readlines()
    fixed_atoms = None  
    for line in lines:
        if 'QMAtoms' in line:
            QM_atoms = re.search(r'\{(.*?)\}', line).group(1)
        if 'ActiveAtoms' in line:
            Active_atoms = re.search(r'\{(.*?)\}', line).group(1)
        if 'OptRegion_FixedAtoms' in line:
            fixed_atoms = re.search(r'\{(.*?)\}', line).group(1)    
        if 'pdbfile' in line:
            charge = re.search(r'(-?\d+)', line).group(1)    
    
    if fixed_atoms is not None and return_fixed == True:
        return QM_atoms, Active_atoms, fixed_atoms, charge
    else:
        return QM_atoms, Active_atoms, charge
            

def extract_fixed_atoms(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    capturing = False
    numbers = []

    for line in lines:
        if "Fixed atoms used in optimizer" in line:
            capturing = True
            numbers.extend(re.findall(r'\d+', line))  # Extract all numbers in the line
        elif "Constrain bonds and angle of MM waters" in line:
            capturing = False
            break
        elif capturing:
            numbers.extend(re.findall(r'\d+', line))  # Extract all numbers in the line

    return list(map(int, numbers)) if numbers else None

def write_scan_script(QM_atoms_str,active_atoms_str,fixed_atoms_str,total_QM_charge,scan_string,output_path):
    # Define the input and output file paths
    input_file = "scripts/template_scan_script.inp"    
    # Open the input file and read its contents
    with open(input_file, 'r') as file:
        content = file.read()
    
    # Replace the {} placeholders with the variable values
    content = content.replace("{}", "{" + QM_atoms_str + "}" , 1)  # First occurrence
    content = content.replace("{}", "{" + active_atoms_str + "}", 1)  # Second occurrence
    content = content.replace("{}", "{" + fixed_atoms_str + "}", 1)  # Third occurrence
    
    # add the scan line
    content = content.replace("Scan B", scan_string, 1)  # Third occurrence


    # Add the custom line to the end
    custom_line = "*pdbfile " + str(total_QM_charge) +" 1 opt.pdb\n"
    content += custom_line
    
    # Write the modified content to the output file
    file_name = "scan.inp"
    output_file = output_path + file_name
    # Check if the directory exists
    if not os.path.exists(output_path):
        print(f"Directory '{output_path}' does not exist. Creating it...")
        os.makedirs(output_path)
    with open(output_file, 'w') as file:
        file.write(content)
    print(f"Complex QM/MM File processed and saved as {output_file}")

    
def find_bonded_atoms(universe, atom_index):
    """
    Find all atoms recursively bonded to the given atom index.
    
    Parameters:
    universe (MDAnalysis.Universe): The molecular system.
    atom_index (int): The index of the starting atom.
    
    Returns:
    set: A set of atom indices that are bonded to the given atom.
    """
    bonded_atoms = set()
    to_explore = {atom_index}
    
    while to_explore:
        current_atom = to_explore.pop()
        if current_atom in bonded_atoms:
            continue
        bonded_atoms.add(current_atom)
        neighbors = set(
            atom.index for bond in universe.bonds if current_atom in (bond[0].index, bond[1].index) 
            for atom in bond if atom.index != current_atom
        )
        to_explore.update(neighbors - bonded_atoms)
    
    return bonded_atoms

def extract_potential_energy(md_output_file_path):
    """
    Extracts the step numbers and potential energy (EPtot) values from an Amber output file (equil.out).
    Parameters:
        file_path (str): Path to the Amber output file.
    Returns:
        tuple: Two lists - one with step numbers and one with potential energies.
    """
    steps = []
    potential_energies = []

    # Regular expression to match the relevant lines
    step_pattern = re.compile(r"NSTEP\s+=\s+(\d+)")
    energy_pattern = re.compile(r"EPtot\s+=\s+([\-\d\.]+)")

    with open(md_output_file_path, 'r') as file:
        step = None
        for line in file:
            # Check for step number
            step_match = step_pattern.search(line)
            if step_match:
                step = int(step_match.group(1))

            # Check for potential energy
            energy_match = energy_pattern.search(line)
            if energy_match and step is not None:
                potential_energy = float(energy_match.group(1))
                steps.append(step)
                potential_energies.append(potential_energy)
                step = None  # Reset step to ensure correct pairing
    # the last value is fluctuations
    steps = steps[:-1]
    potential_energies = potential_energies[:-1]
    return steps, potential_energies

def write_cpptraj_input(md_output_file_path,frame_number):
    cpptraj_content = f"""parm complex_solvated.prmtop
        trajin mdcrd_equil.nc {frame_number} {frame_number}
        autoimage :1-1114
        trajout equil_min_energy_frame.pdb
        go
        quit
        """
    with open(md_output_file_path + "cpptraj.in", "w") as file:
        file.write(cpptraj_content)
    print("cpptraj.in written successfully!")
    return None


def find_lowest_energy_frame(md_directory,md_filename,start_frame_to_search):
    steps, all_energies = extract_potential_energy(md_directory+md_filename)
    energies_to_search = all_energies[start_frame_to_search:]
    min_energy = min(energies_to_search)
    min_frame = start_frame_to_search + energies_to_search.index(min_energy)
    # cpptraj uses 1 indexing 
    write_cpptraj_input(md_directory,min_frame + 1)
    
    # Run the bash script from the specified directory
    result = subprocess.run(
        ["bash", os.path.join(PREP_DIR, "scripts", "format_MD.sh"), md_directory],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Error: Cpptraj Script failed.")
        print(result.stderr)  # Print the error output
        print(result.stdout)
    else:
        print('tleap for MD processing was successful')
        #print(result.stdout)  # Print the normal output


# ---------------------------------------------------------------------------
# Pose-independent forcefield: mol2 parsing + coordinate transfer
# ---------------------------------------------------------------------------

def _read_lines(path):
    with open(path) as fh:
        return fh.readlines()


def get_mol2_bonds(mol2_path):
    '''
    OBJECTIVE:
        Parse the @<TRIPOS>BOND block of a mol2 file.
        Returns a list of (i0, j0, order_str) tuples with 0-based atom indices
        (mol2 atom ids are 1-based) and the SYBYL/GAFF bond order string
        ('1', '2', '3', 'ar', 'am', ...).

        NOTE: antechamber pdb->mol2 in this pipeline writes every bond as '1'
        (connectivity only; hybridization lives in the GAFF atom types instead), so
        find_rotatable_bonds relies on get_mol2_atom_types, not this order field.
    '''
    bonds = []
    in_bond = False
    for line in _read_lines(mol2_path):
        s = line.strip()
        if s.startswith('@<TRIPOS>'):
            in_bond = s.upper() == '@<TRIPOS>BOND'
            continue
        if in_bond and s:
            parts = s.split()
            if len(parts) >= 4:
                bonds.append((int(parts[1]) - 1, int(parts[2]) - 1, parts[3]))
    return bonds


def get_mol2_atom_types(mol2_path):
    '''
    OBJECTIVE:
        Parse the @<TRIPOS>ATOM block of a mol2 file and return the per-atom
        GAFF/SYBYL atom type strings (column 6) in file order (0-based).
    '''
    types = []
    in_atom = False
    for line in _read_lines(mol2_path):
        s = line.strip()
        if s.startswith('@<TRIPOS>'):
            in_atom = s.upper() == '@<TRIPOS>ATOM'
            continue
        if in_atom and s:
            parts = s.split()
            if len(parts) >= 6:
                types.append(parts[5])
    return types


def write_placed_mol2(ff_mol2_path, placed_positions, out_mol2_path, placed_elements=None):
    '''
    OBJECTIVE:
        Copy ff_mol2_path, overwriting the x/y/z of atom i with placed_positions[i],
        and write the result to out_mol2_path. Atom names/types/charges and the
        @<TRIPOS>BOND block are left untouched.

        Assumes the mol2 atom order matches placed_positions order. Both derive from
        the same clean initial pdb (antechamber preserves input atom order), so this
        is a safe per-index substitution. If placed_elements is given, each mol2 atom
        name's leading letter is checked against the corresponding placed element and
        a mismatch raises immediately (guards the atom-order assumption).
    '''
    placed_positions = np.asarray(placed_positions, dtype=float)
    out = []
    in_atom = False
    atom_i = 0
    for line in _read_lines(ff_mol2_path):
        s = line.strip()
        if s.startswith('@<TRIPOS>'):
            in_atom = s.upper() == '@<TRIPOS>ATOM'
            out.append(line)
            continue
        if in_atom and s:
            parts = s.split()
            atom_name = parts[1]
            if atom_i >= len(placed_positions):
                raise ValueError(
                    f"{ff_mol2_path} has more ATOM lines than placed positions "
                    f"({len(placed_positions)})")
            if placed_elements is not None:
                elem = str(placed_elements[atom_i]).strip()
                if elem and atom_name[0].upper() != elem[0].upper():
                    raise ValueError(
                        f"Atom-order mismatch at index {atom_i}: mol2 atom "
                        f"'{atom_name}' vs placed element '{elem}'. Aborting transfer.")
            x, y, z = placed_positions[atom_i]
            out.append("{:>7} {:<8} {:>10.4f} {:>10.4f} {:>10.4f} {}\n".format(
                parts[0], atom_name, x, y, z, " ".join(parts[5:])))
            atom_i += 1
            continue
        out.append(line)
    if atom_i != len(placed_positions):
        raise ValueError(
            f"{ff_mol2_path} has {atom_i} ATOM lines but {len(placed_positions)} "
            f"placed positions")
    with open(out_mol2_path, 'w') as fh:
        fh.writelines(out)
    print(f"Wrote placed coordinates into {out_mol2_path} ({atom_i} atoms)")
    return out_mol2_path


# ---------------------------------------------------------------------------
# Soft multi-torsion clash reduction
# ---------------------------------------------------------------------------

_VDW_RADII = {'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80,
              'F': 1.47, 'CL': 1.75, 'BR': 1.85, 'MG': 1.73}


def _vdw(elem):
    return _VDW_RADII.get(str(elem).upper(), 1.70)


def _build_adjacency(bonds, n_atoms):
    adj = {i: set() for i in range(n_atoms)}
    for a, b, _order in bonds:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _bfs_component(adj, start, blocked_edge=None, avoid=None):
    '''Atoms reachable from start. blocked_edge=(u,v) is never traversed; avoid is a
    node never entered.'''
    from collections import deque
    seen = {start}
    dq = deque([start])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if avoid is not None and v == avoid:
                continue
            if blocked_edge is not None and {u, v} == set(blocked_edge):
                continue
            if v not in seen:
                seen.add(v)
                dq.append(v)
    return seen


def get_rotatable_subtree(bonds, proximal, distal, n_atoms):
    '''
    OBJECTIVE:
        Atoms on the distal side of the proximal-distal bond (BFS from distal without
        crossing back through proximal). Includes distal and its hydrogens; excludes
        proximal. These are the atoms rotated when twisting that bond.
    '''
    adj = _build_adjacency(bonds, n_atoms)
    return _bfs_component(adj, distal, avoid=proximal)


# GAFF sp3 (rotation-permissive) heavy-atom types. A bond is only treated as a
# rotatable single bond if at least one endpoint is sp3 (or the bond is force-included);
# a bond between two non-sp3 (sp2/sp/aromatic) heavy atoms is assumed to be a double,
# triple, aromatic, amide or conjugated bond and is left rigid.
_GAFF_SP3_TYPES = {'c3', 'cx', 'cy', 'n3', 'n4', 'os', 'oh', 'sh', 'ss', 's4', 's6',
                   'p3', 'p4', 'p5'}


def _type_element(gaff_type):
    return str(gaff_type)[0].upper()


def _is_sp3(gaff_type):
    return str(gaff_type).lower() in _GAFF_SP3_TYPES


def find_rotatable_bonds(bonds, n_atoms, atom_types, head_indexes, force_bonds=None):
    '''
    OBJECTIVE:
        Identify rotatable tail bonds. atom_types are the per-atom GAFF types from the
        mol2 (element is taken as the leading letter). A bond (a,b) qualifies if:
        neither endpoint is hydrogen; the mol2 order is not an explicit multiple/
        aromatic/amide bond; it is acyclic (removing it disconnects a and b); all
        head_indexes lie on one side (so the placed head never moves); the distal
        endpoint has heavy-atom degree >= 2 (a real torsion, not a terminal group);
        and at least one endpoint is sp3 (so it is a genuine single bond) -- unless the
        bond is in force_bonds. Bond orders in this pipeline are all written as '1', so
        the sp3 test, not the order, is what protects double/aromatic bonds.

        force_bonds: iterable of (i,j) always permitted through the sp3 test (e.g. the
        C2-R bond, which is single by definition in an alpha-keto acid).

        Returns a list of (proximal, distal, subtree) with proximal on the head side
        and subtree the sorted distal-side atoms, ordered by subtree size descending
        (root -> leaf) for cyclic coordinate descent.
    '''
    force = set(frozenset(fb) for fb in (force_bonds or []))
    rigid_orders = {'2', '3', 'ar', 'am', 'du'}
    adj = _build_adjacency(bonds, n_atoms)
    heavy_deg = {i: sum(1 for j in adj[i] if _type_element(atom_types[j]) != 'H')
                 for i in range(n_atoms)}
    head = set(head_indexes)
    rot = []
    for a, b, order in bonds:
        if order in rigid_orders:
            continue
        if _type_element(atom_types[a]) == 'H' or _type_element(atom_types[b]) == 'H':
            continue
        comp_a = _bfs_component(adj, a, blocked_edge=(a, b))
        if b in comp_a:
            continue  # ring bond
        comp_b = set(range(n_atoms)) - comp_a
        if head.issubset(comp_a):
            proximal, distal, distal_side = a, b, comp_b
        elif head.issubset(comp_b):
            proximal, distal, distal_side = b, a, comp_a
        else:
            continue  # head atoms split across the bond -> would move the head
        if heavy_deg[distal] < 2:
            continue  # terminal group -> nothing meaningful rotates
        if frozenset((a, b)) not in force and not (_is_sp3(atom_types[a]) or _is_sp3(atom_types[b])):
            continue  # both sp2/sp/aromatic -> assume double/conjugated, keep rigid
        rot.append((proximal, distal, sorted(distal_side)))
    rot.sort(key=lambda t: len(t[2]), reverse=True)
    return rot


def make_soft_clash_scorer(complex_universe, tail_indices, bonds_complex, elements_all,
                           factor=0.6, cutoff=3.5):
    '''
    OBJECTIVE:
        Build a scorer(universe) -> float that sums soft van der Waals overlap
        penalties between the moving tail atoms and the (static) environment
        (receptor + fixed head atoms). For each tail-environment pair closer than
        r_min = factor*(vdw_i + vdw_j) the penalty is (r_min - d)**2.

        bonds_complex are the DON bonds expressed in complex-index space; 1-2 and 1-3
        tail<->environment pairs are excluded so real bonds are not penalized.
    '''
    from scipy.spatial import cKDTree
    n_all = len(complex_universe.atoms)
    tail_indices = list(tail_indices)
    tail_set = set(tail_indices)
    env_indices = [i for i in range(n_all) if i not in tail_set]
    env_positions = complex_universe.atoms.positions[env_indices].astype(float).copy()
    env_tree = cKDTree(env_positions)
    env_r = np.array([_vdw(elements_all[i]) for i in env_indices])
    tail_r = np.array([_vdw(elements_all[i]) for i in tail_indices])
    tail_idx_arr = np.array(tail_indices)
    env_pos_of = {ci: k for k, ci in enumerate(env_indices)}

    # DON adjacency in complex space (receptor has no bonds to DON)
    adj = {}
    for a, b, _o in bonds_complex:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    # env neighbours within 2 bonds of each tail atom, in env-array positions
    excluded = [set() for _ in tail_indices]
    for lt, ci_t in enumerate(tail_indices):
        near = set(adj.get(ci_t, ()))
        for n in list(near):
            near |= adj.get(n, set())
        near.discard(ci_t)
        for ci_e in near:
            if ci_e in env_pos_of:
                excluded[lt].add(env_pos_of[ci_e])

    def scorer(universe):
        tail_pos = universe.atoms.positions[tail_idx_arr].astype(float)
        neigh = env_tree.query_ball_point(tail_pos, cutoff)
        total = 0.0
        for lt, elist in enumerate(neigh):
            if not elist:
                continue
            rt = tail_r[lt]
            pt = tail_pos[lt]
            excl = excluded[lt]
            for k in elist:
                if k in excl:
                    continue
                rmin = factor * (rt + env_r[k])
                d = np.linalg.norm(pt - env_positions[k])
                if d < rmin:
                    total += (rmin - d) ** 2
        return total

    return scorer


def minimize_clashes_ccd(complex_universe, rotatable_bonds_complex, scorer,
                         step=15, max_passes=4, refine_step=5):
    '''
    OBJECTIVE:
        Cyclic coordinate descent over tail torsions minimizing scorer. Each element
        of rotatable_bonds_complex is (proximal, distal, subtree) in complex indices.
        Each pass scans every rotatable bond over 0..360 deg (step deg; refine_step on
        the final pass) and commits the best angle; passes repeat until no bond
        improves or max_passes is reached.
    '''
    current = complex_universe.copy()
    for p in range(max_passes):
        this_step = step if p < max_passes - 1 else refine_step
        improved = False
        for proximal, distal, subtree in rotatable_bonds_complex:
            best_angle, best_score = 0, scorer(current)
            for angle in range(this_step, 360, this_step):
                trial = rotate_atoms(current, proximal, distal, subtree, angle)
                s = scorer(trial)
                if s < best_score - 1e-9:
                    best_score, best_angle = s, angle
            if best_angle != 0:
                current = rotate_atoms(current, proximal, distal, subtree, best_angle)
                improved = True
        print(f"CCD pass {p + 1}: score = {scorer(current):.4f}")
        if not improved:
            break
    return current


def expand_for_terminal_prolines(full_universe, selected_resids, verbose=True):
    '''
    Given the resids chosen for a cluster, pull in the residue preceding any
    proline that would sit at the N-terminus of a contiguous run. Proline's
    N-Ca bond is part of the pyrrolidine ring, so it cannot be the sp3 cut point
    (see add_backbone_terminal_caps); including resid-1 makes the proline interior
    so the cut lands on a normal residue instead.

    Returns a de-duplicated, sorted list of resids (the input plus any pulled-in
    predecessors). Predecessors that are themselves proline are handled by
    repeating the expansion until it stabilizes.
    '''
    resids = set(int(r) for r in selected_resids)

    # protein resids in the full frame — run detection considers only these so a
    # neighbouring water/cofactor resid can't merge into a protein run (matches the
    # driver's resid-based terminal-CA logic)
    protein_resids = set(int(r) for r in
                         full_universe.select_atoms("protein").residues.resids)

    def _first_of_each_protein_run(resid_set):
        # split selected PROTEIN resids into contiguous runs, return each run's first
        ordered = sorted(r for r in resid_set if r in protein_resids)
        firsts = []
        for prev, cur in zip([None] + ordered, ordered):
            if prev is None or cur != prev + 1:
                firsts.append(cur)
        return firsts

    # look up a resid's resname in the full frame (protein residue)
    def _resname(resid):
        sel = full_universe.select_atoms(f"protein and resid {resid}")
        return sel.residues[0].resname if len(sel.residues) else None

    changed = True
    while changed:
        changed = False
        for first in _first_of_each_protein_run(resids):
            if _resname(first) != 'PRO':
                continue
            prev = first - 1
            if prev in resids:
                continue
            if _resname(prev) is None:
                if verbose:
                    print(f"WARN: N-terminal PRO {first} has no preceding protein "
                          f"residue in the frame; ring cut cannot be avoided")
                continue
            resids.add(prev)
            changed = True
            if verbose:
                print(f"Included residue {prev} to keep N-terminal PRO {first} "
                      f"interior (avoid cutting the proline ring)")

    return sorted(resids)


def _unit(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def add_backbone_terminal_caps(universe, cap_h_dist=1.09,
                               cap_name='HL', verbose=True):
    '''
    OBJECTIVE:
        Truncate the backbone at the terminal residues of each contiguous run of
        protein residues (consecutive resids within a chain) using the traditional
        HIMO cluster-model convention: cut only the sp3 single bond at the terminal
        residue's Ca and cap that Ca with a hydrogen placed along the original bond
        vector. Interior peptide bonds are left intact. Non-protein residues are
        ignored.

        For each run:
            N-terminal residue -> cut N-Ca. Remove the backbone N and its amide H,
                                  cap Ca with an H along the Ca->N vector. The
                                  residue keeps its own C=O (bonds to the next
                                  residue in the run).
            C-terminal residue -> cut Ca-C. Remove the carbonyl C and O, cap Ca
                                  with an H along the Ca->C vector. The residue
                                  keeps its N/H (bonds to the previous residue).
            isolated residue (run length 1) -> both cuts (Ca + sidechain only).

        Backbone removal is charge-neutral (a neutral N-H or C=O is swapped for a
        C-H). The cap is placed at cap_h_dist along the original bond direction and
        is intended to be relaxed by the subsequent optimization.

        Proline guard: proline's N-Ca bond is part of the pyrrolidine ring, so an
        N-terminal proline is NOT cut here (its N cap is skipped with a warning).
        The selection stage (expand_for_terminal_prolines) is expected to have
        pulled in the preceding residue so proline is no longer N-terminal.

    Returns a NEW MDAnalysis Universe with the dangling backbone atoms removed and
    the cap hydrogens inserted immediately after their parent residue's atoms. The
    input universe is left unmodified.
    '''
    protein = universe.select_atoms("protein")
    if len(protein) == 0:
        print("No protein residues found; returning a copy of the input.")
        return universe.copy()

    # order protein residues by chain then resid, then split into contiguous runs
    prot_residues = sorted(protein.residues, key=lambda r: (str(r.segid), int(r.resid)))
    runs = [[prot_residues[0]]]
    for prev, res in zip(prot_residues, prot_residues[1:]):
        same_chain = str(res.segid) == str(prev.segid)
        consecutive = int(res.resid) == int(prev.resid) + 1
        if same_chain and consecutive:
            runs[-1].append(res)
        else:
            runs.append([res])

    n_cap_resindexes = set(run[0].resindex for run in runs)   # N-terminal of each run
    c_cap_resindexes = set(run[-1].resindex for run in runs)  # C-terminal of each run

    def _named_atom(res, name):
        sel = res.atoms.select_atoms(f"name {name}")
        return sel[0] if len(sel) else None

    # {resindex: [(position, name), ...]} of cap hydrogens to insert and
    # {resindex: set(atom.index)} of dangling backbone atoms to drop
    new_hs = {}
    drop_indices = {}
    for res in universe.residues:
        caps = []
        drops = set()
        CA = _named_atom(res, 'CA')
        # N-terminal cut: remove backbone N (+amide H), cap Ca along Ca->N
        if res.resindex in n_cap_resindexes:
            N = _named_atom(res, 'N')
            if res.resname == 'PRO':
                if verbose:
                    print(f"WARN: resid {res.resid} is an N-terminal PRO; N-Ca is a "
                          f"ring bond, skipping N cut (include the previous residue "
                          f"at selection to avoid this)")
            elif N is None or CA is None:
                if verbose:
                    print(f"WARN: resid {res.resid} missing N/CA, skipping N cut")
            else:
                pos = CA.position + _unit(N.position - CA.position) * cap_h_dist
                caps.append((pos, cap_name))
                drops.add(N.index)
                amide_h = _named_atom(res, 'H') or _named_atom(res, 'HN')
                if amide_h is not None:
                    drops.add(amide_h.index)
        # C-terminal cut: remove carbonyl C and O, cap Ca along Ca->C
        if res.resindex in c_cap_resindexes:
            C = _named_atom(res, 'C')
            O = _named_atom(res, 'O')
            if C is None or CA is None:
                if verbose:
                    print(f"WARN: resid {res.resid} missing C/CA, skipping C cut")
            else:
                pos = CA.position + _unit(C.position - CA.position) * cap_h_dist
                caps.append((pos, cap_name))
                drops.add(C.index)
                if O is not None:
                    drops.add(O.index)
        if caps:
            new_hs[res.resindex] = caps
        if drops:
            drop_indices[res.resindex] = drops

    # ---- rebuild a universe with dangling atoms removed + caps inserted ----
    seg_list = list(universe.segments)
    seg_index_map = {s.segindex: i for i, s in enumerate(seg_list)}
    res_list = list(universe.residues)

    old_names = universe.atoms.names
    old_types = universe.atoms.types
    old_pos = universe.atoms.positions

    names, types, positions, atom_resindex = [], [], [], []
    for i, res in enumerate(res_list):
        drops = drop_indices.get(res.resindex, set())
        for atom in res.atoms:
            if atom.index in drops:
                continue
            names.append(old_names[atom.index])
            types.append(old_types[atom.index])
            positions.append(old_pos[atom.index])
            atom_resindex.append(i)
        for pos, name in new_hs.get(res.resindex, []):
            names.append(name)
            types.append('H')
            positions.append(pos)
            atom_resindex.append(i)

    residue_segindex = [seg_index_map[r.segment.segindex] for r in res_list]

    new_u = mda.Universe.empty(
        n_atoms=len(names),
        n_residues=len(res_list),
        n_segments=len(seg_list),
        atom_resindex=np.array(atom_resindex),
        residue_segindex=np.array(residue_segindex),
        trajectory=True,
    )
    new_u.add_TopologyAttr('name', names)
    new_u.add_TopologyAttr('type', types)
    new_u.add_TopologyAttr('element', list(types))
    new_u.add_TopologyAttr('resid', [r.resid for r in res_list])
    new_u.add_TopologyAttr('resname', [r.resname for r in res_list])
    new_u.add_TopologyAttr('segid', [s.segid for s in seg_list])
    new_u.atoms.positions = np.array(positions, dtype=np.float32)

    n_added = sum(len(v) for v in new_hs.values())
    if verbose:
        print(f"Added {n_added} capping hydrogens across {len(runs)} protein run(s).")
    return new_u


# ---------------------------------------------------------------------------
# ASE + PLUMED + UMA well-tempered metadynamics run generators
# (templated verbatim on 5EJ5/metadynamics/first_half/6/step_1/md_uma_plumed.py
#  and submit_md_uma_plumed.sh; only the tokenized values are substituted)
# ---------------------------------------------------------------------------
# The metadynamics driver lives in a standalone template file next to this
# module (md_uma_plumed_template.py), read at write time; its __TOKEN__
# placeholders are substituted in write_md_uma_plumed().
_MD_UMA_PLUMED_TEMPLATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "md_uma_plumed_template.py")


def _load_md_uma_plumed_template():
    with open(_MD_UMA_PLUMED_TEMPLATE_FILE) as f:
        return f.read()


_SUBMIT_MD_UMA_PLUMED_TEMPLATE = r'''#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=p30041
#SBATCH --partition=gengpu
#SBATCH --constraint=sxm
#SBATCH --gres=gpu:1
#SBATCH --exclude=qgpu2005,qgpu2006
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --job-name=__JOB_NAME__
#SBATCH --output=%x_%j.out

# ASE + PLUMED metadynamics (heat -> equilibrate -> well-tempered MtD) driving
# UMA in-process on the GPU. Run from this directory:  sbatch submit_md_uma_plumed.sh
#
# One-time setup (PLUMED python backend, not installed by default):
#   conda install -p /projects/p30041/gbf4422/envs/uma-env -c conda-forge py-plumed

source ~/.bashrc
ulimit -s unlimited

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

UMA_PYTHON=/projects/p30041/gbf4422/envs/uma-env/bin/python3

"$UMA_PYTHON" md_uma_plumed.py
'''


def write_md_uma_plumed(output_dir, charge, mult, cv_atoms, frozen_atoms,
                        structure="protonated_cluster.pdb",
                        ca_restraint_atoms=(), ca_restraint_k=2.0,
                        n_heat=10000, n_equil=20000, n_mtd=2000000,
                        sigma=0.1, height=0.5, pace=50,
                        biasfactor=27.6667,  # (T + deltaT_bias) / T = (300 + 8000) / 300
                        wall_lo=1.2, kappa_lo=50.0, wall_hi=6.5, kappa_hi=100.0,
                        grid_min=0.8, grid_max=8.5, grid_bin=360,
                        cv2_pt=None,
                        sigma2=0.05, wall2_lo=-4.5, kappa2_lo=100.0,
                        wall2_hi=2.0, kappa2_hi=100.0,
                        grid2_min=-6.0, grid2_max=3.5, grid2_bin=474,
                        no_wall_hi=5.5, no_kappa=100.0,
                        seed=None):
    """Write an ASE+PLUMED+UMA well-tempered metadynamics driver (md_uma_plumed.py)
    into output_dir, templated verbatim on the reference step_1 driver. Only the
    per-run values are substituted; MtD scalars default to the reference step_1
    values so the emitted file matches the reference unless overridden.

    Energy params are in kcal/mol: ``height`` (hill height) and ``kappa_lo/kappa_hi``
    (wall force constants, kcal/mol/A^2); the driver converts them to PLUMED kJ/mol
    at the point of use.

    cv_atoms : (int, int) — 0-based indices of the two atoms whose distance is the CV.
    cv2_pt : (int, int, int) | None — if given, a SECOND CV that is a PROTON-TRANSFER
        coordinate cv2 = d(N,H) - d(O,H) built from three 0-based atom indices (N = proton
        donor, O = proton acceptor, H = the transferring proton). cv2 is negative with H on N
        (reactant), ~0 at the shared TS, positive with H on O (product). The run becomes 2-D
        over (d1, cv2); cv2 gets its own walls (wall2_*/kappa2_*, in cv2 units, e.g. -4.5/+2.0),
        sigma (sigma2) and grid (grid2_*, which must span those walls). A d(N,O) UPPER wall at
        ``no_wall_hi`` (KAPPA ``no_kappa`` kcal/mol/A^2) keeps the proton near its donor/acceptor
        pair so cv2 stays well-defined. All walls are labeled and their bias (plus metad.bias)
        is printed to COLVAR alongside d1, cv2, dNO. None (the default) leaves the emitted
        driver a 1-D run over d1 only.
    no_wall_hi, no_kappa : d(N,O) upper-wall position (A) and force constant (kcal/mol/A^2),
        used only in the ``cv2_pt`` path.
    frozen_atoms : iterable[int] — 0-based indices held fixed (fragment 2).
    ca_restraint_atoms : iterable[int] — 0-based indices kept mobile but harmonically
        tethered to their initial positions (soft positional restraint), with force
        constant ``ca_restraint_k`` (kcal/mol/A^2). Empty by default (no restraints).
    seed : int | None — seeds the driver's RNG (numpy Generator), used for the initial
        MaxwellBoltzmannDistribution velocities and the Langevin thermostat noise, so the
        run is reproducible. None (the default) leaves it unseeded.

    Water containment is handled in-driver by flat-bottom Hookean tethers on every
    water O (active in all phases), so no PLUMED solvent wall is emitted.
    """
    cv0, cv1 = cv_atoms
    fr_str = " ".join(str(int(i)) for i in frozen_atoms)
    car_str = " ".join(str(int(i)) for i in ca_restraint_atoms)
    expected_waters = mda.Universe(os.path.join(output_dir, structure)).select_atoms(
        "resname WAT and name O").n_atoms
    text = _load_md_uma_plumed_template()
    for token, value in (
        ("__STRUCTURE__", structure),
        ("__CHARGE__", str(charge)),
        ("__MULT__", str(mult)),
        ("__NHEAT__", str(n_heat)),
        ("__NEQUIL__", str(n_equil)),
        ("__NMTD__", str(n_mtd)),
        ("__CV0__", str(int(cv0))),
        ("__CV1__", str(int(cv1))),
        ("__SIGMA__", str(sigma)),
        ("__HEIGHT__", str(height)),
        ("__PACE__", str(pace)),
        ("__BIASFACTOR__", str(biasfactor)),
        ("__WALL_LO__", str(wall_lo)),
        ("__KAPPA_LO__", str(kappa_lo)),
        ("__WALL_HI__", str(wall_hi)),
        ("__KAPPA_HI__", str(kappa_hi)),
        ("__GRID_MIN__", str(grid_min)),
        ("__GRID_MAX__", str(grid_max)),
        ("__GRID_BIN__", str(grid_bin)),
        ("__CV2_PT__",
         "None" if cv2_pt is None
         else f"({int(cv2_pt[0])}, {int(cv2_pt[1])}, {int(cv2_pt[2])})"),
        ("__NO_WALL_HI__", str(no_wall_hi)),
        ("__NO_KAPPA__", str(no_kappa)),
        ("__SIGMA2__", str(sigma2)),
        ("__WALL2_LO__", str(wall2_lo)),
        ("__KAPPA2_LO__", str(kappa2_lo)),
        ("__WALL2_HI__", str(wall2_hi)),
        ("__KAPPA2_HI__", str(kappa2_hi)),
        ("__GRID2_MIN__", str(grid2_min)),
        ("__GRID2_MAX__", str(grid2_max)),
        ("__GRID2_BIN__", str(grid2_bin)),
        ("__FR_STR__", fr_str),
        ("__CA_RESTRAINT_STR__", car_str),
        ("__CA_RESTRAINT_K__", str(ca_restraint_k)),
        ("__EXPECTED_WATERS__", str(int(expected_waters))),
        ("__SEED__", str(int(seed)) if seed is not None else "None"),
    ):
        text = text.replace(token, value)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "md_uma_plumed.py")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"wrote {out_path}")
    return out_path


def write_md_submit_script(output_dir, job_name):
    """Write submit_md_uma_plumed.sh (SLURM gengpu/sxm gpu:1, uma-env python) into
    output_dir, templated on the reference. Only the job name is substituted."""
    text = _SUBMIT_MD_UMA_PLUMED_TEMPLATE.replace("__JOB_NAME__", job_name)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "submit_md_uma_plumed.sh")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"wrote {out_path}")
    return out_path

