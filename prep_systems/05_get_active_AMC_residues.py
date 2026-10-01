import MDAnalysis as mda
import os
import csv
import re
import warnings
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import *

# Suppress warnings specific to MDAnalysis
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")

# Directory holding this script and its bundled helper scripts (scripts/).
PREP_DIR = os.path.dirname(os.path.abspath(__file__))
# dictionary of charged protein residues with key being the Amber resname and the value being the net charge of that residue
residue_charge_dict = {'ARG':1,'LYS':1,'HIP':1,'ASP':-1,'GLU':-1,'MG':2,'TPP':-3}

# Fixed-shell / free-region scheme: the full cluster is every protein residue with any atom
# within R_FIXED of any atom of the substrate, intermediate, or Mg; of those, only residues
# within R_FREE are allowed to move in the optimization, the rest of the shell is frozen.
# R_FIXED is constant; R_FREE is swept so the free region can be converged. One CSV is written
# per R_FREE value.

# PARAMS TO SET
R_FIXED = 5          # Angstroms, outer cluster radius around the substrate/intermediate/Mg (constant)
R_FREE_SWEEP = [5]   # Angstroms, free-region radii around the substrate/intermediate/Mg
WATER_DISTANCE = 10  # Angstroms, any waters near protein
MIN_ENERGY_FRAME = True   # True finds the lowest-energy frame in the trajectory, False uses the last frame

substrate = 'ACC'
intermediate = 'INO'
intermediate_resid = 1116 # need to specify if there is more than one INO in the system
Mg_resid = 557 # need to specify if there is more than one MG in the system

# where we will read and write files to/from
HEAD_DIR = '/projects/p30041/gbf4422/5EJ5/MD_Final/AMC/'
OUTPUT_DIR = f"/projects/p30041/gbf4422/Generate_QMMM_Geometries/5EJ5/Metadynamics_Clusters/Second_Half_Active_Residues/R_Fixed_0{R_FIXED}/"
SUBSTRATES_TO_PROCESS = [1,2,4,6]  # homo-coupled donor/acceptor indices to process

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

def write_cpptraj_and_run(md_input_file_path, md_directory, frame_number=None, out_pdb="equilibrated_system.pdb"):

    if frame_number is not None:
        trajin_line = f"trajin {md_input_file_path} {frame_number} {frame_number}"
    else:
        trajin_line = f"trajin {md_input_file_path} lastframe"

    cpptraj_content = f"""parm complex_solvated.prmtop
        {trajin_line}
        autoimage :1-1117
        trajout {out_pdb}
        go
        quit
        """
    with open(md_directory + "cpptraj.in", "w") as file:
        file.write(cpptraj_content)
    print("cpptraj.in written successfully!")

    # Run the bash script from the specified directory
    result = subprocess.run(
        ["bash", os.path.join(PREP_DIR, "scripts", "format_MD.sh"), md_directory],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Error: Script failed.")
        print(result.stdout)  # Print the normal output
        print(result.stderr)  # Print the error output
    else:
        print('tleap for MD processing was successful')
        print(result.stdout)  # Print the normal output

    return None

def find_lowest_energy_frame(md_directory, md_filenames, start_frame_to_search=0):

    lowest_min_energy = 0
    min_filename = None
    # iterate through any number of files specified looking for the lowest energy frame
    for md_filename in md_filenames:
        steps, all_energies = extract_potential_energy(md_directory+md_filename)
        energies_to_search = all_energies[start_frame_to_search:]
        min_energy = min(energies_to_search)
        if min_energy < lowest_min_energy:
            lowest_min_energy = min_energy
            min_filename = md_filename

    # extract lowest frame
    steps, all_energies = extract_potential_energy(md_directory+min_filename)
    energies_to_search = all_energies[start_frame_to_search:]
    min_energy = min(energies_to_search)
    min_frame = start_frame_to_search + energies_to_search.index(min_energy)

    # cpptraj uses 1 indexing
    file_start = min_filename.split(".")[0]

    write_cpptraj_and_run(f"mdcrd_{file_start}.nc", md_directory, min_frame + 1)

    return None

def extract_run_lowest_energy_frame(md_directory, md_filename, start_frame_to_search, out_pdb):
    """
    Extract the lowest-energy frame from a single production run (md_filename), writing
    it to out_pdb. Mirrors find_lowest_energy_frame but scopes the search to one run so
    multiple per-run frames can be sampled without overwriting each other.
    """
    steps, all_energies = extract_potential_energy(md_directory + md_filename)
    energies_to_search = all_energies[start_frame_to_search:]
    min_energy = min(energies_to_search)
    min_frame = start_frame_to_search + energies_to_search.index(min_energy)

    # cpptraj uses 1 indexing
    file_start = md_filename.split(".")[0]
    write_cpptraj_and_run(f"mdcrd_{file_start}.nc", md_directory, min_frame + 1, out_pdb=out_pdb)
    return None

def build_trimmed_frame(md_dir, raw_pdb, trimmed_pdb):
    """
    Load a raw extracted frame (raw_pdb), keep the protein/substrate/intermediate/TPP/Mg
    plus nearby waters, renumber residues from 1, write the trimmed PDB, fix up the
    protein files, and return the reloaded trimmed Universe.
    """
    complex = mda.Universe(md_dir + raw_pdb)

    # get all atoms near the protein excluding NaCl
    trimmed_complex_initial = complex.select_atoms(
        f"(protein or resname {substrate} or resname {intermediate} or resname TPP or resname MG) "
        f"or ((around {WATER_DISTANCE} protein) and resname WAT)").residues

    # Renumber residues manually
    for i, residue in enumerate(trimmed_complex_initial):
        residue.resid = i + 1  # Assign new resid starting from 1

    write_universe(md_dir + trimmed_pdb, trimmed_complex_initial)
    edit_protein_files(md_dir + trimmed_pdb, extra_resnames=[f"{intermediate}", "TPP", "MG"])
    return mda.Universe(md_dir + trimmed_pdb)


all_fixed_residues = set()           # residues within R_FIXED of substrate/intermediate/Mg (whole cluster shell)
all_free_residues = {r: set() for r in R_FREE_SWEEP}  # residues within each R_FREE of substrate/intermediate/Mg
residue_substrate_counts = Counter()  # cluster residue id -> number of substrate systems it appears in
if intermediate_resid is not None:
    reference_sel = f"resname {substrate} or resid {intermediate_resid} or resid {Mg_resid}"  # substrate + INO + MG
else:
    reference_sel = f"resname {substrate} or resname {intermediate} or resname MG"  # substrate + INO + MG
trimmed_complex = None      # keep the last-loaded frame as a representative for proline expansion

all_substrates = [str(i) for i in SUBSTRATES_TO_PROCESS]
substrates_to_skip = None

for donor_substrate in all_substrates:
    for acceptor_substrate in all_substrates:
        # for homo-coupled pairs only
        if donor_substrate != acceptor_substrate:
            continue
        if substrates_to_skip is not None:
            if donor_substrate in substrates_to_skip or acceptor_substrate in substrates_to_skip:
                continue

        MD_dir = f"{HEAD_DIR}{donor_substrate}/{donor_substrate}_{acceptor_substrate}/MD/run/"

        if MIN_ENERGY_FRAME:
            find_lowest_energy_frame(MD_dir, ["equil.out"])
        else:
            write_cpptraj_and_run("mdcrd_equil.nc", MD_dir)

        trimmed_complex = build_trimmed_frame(MD_dir, "equilibrated_system.pdb", "equilibrated_system_trimmed.pdb")

        # fixed shell: the substrate/intermediate/Mg themselves, plus any residue with any
        # atom within R_FIXED of them. "around" alone excludes the reference atoms from its
        # result, so the substrate/INO/MG have to be OR'd back in explicitly or they'd never
        # appear in the cluster.
        fixed = trimmed_complex.select_atoms(
            f"(byres (({reference_sel}) or (around {R_FIXED} ({reference_sel})))) "
            f"and not resname WAT")
        fixed_resids = set(int(r) for r in fixed.residues.resids)

        print(f"[{donor_substrate}] equilibrated_system_trimmed.pdb cluster total ({len(fixed_resids)})")

        all_fixed_residues.update(fixed_resids)
        # count how many substrate systems each cluster residue appears in
        residue_substrate_counts.update(fixed_resids)

        # free region: the substrate/intermediate/Mg themselves, plus residues within each
        # swept R_FREE of them (same reasoning as the fixed shell above)
        for r in R_FREE_SWEEP:
            free = trimmed_complex.select_atoms(
                f"(byres (({reference_sel}) or (around {r} ({reference_sel})))) "
                f"and not resname WAT")
            all_free_residues[r].update(set(int(x) for x in free.residues.resids))

# pull in the residue before any N-terminal proline so the HIMO sp3 cut never lands
# on the proline ring (protein topology is identical across substrates, so the last
# loaded frame is a valid representative). CLUSTER is the real backbone-cut boundary;
# the free region only pins/frees whole residues (no cut) but is expanded too for consistency.
cluster_residues = expand_for_terminal_prolines(trimmed_complex, sorted(all_fixed_residues))
print(f"CLUSTER RESIDUES ({len(cluster_residues)}): {cluster_residues}")

# Write one CSV per R_FREE value. CLUSTER is constant across the sweep; only the FREE row
# grows with R_FREE. Everything in CLUSTER but not in FREE is frozen by Second_Half_Full_Run.
if not os.path.exists(OUTPUT_DIR):
    print(f"Directory '{OUTPUT_DIR}' does not exist. Creating it...")
    os.makedirs(OUTPUT_DIR)
for r in R_FREE_SWEEP:
    free_residues = expand_for_terminal_prolines(
        trimmed_complex, sorted(all_free_residues[r]))
    output_file = f"{OUTPUT_DIR}AMC_rfixed_{R_FIXED}_rfree_{r}_active_residues.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['CLUSTER RESIDUES'] + cluster_residues)
        writer.writerow(['FREE RESIDUES'] + free_residues)
    print(f"FREE RESIDUES r={r} ({len(free_residues)}) -> {output_file}")

# cumulative frequency: how many cluster residues appear in at most n substrate systems
# (n = 1..number of systems)
n_systems = list(range(1, len(all_substrates) + 1))   # 1..number of substrate systems
residues_per_n = [sum(1 for c in residue_substrate_counts.values() if c == n)
                  for n in n_systems]
cumulative_residues = list(np.cumsum(residues_per_n))  # residues appearing in <= n systems
plt.figure(dpi=300)
plt.bar(n_systems, cumulative_residues)
plt.xlabel("Number of substrate systems")
plt.ylabel("Cumulative number of residues (<= n systems)")
plt.xticks(n_systems)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}residue_substrate_occurrence.png")
plt.close()
print(f"residue-occurrence cumulative chart -> {OUTPUT_DIR}residue_substrate_occurrence.png")
