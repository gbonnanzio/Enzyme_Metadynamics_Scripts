import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd as calc_rmsd
import os
import csv
import re
import warnings
import subprocess
from collections import defaultdict
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import *

# Suppress warnings specific to MDAnalysis
warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")

# path specific to this cluster
SYSTEM_PATH = "/projects/p30041/gbf4422/"

# Directory holding this script and its bundled helper scripts (scripts/).
PREP_DIR = os.path.dirname(os.path.abspath(__file__))

# dictionary of charged protein residues with key being the Amber resname and the value being the net charge of that residue
residue_charge_dict = {'ARG':1,'LYS':1,'HIP':1,'ASP':-1,'GLU':-1,'MG':2,'TPP':-3}

# PARAMS TO SET
R_FIXED = 6         # Angstroms, outer cluster radius around the substrate/intermediate/Mg (constant)
R_FREE_SWEEP = [6]  # Angstroms, free-region radii around the substrate/intermediate/Mg
WATER_DISTANCE = 10 # Angstroms, any waters near protein
MIN_ENERGY_FRAME = False # set to true to find minimum energy frame in the trajectory, False sets to last frame

# RMSD-based frame selection: scan the trailing RMSD_WINDOW_NS of the production trajectory and
# pick the frame where the substrate's alpha-keto-acid core is closest (post-superposition) to
# its pose in min.rst. Mutually exclusive with MIN_ENERGY_FRAME (validated below).
MIN_RMSD_FRAME = True             # set to true to find the lowest-RMSD-to-min.rst frame
RMSD_WINDOW_NS = 10.0              # ns; trailing window scanned, at full frame resolution
RMSD_REFERENCE_FILE = "min.rst"    # bare filename, resolved inside each substrate's MD/run/ dir
RMSD_ALIGN_SELECT = "backbone"     # superposition selection (matches MD_utils.py's RMSD/RMSF convention)
RMSD_ATOM_KEYS = ['C2', 'O1', 'C3', 'O2', 'O3']  # alpha-keto-acid atoms (excludes 'R', the variable group)

if MIN_ENERGY_FRAME and MIN_RMSD_FRAME:
    raise ValueError("MIN_ENERGY_FRAME and MIN_RMSD_FRAME are mutually exclusive - set at most one to True")

# where we will read and write files to/from
HEAD_DIR = f"{SYSTEM_PATH}5EJ5/MD_200/DMC/"
OUTPUT_DIR = f"./5EJ5/Cluster_Sizes/First_Half/R_Fixed_0{R_FIXED}/"
SUBSTRATES_TO_PROCESS = [6]

# H-bond-partner augmentation: pull residues that H-bond into the pocket (e.g. a sidechain
# reaching in from just outside R_FIXED) into the fixed/free residue sets, using occupancy
# over the production trajectory rather than a single geometric frame.
HBOND_POCKET_CUTOFF = R_FIXED      # Angstroms; pocket radius for the H-bond pass (defaults to R_FIXED)
HBOND_D_A_CUTOFF = 3.5             # Angstroms; donor-acceptor distance cutoff for an H-bond
HBOND_D_H_A_ANGLE = 150.0          # degrees; minimum donor-H-acceptor angle for an H-bond
HBOND_WINDOW_NS = 50.0             # ns; trailing window of the production trajectory analyzed (full frame resolution, no striding)
HBOND_OCCUPANCY_THRESHOLD = 50.0   # a pair qualifies if its occupancy exceeds this % in at least one substrate
HBOND_RESULTS_DIR = os.path.join(OUTPUT_DIR, "hbond_results")
INCLUDE_HBOND_PARTNERS = True      # master toggle for the H-bond-partner augmentation
HBOND_PARTNERS_FREE = True         # pulled-in partners also join every free-region set

RMSD_RESULTS_DIR = os.path.join(OUTPUT_DIR, "rmsd_frame_scan")

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

def discover_equil_stage_files(md_directory):
    """
    Find every production trajectory stage in md_directory, matching
    mdcrd_equil.nc, mdcrd_equil2.nc, mdcrd_equil3.nc, ... (a restart continuation chain).

    Returns (latest_filename, ordered_paths):
        latest_filename - bare filename of the highest-numbered stage (what
                           write_cpptraj_and_run needs for last-frame extraction)
        ordered_paths    - absolute paths of every stage, in ascending order (for a
                           multi-file MDAnalysis Universe spanning the whole chain)
    """
    pattern = re.compile(r"^mdcrd_equil(\d*)\.nc$")
    staged = []
    for filename in os.listdir(md_directory):
        match = pattern.match(filename)
        if match:
            stage_num = int(match.group(1)) if match.group(1) else 1
            staged.append((stage_num, filename))
    if not staged:
        raise FileNotFoundError(f"No mdcrd_equil*.nc files found in {md_directory}")

    staged.sort(key=lambda pair: pair[0])
    ordered_filenames = [filename for _, filename in staged]
    ordered_paths = [os.path.join(md_directory, filename) for filename in ordered_filenames]
    return ordered_filenames[-1], ordered_paths

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
    Load a raw extracted frame (raw_pdb), keep the protein/substrate/intermediate/Mg plus
    nearby waters, renumber residues from 1, write the trimmed PDB, fix up the protein
    files, and return the reloaded trimmed Universe.
    """
    complex = mda.Universe(md_dir + raw_pdb)

    # get all atoms near the protein excluding NaCl
    trimmed_complex_initial = complex.select_atoms(
        f"(protein or resname {substrate} or resname {intermediate} or resname MG) "
        f"or ((around {WATER_DISTANCE} protein) and resname WAT)").residues

    # Renumber residues manually
    for i, residue in enumerate(trimmed_complex_initial):
        residue.resid = i + 1  # Assign new resid starting from 1

    write_universe(md_dir + trimmed_pdb, trimmed_complex_initial)
    edit_protein_files(md_dir + trimmed_pdb, extra_resnames=[f"{intermediate}", "MG"])
    return mda.Universe(md_dir + trimmed_pdb)


def compute_pocket_membership(u, ref_sel, cutoff, start, stop, step):
    """Per-frame set of protein resids within `cutoff` A of the reference.

    Returns {frame_index: set(resid)}. Any reference residue that is itself a
    protein residue is always included (an `around` selection excludes the
    reference atoms themselves); here the reference (TPP/MG/DON) is non-protein,
    so nothing extra is added.
    """
    pocket = {}
    sel_str = f"byres (protein and around {cutoff} ({ref_sel}))"
    ref_protein = set(u.select_atoms(f"protein and ({ref_sel})").residues.resids)
    for ts in u.trajectory[start:stop:step]:
        shell = u.select_atoms(sel_str)
        pocket[ts.frame] = set(shell.residues.resids) | ref_protein
    return pocket


def trailing_window_frame_range(u, window_ns):
    """(start, stop) frame indices for the trailing `window_ns` of u.trajectory.

    Relative to the CURRENT end of the trajectory (not an absolute simulation-time
    target), so this works correctly on a still-running production job. Falls back to
    using every available frame if per-frame spacing can't be determined.
    """
    n_total = len(u.trajectory)
    if n_total <= 1:
        return 0, n_total

    dt_ps = u.trajectory[1].time - u.trajectory[0].time
    if dt_ps <= 0:
        dt_ps = u.trajectory.dt
    dt_ns = dt_ps / 1000.0

    if dt_ns <= 0:
        print(f"WARN: could not determine per-frame spacing for trailing-window "
              f"selection; using all {n_total} available frames")
        n_window_frames = n_total
    else:
        n_window_frames = int(round(window_ns / dt_ns))
    n_window_frames = max(1, min(n_window_frames, n_total))

    start = max(0, n_total - n_window_frames)
    stop = n_total
    print(f"Trailing window: {n_total} total frames, dt={dt_ns:.4f} ns/frame, "
          f"window={window_ns} ns -> frames [{start}:{stop}) "
          f"({stop - start} frames analyzed)")
    return start, stop


def plot_bar(labels, values, title, out_png, ylabel="Occupancy (% of frames)",
             ymax=100):
    """Vertical bar chart of `values` against string `labels`."""
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(labels)), 6))
    ax.bar(range(len(labels)), values, color="steelblue")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_rmsd_scan(times_ps, values, best_idx, title, out_png):
    """Line plot of RMSD vs time, with the chosen (lowest-RMSD) frame marked."""
    fig, ax = plt.subplots(figsize=(8, 5))
    times_ns = [t / 1000.0 for t in times_ps]
    ax.plot(times_ns, values, color="steelblue", marker="o", markersize=3, linewidth=1)
    ax.scatter([times_ns[best_idx]], [values[best_idx]], color="crimson", zorder=5,
               label=f"selected frame ({values[best_idx]:.3f} A)")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("RMSD to min.rst (A)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def find_lowest_rmsd_frame(md_dir, prmtop_path, stage_paths, reference_file, substrate,
                            align_select, atom_keys, window_ns, results_dir, substrate_label):
    """Pick the trailing-window frame whose alpha-keto-acid core (atom_keys) is closest,
    after superposition on align_select, to its pose in reference_file (e.g. min.rst).

    Returns (stage_basename, local_frame_number_1_based) for write_cpptraj_and_run, mapping
    the globally-best frame back to whichever stage file actually contains it (the window can
    span a restart boundary, so the best frame need not be in the latest stage file).
    """
    # min.rst is an AMBERRESTART-convention netCDF file, which MDAnalysis's NCDF reader
    # rejects outright (confirmed against real data: "Conventions must contain 'AMBER'").
    # cpptraj already handles both conventions transparently, so convert it to a plain PDB
    # first via the same cpptraj pipeline used for every other frame extraction in this script.
    reference_pdb = "min_reference.pdb"
    write_cpptraj_and_run(reference_file, md_dir, out_pdb=reference_pdb)
    ref_path = os.path.join(md_dir, reference_pdb)
    ref_u = mda.Universe(ref_path)
    ref_substrate_ag = ref_u.select_atoms(f"resname {substrate}")
    aka_local = get_substrate_aka_indexes(ref_substrate_ag)
    if aka_local is None or any(k not in aka_local for k in atom_keys):
        raise ValueError(
            f"get_substrate_aka_indexes could not identify all of {atom_keys} "
            f"in {ref_path} (resname {substrate})")

    # global indices are valid on ANY frame sharing this topology (reference or production),
    # since atom ordering/identity never changes between frames - only coordinates do.
    global_indices = [int(ref_substrate_ag[aka_local[k]].index) for k in atom_keys]
    ref_coords = ref_u.atoms.positions[global_indices]

    print(f"RMSD reference : {ref_path}")
    print(f"RMSD atoms     : {dict(zip(atom_keys, global_indices))}")
    print(f"RMSD align sel : {align_select}")

    u = mda.Universe(prmtop_path, stage_paths)
    start, stop = trailing_window_frame_range(u, window_ns)

    frames, times_ps, rmsd_values = [], [], []
    for ts in u.trajectory[start:stop]:
        align.alignto(u, ref_u, select=align_select)
        mobile_coords = u.atoms.positions[global_indices]
        r = calc_rmsd(mobile_coords, ref_coords, superposition=False)
        frames.append(ts.frame)
        times_ps.append(ts.time)
        rmsd_values.append(r)

    best_pos = int(np.argmin(rmsd_values))
    best_global_frame = frames[best_pos]
    print(f"Lowest-RMSD frame: global frame {best_global_frame} "
          f"(t={times_ps[best_pos]/1000.0:.3f} ns), RMSD={rmsd_values[best_pos]:.3f} A")

    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, f"DMC_{substrate_label}_rmsd_frame_scan.csv")
    pd.DataFrame({"frame": frames, "time_ps": times_ps, "rmsd": rmsd_values}).to_csv(
        out_csv, index=False)
    print(f"RMSD scan CSV written to: {out_csv}")
    out_png = os.path.join(results_dir, f"DMC_{substrate_label}_rmsd_frame_scan.png")
    plot_rmsd_scan(times_ps, rmsd_values, best_pos,
                   f"Substrate {substrate_label}: RMSD of alpha-keto-acid core to min.rst "
                   f"(trailing {window_ns} ns)", out_png)
    print(f"RMSD scan plot written to: {out_png}")

    # map the global frame index back to (stage file, local 1-based frame number) for cpptraj,
    # since the trailing window can span a restart boundary.
    cumulative = 0
    for path in stage_paths:
        n = len(mda.Universe(prmtop_path, path).trajectory)
        if best_global_frame < cumulative + n:
            local_frame = best_global_frame - cumulative
            return os.path.basename(path), local_frame + 1
        cumulative += n
    raise IndexError(
        f"global frame {best_global_frame} not found within stage_paths {stage_paths} "
        f"(total frames counted: {cumulative})")


def analyze_substrate_hbonds(substrate_label, prmtop_path, trajectory_paths, ref_sel,
                              pocket_cutoff, d_a_cutoff, d_h_a_angle, window_ns,
                              results_dir, top_n=30):
    """Rank active-site protein-protein H-bond pairs by occupancy, over the trailing
    `window_ns` of the (possibly multi-file/restart-chained) production trajectory for
    one substrate. Only sidechain-involved protein-protein H-bonds are counted
    (backbone-backbone excluded); a pair qualifies only if at least one partner residue
    is within `pocket_cutoff` of `ref_sel` in that frame. Writes a per-substrate ranked
    CSV + bar chart to `results_dir` and returns the ranked DataFrame.
    """
    print(f"Substrate      : {substrate_label}")
    print(f"Topology       : {prmtop_path}")
    print(f"Trajectory     : {trajectory_paths}")
    print(f"Reference      : {ref_sel}")
    print(f"Pocket cutoff  : {pocket_cutoff} A")
    print(f"H-bond D-A/ang : {d_a_cutoff} A / {d_h_a_angle} deg")
    print("Loading universe ...", flush=True)

    u = mda.Universe(prmtop_path, trajectory_paths)
    start, stop = trailing_window_frame_range(u, window_ns)
    n_frames = stop - start

    # Backbone atom indices, used to enforce the sidechain-involved rule.
    bb = set(u.select_atoms("protein and backbone").indices)

    # --- 1) Detect all protein hydrogen bonds -------------------------------
    print("Guessing protein donors/acceptors and running H-bond analysis ...",
          flush=True)
    hba = HydrogenBondAnalysis(
        universe=u,
        d_a_cutoff=d_a_cutoff,
        d_h_a_angle_cutoff=d_h_a_angle,
        update_selections=False,
    )
    hba.hydrogens_sel = hba.guess_hydrogens("protein")
    hba.acceptors_sel = hba.guess_acceptors("protein")
    hba.run(start=start, stop=stop, step=1, verbose=True)
    hbonds = hba.results.hbonds  # cols: frame, donor_ix, H_ix, acceptor_ix, dist, angle
    print(f"Raw H-bond records: {len(hbonds)}", flush=True)

    # --- 2) Per-frame pocket membership -------------------------------------
    print("Computing per-frame pocket membership ...", flush=True)
    pocket = compute_pocket_membership(u, ref_sel, pocket_cutoff, start, stop, 1)

    # --- 3) Post-filter + aggregate by residue pair -------------------------
    resids = u.atoms.resids
    resnames = u.atoms.resnames

    pair_frames = defaultdict(set)          # frozenset({rid_a, rid_b}) -> {frame}
    pair_ever_in_pocket = defaultdict(lambda: [False, False])  # per-residue flag

    for row in hbonds:
        frame = int(row[0])
        d_ix, a_ix = int(row[1]), int(row[3])
        rid_d, rid_a = int(resids[d_ix]), int(resids[a_ix])

        if rid_d == rid_a:
            continue  # intra-residue
        if d_ix in bb and a_ix in bb:
            continue
        shell = pocket.get(frame, set())
        d_in = rid_d in shell
        a_in = rid_a in shell
        if not (d_in or a_in):
            continue

        key = frozenset((rid_d, rid_a))
        pair_frames[key].add(frame)
        ordered = sorted(key)
        flags = pair_ever_in_pocket[key]
        for rid, is_in in ((rid_d, d_in), (rid_a, a_in)):
            if is_in:
                flags[ordered.index(rid)] = True

    # --- 4) Occupancy + ranking ---------------------------------------------
    def label(rid):
        idx = np.where(resids == rid)[0][0]
        return f"{resnames[idx]}{rid}"

    records = []
    for key, frames in pair_frames.items():
        rid_a, rid_b = sorted(key)
        flags = pair_ever_in_pocket[key]
        records.append({
            "resA": label(rid_a),
            "resB": label(rid_b),
            "resid_A": rid_a,
            "resid_B": rid_b,
            "frames_bonded": len(frames),
            "n_frames": n_frames,
            "occupancy_pct": round(100.0 * len(frames) / n_frames, 2),
            "A_in_pocket": bool(flags[0]),
            "B_in_pocket": bool(flags[1]),
        })

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values("occupancy_pct", ascending=False).reset_index(drop=True)

    # --- 5) Output -----------------------------------------------------------
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir,
                           f"DMC_{substrate_label}_sidechain_hbond_pairs.csv")
    df.to_csv(out_csv, index=False)

    print(f"Analyzed {n_frames} frames; found {len(df)} qualifying residue pairs.")
    print(f"CSV written to: {out_csv}")
    if not df.empty:
        top = df.head(top_n)
        print(f"Top {min(10, len(top))} pairs by occupancy:")
        print(top.head(10)[["resA", "resB", "occupancy_pct",
                            "frames_bonded", "n_frames"]].to_string(index=False))
        out_png = os.path.join(
            results_dir, f"DMC_{substrate_label}_sidechain_hbond_pairs.png")
        plot_bar(
            [f"{a}-{b}" for a, b in zip(top["resA"], top["resB"])],
            list(top["occupancy_pct"]),
            f"Substrate {substrate_label}: active-site protein-protein H-bond pairs "
            f"(top {len(top)}, cutoff {pocket_cutoff} A)",
            out_png)
        print(f"Bar chart written to: {out_png}")
    return df


def build_combined(results, threshold):
    """Merge per-substrate DataFrames into one wide table of >threshold pairs.

    `results` maps substrate number -> per-substrate DataFrame. Returns a wide
    DataFrame with one occ_sub{N} column per substrate plus max_occupancy_pct and
    n_substrates_over_thresh, filtered to pairs exceeding `threshold` in at least
    one substrate and sorted by max occupancy descending.
    """
    if not results:
        return pd.DataFrame()

    labels = {}
    occ = defaultdict(dict)  # (rid_a, rid_b) -> {substrate: occupancy_pct}
    for sub, df in results.items():
        for _, r in df.iterrows():
            key = (int(r["resid_A"]), int(r["resid_B"]))
            labels[key] = (r["resA"], r["resB"])
            occ[key][sub] = float(r["occupancy_pct"])

    subs = sorted(results.keys())
    records = []
    for key, per_sub in occ.items():
        resA, resB = labels[key]
        row = {"resA": resA, "resB": resB, "resid_A": key[0], "resid_B": key[1]}
        vals = []
        for sub in subs:
            v = per_sub.get(sub, 0.0)
            row[f"occ_sub{sub}"] = round(v, 2)
            vals.append(v)
        row["max_occupancy_pct"] = round(max(vals), 2)
        row["n_substrates_over_thresh"] = int(sum(v > threshold for v in vals))
        records.append(row)

    combined = pd.DataFrame.from_records(records)
    if combined.empty:
        return combined
    combined = combined[combined["max_occupancy_pct"] > threshold]
    combined = combined.sort_values(
        "max_occupancy_pct", ascending=False).reset_index(drop=True)
    return combined


def augment_with_hbond_partners(fixed_residues, free_residues_by_r, combined_df,
                                 include=True, partners_free=True):
    """Pull H-bond partners of cutoff residues into the fixed/free residue sets.

    `combined_df` (from build_combined) lists every protein-protein residue pair with
    >threshold occupancy in at least one substrate. For each pair where exactly ONE
    residue is already in `fixed_residues`, the outside partner is added so the H-bond
    is represented whole. A single pass over the original membership - added residues
    do not recursively pull in their own partners. Added residues join every free-region
    set in `free_residues_by_r` only if `partners_free`.

    Returns (new_fixed_residues, new_free_residues_by_r, added), where `added` maps
    newly-added resid -> (added_label, partner_resid, partner_label, occupancy).
    """
    if not include or combined_df.empty:
        if not include:
            print("[hbond] INCLUDE_HBOND_PARTNERS is off; skipping augmentation.")
        else:
            print("[hbond] combined H-bond table is empty - no partners to pull in.")
        return (set(fixed_residues),
                {r: set(s) for r, s in free_residues_by_r.items()},
                {})

    cluster = set(int(r) for r in fixed_residues)
    added = {}
    for _, row in combined_df.iterrows():
        ra, rb = int(row["resid_A"]), int(row["resid_B"])
        la, lb = row["resA"], row["resB"]
        occ = row["max_occupancy_pct"]
        a_in, b_in = ra in cluster, rb in cluster
        if a_in and not b_in:
            added.setdefault(rb, (lb, ra, la, occ))
        elif b_in and not a_in:
            added.setdefault(ra, (la, rb, lb, occ))

    if added:
        print(f"[hbond] pulled {len(added)} residue(s) into the fixed/cluster set via "
              f"in-memory combined H-bond table:")
        for rid in sorted(added):
            label, _pid, plabel, occ = added[rid]
            print(f"  + {label} (resid {rid}) - H-bonds {plabel}, max occ {occ}%")
    else:
        print("[hbond] no out-of-cutoff H-bond partners to pull into the fixed/cluster set.")

    new_fixed = cluster | set(added)
    if partners_free:
        new_free_by_r = {r: (set(s) | set(added)) for r, s in free_residues_by_r.items()}
    else:
        new_free_by_r = {r: set(s) for r, s in free_residues_by_r.items()}
    return new_fixed, new_free_by_r, added


substrate = 'DON'
intermediate = 'TPP'
intermediate_resid = 557 # need to specify if there is more than one TPP in the system
Mg_resid = 558 # need to specify if there is more than one MG in the system

all_fixed_residues = set()           # residues within R_FIXED of substrate/intermediate/Mg (whole cluster shell)
all_free_residues = {r: set() for r in R_FREE_SWEEP}  # residues within each R_FREE of substrate/intermediate/Mg
residue_substrate_counts = Counter()  # cluster residue id -> number of substrate systems it appears in
reference_sel = f"resname {substrate} or resid {intermediate_resid} or resid {Mg_resid}"  # substrate + TPP + MG

all_substrates = [str(i) for i in SUBSTRATES_TO_PROCESS] #
substrates_to_skip = None
trimmed_complex = None      # keep the last-loaded frame as a representative for proline expansion
substrate_hbond_results = {}  # donor_substrate -> ranked H-bond-pair DataFrame
hbond_failed = {}
for donor_substrate in all_substrates:
    if substrates_to_skip is not None and donor_substrate in substrates_to_skip:
        continue

    # find the lowest energy frame(s) from the production runs of the MD simulations
    MD_dir = f"{HEAD_DIR}{donor_substrate}/MD/run/"
    prmtop_path = os.path.join(MD_dir, "complex_solvated.prmtop")
    latest_stage_filename, stage_paths = discover_equil_stage_files(MD_dir)

    if MIN_ENERGY_FRAME:
        # TODO: still hardcoded to the old equil3-only staging; generalize with a
        # discover_equil_stage_files-style helper for *.out files if this is ever re-enabled.
        find_lowest_energy_frame(MD_dir, ["equil3.out"])
    elif MIN_RMSD_FRAME:
        rmsd_stage_filename, rmsd_local_frame = find_lowest_rmsd_frame(
            MD_dir, prmtop_path, stage_paths, RMSD_REFERENCE_FILE, substrate,
            RMSD_ALIGN_SELECT, RMSD_ATOM_KEYS, RMSD_WINDOW_NS, RMSD_RESULTS_DIR,
            donor_substrate)
        write_cpptraj_and_run(rmsd_stage_filename, MD_dir, frame_number=rmsd_local_frame)
    else:
        write_cpptraj_and_run(latest_stage_filename, MD_dir)

    trimmed_complex = build_trimmed_frame(MD_dir, "equilibrated_system.pdb", "equilibrated_system_trimmed.pdb")

    # fixed shell: the substrate/intermediate/Mg themselves, plus any residue with any
    # atom within R_FIXED of them. "around" alone excludes the reference atoms from its
    # result, so the substrate/TPP/MG have to be added explicitly
    fixed = trimmed_complex.select_atoms(
        f"(byres (({reference_sel}) or (around {R_FIXED} ({reference_sel})))) "
        f"and not resname WAT")
    fixed_resids = set(int(r) for r in fixed.residues.resids)
    all_fixed_residues.update(fixed_resids)
    # count how many substrate systems each cluster residue appears in (single-frame path only)
    residue_substrate_counts.update(fixed_resids)
    # free region: the substrate/intermediate/Mg themselves, plus residues within each
    # swept R_FREE of them (same reasoning as the fixed shell above)
    for r in R_FREE_SWEEP:
        free = trimmed_complex.select_atoms(
            f"(byres (({reference_sel}) or (around {r} ({reference_sel})))) "
            f"and not resname WAT")
        all_free_residues[r].update(set(int(x) for x in free.residues.resids))

    # H-bond occupancy pass over the trailing HBOND_WINDOW_NS of this substrate's
    # production trajectory (all mdcrd_equil*.nc stages concatenated): finds residues
    # that reach into the pocket via a sidechain H-bond even if their backbone sits
    # outside R_FIXED. Per-substrate failures don't abort the geometric extraction.
    try:
        substrate_hbond_results[donor_substrate] = analyze_substrate_hbonds(
            donor_substrate, prmtop_path, stage_paths, reference_sel,
            HBOND_POCKET_CUTOFF, HBOND_D_A_CUTOFF, HBOND_D_H_A_ANGLE,
            HBOND_WINDOW_NS, HBOND_RESULTS_DIR)
    except Exception as exc:
        print(f"!! Substrate {donor_substrate} H-bond analysis FAILED: {exc}")
        hbond_failed[donor_substrate] = str(exc)


# combine the per-substrate H-bond tables and pull any out-of-cutoff partner into the
# fixed/free residue sets BEFORE proline-boundary expansion, so newly added residues
# also get their own proline check.
combined_hbond_df = build_combined(substrate_hbond_results, HBOND_OCCUPANCY_THRESHOLD)
os.makedirs(HBOND_RESULTS_DIR, exist_ok=True)
combined_hbond_csv = os.path.join(HBOND_RESULTS_DIR, "combined_hbond_pairs.csv")
combined_hbond_df.to_csv(combined_hbond_csv, index=False)
print(f"Combined H-bond pairs >{HBOND_OCCUPANCY_THRESHOLD}% in any substrate: "
      f"{len(combined_hbond_df)} -> {combined_hbond_csv}")
if hbond_failed:
    print(f"Substrates with FAILED H-bond analysis: {hbond_failed}")
if not combined_hbond_df.empty:
    combined_hbond_png = os.path.join(HBOND_RESULTS_DIR, "combined_hbond_pairs.png")
    top_hbond = combined_hbond_df.head(30)
    plot_bar(
        [f"{a}-{b}" for a, b in zip(top_hbond["resA"], top_hbond["resB"])],
        list(top_hbond["max_occupancy_pct"]),
        f"Max active-site H-bond occupancy across substrates "
        f"(>{HBOND_OCCUPANCY_THRESHOLD}%, cutoff {HBOND_POCKET_CUTOFF} A)",
        combined_hbond_png,
        ylabel="Max occupancy across substrates (%)")
    print(f"Combined H-bond bar chart written to: {combined_hbond_png}")

all_fixed_residues, all_free_residues, hbond_added_resids = augment_with_hbond_partners(
    all_fixed_residues, all_free_residues, combined_hbond_df,
    include=INCLUDE_HBOND_PARTNERS, partners_free=HBOND_PARTNERS_FREE)

# pull in the residue before any N-terminal proline so the HIMO sp3 cut never lands
# on the proline ring (protein topology is identical across substrates, so the last
# loaded frame is a valid representative). CLUSTER is the real backbone-cut boundary;
# the free region only pins/frees whole residues (no cut) but is expanded too for consistency.
cluster_residues = expand_for_terminal_prolines(trimmed_complex, sorted(all_fixed_residues))
print(f"CLUSTER RESIDUES ({len(cluster_residues)}): {cluster_residues}")
# Write one CSV per R_FREE value. CLUSTER is constant across the sweep; only the FREE row
# grows with R_FREE. Everything in CLUSTER but not in FREE is frozen by DMC_Full_Run.
if not os.path.exists(OUTPUT_DIR):
    print(f"Directory '{OUTPUT_DIR}' does not exist. Creating it...")
    os.makedirs(OUTPUT_DIR)
for r in R_FREE_SWEEP:
    free_residues = expand_for_terminal_prolines(
        trimmed_complex, sorted(all_free_residues[r]))
    output_file = f"{OUTPUT_DIR}DMC_rfixed_{R_FIXED}_rfree_{r}_active_residues.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['CLUSTER RESIDUES'] + cluster_residues)
        writer.writerow(['FREE RESIDUES'] + free_residues)
    print(f"FREE RESIDUES r={r} ({len(free_residues)}) -> {output_file}")
# cumulative frequency: how many cluster residues appear in at most n substrate systems
# (n = 1..number of systems)

n_systems = list(range(1, len(all_substrates) + 1))   # 1..20
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
