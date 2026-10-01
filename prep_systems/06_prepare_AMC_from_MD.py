"""06_prepare_AMC_from_MD.py — prep-only cluster-model builder for the AMC stage.

Standalone mirror of Second_Half_Full_Run.py's stage 07 (prep_amc), structured exactly like
03_prepare_DMC_from_MD.py: reads the trimmed MD frame and the CLUSTER/FREE residue sets
written by 05_get_active_AMC_residues.py, builds the cluster model (protein cluster residues +
cofactors + selected waters), caps the truncated backbone, computes the fixed-shell frozen-atom
set, and writes the geometry + active_region.pdb + free_water_resids.dat.

Like stage 03, the "calc" for this pipeline is a well-tempered metadynamics run, not a
stationary opt: after building the cluster, this stage writes the metadynamics driver
(md_uma_plumed.py + submit script) into the SAME prep directory. The AMC collective variable is
the forming C-C bond between INO's reactive carbon (the TPP-derived nucleophile) and the
acceptor's reactive carbon — the second-half analog of stage 03's ThDP-C1 / donor-C2 bond.
"""

import csv
import os
import warnings

import numpy as np
import MDAnalysis as mda

from utils import *  # write_universe, edit_protein_files, get_substrate_aka_indexes,
                     # add_backbone_terminal_caps, expand_for_terminal_prolines, ...

warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")

# ---------------------------------------------------------------------------
# CONFIG — set once (mirrors the relevant subset of Second_Half_Full_Run.py)
# ---------------------------------------------------------------------------
R_FIXED = 5  # distance from ACC/INO/MG where protein residues are included (matches stage 12)
R_FREE = 5   # distance from ACC/INO/MG where residues (and waters, in WATER_FREE_MODE
             # "distance") are active
# --- water handling: two independent controls ------------------------------
# (1) INCLUSION — which waters are part of the cluster model (see _cluster_water_selection):
#   "distance" — waters within R_WATER_SELECTION of the reactive core (ACC/INO/MG)
#   "residues" — waters within R_WATER_SELECTION of ANY cluster residue
WATER_SELECTION_MODE = "distance"
R_WATER_SELECTION = 10  # Ang; radius used by WATER_SELECTION_MODE
# (2) MOBILITY — which included waters may move (see _free_water_resids):
#   "all"      — every water in the cluster is free (default)
#   "none"     — every water is frozen
#   "distance" — waters within R_FREE of the reactive core are free
WATER_FREE_MODE = "all"
# how Cα atoms in the free region are frozen:
#   "terminal" — only free residues at a one-sided boundary with the fixed shell (default)
#   "all"      — every Cα of every free protein residue
CA_FREEZE_MODE = "terminal"
# if set to an int X, keep only the X waters closest to the acceptor C2 atom and remove all
# other waters; None keeps every water selected by WATER_SELECTION_MODE (the default).
N_CLOSEST_WATERS = None

# --- whole-system solvation (added AFTER the cluster is built) --------------
# When True, the fully built + capped cluster (protonated_cluster.pdb) is wrapped
# in an explicit spherical water droplet (MDAnalysis lattice fill) and the added
# waters are kept mobile but contained by the driver's per-water Hookean tethers
# (the cluster MD is non-periodic, so uncontained bulk water would evaporate). The
# added waters carry no net charge, so the QM charge is unchanged.
SOLVATE_SYSTEM = False           # master toggle
# how the added water is shaped:
#   "shell"   — a skin hugging the molecular surface: keep waters within
#               SOLVATION_SHELL Ang of ANY solute atom (few waters; recommended)
#   "droplet" — fill the whole sphere enclosing the cluster (radius = farthest
#               solute atom from center + SOLVATION_SHELL); corners included (many waters)
SOLVATION_MODE = "shell"
SOLVATION_SHELL = 3.0            # Ang; shell skin thickness ("shell") or sphere padding ("droplet")
SOLVATION_CENTER = "solute"      # droplet only: "solute" (whole-cluster centroid) or "core" (ACC/INO/MG centroid)
SOLVATION_CLASH = 2.4            # Ang; drop lattice waters whose O is within this of any solute atom
WATER_DENSITY = 0.0333           # waters / Ang^3 (bulk ~1 per 29.9 Ang^3); sets lattice spacing
# DEPRECATED: the PLUMED spherical containment wall was replaced by the driver's
# per-water Hookean tethers (md_uma_plumed.py), which contain every water; these two
# constants are no longer used.
SOLVATION_WALL_KAPPA = 150.0     # kJ/mol/Ang^2 for the (removed) PLUMED containment wall
SOLVATION_WALL_OFFSET = 1.0      # Ang; (removed) wall radius = droplet radius + this

HEAD_DIR = "/projects/p30041/gbf4422/5EJ5/MD_Final/AMC/"               # output root (as in stage 03)
# Unlike DMC (stage 03 reads + writes the same MD_Final tree), the AMC equilibrated frames
# are trimmed into MD_Equilibration by 05_get_active_AMC_residues.py, so inputs are read from
# there while the cluster prep + metadynamics run are written under MD_Final/AMC.
MD_HEAD_DIR = "/projects/p30041/gbf4422/5EJ5/MD_Final/AMC/"  # trimmed frames read here
ACTIVE_RESIDUES_FILEPATH = (
    f"/projects/p30041/gbf4422/Generate_QMMM_Geometries/5EJ5/Metadynamics_Clusters/Second_Half_Active_Residues/"
    f"R_Fixed_0{R_FIXED}/AMC_rfixed_{R_FIXED}_rfree_{R_FREE}_active_residues.csv")
# combined active-site H-bond pairs (from analyze_active_site_hbonds_AMC.py): any pair with
# >50% occupancy in at least one substrate. When a cutoff residue H-bonds a partner just
# outside the cutoff, that partner is pulled into the cluster (see
# _augment_cluster_with_hbond_partners).
COMBINED_HBOND_CSV = (
    "/projects/p30041/gbf4422/5EJ5/MD_Equilibration/hbond_results_AMC/"
    "combined_over50_hbond_pairs.csv")
INCLUDE_HBOND_PARTNERS = False   # master toggle for the H-bond-partner augmentation
HBOND_PARTNERS_FREE = True      # False: pulled-in partners join the frozen fixed shell (CLUSTER,
                                # not FREE); True: also make them active

SUBSTRATES = [6]  # homo-coupled donor==acceptor substrates to prep

# --- step_3 metadynamics run generation (fills the old "backend" stub) -------
# In this pipeline the AMC "calc" is a well-tempered metadynamics run, not a
# stationary opt. After building the cluster, stage 13 writes the metadynamics
# driver (md_uma_plumed.py + submit script) into the SAME prep directory
# (output_dir), templated on 5EJ5/metadynamics/second_half/6/step_3/, with
# CHARGE/MULT/FROZEN_ATOMS/CV_ATOMS wired in from the prep.
WRITE_METADYNAMICS = True
FORCE_OVERWRITE_RUN = False  # if False, skip a run dir that already has a HILLS file
# Optional SECOND collective variable for the AMC step: the proton-transfer distance
# between the acceptor's carbonyl oxygen (ACC "O1") and the proton on INO's hydroxyl
# (INO "H1"). With this on, step_3 is 2-D well-tempered metadynamics over (forming C-C
# bond, O...H proton transfer); off, it is the 1-D C-C run (identical to stage 03).
INCLUDE_PROTON_CV = True

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
intermediate_resid = 1116  # need to specify if there is more than one INO in the system
Mg_resid = 557             # need to specify if there is more than one MG in the system
reference_sel = f"resname ACC or resid {intermediate_resid} or resid {Mg_resid}"
SUBSTRATE = "ACC"          # acceptor residue name
INTERMEDIATE = "INO"       # TPP-derived intermediate residue name
ACC_BASE_CHARGE = -1  # the acceptor carries a CO2 group
INO_BASE_CHARGE = -3  # INO with the alkoxide protonated
EXTRA_NEG_SUBSTRATES = {"4", "6", "11", "16"}  # carry an additional -1 charge
RESIDUE_CHARGE_DICT = {
    "ARG": 1, "LYS": 1, "HIP": 1, "ASP": -1, "GLU": -1, "MG": 2, "TPP": -3,
}


# ---------------------------------------------------------------------------
# residue-set loading (from the 12 CSV) + H-bond partner augmentation
# ---------------------------------------------------------------------------
def _load_residue_sets():
    """CLUSTER_RESIDS build the QM region; FREE_RESIDS are allowed to move; FIXED_RESIDS
    (the rest of the shell) are frozen in every stationary optimization."""
    csv_data = {}
    with open(ACTIVE_RESIDUES_FILEPATH, newline="") as f:
        for row in csv.reader(f):
            if row:
                csv_data[row[0]] = [int(x) for x in row[1:]]
    return csv_data["CLUSTER RESIDUES"], csv_data["FREE RESIDUES"]


def _augment_cluster_with_hbond_partners(cluster_resids, free_resids):
    """Pull H-bond partners of cutoff residues into the cluster.

    Reads COMBINED_HBOND_CSV (every protein-protein residue pair with >50% H-bond occupancy
    in at least one substrate). For each pair where exactly ONE residue is already in
    ``cluster_resids`` and the other is not, the outside partner is added so the H-bond is
    represented whole. A single pass over the original cluster membership — added residues do
    not recursively pull in their own partners. Added residues join CLUSTER only unless
    HBOND_PARTNERS_FREE, in which case they are also added to FREE (active).
    """
    if not INCLUDE_HBOND_PARTNERS:
        return list(cluster_resids), list(free_resids), {}
    if not os.path.exists(COMBINED_HBOND_CSV):
        raise FileNotFoundError(
            f"combined H-bond CSV not found: {COMBINED_HBOND_CSV}\n"
            f"Generate it with analyze_active_site_hbonds_AMC.py or set "
            f"INCLUDE_HBOND_PARTNERS = False.")

    cluster = set(int(r) for r in cluster_resids)
    added = {}
    with open(COMBINED_HBOND_CSV, newline="") as f:
        for row in csv.DictReader(f):
            ra, rb = int(row["resid_A"]), int(row["resid_B"])
            la, lb = row["resA"], row["resB"]
            occ = row.get("max_occupancy_pct", "")
            a_in, b_in = ra in cluster, rb in cluster
            if a_in and not b_in:
                added.setdefault(rb, (lb, ra, la, occ))
            elif b_in and not a_in:
                added.setdefault(ra, (la, rb, lb, occ))

    if added:
        print(f"[hbond] pulled {len(added)} residue(s) into the cluster via "
              f"combined H-bond CSV ({os.path.basename(COMBINED_HBOND_CSV)}):")
        for rid in sorted(added):
            label, _pid, plabel, occ = added[rid]
            print(f"  + {label} (resid {rid}) — H-bonds {plabel}, max occ {occ}%")
    else:
        print("[hbond] no out-of-cutoff H-bond partners to pull into the cluster.")

    new_cluster = sorted(cluster | set(added))
    if HBOND_PARTNERS_FREE:
        new_free = sorted(set(int(r) for r in free_resids) | set(added))
    else:
        new_free = list(free_resids)
    return new_cluster, new_free, added


CLUSTER_RESIDS, FREE_RESIDS = _load_residue_sets()
CLUSTER_RESIDS, FREE_RESIDS, HBOND_ADDED_RESIDS = _augment_cluster_with_hbond_partners(
    CLUSTER_RESIDS, FREE_RESIDS)
FIXED_RESIDS = sorted(set(CLUSTER_RESIDS) - set(FREE_RESIDS))


# ---------------------------------------------------------------------------
# region / freeze helpers (mirrors Second_Half_Full_Run.py)
# ---------------------------------------------------------------------------
def _cluster_water_selection():
    """MDAnalysis selection for which waters are INCLUDED in the cluster model, per
    WATER_SELECTION_MODE (radius R_WATER_SELECTION)."""
    if WATER_SELECTION_MODE == "distance":
        return f"resname WAT and byres (around {R_WATER_SELECTION} ({reference_sel}))"
    elif WATER_SELECTION_MODE == "residues":
        cluster_ids = " ".join(str(r) for r in CLUSTER_RESIDS)
        return (f"resname WAT and byres (around {R_WATER_SELECTION} "
                f"(resid {cluster_ids} and not resname WAT))")
    raise ValueError(
        f"unknown WATER_SELECTION_MODE {WATER_SELECTION_MODE!r} — "
        f"expected 'distance' or 'residues'")


def _free_water_resids(universe, working_dir, build=False):
    """Resids of the waters allowed to move, defined ONCE at stage 13 (build=True) and
    persisted to ``{working_dir}free_water_resids.dat`` so later stages reuse it verbatim.

    Which waters are free is set by WATER_FREE_MODE ("all"/"none"/"distance"). Stage 13
    always (re)builds and overwrites the file; later (build=False) stages read the cache and
    raise if it is missing. The reactive core is referenced as ``not protein and not resname
    WAT`` so the build is stage-agnostic.
    """
    path = os.path.join(working_dir, "free_water_resids.dat")
    if not build:
        if os.path.exists(path):
            with open(path) as f:
                return [int(x) for x in f.read().split()]
        raise FileNotFoundError(
            f"canonical free-water list not found: {path}\n"
            f"It is built once at stage 13; run stage 13 for this substrate/head-dir "
            f"before any later stage.")
    if WATER_FREE_MODE == "none":
        free_wat = universe.atoms[0:0]
    elif WATER_FREE_MODE == "all":
        free_wat = universe.select_atoms("resname WAT")
    elif WATER_FREE_MODE == "distance":
        free_wat = universe.select_atoms(
            f"resname WAT and (byres (around {R_FREE} (not protein and not resname WAT)))")
    else:
        raise ValueError(
            f"unknown WATER_FREE_MODE {WATER_FREE_MODE!r} — "
            f"expected 'all', 'none', or 'distance'")
    resids = sorted(set(int(r) for r in free_wat.residues.resids))
    os.makedirs(working_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write(" ".join(str(r) for r in resids) + "\n")
    print(f"[free-water] built canonical free-water set (mode={WATER_FREE_MODE!r}, "
          f"{len(resids)} waters) -> {path}")
    return resids


def _frozen_ca_indices(universe):
    """0-based indices of the free-region Cα atoms to freeze, per CA_FREEZE_MODE
    ("terminal" or "all"). Empty-selection cases are guarded to avoid a malformed
    selection string."""
    protein_resids = set(int(r) for r in universe.select_atoms("protein").residues.resids)
    free_in_frame = protein_resids & set(FREE_RESIDS)
    if CA_FREEZE_MODE == "terminal":
        resids = [r for r in free_in_frame
                  if (r - 1) not in protein_resids or (r + 1) not in protein_resids]
    elif CA_FREEZE_MODE == "all":
        resids = sorted(free_in_frame)
    else:
        raise ValueError(
            f"unknown CA_FREEZE_MODE {CA_FREEZE_MODE!r} — expected 'terminal' or 'all'")

    if not resids:
        return []
    ca = universe.select_atoms(
        "protein and name CA and resid " + " ".join(str(r) for r in resids))
    return [int(a.index) for a in ca]


def _fixed_shell_indices(universe, working_dir, build_free_waters=False):
    """0-based atom indices frozen in every stationary optimization: every atom of the
    fixed-shell protein residues (CLUSTER minus FREE), plus the cluster waters not in the
    canonical free-water set, plus the free-region CA atoms selected by CA_FREEZE_MODE.
    Recomputed per stage (mda.Merge reorders atoms); protein resids are stable and the free
    waters are pinned by the persisted resid list."""
    fixed_prot = universe.atoms[0:0]
    if FIXED_RESIDS:
        fixed_prot = universe.select_atoms(
            "protein and resid " + " ".join(str(r) for r in FIXED_RESIDS))
    free_wat_resids = _free_water_resids(universe, working_dir, build=build_free_waters)
    if free_wat_resids:
        frozen_wat = universe.select_atoms(
            "resname WAT and not resid " + " ".join(str(r) for r in free_wat_resids))
    else:
        frozen_wat = universe.select_atoms("resname WAT")
    frozen_ca = _frozen_ca_indices(universe)
    return sorted(set(int(a.index) for a in (fixed_prot + frozen_wat)) | set(frozen_ca))


def _write_active_region(universe, frozen_indices, output_dir, label):
    """Write just the active (free) region of `universe` to active_region.pdb and print an
    active/frozen/total atom count. The active set is the complement of frozen_indices."""
    n_total = len(universe.atoms)
    active = sorted(set(range(n_total)) - set(int(i) for i in frozen_indices))
    universe.atoms[active].write(os.path.join(output_dir, "active_region.pdb"))
    print(f"[{label}] active={len(active)} frozen={n_total - len(active)} "
          f"total={n_total} -> {output_dir}active_region.pdb")


# ---------------------------------------------------------------------------
# STAGE 13 — AMC cluster-model prep from the MD frame (prep-only)
# ---------------------------------------------------------------------------
def prep_amc(substrate):
    intermediate = INTERMEDIATE
    # nested AMC system layout (homo-coupled donor==acceptor): e.g. 6/6_6/
    system_dir = f"{substrate}/{substrate}_{substrate}/"
    md_dir = f"{MD_HEAD_DIR}{system_dir}MD/run/"
    output_dir = f"{HEAD_DIR}{system_dir}Step_03/"
    os.makedirs(output_dir, exist_ok=True)

    trimmed_complex = mda.Universe(f"{md_dir}equilibrated_system_trimmed.pdb")

    # per-substrate net charges (fresh dict copy so it isn't mutated across substrates). ACC
    # gets its own base charge (+extra if the acceptor is in the extra-negative set) and INO
    # gets its own base charge (+extra if the donor is in the extra-negative set) — homo-coupled
    # here, so donor == acceptor == substrate, but kept as separate lookups for clarity.
    charge_dict = dict(RESIDUE_CHARGE_DICT)
    acceptor_extra = -1 if substrate in EXTRA_NEG_SUBSTRATES else 0
    donor_extra = -1 if substrate in EXTRA_NEG_SUBSTRATES else 0
    charge_dict[SUBSTRATE] = ACC_BASE_CHARGE + acceptor_extra
    charge_dict[INTERMEDIATE] = INO_BASE_CHARGE + donor_extra

    # the QM region is the whole CLUSTER set from the CSV (stage 12) plus the cofactors and
    # the waters chosen by WATER_SELECTION_MODE / R_WATER_SELECTION. Which of those waters are
    # frozen vs. mobile is a separate control (WATER_FREE_MODE); _fixed_shell_indices freezes
    # the frozen ones.
    protein_substrate_cofactors_sel = "resid " + " ".join(str(r) for r in CLUSTER_RESIDS)
    if N_CLOSEST_WATERS is None:
        water_sel = _cluster_water_selection()
    else:
        # keep only the N_CLOSEST_WATERS waters nearest the acceptor C2 atom (the acceptor
        # carbon of the reactive C-C bond); every other water is dropped.
        acc_u = trimmed_complex.select_atoms(f"resname {SUBSTRATE}")
        c2_pos = acc_u[get_substrate_aka_indexes(acc_u)["C2"]].position
        water_res = trimmed_complex.select_atoms("resname WAT").residues
        ranked = sorted(
            water_res,
            key=lambda res: np.linalg.norm(res.atoms.positions - c2_pos, axis=1).min())
        keep_resids = sorted(int(res.resid) for res in ranked[:N_CLOSEST_WATERS])
        print(f"[n-closest-waters] keeping {len(keep_resids)}/{len(water_res)} waters "
              f"closest to acceptor C2 (requested {N_CLOSEST_WATERS})")
        # guard the empty case (X=0 or no waters): match nothing explicitly instead of the
        # malformed "resname WAT and resid " selection.
        water_sel = ("resname WAT and resid " + " ".join(str(r) for r in keep_resids)
                     if keep_resids else "resname WAT and not resname WAT")
    cluster_universe = trimmed_complex.select_atoms(
        f"({protein_substrate_cofactors_sel}) or ({water_sel})")

    total_qm_charge = sum(charge_dict.get(res.resname, 0)
                          for res in cluster_universe.residues)

    write_universe(f"{output_dir}initial_cluster.pdb", cluster_universe)
    edit_protein_files(f"{output_dir}initial_cluster.pdb",
                       extra_resnames=[intermediate, "MG"])
    cluster_universe = mda.Universe(f"{output_dir}initial_cluster.pdb")

    # cap terminal backbone N/C, then re-annotate TER records
    protonated_universe = add_backbone_terminal_caps(universe=cluster_universe)
    write_universe(f"{output_dir}protonated_cluster.pdb", protonated_universe)
    edit_protein_files(f"{output_dir}protonated_cluster.pdb",
                       extra_resnames=[intermediate, "MG"])

    final_cluster_universe = mda.Universe(f"{output_dir}protonated_cluster.pdb")
    frozen_indices = _fixed_shell_indices(final_cluster_universe, f"{HEAD_DIR}{system_dir}",
                                          build_free_waters=True)

    # Optional whole-system solvation: wrap the capped cluster in an explicit water
    # droplet AFTER the frozen shell is fixed. Waters are appended after every solute
    # atom, so `frozen_indices` stay valid and the droplet waters fall into the active
    # (mobile) set automatically; the driver's per-water Hookean tethers keep every
    # water (cluster + added) from evaporating, so no separate containment wall is needed.
    run_structure = "protonated_cluster.pdb"
    if SOLVATE_SYSTEM:
        # center is used for the droplet fill (if any)
        if SOLVATION_CENTER == "core":
            center = final_cluster_universe.select_atoms(
                f"resname {SUBSTRATE} or resname {intermediate} or resname MG"
            ).center_of_geometry()
        elif SOLVATION_CENTER == "solute":
            center = final_cluster_universe.atoms.center_of_geometry()
        else:
            raise ValueError(f"unknown SOLVATION_CENTER {SOLVATION_CENTER!r} — "
                             f"expected 'solute' or 'core'")
        if SOLVATION_MODE == "droplet":
            radius = float(np.linalg.norm(
                final_cluster_universe.atoms.positions - center, axis=1).max()) + SOLVATION_SHELL
            solvated_universe, _added_resids, added_o_indices = solvate_droplet(
                final_cluster_universe, center=center, radius=radius, mode="droplet",
                clash=SOLVATION_CLASH, density=WATER_DENSITY)
        else:
            solvated_universe, _added_resids, added_o_indices = solvate_droplet(
                final_cluster_universe, mode="shell", surface_dist=SOLVATION_SHELL,
                clash=SOLVATION_CLASH, density=WATER_DENSITY)
        write_universe(f"{output_dir}solvated_cluster.pdb", solvated_universe)
        edit_protein_files(f"{output_dir}solvated_cluster.pdb",
                           extra_resnames=[intermediate, "MG"])
        final_cluster_universe = mda.Universe(f"{output_dir}solvated_cluster.pdb")
        run_structure = "solvated_cluster.pdb"
        print(f"[solvate] substrate {substrate}: mode={SOLVATION_MODE}, added "
              f"{len(added_o_indices)} waters (contained by the driver's water "
              f"tethers; charge unchanged)")

    _write_active_region(final_cluster_universe, frozen_indices, output_dir, "amc")

    print(f"[amc] substrate {substrate}: total QM charge {total_qm_charge}, mult 1")

    # -----------------------------------------------------------------------
    # step_3 metadynamics run generation — the AMC "calc" for this pipeline.
    # The driver + submit script are written into the SAME prep directory
    # (output_dir) so the cluster (protonated_cluster.pdb) and the run live
    # together, matching the reference second_half/<sub>/step_3/ layout.
    # CV = the forming C-C bond: INO reactive carbon (get_ino_indexes "C2") and
    # acceptor reactive carbon (get_substrate_aka_indexes "C2"), as global 0-based
    # indices into protonated_cluster.pdb. Order (INO, acceptor) mirrors stage 03.
    # -----------------------------------------------------------------------
    if WRITE_METADYNAMICS:
        ino_u = final_cluster_universe.select_atoms(f"resname {intermediate}")
        ino_indexes = get_ino_indexes(ino_u, with_hydrogen=INCLUDE_PROTON_CV)
        ino_c_index = int(ino_u[ino_indexes["C2"]].index)
        acc_u = final_cluster_universe.select_atoms(f"resname {SUBSTRATE}")
        acc_indexes = get_substrate_aka_indexes(acc_u)
        acc_c_index = int(acc_u[acc_indexes["C2"]].index)
        cv_atoms = (ino_c_index, acc_c_index)

        # optional second CV: the O...H proton transfer from INO's hydroxyl (INO "H1")
        # to the acceptor carbonyl oxygen (ACC "O1"), driven alongside the C-C bond.
        cv2_atoms = None
        if INCLUDE_PROTON_CV:
            acc_o_index = int(acc_u[acc_indexes["O1"]].index)
            ino_h_index = int(ino_u[ino_indexes["H1"]].index)
            cv2_atoms = (acc_o_index, ino_h_index)

        # neutral-water count for the driver's tether guard (WAT = O/H1/H2, so one
        # O per water residue); the driver tethers each water O and asserts this count.
        expected_waters = final_cluster_universe.select_atoms(
            "resname WAT and name O").n_atoms

        if os.path.exists(os.path.join(output_dir, "HILLS")) and not FORCE_OVERWRITE_RUN:
            print(f"[mtd] {output_dir} already has a HILLS file — skipping generation "
                  f"(set FORCE_OVERWRITE_RUN=True to regenerate).")
        else:
            write_md_uma_plumed(output_dir, charge=total_qm_charge, mult=1,
                                cv_atoms=cv_atoms, cv2_atoms=cv2_atoms,
                                frozen_atoms=frozen_indices,
                                expected_waters=expected_waters,
                                structure=run_structure)
            write_md_submit_script(output_dir, job_name=f"{substrate}-S3")
            print(f"[mtd] substrate {substrate}: run dir {output_dir}, CV atoms "
                  f"{cv_atoms}" + (f" + proton CV {cv2_atoms}" if cv2_atoms else "")
                  + f", charge {total_qm_charge}, frozen {len(frozen_indices)}")


if __name__ == "__main__":
    for acceptor_substrate in (str(s) for s in SUBSTRATES):
        prep_amc(acceptor_substrate)
    print("All done!")
