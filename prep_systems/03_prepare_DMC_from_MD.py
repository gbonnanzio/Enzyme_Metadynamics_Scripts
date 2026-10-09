"""03_prepare_DMC_from_MD.py — prep-only cluster-model builder for the DMC stage.

Standalone mirror of DMC_Full_Run.py's stage 03 (prep_dmc): reads the trimmed MD frame and
the CLUSTER/FREE residue sets written by 02_get_active_DMC_residues.py, builds the cluster
model (protein cluster residues + cofactors + selected waters), caps the truncated backbone,
computes the fixed-shell frozen-atom set, and writes the geometry + active_region.pdb.
All cluster waters are active (mobile); only the fixed-shell protein is frozen.

This script is PREP-ONLY: it does not write or run any optimization job. The optimization
backend (ORCA/ASE + UMA) is intentionally left as an unfinished stub at the end of the loop.
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
# CONFIG — set once (mirrors the relevant subset of DMC_Full_Run.py)
# ---------------------------------------------------------------------------
R_FIXED = 6  # distance from TPP/DON/MG where protein residues are included (matches stage 02)
R_FREE = 6   # distance from TPP/DON/MG where protein residues are active
# --- water handling: INCLUSION only ----------------------------------------
# which waters are part of the cluster model (see _cluster_water_selection):
#   "distance" — waters within R_WATER_SELECTION of the reactive core (DON/TPP/MG)
#   "residues" — waters within R_WATER_SELECTION of ANY cluster residue
# Every included water is active (mobile); none are frozen.
WATER_SELECTION_MODE = "distance"
R_WATER_SELECTION = 10  # Ang; radius used by WATER_SELECTION_MODE
# which free-region Cα atoms are soft-restrained (harmonic tether to their initial MD
# position, strength CA_RESTRAINT_K) instead of hard-frozen:
#   "terminal" — only free residues at a one-sided boundary with the fixed shell (default)
#   "all"      — every Cα of every free protein residue
CA_RESTRAINT_MODE = "terminal"
# harmonic positional restraint strength on those Cα (kcal/mol/Å²)
CA_RESTRAINT_K = 6.0
# if set to an int X, keep only the X waters closest to the substrate C2 atom and remove all
# other waters; None keeps every water selected by WATER_SELECTION_MODE (the default).
N_CLOSEST_WATERS = None

# --- whole-system solvation (added AFTER the cluster is built) --------------
# When True, the fully built + capped cluster (protonated_cluster.pdb) is wrapped
# in an explicit spherical water droplet (MDAnalysis lattice fill) and the added
# waters are kept mobile but contained by a PLUMED spherical wall (the cluster MD
# is non-periodic, so uncontained bulk water would evaporate). The added waters
# carry no net charge, so the QM charge is unchanged.
SOLVATE_SYSTEM = False           # master toggle
# how the added water is shaped:
#   "shell"   — a skin hugging the molecular surface: keep waters within
#               SOLVATION_SHELL Ang of ANY solute atom (few waters; recommended)
#   "droplet" — fill the whole sphere enclosing the cluster (radius = farthest
#               solute atom from center + SOLVATION_SHELL); corners included (many waters)
SOLVATION_MODE = "shell"
SOLVATION_SHELL = 3.0            # Ang; shell skin thickness ("shell") or sphere padding ("droplet")
SOLVATION_CENTER = "solute"      # droplet only: "solute" (whole-cluster centroid) or "core" (DON/TPP/MG centroid)
SOLVATION_CLASH = 2.4            # Ang; drop lattice waters whose O is within this of any solute atom
WATER_DENSITY = 0.0333           # waters / Ang^3 (bulk ~1 per 29.9 Ang^3); sets lattice spacing

HEAD_DIR = "/projects/p30041/gbf4422/5EJ5/Metadynamics_Sensitivity/collective_variables/one_cv/"          # output root
MD_HEAD_DIR = "/projects/p30041/gbf4422/5EJ5/MD_200/DMC/"  # trimmed frames read here
ACTIVE_RESIDUES_FILEPATH = (
    f"/projects/p30041/gbf4422/Generate_QMMM_Geometries/5EJ5/Metadynamics_Clusters/First_Half_Active_Residues/"
    f"R_Fixed_0{R_FIXED}/DMC_rfixed_{R_FIXED}_rfree_{R_FREE}_active_residues.csv")

SUBSTRATES = [6]  # donor substrates to prep

# --- step_1 metadynamics run generation (fills the old "backend" stub) -------
# In this pipeline the DMC "calc" is a well-tempered metadynamics run, not a
# stationary opt. After building the cluster, stage 03 writes the metadynamics
# driver (md_uma_plumed.py + submit script) into the SAME prep directory
# (output_dir), templated on 5EJ5/metadynamics/first_half/6/step_1/, with
# CHARGE/MULT/FROZEN_ATOMS/CV_ATOMS wired in from the prep.
WRITE_METADYNAMICS = True
FORCE_OVERWRITE_RUN = False  # if False, skip a run dir that already has a HILLS file

# --- metadynamics parameters (passed through to write_md_uma_plumed) ---------
# The first CV is always the forming C-C bond distance. When ENABLE_SECOND_CV is True the
# run becomes 2-D well-tempered metadynamics over (C-C distance, proton-transfer coordinate
# cv2 = d(N,H) - d(O,H)); when False the run stays 1-D over the C-C CV only.
MTD_N_HEAT, MTD_N_EQUIL, MTD_N_MTD = 10000, 20000, 1000000   # integrator steps
MTD_SIGMA = 0.1                          # Gaussian width (Å)
MTD_HEIGHT = 0.5                         # hill height (kcal/mol)
MTD_PACE = 200                            # deposition stride (steps)
MTD_BIASFACTOR = 20  # (T + deltaT_bias) / T = (300 + 8000) / 300
MTD_SEED = 42           # int or None; seeds initial velocities + Langevin noise for a
                          # reproducible run (None = unseeded)
MTD_WALL_LO, MTD_KAPPA_LO = 1.2, 50.0    # lower wall AT (Å), KAPPA (kcal/mol/Å²)
MTD_WALL_HI, MTD_KAPPA_HI = 5.5, 100.0   # upper wall AT (Å), KAPPA (kcal/mol/Å²)
MTD_GRID_MIN, MTD_GRID_MAX = 0.8, 7.0
MTD_GRID_BIN = int((MTD_GRID_MAX - MTD_GRID_MIN)//0.02)

# --- optional SECOND CV: proton-transfer coordinate cv2 = d(N,H) - d(O,H) ----
# Captures the HN4' proton transfer from ThDP N4' to the donor carbonyl O1:
#   N = ThDP N4'  (proton donor, literal PDB atom name TPP_DONOR_N_NAME)
#   H = ThDP HN4' (transferring proton, literal PDB atom name TPP_PROTON_NAME)
#   O = donor O1  (proton acceptor; the GEOMETRIC carbonyl O from get_substrate_aka_indexes,
#                  which may be named differently than the atom literally called "O1")
# Sign convention: cv2 < 0 reactant (H on N4'), ~0 shared/TS, cv2 > 0 product (H on O1).
# Expected: reactant ~ -0.5..-0.9 A, TS ~0, product ~ +0.5..+0.9 A.
ENABLE_SECOND_CV = False      # False -> 1-D run over the C-C CV only (original behavior)
TPP_PROTON_NAME  = "HN4'"    # transferring proton on ThDP N4'
TPP_DONOR_N_NAME = "N4'"     # proton-donor nitrogen on ThDP
# cv2 walls/sigma/grid (cv2 is a length difference in A; kappa kcal/mol/A^2). The walls sit
# OUTSIDE both the reactant and product basins so they bound escape without shaping the FES.
MTD2_SIGMA = 0.05
# Lower wall at -4.5: the proton transfer is COUPLED to C-C formation, so at the reactant
# (large CV1) the acceptor O1 is far and d(N,O) can grow as large as the C-C distance (CV1
# wall 5.5 A). With the proton on N (d(N,H) ~1.0) and d(N,O) up to ~5.5, cv2 = d(N,H)-d(O,H)
# can reach ~-4.5; the wall sits just past that geometric limit so it bounds escape without
# clamping the reactant basin. A prior run crashed when cv2 ran past the old grid edge (-2.5).
MTD2_WALL_LO, MTD2_KAPPA_LO = -4.5, 100.0   # cv2 lower wall (reactant side)
MTD2_WALL_HI, MTD2_KAPPA_HI = 2.0, 100.0    # cv2 upper wall (product side)
MTD2_GRID_MIN, MTD2_GRID_MAX = -6.5, 3.5    # must span the cv2 walls (with ~1.5 A margin)
MTD2_GRID_BIN = int((MTD2_GRID_MAX - MTD2_GRID_MIN) // 0.02)
# d(N,O) upper wall raised to 5.5 A to MATCH the CV1 upper wall: because PT is coupled to C-C
# formation, d(N,O) tracks CV1 and can legitimately separate as far as the C-C distance. A
# tighter cap here would silently clamp cv2 above the -4.5 wall and fight CV1.
MTD2_NO_WALL_HI, MTD2_NO_KAPPA = 5.5, 100.0
# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
intermediate_resid = 557  # need to specify if there is more than one TPP in the system
Mg_resid = 558            # need to specify if there is more than one MG in the system
reference_sel = f"resname DON or resid {intermediate_resid} or resid {Mg_resid}"
SUBSTRATE = "DON"
DONOR_BASE_CHARGE = -1  # every donor carries a CO2 group
EXTRA_NEG_SUBSTRATES = {"4", "6", "11", "16"}  # carry an additional -1 charge
RESIDUE_CHARGE_DICT = {
    "ARG": 1, "LYS": 1, "HIP": 1, "ASP": -1, "GLU": -1, "MG": 2, "TPP": -3,
}


# ---------------------------------------------------------------------------
# residue-set loading (from the 02 CSV, which already includes any H-bond-partner
# augmentation 02_get_active_DMC_residues.py performed)
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


CLUSTER_RESIDS, FREE_RESIDS = _load_residue_sets()
FIXED_RESIDS = sorted(set(CLUSTER_RESIDS) - set(FREE_RESIDS))


# ---------------------------------------------------------------------------
# region / freeze helpers (mirrors DMC_Full_Run.py)
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


def _restrained_ca_indices(universe):
    """0-based indices of the free-region Cα atoms to soft-restrain, per CA_RESTRAINT_MODE
    ("terminal" or "all"). These are tethered to their initial positions by the driver
    (strength CA_RESTRAINT_K) rather than frozen. Empty-selection cases are guarded to
    avoid a malformed selection string."""
    protein_resids = set(int(r) for r in universe.select_atoms("protein").residues.resids)
    free_in_frame = protein_resids & set(FREE_RESIDS)
    if CA_RESTRAINT_MODE == "terminal":
        resids = [r for r in free_in_frame
                  if (r - 1) not in protein_resids or (r + 1) not in protein_resids]
    elif CA_RESTRAINT_MODE == "all":
        resids = sorted(free_in_frame)
    else:
        raise ValueError(
            f"unknown CA_RESTRAINT_MODE {CA_RESTRAINT_MODE!r} — expected 'terminal' or 'all'")

    if not resids:
        return []
    ca = universe.select_atoms(
        "protein and name CA and resid " + " ".join(str(r) for r in resids))
    return [int(a.index) for a in ca]


def _fixed_shell_indices(universe):
    """Returns (frozen_indices, ca_restraint_indices), both 0-based.

    frozen_indices — atoms held fixed with FixAtoms: every atom of the fixed-shell
    protein residues (CLUSTER minus FREE). All cluster waters are active (mobile), so no
    water is frozen. ca_restraint_indices — the free-region CA atoms selected by
    CA_RESTRAINT_MODE, which are kept mobile but soft-restrained (not frozen).
    Recomputed per stage (mda.Merge reorders atoms); protein resids are stable."""
    fixed_prot = universe.atoms[0:0]
    if FIXED_RESIDS:
        fixed_prot = universe.select_atoms(
            "protein and resid " + " ".join(str(r) for r in FIXED_RESIDS))
    frozen_indices = sorted(set(int(a.index) for a in fixed_prot))
    ca_restraint_indices = _restrained_ca_indices(universe)
    return frozen_indices, ca_restraint_indices


def _write_active_region(universe, frozen_indices, output_dir, label):
    """Write just the active (free) region of `universe` to active_region.pdb and print an
    active/frozen/total atom count. The active set is the complement of frozen_indices."""
    n_total = len(universe.atoms)
    active = sorted(set(range(n_total)) - set(int(i) for i in frozen_indices))
    universe.atoms[active].write(os.path.join(output_dir, "active_region.pdb"))
    print(f"[{label}] active={len(active)} frozen={n_total - len(active)} "
          f"total={n_total} -> {output_dir}active_region.pdb")


def _write_metadynamics_readme(output_dir, substrate, charge, cv_atoms, cv2_pt,
                               n_frozen, n_ca):
    """Write README_metadynamics.md into output_dir: every metadynamics parameter emitted
    into md_uma_plumed.py plus the reasoning behind each choice. Only called for the 2-D
    proton-transfer run (cv2_pt set)."""
    n_idx, o_idx, h_idx = cv2_pt
    readme = f"""# Well-tempered metadynamics — DMC C-C bond formation + proton transfer

Substrate {substrate}. Run structure: `protonated_cluster.pdb`. QM charge {charge}, mult 1.
Driver: `md_uma_plumed.py` (ASE + PLUMED + UMA); submit with `submit_md_uma_plumed.sh`.

## Collective variables
- **CV1 = d(C,C)** — forming C-C bond, atoms (0-based) {cv_atoms}: ThDP carbanion C1 and
  donor C2. Reactant ~3.5-5 Å, product ~1.5-1.55 Å.
- **CV2 = d(N,H) - d(O,H)** — proton transfer, 0-based N={n_idx} (ThDP N4', {TPP_DONOR_N_NAME}),
  O={o_idx} (donor O1, geometric carbonyl O), H={h_idx} (ThDP {TPP_PROTON_NAME}). Built in
  PLUMED as `cv2: COMBINE ARG=dNH,dOH COEFFICIENTS=1,-1`. Sign: cv2<0 reactant (H on N4'),
  ~0 shared/TS, cv2>0 product (H on O1). Expect reactant ~ -0.5..-0.9 Å, product ~ +0.5..+0.9 Å.

## Walls (soft harmonic LOWER_WALLS/UPPER_WALLS; kappa in kcal/mol/Å²)
- CV1: lower {MTD_WALL_LO} Å (k={MTD_KAPPA_LO}), upper {MTD_WALL_HI} Å (k={MTD_KAPPA_HI}).
  Product (~1.5 Å) sits well inside the lower wall; upper wall bounds dissociation and is
  below GRID_MAX={MTD_GRID_MAX}.
- CV2: lower {MTD2_WALL_LO} Å (k={MTD2_KAPPA_LO}), upper {MTD2_WALL_HI} Å (k={MTD2_KAPPA_HI}).
  Both walls sit OUTSIDE the reactant/product basins (|cv2|~0.5-0.9 Å) so they bound escape
  without shaping the interior FES.
- d(N,O): upper wall {MTD2_NO_WALL_HI} Å (k={MTD2_NO_KAPPA}) keeps the proton near its
  donor/acceptor pair so cv2 stays well-defined (proton can't leave for another acceptor).

## Gaussian hills (well-tempered)
- Sigma: CV1 {MTD_SIGMA} Å, CV2 {MTD2_SIGMA} Å. Starting values (CV1 ~0.1-0.2, CV2 ~0.05-0.1);
  for a tuned run, set each to ~1/3-1/2 of that CV's fluctuation std from a short unbiased run
  in each basin.
- Height {MTD_HEIGHT} kcal/mol, deposition pace {MTD_PACE} steps, bias factor {MTD_BIASFACTOR}.
- Grid: CV1 [{MTD_GRID_MIN}, {MTD_GRID_MAX}] Å / {MTD_GRID_BIN} bins; CV2 [{MTD2_GRID_MIN},
  {MTD2_GRID_MAX}] Å / {MTD2_GRID_BIN} bins (grid spans the walls, ~0.02 Å/bin).

## Schedule & reproducibility
Heat {MTD_N_HEAT} + equilibrate {MTD_N_EQUIL} + metadynamics {MTD_N_MTD} integrator steps.
Seed {MTD_SEED}. Frozen atoms: {n_frozen}; soft Cα restraints: {n_ca} (k={CA_RESTRAINT_K}).

## Logging & convergence checks
`COLVAR` logs d1, cv2, dNO, metad.bias and every wall's bias energy; `walls.log` logs
d1, cv2, d(N,O) plus the water-tether diagnostics. To check:
- Confirm the proton transfers only between N4' and O1 (watch dNO and cv2; if it finds another
  acceptor, switch CV2 to a coordination-number / sum-over-acceptors CV).
- Confirm several recrossings between reactant and product basins; track ΔF vs. time.
- Inspect the 2-D FES: diagonal path = concerted, L-shaped = stepwise.
- Verify no FES feature sits on a wall and product d(C,C)~1.5 Å is well inside the CV1 walls.
"""
    path = os.path.join(output_dir, "README_metadynamics.md")
    with open(path, "w") as f:
        f.write(readme)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# STAGE 03 — DMC cluster-model prep from the MD frame (prep-only)
# ---------------------------------------------------------------------------
def prep_dmc(substrate):
    intermediate = "TPP"
    md_dir = f"{MD_HEAD_DIR}{substrate}/MD/run/"
    #output_dir = f"{HEAD_DIR}{substrate}/Step_01/"
    output_dir = f"{HEAD_DIR}/Step_01/"
    os.makedirs(output_dir, exist_ok=True)

    trimmed_complex = mda.Universe(f"{md_dir}equilibrated_system_trimmed.pdb")

    # per-substrate net charge of the donor (fresh dict copy so it isn't mutated across
    # substrates)
    charge_dict = dict(RESIDUE_CHARGE_DICT)
    additional_charge = -1 if substrate in EXTRA_NEG_SUBSTRATES else 0
    charge_dict[SUBSTRATE] = DONOR_BASE_CHARGE + additional_charge

    # the QM region is the whole CLUSTER set from the CSV (stage 02) plus the cofactors and
    # the waters chosen by WATER_SELECTION_MODE / R_WATER_SELECTION. Every included water is
    # active (mobile); only the fixed-shell protein is frozen (_fixed_shell_indices).
    protein_substrate_cofactors_sel = "resid " + " ".join(str(r) for r in CLUSTER_RESIDS)
    if N_CLOSEST_WATERS is None:
        water_sel = _cluster_water_selection()
    else:
        # keep only the N_CLOSEST_WATERS waters nearest the substrate C2 atom (the donor
        # carbonyl carbon of the reactive C-C bond); every other water is dropped.
        don_u = trimmed_complex.select_atoms(f"resname {SUBSTRATE}")
        c2_pos = don_u[get_substrate_aka_indexes(don_u)["C2"]].position
        water_res = trimmed_complex.select_atoms("resname WAT").residues
        ranked = sorted(
            water_res,
            key=lambda res: np.linalg.norm(res.atoms.positions - c2_pos, axis=1).min())
        keep_resids = sorted(int(res.resid) for res in ranked[:N_CLOSEST_WATERS])
        print(f"[n-closest-waters] keeping {len(keep_resids)}/{len(water_res)} waters "
              f"closest to substrate C2 (requested {N_CLOSEST_WATERS})")
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
    frozen_indices, ca_restraint_indices = _fixed_shell_indices(final_cluster_universe)

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

    _write_active_region(final_cluster_universe, frozen_indices, output_dir, "dmc")

    print(f"[dmc] substrate {substrate}: total QM charge {total_qm_charge}, mult 1")

    # -----------------------------------------------------------------------
    # step_1 metadynamics run generation — the DMC "calc" for this pipeline.
    # The driver + submit script are written into the SAME prep directory
    # (output_dir) so the cluster (protonated_cluster.pdb) and the run live
    # together, matching the reference first_half/<sub>/step_1/ layout.
    # CV = the forming C-C bond: ThDP reactive carbon (get_ThDP_indexes "C1") and
    # donor reactive carbon (get_substrate_aka_indexes "C2"), as global 0-based
    # indices into protonated_cluster.pdb. Order (ThDP, donor) matches the reference.
    # -----------------------------------------------------------------------
    if WRITE_METADYNAMICS:
        tpp_u = final_cluster_universe.select_atoms(f"resname {intermediate}")
        thdp_c_index = int(tpp_u[get_ThDP_indexes(tpp_u)["C1"]].index)
        don_u = final_cluster_universe.select_atoms(f"resname {SUBSTRATE}")
        substrate_indexes = get_substrate_aka_indexes(don_u)
        donor_c_index = int(don_u[substrate_indexes["C2"]].index)
        cv_atoms = (thdp_c_index, donor_c_index)

        # optional second CV: proton-transfer coordinate cv2 = d(N,H) - d(O,H), where
        # N = ThDP N4', H = ThDP HN4' (both by literal atom name, exactly one each in the
        # single-TPP cluster) and O = donor O1 (geometric carbonyl O). All global 0-based.
        cv2_pt = None
        if ENABLE_SECOND_CV:
            def _one_index(sel, what):
                a = final_cluster_universe.select_atoms(sel)
                assert a.n_atoms == 1, (
                    f"expected exactly one {what}, got {a.n_atoms} (selection {sel!r})")
                return int(a[0].index)
            n_index = _one_index(
                f"resname {intermediate} and name {TPP_DONOR_N_NAME}", "ThDP N4'")
            h_index = _one_index(
                f"resname {intermediate} and name {TPP_PROTON_NAME}", "ThDP HN4'")
            o1_index = int(don_u[substrate_indexes["O1"]].index)
            cv2_pt = (n_index, o1_index, h_index)   # (N donor, O acceptor, H proton)

        if os.path.exists(os.path.join(output_dir, "HILLS")) and not FORCE_OVERWRITE_RUN:
            print(f"[mtd] {output_dir} already has a HILLS file — skipping generation "
                  f"(set FORCE_OVERWRITE_RUN=True to regenerate).")
        else:
            write_md_uma_plumed(output_dir, charge=total_qm_charge, mult=1,
                                cv_atoms=cv_atoms, frozen_atoms=frozen_indices,
                                structure=run_structure,
                                ca_restraint_atoms=ca_restraint_indices,
                                ca_restraint_k=CA_RESTRAINT_K,
                                n_heat=MTD_N_HEAT, n_equil=MTD_N_EQUIL, n_mtd=MTD_N_MTD,
                                sigma=MTD_SIGMA, height=MTD_HEIGHT, pace=MTD_PACE,
                                biasfactor=MTD_BIASFACTOR,
                                wall_lo=MTD_WALL_LO, kappa_lo=MTD_KAPPA_LO,
                                wall_hi=MTD_WALL_HI, kappa_hi=MTD_KAPPA_HI,
                                grid_min=MTD_GRID_MIN, grid_max=MTD_GRID_MAX,
                                grid_bin=MTD_GRID_BIN,
                                cv2_pt=cv2_pt,
                                sigma2=MTD2_SIGMA,
                                wall2_lo=MTD2_WALL_LO, kappa2_lo=MTD2_KAPPA_LO,
                                wall2_hi=MTD2_WALL_HI, kappa2_hi=MTD2_KAPPA_HI,
                                grid2_min=MTD2_GRID_MIN, grid2_max=MTD2_GRID_MAX,
                                grid2_bin=MTD2_GRID_BIN,
                                no_wall_hi=MTD2_NO_WALL_HI, no_kappa=MTD2_NO_KAPPA,
                                seed=MTD_SEED)
            write_md_submit_script(output_dir, job_name=f"{substrate}-S1")
            if cv2_pt is not None:
                _write_metadynamics_readme(
                    output_dir, substrate=substrate, charge=total_qm_charge,
                    cv_atoms=cv_atoms, cv2_pt=cv2_pt,
                    n_frozen=len(frozen_indices), n_ca=len(ca_restraint_indices))
            cv2_msg = (f", CV2 (proton transfer) N/O/H atoms {cv2_pt}"
                       if cv2_pt is not None else "")
            print(f"[mtd] substrate {substrate}: run dir {output_dir}, CV atoms "
                  f"{cv_atoms}{cv2_msg}, charge {total_qm_charge}, frozen "
                  f"{len(frozen_indices)}, CA-restrained {len(ca_restraint_indices)} "
                  f"(k={CA_RESTRAINT_K} kcal/mol/Å²)")


if __name__ == "__main__":
    for donor_substrate in (str(s) for s in SUBSTRATES):
        prep_dmc(donor_substrate)
    print("All done!")
