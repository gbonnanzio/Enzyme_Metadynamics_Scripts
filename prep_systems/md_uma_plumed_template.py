#!/projects/p30041/gbf4422/envs/uma-env/bin/python3
"""
Heating -> equilibration -> well-tempered metadynamics with ASE + PLUMED,
driving the UMA (FAIRChem) potential in-process on the GPU.

  1. Heat the active region 100 -> 300 K (Langevin thermostat).
  2. Equilibrate at 300 K.
  3. Well-tempered metadynamics on the forming C-C bond (C2 of ThDP <-> donor
     C1, CV_ATOMS below), with lower/upper walls, biasing handled by PLUMED.

The CV1/CV2 PLUMED walls are active from step 0 (heat), not just phase 3: a
walls-only PLUMED input (no METAD, no grid) is attached before heating, and
is swapped for the full walls+METAD input at the start of phase 3. Without
this, an unbiased heat/equil can drift the CV past the eventual METAD grid
edge and crash PLUMED the instant phase 3 attaches ("Looking for a value
outside the grid"); the walls restrain the CV throughout, so by the time the
grid-based METAD input attaches the CV is already guaranteed in-range.

The atoms are split three ways:
  * FROZEN_ATOMS (0-based) — the fixed protein shell, held rigid with FixAtoms (currently none)
  * CA_RESTRAINT_ATOMS (0-based) — the truncated backbone anchor CA. These are softly tethered 
    to their initial positions by an always-on Hookean restraint (CA_RESTRAINT_K kcal/mol/A^2),
  * everything else — the free active region.

Every water O is held inside a single flat-bottom containment sphere 
(loose Hookean restraint to the centroid of the whole system)

Restart: the metadynamics phase resumes from mtd_restart.traj (latest state
only, written atomically). On resume, HILLS/COLVAR/logs/trajectories are
trimmed back to the checkpoint so nothing is double counted.

Requires the PLUMED python backend in uma-env:
    conda install -p /projects/p30041/gbf4422/envs/uma-env -c conda-forge py-plumed
"""

import os

import numpy as np
import MDAnalysis as mda
from ase import Atoms, units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.constraints import FixAtoms, Hookean
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase.calculators.plumed import Plumed

from fairchem.core import pretrained_mlip, FAIRChemCalculator

# ── system ──────────────────────────────────────────────────────────────
STRUCTURE = "__STRUCTURE__"
CHARGE, MULT = __CHARGE__, __MULT__
MODEL, DEVICE = "uma-s-1p2", "cuda"          # in-process UMA on the GPU

# ── dynamics ────────────────────────────────────────────────────────────
TIMESTEP = 0.5 * units.fs
FRICTION = 1.0 / (100 * units.fs)
T_START, T_HEAT, T_RUN = 100.0, 300.0, 300.0

# ── reproducibility ─────────────────────────────────────────────────────
SEED = __SEED__                              # int for reproducible init velocities/thermostat noise, None for unseeded
RNG = np.random.default_rng(SEED)

# Production: 5 ps heat / 10 ps equil / 500 ps
# N_MTD is the *cumulative* mtd target; raise it and resubmit to extend the run.
N_HEAT, N_EQUIL, N_MTD = __NHEAT__, __NEQUIL__, __NMTD__
DUMP_STRIDE = 200
MTD_OFFSET = N_HEAT + N_EQUIL                # integrator step at which mtd starts

# ── restart ─────────────────────────────────────────────────────────────
RESTART_TRAJ = "mtd_restart.traj"            # latest mtd state (positions, momenta, mtd_step)
CHECKPOINT_EVERY = 1000                      # must be a multiple of PACE and DUMP_STRIDE

# ── energy units: defined in kcal/mol, converted to each consumer's unit at use ─
KCAL_TO_KJ = units.kcal / units.kJ           # kcal/mol -> kJ/mol  (= 4.184)
KCAL_TO_EV = units.kcal / units.mol          # kcal/mol -> eV      (ASE base energy)

# ── metadynamics (lengths in Angstrom; energies in kcal/mol, converted at use) ──
CV_ATOMS = (__CV0__, __CV1__)                               # 0-based (first CV distance); +1 for PLUMED
SIGMA, HEIGHT, PACE = __SIGMA__, __HEIGHT__, __PACE__       # width (A), hill height (kcal/mol)
BIASFACTOR = __BIASFACTOR__                                 # dimensionless
WALL_LO, KAPPA_LO = __WALL_LO__, __KAPPA_LO__               # lower wall, kcal/mol/A^2
WALL_HI, KAPPA_HI = __WALL_HI__, __KAPPA_HI__               # upper wall, kcal/mol/A^2
GRID_MIN, GRID_MAX, GRID_BIN = __GRID_MIN__, __GRID_MAX__, __GRID_BIN__ # 0.02 A/bin

# Optional SECOND collective variable. Two modes:
#   CV2_PT is None                     -> 1-D metadynamics over d1 only.
#   CV2_PT = (N, O, H) 0-based triple  -> 2-D over (d1, cv2) where cv2 is the
#       proton-transfer coordinate cv2 = d(N,H) - d(O,H) (N=donor, O=acceptor,
#       H=transferring proton): cv2<0 reactant (H on N), ~0 TS, cv2>0 product (H on O).
#       A d(N,O) upper wall (NO_WALL_HI/NO_KAPPA) keeps cv2 well-defined.
# In the 2-D mode each CV gets its own walls/sigma/grid and METAD biases the joint space.
CV2_PT = __CV2_PT__
SIGMA2 = __SIGMA2__                                          # width (A) for d2/cv2
WALL2_LO, KAPPA2_LO = __WALL2_LO__, __KAPPA2_LO__           # d2/cv2 lower wall, kcal/mol/A^2
WALL2_HI, KAPPA2_HI = __WALL2_HI__, __KAPPA2_HI__           # d2/cv2 upper wall, kcal/mol/A^2
GRID2_MIN, GRID2_MAX, GRID2_BIN = __GRID2_MIN__, __GRID2_MAX__, __GRID2_BIN__
NO_WALL_HI, NO_KAPPA = __NO_WALL_HI__, __NO_KAPPA__         # d(N,O) upper wall (cv2_pt mode)

# FROZEN ATOMS (0-based) — held fixed with FixAtoms; active region is the complement
fr_str = "__FR_STR__"
FROZEN_ATOMS = [int(i) for i in fr_str.split()]

# SOFT Cα RESTRAINTS (0-based) — kept mobile (part of the active region) but harmonically
# tethered to their initial positions, so the truncated backbone anchors stay near their
# MD positions without being rigidly pinned. Applied as an always-on ASE Hookean (rt=0).
car_str = "__CA_RESTRAINT_STR__"
CA_RESTRAINT_ATOMS = [int(i) for i in car_str.split()]
CA_RESTRAINT_K = __CA_RESTRAINT_K__      # kcal/mol/A^2

# ── water containment (flat-bottom spherical wall, ASE Hookean, every phase) ──
# A single containment sphere centered on the frozen-atom centroid: every water O
# is free inside the wall radius R and feels E = 0.5*k*(r - R)^2 outside it. R is
# the SAME for every water (it does not depend on where a given water sits).
# R is sized off a PERCENTILE of the initial water-O radii, not the single most
# eccentric water: one outlier water (e.g. one legitimately close to the reactive
# core but far from the whole-system centroid) would otherwise balloon R for every
# other water, leaving a huge, mostly-empty cavity that a water can drift through
# before ever feeling the wall (observed as a water "evaporating" out to vacuum).
# A few of the most eccentric initial waters are therefore expected to already sit
# outside R at step 0 (see the startup warning below) — the soft tether pulls them
# back in over the heat/equilibration phases rather than never engaging at all.
EXPECTED_WATERS = __EXPECTED_WATERS__
CONTAIN_PERCENTILE = 99  # percentile (0-100) of initial water-O radii that sets R (auto-size)
CONTAIN_MARGIN = 1.0    # A of head-room past that percentile value
CONTAIN_RADIUS = None   # A; if set, an absolute wall radius used for every water
TETHER_K = 10.0         # kcal/mol/A^2 (soft: 1 A past the wall = 5 kcal/mol ~8 kT)

# Any extra PLUMED lines (appended to the input). Empty by default.
EXTRA_PLUMED_LINES = []


# ── helpers ─────────────────────────────────────────────────────────────
def write_plain_xyz(path, atoms, idx=None, with_energy=False):
    """Append one XYZ frame. idx selects a subset of atoms (no Atoms slicing,
    so constraints are never copied). The energy comment is written only if
    already cached on the calculator — never triggers a fresh evaluation."""
    comment = ""
    if with_energy and atoms.calc is not None:
        try:
            if not atoms.calc.calculation_required(atoms, ["energy"]):
                comment = f"energy={atoms.get_potential_energy():.8f}"
        except Exception:
            comment = ""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    if idx is not None:
        symbols = [symbols[i] for i in idx]
        positions = positions[idx]
    with open(path, "a") as f:
        f.write(f"{len(symbols)}\n{comment}\n")
        for s, (x, y, z) in zip(symbols, positions):
            f.write(f"{s:<2s} {x:18.8f} {y:18.8f} {z:18.8f}\n")


def find_water_oxygens(structure=STRUCTURE):
    """0-based indices of each water oxygen, read straight from the topology:
    every residue named WAT contributes its oxygen. Selecting by residue name is
    robust to O-H geometry (a stretched or transferred proton never changes the
    count), unlike bond-counting. MDAnalysis reads the PDB in file order, so these
    indices line up 1:1 with the ASE Atoms built from the same file."""
    u = mda.Universe(structure)
    water_O = u.select_atoms("resname WAT and name O*")
    return water_O.indices.tolist()


def build_plumed_lines(include_metad=True):
    """PLUMED input: CV definitions + walls are always included. The METAD bias
    (and its grid-dependent PRINT) is appended only when include_metad=True — the
    walls-only input (include_metad=False) is attached before heating so the CV(s)
    are restrained from step 0, well before the METAD grid exists (see module
    docstring: an unrestrained heat/equil can drift the CV past GRID_MIN/MAX and
    crash PLUMED the moment the grid-based input attaches for phase 3)."""
    c1, c2 = CV_ATOMS[0] + 1, CV_ATOMS[1] + 1          # 0-based -> PLUMED 1-based
    klo_kj, khi_kj = KAPPA_LO * KCAL_TO_KJ, KAPPA_HI * KCAL_TO_KJ
    lines = [
        "UNITS LENGTH=A TIME=fs",   # energy left as PLUMED default kJ/mol
        f"d1: DISTANCE ATOMS={c1},{c2}",
        f"LOWER_WALLS ARG=d1 AT={WALL_LO} KAPPA={klo_kj:g} LABEL=lwd1",
        f"UPPER_WALLS ARG=d1 AT={WALL_HI} KAPPA={khi_kj:g} LABEL=uwd1",
    ]
    if CV2_PT is not None:
        # proton-transfer CV: cv2 = d(N,H) - d(O,H). N=donor, O=acceptor, H=proton.
        n, o, h = CV2_PT[0] + 1, CV2_PT[1] + 1, CV2_PT[2] + 1   # 0-based -> PLUMED 1-based
        k2lo_kj, k2hi_kj = KAPPA2_LO * KCAL_TO_KJ, KAPPA2_HI * KCAL_TO_KJ
        kno_kj = NO_KAPPA * KCAL_TO_KJ
        lines += [
            f"dNH: DISTANCE ATOMS={n},{h}",
            f"dOH: DISTANCE ATOMS={o},{h}",
            "cv2: COMBINE ARG=dNH,dOH COEFFICIENTS=1,-1 PERIODIC=NO",
            f"dNO: DISTANCE ATOMS={n},{o}",
            f"LOWER_WALLS ARG=cv2 AT={WALL2_LO} KAPPA={k2lo_kj:g} LABEL=lwcv2",
            f"UPPER_WALLS ARG=cv2 AT={WALL2_HI} KAPPA={k2hi_kj:g} LABEL=uwcv2",
            # d(N,O) upper wall keeps the proton near its donor/acceptor pair so that
            # cv2 stays a well-defined proton-transfer coordinate (the proton can't walk
            # off to another acceptor while the N and O drift apart).
            f"UPPER_WALLS ARG=dNO AT={NO_WALL_HI} KAPPA={kno_kj:g} LABEL=uwno",
        ]
    if not include_metad:
        return lines + EXTRA_PLUMED_LINES
    # kcal/mol -> kJ/mol for the PLUMED input (energy left at the PLUMED default)
    height_kj = HEIGHT * KCAL_TO_KJ
    if CV2_PT is not None:
        metad_arg, sigma_arg = "d1,cv2", f"{SIGMA},{SIGMA2}"
        grid_min = f"{GRID_MIN},{GRID2_MIN}"
        grid_max = f"{GRID_MAX},{GRID2_MAX}"
        grid_bin = f"{GRID_BIN},{GRID2_BIN}"
        # log CV1, CV2, d(N,O), the total metad bias and every wall's bias energy
        print_arg = ("d1,cv2,dNO,metad.bias,"
                     "lwd1.bias,uwd1.bias,lwcv2.bias,uwcv2.bias,uwno.bias")
    else:
        metad_arg, sigma_arg = "d1", f"{SIGMA}"
        grid_min, grid_max, grid_bin = f"{GRID_MIN}", f"{GRID_MAX}", f"{GRID_BIN}"
        print_arg = "d1,metad.bias,lwd1.bias,uwd1.bias"
    lines += [
        (f"metad: METAD ARG={metad_arg} SIGMA={sigma_arg} HEIGHT={height_kj:g} PACE={PACE} "
         f"BIASFACTOR={BIASFACTOR} TEMP={T_RUN} "
         f"GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin} "
         f"GRID_WSTRIDE=1000 GRID_WFILE=grid.dat CALC_RCT FILE=HILLS"),
        f"PRINT ARG={print_arg} STRIDE={DUMP_STRIDE} FILE=COLVAR",
        f"FLUSH STRIDE={DUMP_STRIDE}",
    ]
    return lines + EXTRA_PLUMED_LINES


def write_checkpoint(atoms, mtd_step):
    """Latest state only, written to a temp file and renamed so a kill mid-write
    never leaves a corrupt checkpoint."""
    snap = Atoms(numbers=atoms.get_atomic_numbers(),
                 positions=atoms.get_positions(),
                 momenta=atoms.get_momenta())
    snap.info["mtd_step"] = int(mtd_step)
    tmp = RESTART_TRAJ + ".tmp"
    with Trajectory(tmp, "w") as t:
        t.write(snap)
    os.replace(tmp, RESTART_TRAJ)


def read_checkpoint():
    if not os.path.exists(RESTART_TRAJ):
        return None
    with Trajectory(RESTART_TRAJ) as t:
        if len(t) == 0:
            return None
        snap = t[-1]
    if "mtd_step" not in snap.info:
        raise RuntimeError(
            f"{RESTART_TRAJ} has no mtd_step (written by an older version of "
            "this script, whose step count is unreliable). Move it and the mtd "
            "outputs (HILLS, COLVAR, *_mtd_trj.xyz) aside and start over.")
    return snap


def keep_rows_upto(path, limit):
    """Drop numeric rows whose first column exceeds `limit`; keep comments/headers."""
    if not os.path.exists(path):
        return
    kept = []
    with open(path) as f:
        for line in f:
            s = line.split()
            try:
                if s and not line.startswith("#") and float(s[0]) > limit:
                    continue
            except ValueError:
                pass
            kept.append(line)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(kept)
    os.replace(tmp, path)


def truncate_xyz(path, n_frames, n_per_frame):
    """Keep the first n_frames frames of a plain XYZ file."""
    if not os.path.exists(path):
        return
    n_lines = n_frames * (n_per_frame + 2)
    with open(path, "rb+") as f:
        for _ in range(n_lines):
            if not f.readline():
                return            # shorter than expected: leave it alone
        f.truncate(f.tell())


def main():
    assert CHECKPOINT_EVERY % PACE == 0 and CHECKPOINT_EVERY % DUMP_STRIDE == 0
    assert MTD_OFFSET % CHECKPOINT_EVERY == 0 and MTD_OFFSET % DUMP_STRIDE == 0

    atoms = read(STRUCTURE)
    n_atoms = len(atoms)
    # UMA omol is molecular: no PBC, no cell
    atoms.set_pbc(False)
    atoms.set_cell([0.0, 0.0, 0.0])
    atoms.info["charge"] = CHARGE
    atoms.info["spin"] = MULT
    # Drop PDB-derived per-atom arrays so trajectory .xyz stays clean
    for key in ("occupancy", "bfactor", "residuenames",
                "atomtypes", "residuenumbers"):
        if key in atoms.arrays:
            del atoms.arrays[key]

    if DEVICE == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE='cuda' but torch.cuda.is_available() is False — "
                "no GPU visible (check the SLURM --gres request)."
            )
        print(f"UMA on GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Active/frozen partition (0-based indices, no shift).
    frozen_idx = sorted(set(FROZEN_ATOMS))
    active_idx = sorted(set(range(n_atoms)) - set(frozen_idx))
    cv_all = set(CV_ATOMS)
    if CV2_PT is not None:
        cv_all |= set(CV2_PT)
    if cv_all & set(frozen_idx):
        raise RuntimeError("a CV atom is in FROZEN_ATOMS")
    print(f"active={len(active_idx)} frozen={len(frozen_idx)} atoms", flush=True)

    # ── water tethers: WAT oxygens read from the input PDB, so identical on restart
    water_O = find_water_oxygens(STRUCTURE)
    if len(water_O) != EXPECTED_WATERS:
        raise RuntimeError(f"found {len(water_O)} WAT oxygens, expected {EXPECTED_WATERS} "
                           "(check the WAT residue count in the input PDB)")
    pos0 = atoms.get_positions()
    center = pos0.mean(axis=0)              # whole-system centroid (robust to an empty frozen set)
    r0 = np.linalg.norm(pos0[water_O] - center, axis=1)
    if CONTAIN_RADIUS is not None:
        R = float(CONTAIN_RADIUS)
    else:
        R = float(np.percentile(r0, CONTAIN_PERCENTILE)) + CONTAIN_MARGIN
    print(f"containing {len(water_O)} waters within R={R:.2f} A of the system centroid "
          f"(uniform wall, k={TETHER_K} eV/A^2; outermost initial water at "
          f"{r0.max():.2f} A)", flush=True)
    outside0 = np.where(r0 > R)[0]
    if len(outside0) > 0:
        print(f"NOTE: {len(outside0)}/{len(water_O)} water(s) start beyond R "
              f"(max overshoot {r0[outside0].max() - R:.2f} A) — the flat-bottom "
              f"tether will pull them in during heating/equilibration, before any "
              f"metadynamics bias is applied:", flush=True)
        for i in outside0[np.argsort(-r0[outside0])]:
            print(f"  water_O index {water_O[i]}: r0={r0[i]:.2f} A "
                  f"(+{r0[i] - R:.2f} A over R)", flush=True)

    predictor = pretrained_mlip.get_predict_unit(MODEL, device=DEVICE, inference_settings="turbo")
    base_calc = FAIRChemCalculator(predictor, task_name="omol")
    atoms.calc = base_calc

    # Constraints BEFORE assigning velocities so FixAtoms zeros frozen momenta.
    cons = [FixAtoms(indices=frozen_idx)]
    cons += [Hookean(a1=int(o), a2=center.copy(), k=TETHER_K * KCAL_TO_EV, rt=R)
             for o in water_O]
    # Soft Cα restraints: harmonic tether to each Cα's initial position (rt=0 -> always on).
    cons += [Hookean(a1=int(c), a2=pos0[c].copy(), k=CA_RESTRAINT_K * KCAL_TO_EV, rt=0.0)
             for c in CA_RESTRAINT_ATOMS]
    if CA_RESTRAINT_ATOMS:
        print(f"soft-restraining {len(CA_RESTRAINT_ATOMS)} Cα to their initial positions "
              f"(k={CA_RESTRAINT_K} kcal/mol/A^2)", flush=True)
    atoms.set_constraint(cons)

    snap = read_checkpoint()
    resume = snap is not None
    if not resume:
        # fresh start: move dumps from any earlier attempt aside (they're appended to)
        for p in ("all_heat_trj.xyz", "all_equil_trj.xyz",
                  "all_mtd_trj.xyz", "walls.log"):
            if os.path.exists(p):
                os.replace(p, f"bck.{p}")

    # Langevin: per-atom coupling handles the frozen DOF; no COM removal, since
    # with atoms pinned the centre of mass is not a conserved quantity.
    dyn = Langevin(atoms, TIMESTEP, temperature_K=T_RUN, friction=FRICTION,
                   fixcm=False, rng=RNG)

    if resume:
        # Trim everything written after the checkpoint BEFORE opening the logs.
        done_mtd = int(snap.info["mtd_step"])
        dt_fs = TIMESTEP / units.fs
        step_ckpt = MTD_OFFSET + done_mtd
        # PLUMED re-deposits/re-prints at the checkpoint step itself, so drop it
        for fn in ("HILLS", "COLVAR"):
            keep_rows_upto(fn, done_mtd * dt_fs - 0.5 * dt_fs)
        keep_rows_upto("md.log", step_ckpt * dt_fs / 1000.0 + 0.25 * dt_fs / 1000.0)
        keep_rows_upto("walls.log", step_ckpt + 0.5)
        n_frames = done_mtd // DUMP_STRIDE
        truncate_xyz("all_mtd_trj.xyz", n_frames, n_atoms)

    # Instantaneous T (mobile DOF, accounts for FixAtoms) + energies -> md.log
    mdlog = MDLogger(dyn, atoms, "md.log", header=not resume, stress=False,
                     peratom=False, mode="a" if resume else "w")

    # Tether diagnostics -> walls.log
    wall_f = open("walls.log", "a" if resume else "w")
    if not resume:
        if CV2_PT is not None:
            cv2_hdr = "    cv2(A)    dNO(A)"
        else:
            cv2_hdr = ""
        wall_f.write(f"# step  d1(A){cv2_hdr}  n_beyond_wall  max_overshoot(A)  "
                     f"E_tether(kcal/mol)\n")
    c1, c2 = CV_ATOMS

    def log_walls():
        p = atoms.positions
        over = np.linalg.norm(p[water_O] - center, axis=1) - R
        act = over > 0
        e = 0.5 * TETHER_K * np.sum(over[act] ** 2)
        d1 = np.linalg.norm(p[c1] - p[c2])
        if CV2_PT is not None:
            n, o, h = CV2_PT
            cv2_val = np.linalg.norm(p[n] - p[h]) - np.linalg.norm(p[o] - p[h])
            d_no = np.linalg.norm(p[n] - p[o])
            cv2_col = f" {cv2_val:9.4f} {d_no:9.4f}"
        else:
            cv2_col = ""
        wall_f.write(f"{dyn.nsteps:10d} {d1:8.4f}{cv2_col} {int(act.sum()):4d} "
                     f"{max(over.max(), 0.0):8.3f} {e:12.6f}\n")
        wall_f.flush()

    def attach_phase(all_path):
        dyn.observers.clear()
        dyn.attach(lambda: write_plain_xyz(all_path, atoms, with_energy=True),
                   interval=DUMP_STRIDE)
        dyn.attach(mdlog, interval=DUMP_STRIDE)
        dyn.attach(log_walls, interval=DUMP_STRIDE)

    if not resume:
        MaxwellBoltzmannDistribution(atoms, temperature_K=T_START)
        dyn.set_temperature(temperature_K=T_START)

        # Walls only (no METAD/grid) from step 0: restrains the CV(s) through
        # heat/equil so they can't drift past the METAD grid before phase 3 attaches.
        atoms.calc = Plumed(calc=base_calc, input=build_plumed_lines(include_metad=False),
                            timestep=TIMESTEP, atoms=atoms,
                            kT=units.kB * T_RUN, log="plumed_walls.log", restart=False)

        # ── Phase 1: heat 100 -> 300 K ──────────────────────────────────
        def ramp():
            frac = min(dyn.nsteps / N_HEAT, 1.0)
            dyn.set_temperature(temperature_K=T_START + (T_HEAT - T_START) * frac)

        attach_phase("all_heat_trj.xyz")
        dyn.attach(ramp, interval=1)
        print("heating...", flush=True)
        dyn.run(N_HEAT)

        # ── Phase 2: equilibrate at 300 K ───────────────────────────────
        dyn.set_temperature(temperature_K=T_RUN)
        attach_phase("all_equil_trj.xyz")
        print("equilibrating...", flush=True)
        dyn.run(N_EQUIL)
        done_mtd = 0
    else:
        atoms.set_positions(snap.get_positions())
        atoms.set_momenta(snap.get_momenta())
        # Continue the integrator's step counter: keeps observer intervals aligned
        # and suppresses ASE's step-0 observer call (no duplicate frames/checkpoint).
        dyn.nsteps = MTD_OFFSET + done_mtd
        print(f"resuming metadynamics from step {done_mtd}/{N_MTD}", flush=True)

    # ── Phase 3: well-tempered metadynamics ─────────────────────────────
    # restart=True makes PLUMED read HILLS back and append; istep keeps the
    # PACE/output counters continuous.
    atoms.calc = Plumed(calc=base_calc, input=build_plumed_lines(),
                        timestep=TIMESTEP, atoms=atoms,
                        kT=units.kB * T_RUN, log="plumed.log", restart=resume)
    atoms.calc.istep = done_mtd
    attach_phase("all_mtd_trj.xyz")
    dyn.attach(lambda: write_checkpoint(atoms, dyn.nsteps - MTD_OFFSET),
               interval=CHECKPOINT_EVERY)
    print("metadynamics...", flush=True)
    dyn.run(max(0, N_MTD - done_mtd))
    wall_f.close()

    # Final snapshot back into a PDB (preserves residue/atom names)
    template = read(STRUCTURE)
    template.set_positions(atoms.get_positions())
    write("md_final.pdb", template)
    print("done", flush=True)


if __name__ == "__main__":
    main()
