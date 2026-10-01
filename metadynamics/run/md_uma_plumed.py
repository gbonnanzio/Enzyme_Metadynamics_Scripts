#!/projects/p30041/gbf4422/envs/uma-env/bin/python3
"""
Heating -> equilibration -> well-tempered metadynamics with ASE + PLUMED,
driving the UMA (FAIRChem) potential in-process on the GPU.

  1. Heat the active region 100 -> 300 K (Langevin thermostat).
  2. Equilibrate at 300 K.
  3. Well-tempered metadynamics on the forming C-C bond (C2 of ThDP <-> donor
     C1, CV_ATOMS below), with lower/upper walls, biasing handled by PLUMED.

Alpha carbons of terminal protein residues are frozen to retain protein structure.

Every water O is held with a loose Hookean restraint to the centroid of the frozen atoms
so waters cannot evaporate from the non-periodic cluster. The wall radius
is uniform — the same for every water, independent of where it sits. walls.log
records how often the wall engages (almost never)

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
from ase.neighborlist import natural_cutoffs, NeighborList
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase.calculators.plumed import Plumed

from fairchem.core import pretrained_mlip, FAIRChemCalculator

# ── system ──────────────────────────────────────────────────────────────
STRUCTURE = "protonated_cluster.pdb"
CHARGE, MULT = 0, 1
MODEL, DEVICE = "uma-s-1p2", "cuda"          # in-process UMA on the GPU

# ── dynamics ────────────────────────────────────────────────────────────
TIMESTEP = 0.5 * units.fs
FRICTION = 1.0 / (100 * units.fs)            
T_START, T_HEAT, T_RUN = 100.0, 300.0, 300.0 # temperature ramp

# Production: 5 ps heat / 10 ps equil / 500 ps metadynamics
N_HEAT, N_EQUIL, N_MTD = 10000, 20000, 1000000
# Fast test values were: N_HEAT, N_EQUIL, N_MTD = 2000, 2000, 20000
DUMP_STRIDE = 100
MTD_OFFSET = N_HEAT + N_EQUIL                # integrator step at which mtd starts

# ── restart ─────────────────────────────────────────────────────────────
RESTART_TRAJ = "mtd_restart.traj"            # latest mtd state (positions, momenta, mtd_step)
CHECKPOINT_EVERY = 1000                      # must be a multiple of PACE and DUMP_STRIDE

# ── energy units: defined in kcal/mol, converted to each consumer's unit at use ─
KCAL_TO_KJ = units.kcal / units.kJ           # kcal/mol -> kJ/mol  (= 4.184)
KCAL_TO_EV = units.kcal / units.mol          # kcal/mol -> eV      (ASE base energy)

# ── metadynamics (lengths in Angstrom; energies in kcal/mol, converted at use) ──
CV_ATOMS = (444, 679)                        # 0-based (C2 ThDP, C1 donor); +1 for PLUMED
SIGMA, HEIGHT, PACE = 0.1, 0.5, 50           # width (A), hill height (kcal/mol)
BIASFACTOR = (300.0 + 8000.0) / 300.0        # ~27.7 from ORCA WellTempered 8000 K (dimensionless)
WALL_LO, KAPPA_LO = 1.2, 50.0                # lower wall AT 1.2 A; KAPPA kcal/mol/A^2
WALL_HI, KAPPA_HI = 6.5, 100.0               # upper wall; KAPPA kcal/mol/A^2 
GRID_MIN, GRID_MAX, GRID_BIN = 0.8, 8.5, 360 # ~0.021 A/bin; walls (1.2/6.5) sit well inside the grid

# FROZEN ATOMS (0-based) — held fixed with FixAtoms; active region is the complement
fr_str = "0 12 32 109 130 152 200 209 259 266 285 394 411 428 485 528 549 567 581 593 608 615 627 649 664"
FROZEN_ATOMS = [int(i) for i in fr_str.split()]

# ── water containment (flat-bottom spherical wall, ASE Hookean, every phase) ──
# every water O is free inside the wall radius R and feels E = 0.5*k*(r - R)^2 outside it. 
EXPECTED_WATERS = 128
CONTAIN_MARGIN = 5.0    # A of head-room past the outermost initial water O (auto-size)
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


def build_plumed_lines():
    c1, c2 = CV_ATOMS[0] + 1, CV_ATOMS[1] + 1          # 0-based -> PLUMED 1-based
    # kcal/mol -> kJ/mol for the PLUMED input (energy left at the PLUMED default)
    height_kj = HEIGHT * KCAL_TO_KJ
    klo_kj, khi_kj = KAPPA_LO * KCAL_TO_KJ, KAPPA_HI * KCAL_TO_KJ
    lines = [
        "UNITS LENGTH=A TIME=fs",   # energy left as PLUMED default kJ/mol
        f"d1: DISTANCE ATOMS={c1},{c2}",
        f"LOWER_WALLS ARG=d1 AT={WALL_LO} KAPPA={klo_kj:g}",
        f"UPPER_WALLS ARG=d1 AT={WALL_HI} KAPPA={khi_kj:g}",
        (f"metad: METAD ARG=d1 SIGMA={SIGMA} HEIGHT={height_kj:g} PACE={PACE} "
         f"BIASFACTOR={BIASFACTOR} TEMP={T_RUN} "
         f"GRID_MIN={GRID_MIN} GRID_MAX={GRID_MAX} GRID_BIN={GRID_BIN} "
         f"GRID_WSTRIDE=1000 GRID_WFILE=grid.dat CALC_RCT FILE=HILLS"),
        f"PRINT ARG=d1,metad.bias STRIDE={DUMP_STRIDE} FILE=COLVAR",
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
    if set(CV_ATOMS) & set(frozen_idx):
        raise RuntimeError("a CV atom is in FROZEN_ATOMS")
    print(f"active={len(active_idx)} frozen={len(frozen_idx)} atoms", flush=True)

    # ── water tethers: WAT oxygens read from the input PDB, so identical on restart
    water_O = find_water_oxygens(STRUCTURE)
    if len(water_O) != EXPECTED_WATERS:
        raise RuntimeError(f"found {len(water_O)} WAT oxygens, expected {EXPECTED_WATERS} "
                           "(check the WAT residue count in the input PDB)")
    pos0 = atoms.get_positions()
    center = pos0[frozen_idx].mean(axis=0)
    r0 = np.linalg.norm(pos0[water_O] - center, axis=1)
    R = float(CONTAIN_RADIUS) if CONTAIN_RADIUS is not None else float(r0.max()) + CONTAIN_MARGIN
    if r0.max() > R:
        raise RuntimeError(f"CONTAIN_RADIUS={R} A sits inside the outermost initial "
                           f"water O at {r0.max():.2f} A; raise it or use auto-size.")
    print(f"containing {len(water_O)} waters within R={R:.2f} A of the frozen centroid "
          f"(uniform wall, k={TETHER_K} eV/A^2; outermost initial water at "
          f"{r0.max():.2f} A)", flush=True)

    predictor = pretrained_mlip.get_predict_unit(MODEL, device=DEVICE, inference_settings="turbo")    
    base_calc = FAIRChemCalculator(predictor, task_name="omol")
    atoms.calc = base_calc

    # Constraints BEFORE assigning velocities so FixAtoms zeros frozen momenta.
    cons = [FixAtoms(indices=frozen_idx)]
    cons += [Hookean(a1=int(o), a2=center.copy(), k=TETHER_K * KCAL_TO_EV, rt=R)
             for o in water_O]
    atoms.set_constraint(cons)

    snap = read_checkpoint()
    resume = snap is not None
    if not resume:
        # fresh start: move dumps from any earlier attempt aside (they're appended to)
        for p in ("all_heat_trj.xyz", "active_heat_trj.xyz",
                  "all_equil_trj.xyz", "active_equil_trj.xyz",
                  "all_mtd_trj.xyz", "active_mtd_trj.xyz", "walls.log"):
            if os.path.exists(p):
                os.replace(p, f"bck.{p}")

    # Langevin: per-atom coupling handles the frozen DOF; no COM removal, since
    # with atoms pinned the centre of mass is not a conserved quantity.
    dyn = Langevin(atoms, TIMESTEP, temperature_K=T_RUN, friction=FRICTION,
                   fixcm=False)

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
        truncate_xyz("active_mtd_trj.xyz", n_frames, len(active_idx))

    # Instantaneous T (mobile DOF, accounts for FixAtoms) + energies -> md.log
    mdlog = MDLogger(dyn, atoms, "md.log", header=not resume, stress=False,
                     peratom=False, mode="a" if resume else "w")

    # Tether diagnostics -> walls.log
    wall_f = open("walls.log", "a" if resume else "w")
    if not resume:
        wall_f.write("# step  d1(A)  n_beyond_wall  max_overshoot(A)  E_tether(kcal/mol)\n")
    c1, c2 = CV_ATOMS

    def log_walls():
        p = atoms.positions
        over = np.linalg.norm(p[water_O] - center, axis=1) - R
        act = over > 0
        e = 0.5 * TETHER_K * np.sum(over[act] ** 2)
        d1 = np.linalg.norm(p[c1] - p[c2])
        wall_f.write(f"{dyn.nsteps:10d} {d1:8.4f} {int(act.sum()):4d} "
                     f"{max(over.max(), 0.0):8.3f} {e:12.6f}\n")
        wall_f.flush()

    def attach_phase(all_path, active_path):
        dyn.observers.clear()
        dyn.attach(lambda: write_plain_xyz(all_path, atoms, with_energy=True),
                   interval=DUMP_STRIDE)
        dyn.attach(lambda: write_plain_xyz(active_path, atoms, idx=active_idx),
                   interval=DUMP_STRIDE)
        dyn.attach(mdlog, interval=DUMP_STRIDE)
        dyn.attach(log_walls, interval=DUMP_STRIDE)

    if not resume:
        MaxwellBoltzmannDistribution(atoms, temperature_K=T_START)
        dyn.set_temperature(temperature_K=T_START)

        # ── Phase 1: heat 100 -> 300 K ──────────────────────────────────
        def ramp():
            frac = min(dyn.nsteps / N_HEAT, 1.0)
            dyn.set_temperature(temperature_K=T_START + (T_HEAT - T_START) * frac)

        attach_phase("all_heat_trj.xyz", "active_heat_trj.xyz")
        dyn.attach(ramp, interval=1)
        print("heating...", flush=True)
        dyn.run(N_HEAT)

        # ── Phase 2: equilibrate at 300 K ───────────────────────────────
        dyn.set_temperature(temperature_K=T_RUN)
        attach_phase("all_equil_trj.xyz", "active_equil_trj.xyz")
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
    attach_phase("all_mtd_trj.xyz", "active_mtd_trj.xyz")
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