"""
Diffusion simulation — phenomenological Damköhler model.
"""

import sys
import numpy as np
import readdy
import os
import json

# =========================================================
# CELL AND MEDIA PARAMETERS
# =========================================================

R_CELL   = 5e-6          # m    cell radius (5 um)
T_KELVIN = 310.15        # K    physiological temperature (37C)
KB       = 1.380649e-23  # J/K

# =========================================================
# EFFECTIVE REACTION PARAMETERS
# Edit in run_sweep.py only; values are passed here via argv.
# =========================================================

K_CLEAVE_DEFAULT = 1e-2   # ms-1  effective cleavage rate
K_BIND_DEFAULT   = 1e-2   # ms-1  effective binding rate
R_CLEAVE_DEFAULT = 8.0    # um    cleavage capture radius
R_BIND_DEFAULT   = 10.0   # um    binding capture radius

# =========================================================
# DEFAULT D — gives Da ~ 1 at baseline viscosity
# Da = K_BIND * R_BIND^2 / D  =>  D = K_BIND * R_BIND^2 / Da
# =========================================================

D_DEFAULT = K_BIND_DEFAULT * R_BIND_DEFAULT**2 / 1.0   # Da = 1 by default

# =========================================================
# PARAMETERS — command line overrides defaults
# run_sweep.py always passes explicit values so defaults are
# only used when running simulation.py standalone.
# =========================================================

D_CELL   = float(sys.argv[1]) if len(sys.argv) > 1 else D_DEFAULT
OUTPUT   = sys.argv[2]        if len(sys.argv) > 2 else f"output_D{D_CELL:.6f}.h5"
N_STEPS  = int(sys.argv[3])   if len(sys.argv) > 3 else 1_000_000
K_CLEAVE = float(sys.argv[4]) if len(sys.argv) > 4 else K_CLEAVE_DEFAULT
K_BIND   = float(sys.argv[5]) if len(sys.argv) > 5 else K_BIND_DEFAULT
R_CLEAVE = float(sys.argv[6]) if len(sys.argv) > 6 else R_CLEAVE_DEFAULT
R_BIND   = float(sys.argv[7]) if len(sys.argv) > 7 else R_BIND_DEFAULT

DT = 1e-3   # ms per timestep

Da = K_BIND * R_BIND**2 / D_CELL

print(f"\nDiffusion simulation (phenomenological Da model):")
print(f"  D        = {D_CELL:.6f} um2/ms")
print(f"  Da       = {Da:.4f}  (k_bind={K_BIND} ms-1, r_bind={R_BIND} um)")
print(f"  k_cleave = {K_CLEAVE} ms-1  (effective)")
print(f"  k_bind   = {K_BIND} ms-1  (effective)")
print(f"  r_cleave = {R_CLEAVE} um")
print(f"  r_bind   = {R_BIND} um")
print(f"  N_steps  = {N_STEPS}  ({N_STEPS * DT:.1f} ms simulated)")
print(f"  Output   = {OUTPUT}")

# =========================================================
# GEOMETRY
# =========================================================

custom_units = {'length_unit': 'micrometer', 'time_unit': 'millisecond'}

Lx = 60.0    # um  chamber height (x = vertical axis)
Ly = 100.0   # um  lateral (periodic)
Lz = 100.0   # um  lateral (periodic)

N_CELLS     = 40
delta       = 15.0   # um  gap between hard wall and antibody surface
cell_radius = 5.0    # um


# Antibody surface at the bottom
x_surface = -Lx/2 + delta + 0.5   # = -14.5 um
SURFACE_X = x_surface + cell_radius

X_START_MIN = -9.5   # um  cells start at surface level
X_START_MAX = -9   # um  cells start within 3 um of surface

box_lower = -Lx/2 + 2.0   # = -28.0 um

print(f"\nGeometry:")
print(f"  Chamber        = {Lx} x {Ly} x {Lz} um")
print(f"  Antibody surface at x = {x_surface:.1f} um  (bottom)")
print(f"  Cells start at  x = {X_START_MIN:.1f} to {X_START_MAX:.1f} um  (top)")
print(f"  Fall distance   = {X_START_MIN - x_surface:.1f} um")

# =========================================================
# BUILD SYSTEM
# =========================================================

if os.path.exists(OUTPUT):
    os.remove(OUTPUT)

system = readdy.ReactionDiffusionSystem([Lx, Ly, Lz], unit_system=custom_units)
system.periodic_boundary_conditions = [False, True, True]

system.add_topology_species("Ab_masked", diffusion_constant=0.0)
for i in range(N_CELLS):
    system.add_topology_species(f"Cell_{i}",      diffusion_constant=D_CELL)
    system.add_topology_species(f"Enz_{i}",       diffusion_constant=D_CELL)
    system.add_topology_species(f"Ab_active_{i}", diffusion_constant=0.0)

system.topologies.add_type("AbMasked")
for i in range(N_CELLS):
    system.topologies.add_type(f"CellEnz_{i}")
    system.topologies.add_type(f"AbActive_{i}")
    system.topologies.add_type(f"CellEnzBound_{i}")

# Cell-Enz harmonic bond
for i in range(N_CELLS):
    system.topologies.configure_harmonic_bond(
        f"Cell_{i}", f"Enz_{i}", force_constant=100.0, length=5.0
    )
# Cell_j -- Ab_active_i bond for all (i,j) pairs
for i in range(N_CELLS):
    for j in range(N_CELLS):
        system.topologies.configure_harmonic_bond(
            f"Cell_{j}", f"Ab_active_{i}",
            force_constant=50.0, length=10.0
        )

# Box confinement — keeps cells within chamber
for i in range(N_CELLS):
    for ptype in [f"Cell_{i}", f"Enz_{i}"]:
        system.potentials.add_box(
            particle_type=ptype, force_constant=100.0,
            origin=[box_lower, -Ly / 2, -Lz / 2],
            extent=[Lx / 2 - box_lower, Ly, Lz]
        )

# Cell-cell steric repulsion
for i in range(N_CELLS):
    for j in range(i, N_CELLS):
        system.potentials.add_harmonic_repulsion(
            f"Cell_{i}", f"Cell_{j}",
            force_constant=10.0, interaction_distance=10.0
        )

# Surface confinement — potential keeping cells near the
# antibody surface so reactions can fire. Acts only in x so cells
# remain free to diffuse laterally across the surface.
for i in range(N_CELLS):
    system.potentials.add_box(
        particle_type=f"Cell_{i}",
        force_constant=10.0,
        origin=[x_surface - cell_radius, -Ly/2, -Lz/2],
        extent=[cell_radius * 5, Ly, Lz]
    )

# Cleavage and binding reactions
for i in range(N_CELLS):
    system.topologies.add_spatial_reaction(
        f"cleavage_{i}: CellEnz_{i}(Enz_{i}) + AbMasked(Ab_masked) "
        f"-> CellEnz_{i}(Enz_{i}) + AbActive_{i}(Ab_active_{i})",
        rate=K_CLEAVE, radius=R_CLEAVE
    )
    system.topologies.add_spatial_reaction(
        f"bind_same_{i}: CellEnz_{i}(Cell_{i}) + AbActive_{i}(Ab_active_{i}) "
        f"-> CellEnzBound_{i}(Cell_{i}--Ab_active_{i})",
        rate=K_BIND, radius=R_BIND
    )
    for j in range(N_CELLS):
        if j == i:
            continue
        system.topologies.add_spatial_reaction(
            f"bind_diff_{j}_on_{i}: CellEnz_{j}(Cell_{j}) + AbActive_{i}(Ab_active_{i}) "
            f"-> CellEnzBound_{j}(Cell_{j}--Ab_active_{i})",
            rate=K_BIND, radius=R_BIND
        )

# =========================================================
# INITIAL CONDITIONS
# =========================================================

sim = system.simulation(kernel="CPU")
sim.output_file      = OUTPUT
sim.reaction_handler = "Gillespie"

# Antibody grid on bottom surface
Ab_spacing = 3.0
y_vals = np.linspace(-Ly/2, Ly/2, int(Ly/Ab_spacing), endpoint=False)
z_vals = np.linspace(-Lz/2, Lz/2, int(Lz/Ab_spacing), endpoint=False)
for y in y_vals:
    for z in z_vals:
        sim.add_topology("AbMasked", ["Ab_masked"],
                         np.array([[x_surface, y, z]]))

# Cells placed uniformly near the top — diffuse down to surface
enz_offset = np.array([-5.0, 0.0, 0.0])   # Enz sits 5 um below Cell in x
for i in range(N_CELLS):
    cp = np.array([
        np.random.uniform(X_START_MIN, X_START_MAX),
        np.random.uniform(-Ly/2 + cell_radius, Ly/2 - cell_radius),
        np.random.uniform(-Lz/2 + cell_radius, Lz/2 - cell_radius)
    ])
    top = sim.add_topology(
        f"CellEnz_{i}",
        [f"Cell_{i}", f"Enz_{i}"],
        np.array([cp, cp + enz_offset])
    )
    top.get_graph().add_edge(0, 1)

# =========================================================
# OBSERVABLES AND RUN
# =========================================================

sim.observe.reaction_counts(stride=100)

sim.run(n_steps=N_STEPS, timestep=DT)

# =========================================================
# READ RESULTS
# =========================================================

traj = readdy.Trajectory(OUTPUT)

def decode(x):
    return x.decode() if isinstance(x, bytes) else x

try:
    _, rc = traj.read_observable_reaction_counts()
    counts = {"cleavage": 0, "bind_same": 0, "bind_diff": 0}
    for cat, rd in rc.items():
        if not isinstance(rd, dict):
            continue
        for rname, arr in rd.items():
            rn = decode(rname)
            total = int(np.sum(arr))
            if rn.startswith("cleavage"):
                counts["cleavage"] += total
            elif rn.startswith("bind_same"):
                counts["bind_same"] += total
            elif rn.startswith("bind_diff"):
                counts["bind_diff"] += total
except Exception as e:
    print(f"WARNING: could not read reaction counts: {e}")
    counts = {"cleavage": 0, "bind_same": 0, "bind_diff": 0}

total_binding = counts["bind_same"] + counts["bind_diff"]
fraction_same = (counts["bind_same"] / total_binding
                 if total_binding > 0 else float("nan"))

print(f"\n{'='*50}")
print(f"D               : {D_CELL:.6f} um2/ms")
print(f"Da              : {Da:.4f}")
print(f"Cleavage        : {counts['cleavage']}")
print(f"Same-cell       : {counts['bind_same']}")
print(f"Diff-cell       : {counts['bind_diff']}")
print(f"Total bindings  : {total_binding}")
print(f"Fraction same   : {fraction_same:.3f}")
print(f"{'='*50}")

result = {
    "D":             D_CELL,
    "Da":            Da,
    "k_bind":        K_BIND,
    "k_cleave":      K_CLEAVE,
    "r_bind":        R_BIND,
    "r_cleave":      R_CLEAVE,
    "cleavage":      counts["cleavage"],
    "bind_same":     counts["bind_same"],
    "bind_diff":     counts["bind_diff"],
    "total_binding": total_binding,
    "fraction_same": fraction_same,
}
result_file = OUTPUT.replace(".h5", "_result.json")
with open(result_file, "w") as f:
    json.dump(result, f, indent=2)
print(f"Result saved to {result_file}")


if os.path.exists(OUTPUT):
    os.remove(OUTPUT)
    print(f"Trajectory file deleted (result JSON retained)")


import gc
del traj
del sim
del system
gc.collect()




























































