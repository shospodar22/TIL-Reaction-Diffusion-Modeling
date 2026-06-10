"""
Viscosity sweep — phenomenological Damköhler model.

The sweep varies fluid viscosity η across the experimental range and maps
each η to a Damköhler number Da via the Stokes-Einstein relation:

    D_phys(η) = kBT / (6π η r_cell)
    Da(η)     = k_bind * r_bind^2 / D_phys(η)
              = k_bind * r_bind^2 * 6π η r_cell / (kBT)

Since D_phys is too small for tractable ReaDDy simulation, a simulation
diffusion constant D_sim is derived that gives the same Da:

    D_sim = k_bind * r_bind^2 / Da(η)   [= D_phys in magnitude, but see note]

The reaction parameters k_bind, k_cleave, r_bind, r_cleave are effective
coarse-grained parameters chosen so that Da spans ~0.1 to 100 across the
experimental viscosity range. They are not molecular rate constants. Once
experimental f_same data are available, k_bind can be fitted empirically.

Sweep modes
-----------
  "eta"  : sweep viscosity, derive Da and D_sim from η
  "Da"   : sweep Da directly — useful for mapping the full transition curve

Usage
-----
    python run_sweep.py
"""

import subprocess
import sys
import os
import json
import numpy as np

# =========================================================
# PHYSICAL PARAMETERS
# =========================================================

R_CELL   = 5e-6          # m    cell radius
T_KELVIN = 310.15        # K    physiological temperature (37C)
KB       = 1.380649e-23  # J/K

# =========================================================
# EFFECTIVE REACTION PARAMETERS
# =========================================================

K_CLEAVE = 1   # ms-1  effective cleavage rate  (scaled x100 vs original 1e-3)
K_BIND   = 1e-2   # ms-1  effective binding rate   (scaled x100 vs original 1e-3)
R_CLEAVE = 8.0    # um    cleavage capture radius
R_BIND   = 10.0   # um    binding capture radius

# =========================================================
# SWEEP MODE
# =========================================================

SWEEP_MODE = "eta"   # "eta" or "Da"

# =========================================================
# VISCOSITY RANGE (used when SWEEP_MODE = "eta")
# =========================================================
# Reference values (Pa*s):
#   Water at 20C              : ~1.0e-3
#   Cell culture media (DMEM) : ~9.6e-4
#   10% glycerol              : ~1.3e-3
#   50% glycerol              : ~6e-3
#   0.5% methylcellulose      : ~1e-2 to 1e-1

ETA_MIN = 5e-4
ETA_MAX = 1e-1
N_ETA   = 7
# To use explicit values replace the line below:
# ETA_VALUES = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2, 1e-1]
ETA_VALUES = np.logspace(np.log10(ETA_MIN), np.log10(ETA_MAX), N_ETA).tolist()

# =========================================================
# Da RANGE (used when SWEEP_MODE = "Da")
# =========================================================

Da_MIN = 0.01
Da_MAX = 1000.0
N_Da   = 8

# =========================================================
# SIMULATION PARAMETERS
# =========================================================

N_STEPS = 1_000_000

# =========================================================
# PHYSICS FUNCTIONS
# =========================================================

def compute_D_phys(eta):
    """Physical Stokes-Einstein diffusion (um2/ms)."""
    return (KB * T_KELVIN) / (6 * np.pi * eta * R_CELL) * 1e9

def compute_Da(eta):
    """Physical Damköhler number at viscosity eta."""
    return K_BIND * R_BIND**2 / compute_D_phys(eta)

def compute_D_sim(eta):
    """
    Simulation D value that reproduces the physical Da at viscosity eta.
    D_sim = k_bind * r_bind^2 / Da(eta)
    """
    Da = compute_Da(eta)
    return K_BIND * R_BIND**2 / Da

def binomial_se(f, n):
    if n == 0 or np.isnan(f):
        return float("nan")
    return np.sqrt(f * (1 - f) / n)

# =========================================================
# PRINT SUMMARY
# =========================================================

print(f"\nPhenomenological Damköhler sweep")
print(f"  R_cell   = {R_CELL*1e6:.1f} um")
print(f"  T        = {T_KELVIN} K")
print(f"  k_bind   = {K_BIND} ms-1  (effective)")
print(f"  k_cleave = {K_CLEAVE} ms-1  (effective)")
print(f"  r_bind   = {R_BIND} um")
print(f"  r_cleave = {R_CLEAVE} um")

# =========================================================
# BUILD SWEEP PARAMETERS
# =========================================================

D_MAX_SIM = 10.0   # um2/ms  target D_sim at lowest eta (highest diffusivity)

if SWEEP_MODE == "eta":
    # Step 1: compute physical D values and rescale so max D_sim = D_MAX_SIM
    D_phys_values = [compute_D_phys(eta) for eta in ETA_VALUES]
    D_max         = max(D_phys_values)
    SCALE         = D_MAX_SIM / D_max   # preserves Da: D_sim/D_phys = const for all eta

    print(f"\nRescale factor: {SCALE:.3e}  (max D_sim = {D_MAX_SIM} um2/ms)")
    print(f"\n{'Eta (Pa*s)':>12}  {'D_phys':>14}  {'D_sim':>10}  {'Da (sim)':>10}  (D_max_sim={D_MAX_SIM})")
    print("-" * 55)

    sweep_params = []
    for eta, D_phys in zip(ETA_VALUES, D_phys_values):
        D_sim = D_phys * SCALE                  # rescaled simulation D
        Da    = K_BIND * R_BIND**2 / D_sim      # Da computed from D_sim — correct
        sweep_params.append({
            "label":  f"eta={eta:.2e}",
            "eta":    eta,
            "D_phys": D_phys,
            "D_sim":  D_sim,
            "Da":     Da,
        })
        print(f"{eta:>12.2e}  {D_phys:>14.4e}  {D_sim:>10.4f}  {Da:>10.4f}")

elif SWEEP_MODE == "Da":
    Da_VALUES = np.logspace(np.log10(Da_MIN), np.log10(Da_MAX), N_Da).tolist()

    print(f"\n{'Da':>10}  {'D_sim':>12}")
    print("-" * 26)

    sweep_params = []
    for Da in Da_VALUES:
        D_sim = K_BIND * R_BIND**2 / Da
        sweep_params.append({
            "label": f"Da={Da:.4f}",
            "Da":    Da,
            "D_sim": D_sim,
        })
        print(f"{Da:>10.4f}  {D_sim:>12.6f}")

else:
    raise ValueError(f"Unknown SWEEP_MODE: {SWEEP_MODE}")

print(f"\nTotal runs: {len(sweep_params)}")

# =========================================================
# RUN EACH CONDITION
# =========================================================

PYTHON = sys.executable
SIM    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DahmkohlerModel2.py")

RESULTS = []

for p in sweep_params:
    D_sim = p["D_sim"]
    label = p["label"]
    Da    = p["Da"]
    safe  = label.replace("=", "").replace(".", "p").replace("-", "m")
    output = f"output_{safe}.h5"
    rf     = output.replace(".h5", "_result.json")

    print(f"\nRunning {label}  (D={D_sim:.6f} um2/ms  Da={Da:.4f})")

    ret = subprocess.run(
        [PYTHON, SIM,
         str(D_sim), output, str(N_STEPS),
         str(K_CLEAVE), str(K_BIND), str(R_CLEAVE), str(R_BIND)],
        capture_output=False
    )

    if ret.returncode != 0:
        print(f"ERROR for {label}")
        RESULTS.append({**p, "error": True})
        continue

    if os.path.exists(rf):
        with open(rf) as f:
            r = json.load(f)
        r["label"]   = label
        r["Da"]      = Da
        if "eta" in p:
            r["eta"] = p["eta"]
        RESULTS.append(r)
    else:
        RESULTS.append({**p, "error": "result file not found"})

# =========================================================
# SUMMARY TABLE
# =========================================================

print(f"\n{'='*110}")
print(f"{'Label':>16}  {'eta (Pa*s)':>12}  {'D_sim':>10}  {'Da':>8}  "
      f"{'Cleavage':>9}  {'Same':>6}  {'Diff':>6}  {'Frac Same':>9}  {'SE':>6}")
print(f"{'-'*110}")

for r in RESULTS:
    if "error" in r:
        print(f"  {r.get('label','?')}  ERROR")
        continue
    frac     = r.get("fraction_same", float("nan"))
    N        = r.get("total_binding", 0)
    se       = binomial_se(frac, N)
    frac_str = f"{frac:.3f}" if not np.isnan(frac) else "  n/a"
    se_str   = f"{se:.3f}"   if not np.isnan(se)   else "  n/a"
    eta_str  = f"{r['eta']:.2e}" if "eta" in r else "  n/a"
    print(f"{r.get('label','?'):>16}  "
          f"{eta_str:>12}  "
          f"{r.get('D', 0):>10.6f}  "
          f"{r.get('Da', 0):>8.4f}  "
          f"{r.get('cleavage', 0):>9}  "
          f"{r.get('bind_same', 0):>6}  "
          f"{r.get('bind_diff', 0):>6}  "
          f"{frac_str:>9}  "
          f"{se_str:>6}")

print(f"{'='*110}")

with open("sweep_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
print("\nResults saved to sweep_results.json")
