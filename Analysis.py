"""
Analysis for diffusion simulation results.

Reads JSON result files produced by simulation.py and summarises results.

Outputs:
  - Summary table with SE
  - Per-result detailed printout
  - Three-panel plot with 95% CI error bars:
      Panel 1: fraction same vs Da  (primary — governs physics)
      Panel 2: fraction same vs η   (experimental variable, if eta in results)
      Panel 3: fraction same vs D   (simulation variable)

"""

import sys
import os
import json
import glob
import numpy as np


def load_result(path):
    with open(path) as f:
        return json.load(f)


def binomial_se(f, n):
    if n == 0 or np.isnan(f):
        return float("nan")
    return np.sqrt(f * (1 - f) / n)


def print_result(r):
    frac  = r.get("fraction_same", float("nan"))
    D     = r.get("D", 0)
    Da    = r.get("Da", 0)
    label = r.get("label", f"D={D}")
    N     = r.get("total_binding", 0)
    se    = binomial_se(frac, N)

    frac_str = f"{frac:.3f} ± {se:.3f}" if not np.isnan(se) else "n/a"
    ci_str   = (f"[{frac-1.96*se:.3f}, {frac+1.96*se:.3f}]"
                if not np.isnan(se) else "n/a")

    print(f"\n{'='*50}")
    print(f"Label           : {label}")
    print(f"D               : {D} um2/ms")
    print(f"Da (k*r²/D)     : {Da:.4f}")
    if "eta" in r:
        print(f"Viscosity η     : {r['eta']:.2e} Pa*s")
    print(f"k_bind          : {r.get('k_bind', '?')} ms-1")
    print(f"r_bind          : {r.get('r_bind', '?')} um")
    print(f"Cleavage events : {r.get('cleavage', 0)}")
    print(f"Same-cell       : {r.get('bind_same', 0)}")
    print(f"Diff-cell       : {r.get('bind_diff', 0)}")
    print(f"Total bindings  : {N}")
    print(f"Fraction same   : {frac_str}  (95% CI: {ci_str})")
    print(f"{'='*50}")


def print_table(results):
    results_sorted = sorted(results, key=lambda r: r.get("Da", 0))

    print(f"\n{'='*110}")
    print(f"{'Label':>16}  {'D':>10}  {'Da':>8}  "
          f"{'Cleavage':>9}  {'Same':>6}  {'Diff':>6}  {'Frac Same':>9}  {'SE':>6}")
    print(f"{'-'*110}")

    for r in results_sorted:
        if "error" in r:
            print(f"  {r.get('label','?')}  ERROR")
            continue
        frac     = r.get("fraction_same", float("nan"))
        N        = r.get("total_binding", 0)
        se       = binomial_se(frac, N)
        frac_str = f"{frac:.3f}" if not np.isnan(frac) else "  n/a"
        se_str   = f"{se:.3f}"   if not np.isnan(se)   else "  n/a"
        print(f"{r.get('label','?'):>16}  "
              f"{r.get('D', 0):>10.6f}  "
              f"{r.get('Da', 0):>8.4f}  "
              f"{r.get('cleavage', 0):>9}  "
              f"{r.get('bind_same', 0):>6}  "
              f"{r.get('bind_diff', 0):>6}  "
              f"{frac_str:>9}  "
              f"{se_str:>6}")

    print(f"{'='*110}")


def make_plot(valid):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ok = [r for r in valid
              if not np.isnan(r.get("fraction_same", float("nan")))
              and r.get("total_binding", 0) > 0]

        if not ok:
            print("No plottable results.")
            return

        fracs = np.array([r["fraction_same"] for r in ok])
        Ns    = np.array([r["total_binding"] for r in ok])
        ses   = np.array([binomial_se(f, n) for f, n in zip(fracs, Ns)])
        Das   = np.array([r.get("Da", 0) for r in ok])
        Ds    = np.array([r.get("D", 0) for r in ok])

        has_eta = all("eta" in r for r in ok)
        etas    = np.array([r["eta"] for r in ok]) if has_eta else None

        # Sort by Da
        order = np.argsort(Das)
        fracs, ses, Das, Ds, Ns = (fracs[order], ses[order],
                                    Das[order], Ds[order], Ns[order])
        if has_eta:
            etas = etas[order]

        n_panels = 3 if has_eta else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        if n_panels == 2:
            axes = list(axes)

        def annotate(ax, xs, fs, ns):
            for xi, fi, ni in zip(xs, fs, ns):
                ax.annotate(f"n={ni}", xy=(xi, fi), xytext=(0, 8),
                            textcoords="offset points", ha="center",
                            fontsize=7, color="gray")

        panel = 0

        # ── Panel 1: fraction same vs Da ──
        ax = axes[panel]; panel += 1
        ax.errorbar(Das, fracs, yerr=1.96 * ses,
                    fmt="o-", color="steelblue", lw=2, ms=7,
                    capsize=4, capthick=1.2, elinewidth=1.2,
                    label="fraction same (95% CI)")
        ax.axhline(0.5, color="gray", ls="--", alpha=0.6, label="random (0.5)")
        ax.axvline(1.0, color="orange", ls=":", alpha=0.8)
        ax.text(1.1, 0.08, "Da = 1", ha="left", fontsize=8, color="orange")
        ax.set_xscale("log")
        ax.set_xlabel("Damköhler number  Da = k·r²/D", fontsize=12)
        ax.set_ylabel("Fraction same-cell bindings", fontsize=12)
        ax.set_title("Same-cell fraction vs. Da", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        annotate(ax, Das, fracs, Ns)

        # ── Panel 2: fraction same vs eta (experimental variable) ──
        if has_eta:
            ax = axes[panel]; panel += 1
            ax.errorbar(etas, fracs, yerr=1.96 * ses,
                        fmt="o-", color="coral", lw=2, ms=7,
                        capsize=4, capthick=1.2, elinewidth=1.2,
                        label="fraction same (95% CI)")
            ax.axhline(0.5, color="gray", ls="--", alpha=0.6, label="random (0.5)")
            ax.axvline(1e-3, color="cornflowerblue", ls=":", alpha=0.7)
            ax.text(1.1e-3, 0.08, "water\n(1 mPa·s)",
                    ha="left", fontsize=8, color="cornflowerblue")
            ax.set_xscale("log")
            ax.set_xlabel("Fluid viscosity η (Pa·s)", fontsize=12)
            ax.set_ylabel("Fraction same-cell bindings", fontsize=12)
            ax.set_title("Same-cell fraction vs. viscosity", fontsize=13)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            annotate(ax, etas, fracs, Ns)

        # ── Panel 3: fraction same vs D ──
        ax = axes[panel]; panel += 1
        ax.errorbar(Ds, fracs, yerr=1.96 * ses,
                    fmt="o-", color="mediumseagreen", lw=2, ms=7,
                    capsize=4, capthick=1.2, elinewidth=1.2,
                    label="fraction same (95% CI)")
        ax.axhline(0.5, color="gray", ls="--", alpha=0.6, label="random (0.5)")
        ax.set_xscale("log")
        ax.set_xlabel("Diffusion constant D (µm²/ms)", fontsize=12)
        ax.set_ylabel("Fraction same-cell bindings", fontsize=12)
        ax.set_title("Same-cell fraction vs. D", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        annotate(ax, Ds, fracs, Ns)

        plt.tight_layout()
        plt.savefig("sweep_plot.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved to sweep_plot.png")

    except ImportError:
        print("matplotlib not available — skipping plot")


# =========================================================
# MAIN
# =========================================================

if len(sys.argv) > 1:
    paths = sys.argv[1:]
else:
    paths = sorted(glob.glob("*_result.json"))
    if not paths:
        print("No result JSON files found in current directory.")
        print("Usage: python analysis.py <result.json> [<result2.json> ...]")
        sys.exit(1)
    print(f"Found {len(paths)} result file(s)")

results = []
for path in paths:
    if not os.path.exists(path):
        print(f"WARNING: not found: {path}")
        continue
    try:
        raw = load_result(path)
        if isinstance(raw, list):
            results.extend(raw)
        else:
            results.append(raw)
    except Exception as e:
        print(f"WARNING: could not load {path}: {e}")

valid   = [r for r in results if "error" not in r]
errored = [r for r in results if "error" in r]

if errored:
    print(f"\n{len(errored)} errored result(s):")
    for r in errored:
        print(f"  {r.get('label','?')}: {r.get('error','')}")

if not valid:
    print("No valid results to display.")
    sys.exit(1)

if len(valid) == 1:
    print_result(valid[0])
else:
    print_table(valid)
    for r in sorted(valid, key=lambda r: r.get("Da", 0)):
        print_result(r)
    make_plot(valid)
