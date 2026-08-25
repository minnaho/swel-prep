#!/usr/bin/env python3
"""
Concurrent plotting framework for the SWEL mc60 project.

Runs any subset of the plotting/calculation scripts in this directory in parallel
using Python's ProcessPoolExecutor. Each script is executed as a subprocess with
CWD set to this directory, so all relative paths in the scripts resolve correctly.

MPLBACKEND is forced to 'Agg' so scripts run headlessly without a display.

Usage:
    python run_plots.py                          # run everything (4 workers)
    python run_plots.py --list                   # list all scripts + categories
    python run_plots.py --dry-run                # preview without executing
    python run_plots.py --categories pdf         # run only PDF-style plots
    python run_plots.py --scripts plot_ke plot_energy_cascade
    python run_plots.py --workers 8 --timeout 3600
    python run_plots.py --exclude calc_ke_surf calc_ke_bot   # skip heavy jobs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Script registry
# Each entry: script_stem -> (category, description)
# Categories: spectral | pdf | map | snapshot | util
# ---------------------------------------------------------------------------

SCRIPTS: Dict[str, Tuple[str, str]] = {
    # --- Spectral / heavy computation ---
    "calc_ke_surf": (
        "spectral",
        "Compute KE spectra from surface velocity; writes ke_spectra_comparison.npz",
    ),
    "calc_ke_surf_20190421_20190423": (
        "spectral",
        "Compute KE spectra from surface velocity, 2019-04-21 00:00-04-23 00:00 window only",
    ),
    "calc_ke_bot": (
        "spectral",
        "Compute KE spectra from bottom velocity",
    ),
    "calc_ke_10m": (
        "spectral",
        "Compute KE spectra from 10m-zsliced velocity; writes ke_spectra_10m_comparison.npz",
    ),
    "calc_ke_50m": (
        "spectral",
        "Compute KE spectra from 50m-zsliced velocity; writes ke_spectra_50m_comparison.npz",
    ),
    "save_vort": (
        "spectral",
        "Calculate vorticity field and save to NetCDF",
    ),

    # --- PDF / spectral plots (read pre-computed data, fast) ---
    "plot_ke": (
        "pdf",
        "Rotary, folded, and total KE spectra from ke_spectra_comparison.npz",
    ),
    "plot_energy_cascade": (
        "pdf",
        "Total-domain KE cascade with log-log slope fits",
    ),
    "plot_energy_cascade_20190421_20190423": (
        "pdf",
        "Total-domain KE cascade with slope fits, 2019-04-21 00:00-04-23 00:00 window only",
    ),
    "plot_energy_cascade_coastal": (
        "pdf",
        "Coastal-mask KE cascade with slope fits",
    ),
    "plot_energy_cascade_10m": (
        "pdf",
        "Total-domain KE cascade at 10m depth from ke_spectra_10m_comparison.npz",
    ),
    "plot_energy_cascade_50m": (
        "pdf",
        "Total-domain KE cascade at 50m depth from ke_spectra_50m_comparison.npz",
    ),
    "plot_pdf_biomass": (
        "pdf",
        "Phytoplankton biomass probability density function",
    ),
    "plot_pdf_pv": (
        "pdf",
        "Potential vorticity PDF (full domain + coastal)",
    ),
    "plot_pdf_vort": (
        "pdf",
        "Vorticity PDF from postprocessing NPZ files",
    ),
    "plot_pdf_vort_100m": (
        "pdf",
        "Vorticity PDF from postprocessing NPZ files, h <= 100 m only",
    ),
    "plot_pdf_vort_surf": (
        "pdf",
        "Surface vorticity PDF (full domain + coastal)",
    ),
    "plot_profile_zslice": (
        "pdf",
        "Depth profiles of all zsliced vars; full domain vs coastal mask, all 4 scenarios overlaid",
    ),
    "plot_profile_zslice_100m": (
        "pdf",
        "Depth profiles of all zsliced vars, h<=100m domain only, all 6 scenarios + diff from notidesnowec",
    ),
    "plot_profile_zslice_100m_coastal": (
        "pdf",
        "Depth profiles of all zsliced vars, h<=100m / coastal (10km) / diff panels, all 6 scenarios",
    ),
    "plot_profile_zslice_shelf_offshore": (
        "pdf",
        "Depth profiles of all zsliced vars, 4 panels: h<=100m / diff / h>100m / diff, all 6 scenarios",
    ),
    "plot_profile_zslice_shelf_offshore_4": (
        "pdf",
        "Depth profiles of all zsliced vars, 4 panels: h<=100m / diff / h>100m / diff, notidesnowec/tidesnowec/notidesampwec/tidesampwec only",
    ),
    "plot_profile_drhodz_100m": (
        "pdf",
        "d(rho)/dz depth profile, h<=100m domain mean, all scenarios + diff from notidesnowec",
    ),
    "plot_profile_drhodz_coastal": (
        "pdf",
        "d(rho)/dz depth profile, coastal (10km) domain mean, all scenarios + diff from notidesnowec",
    ),

    # --- Map / static plots ---
    "plot_depth": (
        "map",
        "Bathymetry depth map (cartopy)",
    ),
    "plot_blowups": (
        "map",
        "Zoom panels with velocity vectors at a specific location",
    ),
    "plot_blowups_hz": (
        "map",
        "High-zoom bathymetry blowup",
    ),
    "plot_blowups_hzupdated": (
        "map",
        "Optimised zoom map with rasterized pcolormesh",
    ),
    "plot_map": (
        "map",
        "Nested-domain extent map (sfshelf60 + mc60 grids)",
    ),
    "plot_map_w_rms_pycnocline": (
        "map",
        "Sub-pycnocline w RMS (all scenarios) and RMSE vs notidesnowec maps",
    ),
    "plot_bl_depth": (
        "map",
        "Time-mean surface/bottom boundary layer depth maps (Akt > 1e-4 criterion, all scenarios)",
    ),
    "plot_w_rmse_wec_shelf_std": (
        "pdf",
        "Shelf RMSE(depth) of w vs notidesnowec, with Delta-std overlay",
    ),
    "plot_w_rmse_wec_offshore_std": (
        "pdf",
        "Offshore RMSE(depth) of w vs notidesnowec, with Delta-std overlay",
    ),
    "plot_vort_rmse_wec_shelf_std": (
        "pdf",
        "Shelf RMSE(depth) of zeta/f vs notidesnowec, with Delta-std overlay",
    ),
    "plot_vort_rmse_wec_offshore_std": (
        "pdf",
        "Offshore RMSE(depth) of zeta/f vs notidesnowec, with Delta-std overlay",
    ),
    "plot_rmse_std_grid": (
        "pdf",
        "2x4 grid of w/vort/dudz/drhodz RMSE+Delta-std profiles (shelf/offshore) in one figure",
    ),

    # --- Snapshot / time-loop plots (loop over all history files, slow) ---
    "plot_cs_surf": (
        "snapshot",
        "Cross-section + surface for single scenario (w + ptrace)",
    ),
    "plot_cs_w_NO3": (
        "snapshot",
        "Cross-section + surface for 4 scenarios (w + NO3)",
    ),
    "plot_surf_ptrace": (
        "snapshot",
        "1×3 surface ptrace panels for 3 scenarios",
    ),
    "plot_surf_rtrace": (
        "snapshot",
        "1×3 surface rtrace panels for 3 scenarios",
    ),
    "plot_surf_rtrace_ptrace_zslice": (
        "snapshot",
        "2×2 zsliced-surface (depth=0m) rtrace+ptrace panels, 4 scenarios",
    ),
    "plot_surf_no3_phytoc_zslice": (
        "snapshot",
        "2×2 zsliced-surface (depth=0m) NO3+phytoC panels, 4 scenarios",
    ),
    "plot_surf_3ptrace": (
        "snapshot",
        "2×2 surface ptrace panels for 4 scenarios",
    ),
    "plot_surf_3rtrace": (
        "snapshot",
        "2×2 surface rtrace panels for 4 scenarios",
    ),
    "plot_vorticity": (
        "snapshot",
        "Vorticity snapshots comparing WEC vs no-WEC",
    ),
    "plot_wind_amp": (
        "snapshot",
        "Wind speed and wave amplitude time series",
    ),

    # --- Utilities ---
    "make_mask": (
        "util",
        "Build the 10 km coastal mask; writes coastal_mask.nc",
    ),
    "plot_pipes": (
        "util",
        "Exploratory pipe-tracer plots (incomplete)",
    ),
    "plot_rivers": (
        "util",
        "Exploratory river-tracer plots (incomplete)",
    ),
}

CATEGORIES = sorted({cat for cat, _ in SCRIPTS.values()})
PLOT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess so matplotlib is truly isolated)
# ---------------------------------------------------------------------------

def _run_script(
    name: str,
    python: str,
    timeout: Optional[int],
    extra_env: dict,
) -> dict:
    script = PLOT_DIR / f"{name}.py"
    env = {**os.environ, **extra_env}
    t0 = time.time()
    try:
        proc = subprocess.run(
            [python, str(script)],
            cwd=str(PLOT_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return {
            "name": name,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed": time.time() - t0,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "returncode": -1,
            "stdout": "",
            "stderr": "",
            "elapsed": float(timeout or 0),
            "error": "TIMEOUT",
        }
    except Exception as exc:
        return {
            "name": name,
            "returncode": -1,
            "stdout": "",
            "stderr": "",
            "elapsed": time.time() - t0,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_plots.py",
        description="Run SWEL plotting scripts concurrently.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
Categories
----------
  {', '.join(CATEGORIES)}

Examples
--------
  # Run all scripts with 4 parallel workers (default):
  python run_plots.py

  # Preview what would run without executing:
  python run_plots.py --dry-run

  # Run only fast PDF plots:
  python run_plots.py --categories pdf

  # Run specific scripts:
  python run_plots.py --scripts plot_ke plot_energy_cascade

  # Run everything except the heavy spectral calculations:
  python run_plots.py --exclude calc_ke_surf calc_ke_bot save_vort

  # 8 workers, 1-hour timeout per script, verbose output:
  python run_plots.py --workers 8 --timeout 3600 --verbose

  # List all registered scripts:
  python run_plots.py --list
""",
    )

    sel = parser.add_argument_group("selection")
    sel.add_argument(
        "--scripts",
        nargs="+",
        metavar="SCRIPT",
        help="Explicit list of script names to run (without .py extension).",
    )
    sel.add_argument(
        "--categories",
        nargs="+",
        choices=CATEGORIES,
        metavar="CAT",
        help=f"Run all scripts in the given categories. Choices: {CATEGORIES}",
    )
    sel.add_argument(
        "--exclude",
        nargs="+",
        metavar="SCRIPT",
        default=[],
        help="Script names to skip (applied after --scripts / --categories).",
    )

    run = parser.add_argument_group("execution")
    run.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel worker processes (default: 4).",
    )
    run.add_argument(
        "--timeout",
        type=int,
        default=None,
        metavar="SEC",
        help="Per-script timeout in seconds. No limit by default.",
    )
    run.add_argument(
        "--python",
        default=sys.executable,
        metavar="PATH",
        help="Python interpreter to use (default: the one running this script).",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing anything.",
    )
    out.add_argument(
        "--list",
        action="store_true",
        help="Print all registered scripts with categories and exit.",
    )
    out.add_argument(
        "--verbose",
        action="store_true",
        help="Print stdout/stderr for every script, not just failures.",
    )

    return parser


def resolve_targets(args: argparse.Namespace) -> List[str]:
    if args.scripts:
        unknown = sorted(set(args.scripts) - set(SCRIPTS))
        if unknown:
            print(
                f"ERROR: Unknown scripts: {', '.join(unknown)}\n"
                "       Use --list to see all registered names.",
                file=sys.stderr,
            )
            sys.exit(1)
        targets = list(args.scripts)
    elif args.categories:
        targets = [n for n, (cat, _) in SCRIPTS.items() if cat in args.categories]
    else:
        targets = list(SCRIPTS.keys())

    targets = [t for t in targets if t not in args.exclude]

    # Only keep scripts that actually exist on disk
    missing = [t for t in targets if not (PLOT_DIR / f"{t}.py").exists()]
    if missing:
        print(
            f"WARNING: {len(missing)} script(s) not found on disk and will be skipped:\n"
            + "".join(f"  {m}.py\n" for m in missing),
            file=sys.stderr,
        )
    targets = [t for t in targets if t not in missing]
    return targets


def print_result(r: dict, verbose: bool) -> None:
    if r["error"]:
        status = r["error"]
    elif r["returncode"] == 0:
        status = "OK"
    else:
        status = f"EXIT {r['returncode']}"

    print(f"  [{status:>10}]  {r['name']:<32}  {r['elapsed']:6.1f}s")

    show_output = verbose or r["returncode"] != 0
    if show_output:
        if r["stdout"].strip():
            for line in r["stdout"].strip().splitlines()[-20:]:
                print(f"             stdout: {line}")
        if r["stderr"].strip():
            for line in r["stderr"].strip().splitlines()[-20:]:
                print(f"             stderr: {line}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --list
    if args.list:
        col1, col2 = 32, 12
        header = f"  {'Script':<{col1}} {'Category':<{col2}} Description"
        print(f"\n{header}")
        print("  " + "-" * (len(header) - 2))
        for name, (cat, desc) in sorted(SCRIPTS.items()):
            on_disk = "  " if (PLOT_DIR / f"{name}.py").exists() else "* "
            print(f"{on_disk}{name:<{col1}} {cat:<{col2}} {desc}")
        print("\n  * script file not found on disk\n")
        return

    targets = resolve_targets(args)

    if not targets:
        print("No scripts selected.", file=sys.stderr)
        sys.exit(1)

    # --dry-run
    if args.dry_run:
        print(f"\nDry run — {len(targets)} script(s) would run ({args.workers} worker(s)):\n")
        for name in targets:
            cat, desc = SCRIPTS[name]
            print(f"  [{cat:<8}]  {name}.py")
            print(f"             {desc}")
        return

    # Force non-interactive matplotlib backend for all child processes
    extra_env = {"MPLBACKEND": "Agg"}

    print(
        f"\nRunning {len(targets)} script(s) with {args.workers} worker(s)"
        + (f", timeout {args.timeout}s each" if args.timeout else "")
        + " ...\n"
    )

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_script, name, args.python, args.timeout, extra_env): name
            for name in targets
        }
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            print_result(r, args.verbose)

    n_ok = sum(1 for r in results if r["returncode"] == 0)
    n_fail = len(results) - n_ok
    wall = sum(r["elapsed"] for r in results)

    print(f"\n{'─'*55}")
    print(f"  {n_ok}/{len(results)} succeeded   {n_fail} failed   wall time {wall:.1f}s")
    print(f"{'─'*55}\n")

    if n_fail:
        print("Failed scripts:")
        for r in results:
            if r["returncode"] != 0:
                tag = r["error"] or f"exit {r['returncode']}"
                print(f"  {r['name']:<32}  [{tag}]")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
