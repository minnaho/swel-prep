#!/usr/bin/env python3
"""
Concurrent runner for the box-averaged cross-section scripts in this directory.

Same pattern as ../run_plots.py: runs any subset of the plot_cs_*_box.py
scripts in parallel using Python's ProcessPoolExecutor, each as a subprocess
with CWD set to this directory (plot/boxavg/) so `import cs_boxavg as cb`
and every relative path resolve correctly.

MPLBACKEND is forced to 'Agg' so scripts run headlessly without a display.

Usage:
    python run_boxavg.py                          # run everything (4 workers)
    python run_boxavg.py --list                   # list all scripts + categories
    python run_boxavg.py --dry-run                # preview without executing
    python run_boxavg.py --categories diag         # run only the ts/tn diagonal family
    python run_boxavg.py --scripts plot_cs_diag_box plot_cs_diag_rho_box
    python run_boxavg.py --workers 8 --timeout 3600
    python run_boxavg.py --exclude plot_cs_diag_bgcdia_box   # skip a slow one
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
# Categories: diag | snapshot | diff
# ---------------------------------------------------------------------------

SCRIPTS: Dict[str, Tuple[str, str]] = {
    # --- Group 1: diagonal ts/tn cross-sections, native s_rho, box-averaged ---
    "plot_cs_diag_box": (
        "diag",
        "w cross-section, 6 scenarios, box-averaged",
    ),
    "plot_cs_diag_akt_box": (
        "diag",
        "Akt (s_w) cross-section, 6 scenarios, box-averaged",
    ),
    "plot_cs_diag_akv_box": (
        "diag",
        "Akv (s_w) cross-section, 6 scenarios, box-averaged",
    ),
    "plot_cs_diag_no3_box": (
        "diag",
        "NO3 cross-section, 6 scenarios, box-averaged",
    ),
    "plot_cs_diag_o2_box": (
        "diag",
        "O2 cross-section, 6 scenarios, box-averaged",
    ),
    "plot_cs_diag_rho_box": (
        "diag",
        "rho (sigma-t) cross-section, 6 scenarios, box-averaged",
    ),
    "plot_cs_diag_totc_box": (
        "diag",
        "Total phyto C cross-section, 6 scenarios, box-averaged",
    ),
    "plot_cs_diag_bgcdia_box": (
        "diag",
        "18 DIAT/SP limitation + uptake diagnostics, 3 scenarios, box-averaged",
    ),
    "plot_cs_ini_box": (
        "diag",
        "Initial-condition tides vs no-tides comparison, box-averaged",
    ),

    # --- Group 2: ts/tn/mid single-snapshot cross-sections, box-averaged ---
    "plot_cs_pv_snap_box": (
        "snapshot",
        "Potential vorticity snapshot, 4 scenarios, box-averaged",
    ),
    "plot_cs_vorticity_snap_box": (
        "snapshot",
        "Normalized vorticity (zeta/f) snapshot, 4 scenarios, box-averaged",
    ),

    # --- Group 4: zsliced time-mean diffs, box-averaged on the fixed depth grid ---
    "plot_cs_diag_avg_diff_box": (
        "diff",
        "Time-mean diff from notidesnowec, 7 variables, box-averaged",
    ),
    "plot_cs_diag_drhodz_diff_box": (
        "diff",
        "Time-mean drho/dz diff from notidesnowec, box-averaged",
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
        prog="run_boxavg.py",
        description="Run the box-averaged plot_cs_*_box.py scripts concurrently.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
Categories
----------
  {', '.join(CATEGORIES)}

Examples
--------
  # Run all scripts with 4 parallel workers (default):
  python run_boxavg.py

  # Preview what would run without executing:
  python run_boxavg.py --dry-run

  # Run only the ts/tn diagonal family:
  python run_boxavg.py --categories diag

  # Run specific scripts:
  python run_boxavg.py --scripts plot_cs_diag_box plot_cs_diag_rho_box

  # Run everything except the slow 18-variable bgcdia loop:
  python run_boxavg.py --exclude plot_cs_diag_bgcdia_box

  # 8 workers, 1-hour timeout per script, verbose output:
  python run_boxavg.py --workers 8 --timeout 3600 --verbose

  # List all registered scripts:
  python run_boxavg.py --list
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
