"""
Zslice Akt and Akv from mc60 his files onto a non-uniform z grid.

Choose which scenario to process at the top (SCENARIO), then run instances
in parallel — one per scenario — writing output to
/data/project1/minnaho/swel/zslicefull/<scenario>/ak/.

NOTE: Akt and Akv are defined on the w-grid (s_w, N+1 = 101 levels, cell
faces) rather than on s_rho (N = 100 levels, cell centres).  The zslice
tool handles s_w variables differently from s_rho ones; verify the output
depth coordinate matches expectations before use.

Variables:
    Akv  — vertical viscosity coefficient      (m2 s-1)
    Akt  — vertical thermal diffusivity        (m2 s-1)

Z grid (same as zslice_uniform.py, 157 levels):
    0 to -50 m      : every 1 m
    -55 to -300 m   : every 5 m
    -330 to -1980 m : every 30 m

Output: one z_<basename>.nc per input, written to
    /data/project1/minnaho/swel/zslicefull/<scenario>/ak/
"""

import os
import glob
import shutil
import subprocess
import numpy as np

# ── choose scenario ─────────────────────────────────────────────────────────
SCENARIO = 'tidesampwec'  # 'tideswec' | 'tidesnowec' | 'notidesnowec' | 'notideswec' | 'ampwec' | 'tidesampwec'

# ── paths ────────────────────────────────────────────────────────────────────
SCENARIO_ROOTS = {
    'tideswec':     '/data/project3/minnaho/swel/tides/mc60/wec',
    'tidesnowec':   '/data/project3/minnaho/swel/tides/mc60/nowec/output',
    'notidesnowec': '/data/project3/minnaho/swel/notides/mc60/nowec',
    'notideswec':   '/data/project3/minnaho/swel/notides/mc60/wec/rerun',
    'ampwec':       '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything',
    'tidesampwec':  '/data/project3/minnaho/swel/tides/mc60/ampwec',
}

VARS    = 'Akt,Akv'
DEST_ROOT = '/data/project1/minnaho/swel/zslicefull'
GRID      = '/data/project3/minnaho/project9copy/swel/plot/mc60_grd.nc'

# ── z grid ───────────────────────────────────────────────────────────────────
depths = np.unique(np.concatenate([
    np.arange(0,   51,   1),
    np.arange(55,  301,  5),
    np.arange(330, 1981, 30),
])).astype(int)   # 157 positive integer depths; invoked as negatives

neg_depth_args = ' '.join(str(-d) for d in depths)

# ── run ───────────────────────────────────────────────────────────────────────
scen_root = SCENARIO_ROOTS[SCENARIO]

# ampwec uses a flat layout (his files directly in notrace/, no his/ subdir)
src_dir = scen_root if SCENARIO == 'ampwec' else os.path.join(scen_root, 'his')

# ampwec output goes to notidesampwec/ to match the zslice destination convention
dest_scen = 'notidesampwec' if SCENARIO == 'ampwec' else SCENARIO
out_dir = os.path.join(DEST_ROOT, dest_scen, 'ak')
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(src_dir, 'mc60_his.*.nc')))
print(f'[{SCENARIO}] zslice ak: {len(files)} files ...')
for b in [os.path.basename(f) for f in files]:
    subprocess.call(
        f'zslice {neg_depth_args} --vars={VARS} {GRID} {b}',
        shell=True, cwd=src_dir,
    )
    shutil.move(os.path.join(src_dir, 'z_' + b), out_dir)
