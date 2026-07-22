"""
Zslice TOT_PROD from mc60_bgc_dia_avg files onto a non-uniform z grid.

Choose which scenario to process at the top (SCENARIO), then run four
instances in parallel — one per scenario — writing output to
/data/project1/minnaho/swel/zslicefull/<scenario>/dia/.

Z grid:
    0 to -200 m : every 2 m
    = 101 z levels per file

Variables:
    dia/: TOT_PROD
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
    'notidesnowec': '/data/project3/minnaho/swel/notides/mc60/nowec/output',
    'notideswec':   '/data/project3/minnaho/swel/notides/mc60/wec/rerun',
    'ampwec':       '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything',
    'tidesampwec':  '/data/project3/minnaho/swel/tides/mc60/ampwec',
}

NPP_VAR = 'TOT_PROD'

DEST_ROOT = '/data/project1/minnaho/swel/zslicefull'
GRID      = '/data/project3/minnaho/project9copy/swel/plot/mc60_grd.nc'

# ── z grid ───────────────────────────────────────────────────────────────────
depths = np.arange(0, 201, 2).astype(int)   # 101 positive integer depths; invoked as negatives

neg_depth_args = ' '.join(str(-d) for d in depths)

# ── run ───────────────────────────────────────────────────────────────────────
# ampwec uses a flat layout (no dia/ subdir)
scen_root = SCENARIO_ROOTS[SCENARIO]
src_dir   = scen_root if SCENARIO == 'ampwec' else os.path.join(scen_root, 'dia')
out_dir   = os.path.join(DEST_ROOT, SCENARIO, 'dia')
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(src_dir, 'mc60_bgc_dia_avg.*.nc')))
print(f'[{SCENARIO}] zslice npp: {len(files)} files ...')
for b in [os.path.basename(f) for f in files]:
    subprocess.call(
        f'zslice {neg_depth_args} --vars={NPP_VAR} {GRID} {b}',
        shell=True, cwd=src_dir,
    )
    shutil.move(os.path.join(src_dir, 'z_' + b), out_dir)
