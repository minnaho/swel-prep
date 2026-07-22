"""
Zslice all relevant mc60 his variables onto a non-uniform z grid.

Choose which scenario to process at the top (SCENARIO), then run four
instances in parallel — one per scenario — writing output to
/data/project1/minnaho/swel/zslicefull/<scenario>/.

Z grid:
    0 to -50 m  : every 1 m
    -50 to -300 m : every 5 m
    -300 to -2350 m : every 30 m
    = 157 z levels per file

Variables:
    his/: ptrace, rtrace, w, rho, u, v

BGC variables are handled separately by zslice_bgc.py.

Output: one z_<basename>.nc per input, all depths stacked into a single file.
"""

import os
import glob
import shutil
import subprocess
import numpy as np

# ── choose scenario ─────────────────────────────────────────────────────────
SCENARIO = 'tidesampwec'  # 'tideswec' | 'tidesnowec' | 'notidesnowec' | 'notideswec' | 'notidesampwec' | 'tidesampwec'

# ── paths ────────────────────────────────────────────────────────────────────
SCENARIO_ROOTS = {
    'tideswec':      '/data/project3/minnaho/swel/tides/mc60/wec',
    'tidesnowec':    '/data/project3/minnaho/swel/tides/mc60/nowec/output',
    'notidesnowec':  '/data/project3/minnaho/swel/notides/mc60/nowec/output',
    'notideswec':    '/data/project3/minnaho/swel/notides/mc60/wec/rerun',
    'notidesampwec': '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything',
    'tidesampwec':   '/data/project3/minnaho/swel/tides/mc60/ampwec',
}

SOURCES = [
    ('his', 'mc60_his.*.nc', 'ptrace,rtrace,w,rho,u,v'),
]

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
out_dir   = os.path.join(DEST_ROOT, SCENARIO)
os.makedirs(out_dir, exist_ok=True)

for src_sub, pattern, vars_csv in SOURCES:
    # notidesampwec uses a flat layout (his files directly in everything/, no his/ subdir)
    src_dir = scen_root if SCENARIO == 'notidesampwec' else os.path.join(scen_root, src_sub)
    files     = sorted(glob.glob(os.path.join(src_dir, pattern)))
    basenames = [os.path.basename(f) for f in files]
    print(f'[{SCENARIO}] zslice {src_sub}: {len(files)} files ...')
    for b in basenames:
        subprocess.call(
            f'zslice {neg_depth_args} --vars={vars_csv} {GRID} {b}',
            shell=True, cwd=src_dir,
        )
        shutil.move(os.path.join(src_dir, 'z_' + b), out_dir)
