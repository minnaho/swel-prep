"""
Zslice TOT_PROD from mc60_bgc_dia_avg files onto a non-uniform z grid — notidesampwec only.

Output: /data/project1/minnaho/swel/zslicefull/notidesampwec/dia/

Z grid:
    0 to -200 m : every 2 m
    = 101 z levels per file

NOTE: The notidesampwec (notrace) scenario likely has incorrect tracer indices.
Verify TOT_PROD values against a known-good scenario before using for analysis.
"""

import os
import glob
import shutil
import subprocess
import numpy as np

SCENARIO  = 'notidesampwec'
src_dir   = '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything'
NPP_VAR   = 'TOT_PROD'
DEST_ROOT = '/data/project1/minnaho/swel/zslicefull'
GRID      = '/data/project3/minnaho/project9copy/swel/plot/mc60_grd.nc'

# ── z grid ───────────────────────────────────────────────────────────────────
depths = np.arange(0, 201, 2).astype(int)   # 101 positive integer depths; invoked as negatives

neg_depth_args = ' '.join(str(-d) for d in depths)

# ── run ───────────────────────────────────────────────────────────────────────
out_dir = os.path.join(DEST_ROOT, SCENARIO, 'dia')
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(src_dir, 'mc60_bgc_dia_avg.*.nc')))
print(f'[{SCENARIO}] zslice npp: {len(files)} files ...')
for b in [os.path.basename(f) for f in files]:
    subprocess.call(
        f'zslice {neg_depth_args} --vars={NPP_VAR} {GRID} {b}',
        shell=True, cwd=src_dir,
    )
    shutil.move(os.path.join(src_dir, 'z_' + b), out_dir)
