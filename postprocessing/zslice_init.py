"""
Zslice an initial condition (mc60_ini) file onto a fixed z grid.

Comment/uncomment the CASE block to switch between tides and notides.
Output lands next to the ini file as z_mc60_ini.<timestamp>.nc.

Z grid: same non-uniform grid as the rest of the zslice pipeline (157 levels).
"""

import os
import subprocess
import numpy as np

GRID = '/data/project3/minnaho/project9copy/swel/plot/mc60_grd.nc'

# ── choose case (comment out the one you don't want) ────────────────────────
#INI = '/data/project3/minnaho/project9copy/swel/tides/mc60/rst/mc60_ini.20190415110120.nc'

INI = '/data/project3/minnaho/project9copy/swel/notides/mc60/rst/mc60_ini.20190415110120.nc'

# ── z grid ───────────────────────────────────────────────────────────────────
depth_levs = np.unique(np.concatenate([
    np.arange(0,   51,   1),
    np.arange(55,  301,  5),
    np.arange(330, 1981, 30),
])).astype(int)   # 157 levels; passed as negatives to zslice
neg_depth_args = ' '.join(str(-d) for d in depth_levs)

# rho-grid 3D vars only — exclude u/v/ubar/vbar/u_slow/v_slow (U/V grid) and
# w (s_w grid); zslice errors on non-rho-grid variables
VARS = ('temp,salt,'
        'NO3,PO4,SiO3,Fe,O2,NH4,DIC,Alk,DOC,DON,DONR,DOP,DOPR,DOFE,NO2,N2,N2O,'
        'SPC,SPCHL,SPFE,SPCACO3,DIATC,DIATCHL,DIATFE,DIATSI,DIAZC,DIAZCHL,DIAZFE,ZOOC')

# ── run zslice ────────────────────────────────────────────────────────────────
src_dir  = os.path.dirname(INI)
basename = os.path.basename(INI)

print(f'zslicing {basename} ...')
subprocess.call(
    f'zslice {neg_depth_args} --vars={VARS} {GRID} {basename}',
    shell=True, cwd=src_dir,
)
print(f'done -> {os.path.join(src_dir, "z_" + basename)}')
