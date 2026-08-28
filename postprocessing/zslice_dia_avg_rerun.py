"""
Zslice ALL 3-D variables from mc60_bgc_dia_avg files onto the same non-uniform
z grid as zslice_npp.py, for a rerun subset of scenarios/dates.

Unlike zslice_npp.py (which selects only TOT_PROD via --vars=TOT_PROD), this
script omits --vars so zslice interpolates every 3-D field present in the
file: TOT_PROD, TOT_CHL, POC_FLUX_IN, PAR. The file's 2-D fields (pCO2air,
FG_O2, FG_CO2, SCHMIDT_CO2, PV_CO2, PH, CO2STAR, PCO2OC, DCO2STAR, FG_N2O,
FG_N2) have no depth axis and are skipped automatically by zslice.

Only processes files whose filename date stamp (YYYYMMDD) falls within
[DATE_START, DATE_END] inclusive — scoped to a rerun of tidesampwec,
tidesnowec, notidesnowec, and ampwec covering 2019-04-20 through 2019-04-22.
ampwec's rerun source is /data/project3/minnaho/swel/notides/mc60/wec/ampwec/
dia/rerun_bgcdia/ -- NOT .../ampwec/everything, a separate flat-layout
continuous run whose dia_avg files lack the LIM/UPTAKE variables entirely.
ampwec's zsliced output still lands under zslicefull/notidesampwec/bgcdia/,
not zslicefull/ampwec/ -- the raw scenario key and the zslicefull directory
name diverge for this one run (same run, different naming convention; see
scenario_style.py's comment), so ZSLICE_DIR_ALIAS maps the output path only.

Z grid (same as zslice_npp.py):
    0 to -200 m : every 2 m
    = 101 z levels per file

Output: /data/project1/minnaho/swel/zslicefull/<scenario>/bgcdia/z_<basename>.nc
(a separate directory from zslice_npp.py's dia/ output, so the full-variable
rerun files don't collide with the existing TOT_PROD-only ones).
"""

import os
import re
import glob
import shutil
import subprocess
import numpy as np

# ── scenarios / date window ─────────────────────────────────────────────────
SCENARIOS  = ['ampwec']   # TEMP: restricted to just the new scenario for this
                          # run so tidesampwec/tidesnowec/notidesnowec aren't
                          # redundantly re-zsliced -- revert to
                          # ['tidesampwec', 'tidesnowec', 'notidesnowec', 'ampwec']
                          # afterward
DATE_START = 20190421   # inclusive, YYYYMMDD parsed from filename stamp
DATE_END   = 20190422   # inclusive

# ── paths ────────────────────────────────────────────────────────────────────
# 'ampwec' here is the BASE root (his/ + dia/rerun_bgcdia/ live directly
# under it) for this scoped rerun -- NOT .../ampwec/everything, which is a
# separate flat-layout continuous run used elsewhere in this codebase; see
# module docstring.
SCENARIO_ROOTS = {
    'tideswec':     '/data/project3/minnaho/swel/tides/mc60/wec',
    'tidesnowec':   '/data/project3/minnaho/swel/tides/mc60/nowec/output',
    'notidesnowec': '/data/project3/minnaho/swel/notides/mc60/nowec',
    'notideswec':   '/data/project3/minnaho/swel/notides/mc60/wec/rerun',
    'ampwec':       '/data/project3/minnaho/swel/notides/mc60/wec/ampwec',
    'tidesampwec':  '/data/project3/minnaho/swel/tides/mc60/ampwec',
}

DEST_ROOT = '/data/project1/minnaho/swel/zslicefull'
GRID      = '/data/project3/minnaho/project9copy/swel/plot/mc60_grd.nc'

# tidesnowec/notidesnowec/ampwec reran only this date window — their rerun
# output lands in a rerun_bgcdia/ subdir under dia/, not dia/ itself
SRC_SUBDIR = {
    'tidesnowec':   'dia/rerun_bgcdia',
    'notidesnowec': 'dia/rerun_bgcdia',
    'ampwec':       'dia/rerun_bgcdia',
}

# zslicefull output directory name diverges from the raw scenario key for
# ampwec (same run, different naming convention -- see module docstring);
# every other scenario here uses its own key as the output dir name too
ZSLICE_DIR_ALIAS = {'ampwec': 'notidesampwec'}

STAMP_RE = re.compile(r'mc60_bgc_dia_avg\.(\d{8})\d{6}\.nc$')

# ── z grid ───────────────────────────────────────────────────────────────────
depths = np.arange(0, 201, 2).astype(int)   # 101 positive integer depths; invoked as negatives

neg_depth_args = ' '.join(str(-d) for d in depths)

# ── run ───────────────────────────────────────────────────────────────────────
for scenario in SCENARIOS:
    scen_root = SCENARIO_ROOTS[scenario]
    src_dir   = os.path.join(scen_root, SRC_SUBDIR.get(scenario, 'dia'))
    out_dir   = os.path.join(DEST_ROOT, ZSLICE_DIR_ALIAS.get(scenario, scenario), 'bgcdia')
    os.makedirs(out_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(src_dir, 'mc60_bgc_dia_avg.*.nc')))
    files = []
    for f in all_files:
        m = STAMP_RE.search(os.path.basename(f))
        if m and DATE_START <= int(m.group(1)) <= DATE_END:
            files.append(f)

    print(f'[{scenario}] zslice dia (all 3D vars): '
          f'{len(files)}/{len(all_files)} files in [{DATE_START}, {DATE_END}] ...')
    for b in [os.path.basename(f) for f in files]:
        subprocess.call(
            f'zslice {neg_depth_args} {GRID} {b}',
            shell=True, cwd=src_dir,
        )
        dest_path = os.path.join(out_dir, 'z_' + b)
        if os.path.exists(dest_path):
            os.remove(dest_path)   # allow overwriting a prior zslice_npp.py / rerun output
        shutil.move(os.path.join(src_dir, 'z_' + b), out_dir)
