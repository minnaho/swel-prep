"""
Fully parallel version of profile_zslice_par.py, restricted to deeper than
the 100 m isobath -- same h>100m masking convention used by the offshore
region in calc_w_std.py / calc_vort_std.py / calc_dudz_rmse_wec.py
(mask[h <= 100] = np.nan), applied to the base rho/u/v masks before any
per-file accumulation, so every output mask ('full', 'coastal') is
implicitly an average over depth > 100 m only. Offshore counterpart of
profile_zslice_par_100m.py, which is the h<=100m (shelf) restriction.

The h0to50/h50to200/h200p depth-bin masks from profile_zslice_par.py are
dropped here, same reasoning as profile_zslice_par_100m.py -- they'd
otherwise silently re-bin within an already h>100m-restricted domain.

Usage:
  # launch all in parallel:
  for scen in tideswec tidesnowec notidesnowec notideswec ampwec tidesampwec; do
    python -u profile_zslice_par_offshore.py $scen his > log_${scen}_his_offshore.txt 2>&1 &
    python -u profile_zslice_par_offshore.py $scen bgc > log_${scen}_bgc_offshore.txt 2>&1 &
    python -u profile_zslice_par_offshore.py $scen dia > log_${scen}_dia_offshore.txt 2>&1 &
    python -u profile_zslice_par_offshore.py $scen ak  > log_${scen}_ak_offshore.txt  2>&1 &
  done

  # once all 24 are done:
  python -u profile_zslice_par_offshore.py --merge

Output per job:  zslice_profiles_offshore_<scenario>_<filetype>.npz
Output of merge: zslice_profiles_offshore.npz
                 zslice_profiles_offshore_coastal.npz
"""

import argparse
import glob
import numpy as np
from netCDF4 import Dataset

ZSLICE_ROOT       = '/data/project1/minnaho/swel/zslicefull'
GRD               = '../plot/mc60_grd.nc'
COASTAL_MASK_FILE = '../plot/coastal_mask.nc'
SCENARIOS   = ['tideswec', 'tidesnowec', 'notidesnowec', 'notideswec', 'ampwec', 'tidesampwec']
SCEN_DIRS   = {'ampwec': 'notidesampwec'}
FILETYPES   = ['his', 'bgc', 'dia', 'ak']

HIS_VARS = ['ptrace', 'rtrace', 'w', 'rho', 'u', 'v']
BGC_VARS = ['NO3', 'NH4', 'SPC', 'DIATC', 'DIAZC',
            'SPCHL', 'DIATCHL', 'DIAZCHL', 'O2', 'DIC', 'DOC']
DIA_VARS = ['TOT_PROD']
# Akt/Akv live in a separate ak/ subdirectory (zslicefull/<scen>/ak/z_mc60_his.*.nc),
# not the root z_mc60_his.*.nc that HIS_VARS reads from -- a distinct filetype
# so adding them doesn't require rerunning the his/bgc/dia jobs.
AK_VARS  = ['Akt', 'Akv']

FILETYPE_VARS = {'his': HIS_VARS, 'bgc': BGC_VARS, 'dia': DIA_VARS, 'ak': AK_VARS}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('scenario', nargs='?', choices=SCENARIOS)
parser.add_argument('filetype', nargs='?', choices=FILETYPES)
parser.add_argument('--merge', action='store_true')
args = parser.parse_args()

if not args.merge and (args.scenario is None or args.filetype is None):
    parser.error('Provide a scenario and filetype, or --merge')

# ---------------------------------------------------------------------------
# Merge mode
# ---------------------------------------------------------------------------
if args.merge:
    out = {}
    for scen in SCENARIOS:
        for ft in FILETYPES:
            fname = f'zslice_profiles_offshore_{scen}_{ft}.npz'
            try:
                d = np.load(fname, allow_pickle=False)
            except FileNotFoundError:
                print(f'missing {fname} — skipping')
                continue
            print(f'loaded {fname}  ({len(d.files)} keys)')
            for k in d.files:
                if k not in out or k not in ('depth', 'depth_dia'):
                    out[k] = d[k]

    np.savez('zslice_profiles_offshore.npz', **out)
    print('saved zslice_profiles_offshore.npz')

    coastal_out = {}
    for k in ('depth', 'depth_dia'):
        if k in out:
            coastal_out[k] = out[k]
    for k, v in out.items():
        if k.startswith('coastal_'):
            coastal_out[k[len('coastal_'):]] = v
    np.savez('zslice_profiles_offshore_coastal.npz', **coastal_out)
    print('saved zslice_profiles_offshore_coastal.npz')
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Single-job mode: one scenario + one filetype
# ---------------------------------------------------------------------------
scen     = args.scenario
filetype = args.filetype
scen_dir = SCEN_DIRS.get(scen, scen)
vars     = FILETYPE_VARS[filetype]

if filetype == 'his':
    files = sorted(glob.glob(f'{ZSLICE_ROOT}/{scen_dir}/z_mc60_his.*.nc'))
elif filetype == 'bgc':
    files = sorted(glob.glob(f'{ZSLICE_ROOT}/{scen_dir}/bgc/z_mc60_bgc.*.nc'))
elif filetype == 'ak':
    files = sorted(glob.glob(f'{ZSLICE_ROOT}/{scen_dir}/ak/z_mc60_his.*.nc'))
else:
    files = sorted(glob.glob(f'{ZSLICE_ROOT}/{scen_dir}/dia/z_mc60_bgc_dia_avg.*.nc'))

print(f'[{scen}/{filetype}] {len(files)} files, vars: {vars}')

# ---------------------------------------------------------------------------
# Load masks
# ---------------------------------------------------------------------------
grd_nc   = Dataset(GRD)
mask_rho = np.array(grd_nc['mask_rho']).astype(float)
h        = np.array(grd_nc['h'])
mask_rho[mask_rho == 0] = np.nan

# restrict to areas deeper than 100 m -- shallower cells become NaN so they
# drop out of every profile average the same way land does
mask_rho[h <= 100] = np.nan

mask_u   = mask_rho[:, :-1].copy()
mask_v   = mask_rho[:-1, :].copy()

coastal_mask = np.array(Dataset(COASTAL_MASK_FILE)['coastal_mask']).astype(float)
coastal_mask[coastal_mask == 0] = np.nan
coastal_mask[h <= 100] = np.nan

ALL_MASKS = {
    'full':    None,
    'coastal': coastal_mask,
}

_base_masks = {'rho': mask_rho, 'u': mask_u, 'v': mask_v}

MASK_BOOLS = {}
for lbl, zmask in ALL_MASKS.items():
    for grid, base in _base_masks.items():
        if zmask is None:
            m = base
        elif grid == 'u':
            m = zmask[:, :-1]
        elif grid == 'v':
            m = zmask[:-1, :]
        else:
            m = zmask
        MASK_BOOLS[(lbl, grid)] = np.isfinite(m).reshape(-1)


def _var_grid(v):
    if v == 'u': return 'u'
    if v == 'v': return 'v'
    return 'rho'


def _fill_to_nan(arr):
    return np.where(np.abs(arr) > 1e30, np.nan, arr)


# ---------------------------------------------------------------------------
# Accumulate
# ---------------------------------------------------------------------------
sums   = {lbl: {v: None for v in vars} for lbl in ALL_MASKS}
counts = {lbl: {v: None for v in vars} for lbl in ALL_MASKS}
depth  = None

for fi, f in enumerate(files):
    print(f'  [{fi+1}/{len(files)}] {f.split("/")[-1]}', flush=True)
    with Dataset(f) as nc:
        if depth is None and 'depth' in nc.variables:
            depth = np.array(nc.variables['depth'][:])
        for v in vars:
            if v not in nc.variables:
                continue
            arr = _fill_to_nan(np.array(nc.variables[v][:]))
            if arr.ndim == 3:      # no time dim (single-record avg files)
                arr = arr[np.newaxis]
            if arr.ndim != 4:
                continue
            t_sz, z_sz = arr.shape[0], arr.shape[1]
            arr_flat = arr.reshape(t_sz, z_sz, -1)
            grid = _var_grid(v)
            for lbl in ALL_MASKS:
                m   = MASK_BOOLS[(lbl, grid)]
                sub = arr_flat[:, :, m]
                cs  = np.nansum(sub, axis=(0, 2))
                cc  = np.sum(~np.isnan(sub), axis=(0, 2))
                if sums[lbl][v] is None:
                    sums[lbl][v]   = cs
                    counts[lbl][v] = cc.astype(np.int64)
                else:
                    sums[lbl][v]   += cs
                    counts[lbl][v] += cc

# ---------------------------------------------------------------------------
# Build output
# ---------------------------------------------------------------------------
out = {}
if depth is not None:
    depth_key = 'depth_dia' if filetype == 'dia' else 'depth'
    out[depth_key] = depth

for lbl in ALL_MASKS:
    prefix = '' if lbl == 'full' else f'{lbl}_'
    for v in vars:
        if sums[lbl][v] is None:
            continue
        prof = np.where(counts[lbl][v] > 0,
                        sums[lbl][v] / np.maximum(counts[lbl][v], 1), np.nan)
        out[f'{prefix}{v}_{scen}'] = prof
        if lbl == 'full':
            print(f'  {v}: min={np.nanmin(prof):.3g}  max={np.nanmax(prof):.3g}')

fname = f'zslice_profiles_offshore_{scen}_{filetype}.npz'
np.savez(fname, **out)
print(f'saved {fname}')
