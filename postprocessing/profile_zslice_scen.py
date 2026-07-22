"""
Per-scenario version of profile_zslice.py — run one scenario at a time so
all five can be launched in parallel.

Usage:
# run each scenario in parallel:
python profile_zslice_scen.py tideswec     &
python profile_zslice_scen.py tidesnowec   &
python profile_zslice_scen.py notidesnowec &
python profile_zslice_scen.py notideswec   &
python profile_zslice_scen.py ampwec       &
wait

python profile_zslice_scen.py --merge
# once all five are done, merge into combined NPZs:

Output per scenario run: zslice_profiles_<scenario>.npz
Output of --merge:        zslice_profiles.npz
                          zslice_profiles_coastal.npz
"""

import argparse
import glob
import numpy as np
from netCDF4 import Dataset

ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'
GRD               = '../plot/mc60_grd.nc'
COASTAL_MASK_FILE = '../plot/coastal_mask.nc'
SCENARIOS   = ['tideswec', 'tidesnowec', 'notidesnowec', 'notideswec', 'ampwec']
SCEN_DIRS   = {'ampwec': 'notidesampwec'}
HIS_VARS    = ['ptrace', 'rtrace', 'w', 'rho', 'u', 'v']
BGC_VARS    = ['NO3', 'NH4', 'SPC', 'DIATC', 'DIAZC',
               'SPCHL', 'DIATCHL', 'DIAZCHL', 'O2', 'DIC', 'DOC']
DIA_VARS    = ['TOT_PROD']

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('scenario', nargs='?', choices=SCENARIOS,
                    help='Scenario to process')
parser.add_argument('--merge', action='store_true',
                    help='Merge all per-scenario NPZs into combined output files')
args = parser.parse_args()

if not args.merge and args.scenario is None:
    parser.error('Provide a scenario name or --merge')

# ---------------------------------------------------------------------------
# Merge mode: load per-scenario NPZs and combine
# ---------------------------------------------------------------------------
if args.merge:
    out = {}
    for scen in SCENARIOS:
        fname = f'zslice_profiles_{scen}.npz'
        d = np.load(fname, allow_pickle=False)
        print(f'loaded {fname}  ({len(d.files)} keys)')
        for k in d.files:
            if k not in out:
                out[k] = d[k]
            elif k not in ('depth', 'depth_dia'):
                out[k] = d[k]   # scenario keys are unique; depth is the same

    np.savez('zslice_profiles.npz', **out)
    print('saved zslice_profiles.npz')

    coastal_out = {'depth': out['depth']}
    if 'depth_dia' in out:
        coastal_out['depth_dia'] = out['depth_dia']
    for k, v in out.items():
        if k.startswith('coastal_'):
            coastal_out[k[len('coastal_'):]] = v
    np.savez('zslice_profiles_coastal.npz', **coastal_out)
    print('saved zslice_profiles_coastal.npz')
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Single-scenario mode
# ---------------------------------------------------------------------------
scen = args.scenario

grd_nc   = Dataset(GRD)
mask_rho = np.array(grd_nc['mask_rho']).astype(float)
h        = np.array(grd_nc['h'])
mask_rho[mask_rho == 0] = np.nan
mask_u   = mask_rho[:, :-1].copy()
mask_v   = mask_rho[:-1, :].copy()

coastal_mask = np.array(Dataset(COASTAL_MASK_FILE)['coastal_mask']).astype(float)
coastal_mask[coastal_mask == 0] = np.nan


def _make_h_mask(h_lo, h_hi):
    m = mask_rho.copy()
    m[(h <= h_lo) | (h > h_hi)] = np.nan
    return m


ALL_MASKS = {
    'full':     None,
    'coastal':  coastal_mask,
    'h0to50':   _make_h_mask(0,   50),
    'h50to200': _make_h_mask(50,  200),
    'h200p':    _make_h_mask(200, np.inf),
}

# Pre-compute flat boolean masks per (label, grid-type) to avoid repeated
# broadcasting of the full spatial mask inside the accumulation loop.
_GRIDS = ('rho', 'u', 'v')
_base_masks = {
    'rho': mask_rho,
    'u':   mask_u,
    'v':   mask_v,
}

MASK_BOOLS = {}   # (lbl, grid) -> 1-D boolean array over flattened spatial dim
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


def accumulate_files(files, vars):
    """
    Open each file once; read each variable once; apply all masks via boolean
    indexing (avoids full-size broadcast temporaries for sparse zone masks).
    Returns (depth, {mask_label: {var: mean_profile}}).
    """
    sums   = {lbl: {v: None for v in vars} for lbl in ALL_MASKS}
    counts = {lbl: {v: None for v in vars} for lbl in ALL_MASKS}
    depth  = None
    for f in files:
        with Dataset(f) as nc:
            if depth is None:
                depth = np.array(nc.variables['depth'][:])
            for v in vars:
                if v not in nc.variables:
                    continue
                arr = _fill_to_nan(np.array(nc.variables[v][:]))  # (t,z,eta,xi)
                if arr.ndim == 3:      # no time dim (e.g. single-record avg files)
                    arr = arr[np.newaxis]
                if arr.ndim != 4:
                    continue
                t_sz, z_sz = arr.shape[0], arr.shape[1]
                arr_flat = arr.reshape(t_sz, z_sz, -1)            # (t,z,n_spatial)
                grid = _var_grid(v)
                for lbl in ALL_MASKS:
                    m = MASK_BOOLS[(lbl, grid)]
                    sub = arr_flat[:, :, m]                        # (t,z,n_valid)
                    cs  = np.nansum(sub, axis=(0, 2))
                    cc  = np.sum(~np.isnan(sub), axis=(0, 2))
                    if sums[lbl][v] is None:
                        sums[lbl][v]   = cs
                        counts[lbl][v] = cc.astype(np.int64)
                    else:
                        sums[lbl][v]   += cs
                        counts[lbl][v] += cc
    results = {}
    for lbl in ALL_MASKS:
        results[lbl] = {
            v: np.where(counts[lbl][v] > 0,
                        sums[lbl][v] / np.maximum(counts[lbl][v], 1), np.nan)
            for v in vars if sums[lbl][v] is not None
        }
    return depth, results


scen_dir  = SCEN_DIRS.get(scen, scen)
his_files = sorted(glob.glob(f'{ZSLICE_ROOT}/{scen_dir}/z_mc60_his.*.nc'))
bgc_files = sorted(glob.glob(f'{ZSLICE_ROOT}/{scen_dir}/bgc/z_mc60_bgc.*.nc'))
dia_files = sorted(glob.glob(f'{ZSLICE_ROOT}/{scen_dir}/dia/z_mc60_bgc_dia_avg.*.nc'))
print(f'[{scen}] {len(his_files)} his, {len(bgc_files)} bgc, {len(dia_files)} dia files')

out = {}

depth, his = accumulate_files(his_files, HIS_VARS)
if depth is not None:
    out['depth'] = depth

_, bgc = accumulate_files(bgc_files, BGC_VARS)

depth_dia, dia = accumulate_files(dia_files, DIA_VARS)
if depth_dia is not None:
    out['depth_dia'] = depth_dia

for lbl in ALL_MASKS:
    prefix = '' if lbl == 'full' else f'{lbl}_'
    for v, prof in his[lbl].items():
        out[f'{prefix}{v}_{scen}'] = prof
        if lbl == 'full':
            print(f'  {v}: min={np.nanmin(prof):.3g}  max={np.nanmax(prof):.3g}')
    for v, prof in bgc[lbl].items():
        out[f'{prefix}{v}_{scen}'] = prof
        if lbl == 'full':
            print(f'  {v}: min={np.nanmin(prof):.3g}  max={np.nanmax(prof):.3g}')
    for v, prof in dia[lbl].items():
        out[f'{prefix}{v}_{scen}'] = prof
        if lbl == 'full':
            print(f'  {v}: min={np.nanmin(prof):.3g}  max={np.nanmax(prof):.3g}')

np.savez(f'zslice_profiles_{scen}.npz', **out)
print(f'saved zslice_profiles_{scen}.npz')
