"""
Time- and horizontally-averaged depth profiles for the DIAT/SP nutrient-
limitation and uptake diagnostics zsliced by zslice_dia_avg_rerun.py.
Sibling of profile_zslice_par.py, restricted to the 3 scenarios and the
bgcdia/ output that script produced (tidesampwec, tidesnowec, notidesnowec —
the rerun date window only).

Usage:
  # launch all 3 in parallel:
  for scen in tidesampwec tidesnowec notidesnowec; do
    python -u profile_zslice_bgcdia.py $scen > log_bgcdia_${scen}.txt 2>&1 &
  done

  # once all 3 are done:
  python -u profile_zslice_bgcdia.py --merge

Output per job:  zslice_profiles_bgcdia_<scenario>.npz
Output of merge: zslice_profiles_bgcdia.npz
                 zslice_profiles_bgcdia_coastal.npz
"""

import argparse
import glob
import numpy as np
from netCDF4 import Dataset

ZSLICE_ROOT       = '/data/project1/minnaho/swel/zslicefull'
GRD               = '../plot/mc60_grd.nc'
COASTAL_MASK_FILE = '../plot/coastal_mask.nc'
SCENARIOS = ['tidesampwec', 'tidesnowec', 'notidesnowec']

DIAT_LIM_VARS    = ['DIAT_N_LIM', 'DIAT_FE_LIM', 'DIAT_PO4_LIM',
                     'DIAT_SIO3_LIM', 'DIAT_LIGHT_LIM', 'DIAT_P_LIM']
SP_LIM_VARS      = ['SP_N_LIM', 'SP_FE_LIM', 'SP_PO4_LIM',
                     'SP_LIGHT_LIM', 'SP_P_LIM']
DIAT_UPTAKE_VARS = ['DIAT_NO3_UPTAKE', 'DIAT_NH4_UPTAKE', 'DIAT_NO2_UPTAKE',
                     'DIAT_SI_UPTAKE']
SP_UPTAKE_VARS   = ['SP_NO3_UPTAKE', 'SP_NH4_UPTAKE', 'SP_NO2_UPTAKE']
VARS = DIAT_LIM_VARS + SP_LIM_VARS + DIAT_UPTAKE_VARS + SP_UPTAKE_VARS

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('scenario', nargs='?', choices=SCENARIOS)
parser.add_argument('--merge', action='store_true')
args = parser.parse_args()

if not args.merge and args.scenario is None:
    parser.error('Provide a scenario, or --merge')

# ---------------------------------------------------------------------------
# Merge mode
# ---------------------------------------------------------------------------
if args.merge:
    out = {}
    for scen in SCENARIOS:
        fname = f'zslice_profiles_bgcdia_{scen}.npz'
        try:
            d = np.load(fname, allow_pickle=False)
        except FileNotFoundError:
            print(f'missing {fname} — skipping')
            continue
        print(f'loaded {fname}  ({len(d.files)} keys)')
        for k in d.files:
            if k not in out or k != 'depth':
                out[k] = d[k]

    np.savez('zslice_profiles_bgcdia.npz', **out)
    print('saved zslice_profiles_bgcdia.npz')

    coastal_out = {}
    if 'depth' in out:
        coastal_out['depth'] = out['depth']
    for k, v in out.items():
        if k.startswith('coastal_'):
            coastal_out[k[len('coastal_'):]] = v
    np.savez('zslice_profiles_bgcdia_coastal.npz', **coastal_out)
    print('saved zslice_profiles_bgcdia_coastal.npz')
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Single-job mode: one scenario
# ---------------------------------------------------------------------------
scen = args.scenario

files = sorted(glob.glob(f'{ZSLICE_ROOT}/{scen}/bgcdia/z_mc60_bgc_dia_avg.*.nc'))
print(f'[{scen}] {len(files)} files, vars: {VARS}')

# ---------------------------------------------------------------------------
# Load masks
# ---------------------------------------------------------------------------
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

MASK_BOOLS = {}
for lbl, zmask in ALL_MASKS.items():
    m = mask_rho if zmask is None else zmask
    MASK_BOOLS[lbl] = np.isfinite(m).reshape(-1)


def _fill_to_nan(arr):
    return np.where(np.abs(arr) > 1e30, np.nan, arr)


# ---------------------------------------------------------------------------
# Accumulate
# ---------------------------------------------------------------------------
sums   = {lbl: {v: None for v in VARS} for lbl in ALL_MASKS}
counts = {lbl: {v: None for v in VARS} for lbl in ALL_MASKS}
depth  = None

for fi, f in enumerate(files):
    print(f'  [{fi+1}/{len(files)}] {f.split("/")[-1]}', flush=True)
    with Dataset(f) as nc:
        if depth is None and 'depth' in nc.variables:
            depth = np.array(nc.variables['depth'][:])
        for v in VARS:
            if v not in nc.variables:
                continue
            arr = _fill_to_nan(np.array(nc.variables[v][:]))
            if arr.ndim == 3:      # no time dim (single-record avg files)
                arr = arr[np.newaxis]
            if arr.ndim != 4:
                continue
            t_sz, z_sz = arr.shape[0], arr.shape[1]
            arr_flat = arr.reshape(t_sz, z_sz, -1)
            for lbl in ALL_MASKS:
                m   = MASK_BOOLS[lbl]
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
    out['depth'] = depth

for lbl in ALL_MASKS:
    prefix = '' if lbl == 'full' else f'{lbl}_'
    for v in VARS:
        if sums[lbl][v] is None:
            continue
        prof = np.where(counts[lbl][v] > 0,
                        sums[lbl][v] / np.maximum(counts[lbl][v], 1), np.nan)
        out[f'{prefix}{v}_{scen}'] = prof
        if lbl == 'full':
            print(f'  {v}: min={np.nanmin(prof):.3g}  max={np.nanmax(prof):.3g}')

fname = f'zslice_profiles_bgcdia_{scen}.npz'
np.savez(fname, **out)
print(f'saved {fname}')
