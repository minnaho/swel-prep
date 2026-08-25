"""
Combined h<=100m (shelf) and h>100m (offshore) version of
profile_zslice_bgcdia.py. Computes both masks from a single pass over the
dia_avg files -- the file read is the expensive part, so profile_zslice_
bgcdia_100m.py and profile_zslice_bgcdia_offshore.py were merged into this
one script rather than each re-reading the same files independently.

Masking convention matches profile_zslice_par_100m.py / _offshore.py:
mask_rho[h > 100] = np.nan for the shelf group, mask_rho[h <= 100] = np.nan
for the offshore group, both applied before accumulation. The
h0to50/h50to200/h200p depth-bin masks from profile_zslice_bgcdia.py are
dropped, same reasoning as the two split scripts this replaces -- they'd
otherwise silently re-bin within an already depth-restricted domain.

Usage:
  # launch all 3 in parallel:
  for scen in tidesampwec tidesnowec notidesnowec; do
    python -u profile_zslice_bgcdia_100m_offshore.py $scen > log_bgcdia_100m_offshore_${scen}.txt 2>&1 &
  done

  # once all 3 are done:
  python -u profile_zslice_bgcdia_100m_offshore.py --merge

  # adding a new variable later without re-reading every other variable
  # (merges into the existing per-scenario npz rather than overwriting it):
  for scen in tidesampwec tidesnowec notidesnowec; do
    python -u profile_zslice_bgcdia_100m_offshore.py $scen --vars PAR > log_bgcdia_PAR_${scen}.txt 2>&1 &
  done
  # then re-run --merge as above

Output per job:  zslice_profiles_bgcdia_100m_<scenario>.npz
                 zslice_profiles_bgcdia_offshore_<scenario>.npz
Output of merge: zslice_profiles_bgcdia_100m.npz
                 zslice_profiles_bgcdia_100m_coastal.npz
                 zslice_profiles_bgcdia_offshore.npz
                 zslice_profiles_bgcdia_offshore_coastal.npz
"""

import argparse
import glob
import numpy as np
from netCDF4 import Dataset

ZSLICE_ROOT       = '/data/project1/minnaho/swel/zslicefull'
GRD               = '../plot/mc60_grd.nc'
COASTAL_MASK_FILE = '../plot/coastal_mask.nc'
SCENARIOS = ['tidesampwec', 'tidesnowec', 'notidesnowec']
GROUPS    = ['100m', 'offshore']

DIAT_LIM_VARS    = ['DIAT_N_LIM', 'DIAT_FE_LIM', 'DIAT_PO4_LIM',
                     'DIAT_SIO3_LIM', 'DIAT_LIGHT_LIM', 'DIAT_P_LIM']
SP_LIM_VARS      = ['SP_N_LIM', 'SP_FE_LIM', 'SP_PO4_LIM',
                     'SP_LIGHT_LIM', 'SP_P_LIM']
DIAT_UPTAKE_VARS = ['DIAT_NO3_UPTAKE', 'DIAT_NH4_UPTAKE', 'DIAT_NO2_UPTAKE',
                     'DIAT_SI_UPTAKE']
SP_UPTAKE_VARS   = ['SP_NO3_UPTAKE', 'SP_NH4_UPTAKE', 'SP_NO2_UPTAKE']
PAR_VARS         = ['PAR']
VARS = DIAT_LIM_VARS + SP_LIM_VARS + DIAT_UPTAKE_VARS + SP_UPTAKE_VARS + PAR_VARS

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('scenario', nargs='?', choices=SCENARIOS)
parser.add_argument('--merge', action='store_true')
parser.add_argument('--vars', nargs='+', default=None,
                     help='Restrict accumulation to these vars (e.g. --vars PAR) '
                          'instead of the full VARS list -- for adding a new '
                          'variable without re-reading/re-accumulating every '
                          'other variable. Output is merged into the existing '
                          'per-scenario npz, not overwritten.')
args = parser.parse_args()

if not args.merge and args.scenario is None:
    parser.error('Provide a scenario, or --merge')

if args.vars is not None:
    VARS = [v for v in VARS if v in args.vars]

# ---------------------------------------------------------------------------
# Merge mode
# ---------------------------------------------------------------------------
if args.merge:
    for group in GROUPS:
        out = {}
        for scen in SCENARIOS:
            fname = f'zslice_profiles_bgcdia_{group}_{scen}.npz'
            try:
                d = np.load(fname, allow_pickle=False)
            except FileNotFoundError:
                print(f'missing {fname} — skipping')
                continue
            print(f'loaded {fname}  ({len(d.files)} keys)')
            for k in d.files:
                if k not in out or k != 'depth':
                    out[k] = d[k]

        np.savez(f'zslice_profiles_bgcdia_{group}.npz', **out)
        print(f'saved zslice_profiles_bgcdia_{group}.npz')

        coastal_out = {}
        if 'depth' in out:
            coastal_out['depth'] = out['depth']
        for k, v in out.items():
            if k.startswith('coastal_'):
                coastal_out[k[len('coastal_'):]] = v
        np.savez(f'zslice_profiles_bgcdia_{group}_coastal.npz', **coastal_out)
        print(f'saved zslice_profiles_bgcdia_{group}_coastal.npz')
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Single-job mode: one scenario, both groups
# ---------------------------------------------------------------------------
scen = args.scenario

files = sorted(glob.glob(f'{ZSLICE_ROOT}/{scen}/bgcdia/z_mc60_bgc_dia_avg.*.nc'))
print(f'[{scen}] {len(files)} files, vars: {VARS}')

# ---------------------------------------------------------------------------
# Load masks -- one rho mask restricted to h<=100m, one restricted to h>100m,
# each with a coastal-intersected counterpart
# ---------------------------------------------------------------------------
grd_nc   = Dataset(GRD)
mask_rho = np.array(grd_nc['mask_rho']).astype(float)
h        = np.array(grd_nc['h'])
mask_rho[mask_rho == 0] = np.nan

mask_100m = mask_rho.copy()
mask_100m[h > 100] = np.nan

mask_offshore = mask_rho.copy()
mask_offshore[h <= 100] = np.nan

coastal_mask = np.array(Dataset(COASTAL_MASK_FILE)['coastal_mask']).astype(float)
coastal_mask[coastal_mask == 0] = np.nan

coastal_100m = coastal_mask.copy()
coastal_100m[h > 100] = np.nan

coastal_offshore = coastal_mask.copy()
coastal_offshore[h <= 100] = np.nan

# (group, lbl) -> mask array; lbl 'full' = whole domain within that group's
# depth restriction, 'coastal' = also intersected with the 10km coastal mask
GROUP_MASKS = {
    ('100m',     'full'):    mask_100m,
    ('100m',     'coastal'): coastal_100m,
    ('offshore', 'full'):    mask_offshore,
    ('offshore', 'coastal'): coastal_offshore,
}

MASK_BOOLS = {key: np.isfinite(m).reshape(-1) for key, m in GROUP_MASKS.items()}


def _fill_to_nan(arr):
    return np.where(np.abs(arr) > 1e30, np.nan, arr)


# ---------------------------------------------------------------------------
# Accumulate -- single pass over files, both groups updated per file
# ---------------------------------------------------------------------------
sums   = {key: {v: None for v in VARS} for key in GROUP_MASKS}
counts = {key: {v: None for v in VARS} for key in GROUP_MASKS}
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
            for key, m in MASK_BOOLS.items():
                sub = arr_flat[:, :, m]
                cs  = np.nansum(sub, axis=(0, 2))
                cc  = np.sum(~np.isnan(sub), axis=(0, 2))
                if sums[key][v] is None:
                    sums[key][v]   = cs
                    counts[key][v] = cc.astype(np.int64)
                else:
                    sums[key][v]   += cs
                    counts[key][v] += cc

# ---------------------------------------------------------------------------
# Build output -- one npz per group
# ---------------------------------------------------------------------------
for group in GROUPS:
    out = {}
    if depth is not None:
        out['depth'] = depth

    for lbl in ('full', 'coastal'):
        prefix = '' if lbl == 'full' else 'coastal_'
        key = (group, lbl)
        for v in VARS:
            if sums[key][v] is None:
                continue
            prof = np.where(counts[key][v] > 0,
                            sums[key][v] / np.maximum(counts[key][v], 1), np.nan)
            out[f'{prefix}{v}_{scen}'] = prof
            if lbl == 'full':
                print(f'  [{group}] {v}: min={np.nanmin(prof):.3g}  max={np.nanmax(prof):.3g}')

    fname = f'zslice_profiles_bgcdia_{group}_{scen}.npz'
    # merge into the existing per-scenario npz (if present) instead of
    # overwriting it -- lets --vars target just the new variable(s) without
    # clobbering profiles already computed for everything else
    try:
        existing = dict(np.load(fname, allow_pickle=False))
    except FileNotFoundError:
        existing = {}
    existing.update(out)
    np.savez(fname, **existing)
    print(f'saved {fname}')
