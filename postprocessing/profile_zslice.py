"""
Time- and horizontally-averaged depth profiles for every zsliced variable
across all four scenarios.

Reads:
  /data/project1/minnaho/swel/zslicefull/<scenario>/z_mc60_his.*.nc
  /data/project1/minnaho/swel/zslicefull/<scenario>/z_mc60_bgc.*.nc
  /data/project1/minnaho/swel/zslicefull/<scenario>/dia/z_mc60_bgc_dia_avg.*.nc

Output: zslice_profiles.npz
  depth              : (n_z,)     m, negative downward  (157 levels; his/bgc vars)
  depth_dia          : (n_z_dia,) m, negative downward  (101 levels; dia vars)
  <var>_<scenario>          : (n_z,)  full-domain time/space mean
  coastal_<var>_<scenario>  : (n_z,)  10 km coastal mask mean
  h0to50_<var>_<scenario>   : (n_z,)  mean over 0-50 m bathymetry zone
  h50to200_<var>_<scenario> : (n_z,)  mean over 50-200 m bathymetry zone
  h200p_<var>_<scenario>    : (n_z,)  mean over >200 m bathymetry zone

Also writes zslice_profiles_coastal.npz with keys <var>_<scenario> (no prefix)
for backwards compatibility with plot_profile_zslice.py.
"""

import glob
import numpy as np
from netCDF4 import Dataset

ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'
GRD         = '../plot/mc60_grd.nc'
COASTAL_MASK_FILE = '../plot/coastal_mask.nc'
SCENARIOS   = ['tideswec', 'tidesnowec', 'notidesnowec', 'notideswec', 'ampwec']

# Maps scenario label -> zslice directory name when they differ
SCEN_DIRS   = {'ampwec': 'notidesampwec'}
HIS_VARS    = ['ptrace', 'rtrace', 'w', 'rho', 'u', 'v']
BGC_VARS    = ['NO3', 'NH4', 'SPC', 'DIATC', 'DIAZC',
               'SPCHL', 'DIATCHL', 'DIAZCHL', 'O2', 'DIC', 'DOC']
DIA_VARS    = ['TOT_PROD']

grd_nc   = Dataset(GRD)
mask_rho = np.array(grd_nc['mask_rho']).astype(float)
h        = np.array(grd_nc['h'])
mask_rho[mask_rho == 0] = np.nan
mask_u   = mask_rho[:, :-1].copy()
mask_v   = mask_rho[:-1, :].copy()

coastal_mask = np.array(Dataset(COASTAL_MASK_FILE)['coastal_mask']).astype(float)
coastal_mask[coastal_mask == 0] = np.nan


def _make_h_mask(h_lo, h_hi):
    """Ocean mask restricted to bathymetry depth in (h_lo, h_hi]."""
    m = mask_rho.copy()
    m[(h <= h_lo) | (h > h_hi)] = np.nan
    return m


# 'full' key uses None (full ocean mask via _get_mask); other keys use explicit masks
ALL_MASKS = {
    'full':     None,
    'coastal':  coastal_mask,
    'h0to50':   _make_h_mask(0,   50),
    'h50to200': _make_h_mask(50,  200),
    'h200p':    _make_h_mask(200, np.inf),
}


def _fill_to_nan(arr):
    return np.where(np.abs(arr) > 1e30, np.nan, arr)


def _get_mask(var, zone_mask=None):
    if zone_mask is not None:
        if var == 'u':
            return zone_mask[:, :-1]
        if var == 'v':
            return zone_mask[:-1, :]
        return zone_mask
    if var == 'u':
        return mask_u
    if var == 'v':
        return mask_v
    return mask_rho


def accumulate_files(files, vars):
    """
    Open each file once; read each variable once; apply all masks in one pass.
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
                for lbl, zmask in ALL_MASKS.items():
                    marr = arr * _get_mask(v, zmask)[None, None, :, :]
                    cs   = np.nansum(marr, axis=(0, 2, 3))
                    cc   = np.sum(~np.isnan(marr), axis=(0, 2, 3))
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


out = {}
for scen in SCENARIOS:
    scen_dir  = SCEN_DIRS.get(scen, scen)
    his_files = sorted(glob.glob(f'{ZSLICE_ROOT}/{scen_dir}/z_mc60_his.*.nc'))
    bgc_files = sorted(glob.glob(f'{ZSLICE_ROOT}/{scen_dir}/z_mc60_bgc.*.nc'))
    dia_files = sorted(glob.glob(f'{ZSLICE_ROOT}/{scen_dir}/dia/z_mc60_bgc_dia_avg.*.nc'))
    print(f'[{scen}] {len(his_files)} his, {len(bgc_files)} bgc, {len(dia_files)} dia files')

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

    scen_out = {k: v for k, v in out.items()
                if k.endswith(f'_{scen}') or k in ('depth', 'depth_dia')}
    np.savez(f'zslice_profiles_{scen}.npz', **scen_out)
    print(f'saved zslice_profiles_{scen}.npz')

np.savez('zslice_profiles.npz', **out)
print('saved zslice_profiles.npz')

# write coastal NPZ with unprefixed keys for backwards compatibility
coastal_out = {'depth': out['depth']}
if 'depth_dia' in out:
    coastal_out['depth_dia'] = out['depth_dia']
for k, v in out.items():
    if k.startswith('coastal_'):
        coastal_out[k[len('coastal_'):]] = v
np.savez('zslice_profiles_coastal.npz', **coastal_out)
print('saved zslice_profiles_coastal.npz')
