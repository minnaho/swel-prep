"""
Depth profiles of TOT_PROD (from dia/ zslice files) and Akt, Akv (from ak/
zslice files).  Reads directly from the zsliced NetCDF files and accumulates
profiles in memory — no pre-computed NPZ required.

One PNG per variable (3 total), each with two panels:
  left  — full-domain mean
  right — coastal (10 km mask) mean

Scenarios that have no files for a given variable are silently skipped.
"""

import glob
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import scenario_style as ss

ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'
GRD         = 'mc60_grd.nc'
MASK_FILE   = 'coastal_mask.nc'
NPZ_CACHE   = './figs/profile_ak_npp_cache.npz'

SCENARIOS = ['tideswec', 'tidesnowec', 'notidesnowec', 'notideswec', 'ampwec', 'tidesampwec']
SCEN_DIRS = {'ampwec': 'notidesampwec'}   # label → zslice directory (tidesampwec matches its own zslice dir name)

LABELS = ss.LABELS
COLORS = ss.COLORS

#VARS_DIA = ['TOT_PROD']
VARS_DIA = ['TOT_PROD']
VARS_AK  = ['Akt','Akv']   
#VARS_AK  = []   # skip until zslice_ak.py has finished running

#DEPTH_YLIM = {
#    'TOT_PROD': [-40, 0],
#    'Akt':      [-120, 0],
#    'Akv':      [-120, 0],
#}
DEPTH_YLIM = {
    'TOT_PROD': [-40, 0],
    'Akt':      [-40, 0],
    'Akv':      [-40, 0],
}
VAR_UNITS = {
    'TOT_PROD': r'mmol C m$^{-3}$ d$^{-1}$',
    'Akt':      r'm$^2$ s$^{-1}$',
    'Akv':      r'm$^2$ s$^{-1}$',
}
VAR_LABEL = {
    'TOT_PROD': 'NPP',
    'Akt':      'Akt',
    'Akv':      'Akv',
}

# ── masks ─────────────────────────────────────────────────────────────────────
def _make_masks(nc_path, var='mask_rho'):
    m = np.array(Dataset(nc_path)[var]).astype(float)
    m[m == 0] = np.nan
    return m

mask_full    = _make_masks(GRD, 'mask_rho')
mask_coastal = _make_masks(MASK_FILE, 'coastal_mask')


def _fill_to_nan(arr):
    return np.where(np.abs(arr) > 1e30, np.nan, arr)


def accumulate(files, varnames, mask):
    """
    Accumulate time/space mean profiles from a list of zsliced NetCDF files.
    Handles both 4-D (time, depth, eta, xi) and 3-D (depth, eta, xi) variables.
    Returns (depth_array, {varname: profile}) or (None, {}) if files is empty.
    """
    sums   = {v: None for v in varnames}
    counts = {v: None for v in varnames}
    depth  = None
    for f in files:
        with Dataset(f) as nc:
            if depth is None:
                depth = np.array(nc.variables['depth'][:])
            for v in varnames:
                if v not in nc.variables:
                    continue
                arr = _fill_to_nan(np.array(nc.variables[v][:]))
                if arr.ndim == 3:            # (depth, eta, xi) — dia files
                    arr = arr * mask
                    cs  = np.nansum(arr,         axis=(1, 2))
                    cc  = np.sum(~np.isnan(arr), axis=(1, 2))
                else:                        # (time, depth, eta, xi) — ak/his files
                    arr = arr * mask[None, None, :, :]
                    cs  = np.nansum(arr,         axis=(0, 2, 3))
                    cc  = np.sum(~np.isnan(arr), axis=(0, 2, 3))
                if sums[v] is None:
                    sums[v], counts[v] = cs, cc.astype(np.int64)
                else:
                    sums[v]   += cs
                    counts[v] += cc
    profiles = {v: np.where(counts[v] > 0,
                            sums[v] / np.maximum(counts[v], 1), np.nan)
                for v in varnames if sums[v] is not None}
    return depth, profiles


# ── accumulate per scenario (or load from cache) ──────────────────────────────
results_full    = {}   # {scen: {var: profile}}
results_coastal = {}
depths_dia = None
depths_ak  = None

if os.path.exists(NPZ_CACHE):
    print(f'loading cached profiles from {NPZ_CACHE}')
    d = np.load(NPZ_CACHE)
    depths_dia = d['depths_dia'] if d['depths_dia'].size else None
    depths_ak  = d['depths_ak']  if d['depths_ak'].size  else None
    results_full    = {scen: {} for scen in SCENARIOS}
    results_coastal = {scen: {} for scen in SCENARIOS}
    for key in d.files:
        if key in ('depths_dia', 'depths_ak'):
            continue
        # key format: {full|coastal}_{scen}_{var}  (scen names have no underscores)
        kind, scen, var = key.split('_', 2)
        if scen not in results_full:
            continue
        if kind == 'full':
            results_full[scen][var] = d[key]
        else:
            results_coastal[scen][var] = d[key]
else:
    for scen in SCENARIOS:
        sd = SCEN_DIRS.get(scen, scen)

        dia_files = sorted(glob.glob(f'{ZSLICE_ROOT}/{sd}/dia/z_mc60_bgc_dia_avg.*.nc'))
        ak_files  = sorted(glob.glob(f'{ZSLICE_ROOT}/{sd}/ak/z_mc60_his.*.nc'))
        print(f'[{scen}] {len(dia_files)} dia files, {len(ak_files)} ak files')

        d_dia, p_dia_full    = accumulate(dia_files, VARS_DIA, mask_full)
        _,     p_dia_coastal = accumulate(dia_files, VARS_DIA, mask_coastal)
        if d_dia is not None:
            depths_dia = d_dia

        d_ak, p_ak_full    = accumulate(ak_files, VARS_AK, mask_full)
        _,    p_ak_coastal = accumulate(ak_files, VARS_AK, mask_coastal)
        if d_ak is not None:
            depths_ak = d_ak

        results_full[scen]    = {**p_dia_full,    **p_ak_full}
        results_coastal[scen] = {**p_dia_coastal, **p_ak_coastal}

    os.makedirs('./figs', exist_ok=True)
    save_dict = {
        'depths_dia': depths_dia if depths_dia is not None else np.array([]),
        'depths_ak':  depths_ak  if depths_ak  is not None else np.array([]),
    }
    for scen in SCENARIOS:
        for var, arr in results_full[scen].items():
            save_dict[f'full_{scen}_{var}'] = arr
        for var, arr in results_coastal[scen].items():
            save_dict[f'coastal_{scen}_{var}'] = arr
    np.savez(NPZ_CACHE, **save_dict)
    print(f'saved cache -> {NPZ_CACHE}')


# ── plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 12})
os.makedirs('./figs', exist_ok=True)

all_vars = VARS_DIA + VARS_AK
depth_for = {}
for v in VARS_DIA:
    depth_for[v] = depths_dia
for v in VARS_AK:
    depth_for[v] = depths_ak

for var in all_vars:
    depth = depth_for[var]
    scale = 86400.0 if var == 'TOT_PROD' else 1.0

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

    for scen in SCENARIOS:
        prof_full    = results_full[scen].get(var)
        prof_coastal = results_coastal[scen].get(var)
        if prof_full is None and prof_coastal is None:
            continue
        lw = ss.lw(scen, base_lw=1.5)
        if prof_full is not None:
            axL.plot(prof_full * scale, depth,
                     color=COLORS[scen], linestyle=ss.ls(scen), label=LABELS[scen], linewidth=lw)
        if prof_coastal is not None:
            axR.plot(prof_coastal * scale, depth,
                     color=COLORS[scen], linestyle=ss.ls(scen), linewidth=lw)

    ylim = DEPTH_YLIM[var]
    axL.set_ylim(ylim)

    xlabel = f'{VAR_LABEL[var]} [{VAR_UNITS[var]}]'
    for ax in (axL, axR):
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)

    axL.set_ylabel('depth (m)')
    axL.set_title('full domain')
    axR.set_title('coastal (10 km)')
    axL.legend(loc='best', fontsize=9)

    plt.tight_layout()
    out = f'./figs/profile_ak_npp_{var}.png'
    plt.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'saved -> {out}')
