"""
h>100m-restricted version of calc_wno3_flux_10m.py -- same w'NO3' flux
calculation at the 10 m zslice level, but the horizontal domain is masked to
offshore (h>100m) before any spatial statistic (time series mean, envelope
max/min, PDF) is computed, same masking approach as calc_wno3_flux_20m_offshore.py
(mask[h <= 100] = np.nan applied to the base mask_rho land mask). Offshore
counterpart of calc_wno3_flux_10m_100m.py, which is the h<=100m (shelf)
restriction -- same DEPTH_IDX=10 approach (no START_TRIM), matching that
script's structure rather than the later TARGET_DEPTH/START_TRIM convention
calc_wno3_flux_20m_100m.py/_30m_100m.py use.

Output: wno3_flux_offshore_<scenario>.npz, wno3_env_offshore_<scenario>.npz
"""

import os
import glob
import numpy as np
from netCDF4 import Dataset

FILL_THRESH = 0.9e33

grd = '../mc60_grd.nc'

grdnc = Dataset(grd,'r')
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc==0] = np.nan
h_nc = np.array(grdnc.variables['h'])
masknc[h_nc <= 100] = np.nan   # restrict to h>100m (offshore) only

# 10 m level from the general zslicefull product -- same source as
# calc_wno3_flux_10m.py, just spatially restricted.
ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'
DEPTH_IDX = 10   # confirmed depth[10] == -10.0 m

# key: (zslice subdirectory alias, raw his dir for ocean_time lookup) --
# same tuple shape/values as offshore_flux_zslice.py's scenarios dict
scenarios = {
    'notidesnowec': ('notidesnowec',  '/data/project3/minnaho/swel/notides/mc60/nowec/his'),
    'ampwec':       ('notidesampwec', '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything'),
    'tidesnowec':   ('tidesnowec',    '/data/project3/minnaho/swel/tides/mc60/nowec/output/his'),
    'tidesampwec':  ('tidesampwec',   '/data/project3/minnaho/swel/tides/mc60/ampwec/everything'),
}

# tidesampwec's raw source has a trailing 1-timestep file
# (...20190429110056) whose zslice output has no time dimension at all
TIDESAMPWEC_EXCLUDE = ('20190429110056',)


def get_matched_pairs(zslice_alias):
    zslice_dir = os.path.join(ZSLICE_ROOT, zslice_alias)
    his_files = sorted(glob.glob(os.path.join(zslice_dir, 'z_mc60_his.*.nc')))
    bgc_files = [f for f in sorted(glob.glob(os.path.join(zslice_dir, 'bgc', 'z_mc60_bgc.*.nc')))
                 if 'dia_avg' not in f]
    if zslice_alias == 'tidesampwec':
        his_files = [f for f in his_files if not any(x in f for x in TIDESAMPWEC_EXCLUDE)]
        bgc_files = [f for f in bgc_files if not any(x in f for x in TIDESAMPWEC_EXCLUDE)]
    his_map = {os.path.basename(f).replace('z_mc60_his.', '').replace('.nc', ''): f
               for f in his_files}
    bgc_map = {os.path.basename(f).replace('z_mc60_bgc.', '').replace('.nc', ''): f
               for f in bgc_files}
    common = sorted(set(his_map) & set(bgc_map))
    return [(his_map[s], bgc_map[s], s) for s in common]


def load_masked(path, var):
    data = np.array(Dataset(path)[var][:, DEPTH_IDX, :, :], dtype=np.float32)
    data[data > FILL_THRESH] = np.nan
    data = data*masknc
    return data


def compute_flux(zslice_alias, raw_his_dir):
    pairs = get_matched_pairs(zslice_alias)
    print(f'  {len(pairs)} file pairs found')

    # Load all files, tracking per-file record counts for the time series
    w_list, no3_list, file_meta = [], [], []
    for hf, bf, stamp in pairs:
        w   = load_masked(hf, 'w')
        no3 = load_masked(bf, 'NO3')
        file_meta.append((stamp, w.shape[0]))
        w_list.append(w)
        no3_list.append(no3)

    w_all   = np.concatenate(w_list,   axis=0)  # (total_nt, eta_rho, xi_rho)
    no3_all = np.concatenate(no3_list, axis=0)
    del w_list, no3_list

    # Time mean at each grid point (axis=0 is time)
    mean_w   = np.nanmean(w_all,   axis=0)  # (eta_rho, xi_rho)
    mean_no3 = np.nanmean(no3_all, axis=0)  # (eta_rho, xi_rho)

    flux = (w_all - mean_w) * (no3_all - mean_no3)  # (total_nt, eta_rho, xi_rho)
    # raw (no mean removal) product, for comparison against the parent
    # smode200 solution's raw w*NO3 panel (plot_smode_wno3_10m.py)
    raw_time_series = np.nanmean(w_all * no3_all, axis=(1, 2))  # (total_nt,)
    del w_all, no3_all

    # Time series: spatial mean at every record
    time_series = np.nanmean(flux, axis=(1, 2))  # (total_nt,)

    # Read ocean_time from original his files for the time axis -- zslicefull
    # files have no ocean_time variable at all
    ocean_times  = []
    for stamp, _ in file_meta:
        orig = os.path.join(raw_his_dir, f'mc60_his.{stamp}.nc')
        ocean_times.extend(np.array(Dataset(orig)['ocean_time'][:]).tolist())

    # PDF: all instantaneous w'NO3' values across all time steps and grid points
    all_flux = flux[~np.isnan(flux)]

    plo, phi = np.nanpercentile(all_flux, [0.5, 99.5])
    counts, edges = np.histogram(all_flux, bins=200,
                                 range=(plo, phi), density=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    return dict(
        time_series     = time_series,
        raw_time_series = raw_time_series,
        ocean_times     = np.array(ocean_times),   # seconds since 1995-01-01
        bin_centers     = bin_centers,
        pdf             = counts,
        mean_flux       = float(np.nanmean(all_flux)),
    )

def compute_env(zslice_alias, raw_his_dir):
    pairs = get_matched_pairs(zslice_alias)
    print(f'  {len(pairs)} file pairs found')

    w_list, no3_list, file_meta = [], [], []
    for hf, bf, stamp in pairs:
        w   = load_masked(hf, 'w')
        no3 = load_masked(bf, 'NO3')
        file_meta.append((stamp, w.shape[0]))
        w_list.append(w)
        no3_list.append(no3)

    w_all   = np.concatenate(w_list,   axis=0)
    no3_all = np.concatenate(no3_list, axis=0)
    del w_list, no3_list

    # Time mean at each grid point
    mean_w   = np.nanmean(w_all,   axis=0)
    mean_no3 = np.nanmean(no3_all, axis=0)

    flux = (w_all - mean_w) * (no3_all - mean_no3)
    del w_all, no3_all

    # --- ENVELOPE CALCULATION ---
    # Max and Min at every record across the entire spatial domain
    time_series_max  = np.nanmax(flux, axis=(1, 2))  # (total_nt,)
    time_series_min  = np.nanmin(flux, axis=(1, 2))  # (total_nt,)
    time_series_mean = np.nanmean(flux, axis=(1, 2)) # (total_nt,)

    # Read ocean_time from original his files
    ocean_times  = []
    for stamp, _ in file_meta:
        orig = os.path.join(raw_his_dir, f'mc60_his.{stamp}.nc')
        ocean_times.extend(np.array(Dataset(orig)['ocean_time'][:]).tolist())

    # PDF calculation
    all_flux = flux[~np.isnan(flux)]
    plo, phi = np.nanpercentile(all_flux, [0.5, 99.5])
    counts, edges = np.histogram(all_flux, bins=200, range=(plo, phi), density=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    return dict(
        ts_max      = time_series_max,
        ts_min      = time_series_min,
        ts_mean     = time_series_mean,
        ocean_times = np.array(ocean_times),
        bin_centers = bin_centers,
        pdf         = counts,
        mean_flux   = float(np.nanmean(all_flux)),
    )


for name, (zslice_alias, raw_his_dir) in scenarios.items():
    print(f'Processing {name}...')
    result = compute_flux(zslice_alias, raw_his_dir)
    resultenv = compute_env(zslice_alias, raw_his_dir)
    outfile = f'wno3_flux_offshore_{name}.npz'
    np.savez(outfile, **result)
    outfileenv = f'wno3_env_offshore_{name}.npz'
    np.savez(outfileenv, **resultenv)
    print(f'  mean flux = {result["mean_flux"]:.4e} mmol N m-2 s-1')
    print(f'  saved -> {outfile}')
