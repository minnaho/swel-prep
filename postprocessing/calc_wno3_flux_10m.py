import os
import glob
import numpy as np
from netCDF4 import Dataset
from datetime import datetime

FILL_THRESH = 0.9e33

grd = '../mc60_grd.nc'

grdnc = Dataset(grd,'r')
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc==0] = np.nan

scenarios = {
    'tides_wec':     ('/data/project3/minnaho/swel/tides/mc60/wec/his/zslice_10m',
                      '/data/project3/minnaho/swel/tides/mc60/wec/bgc/zslice_10m'),
    'tides_nowec':   ('/data/project3/minnaho/swel/tides/mc60/nowec/output/his/zslice_10m',
                      '/data/project3/minnaho/swel/tides/mc60/nowec/output/bgc/zslice_10m'),
    'notides_nowec': ('/data/project3/minnaho/swel/notides/mc60/nowec/output/his/zslice_10m',
                      '/data/project3/minnaho/swel/notides/mc60/nowec/output/bgc/zslice_10m'),
    'notides_wec':   ('/data/project3/minnaho/swel/notides/mc60/wec/rerun/his/zslice_10m',
                      '/data/project3/minnaho/swel/notides/mc60/wec/rerun/bgc/zslice_10m'),
}


def get_matched_pairs(his_dir, bgc_dir):
    his_files = sorted(glob.glob(os.path.join(his_dir, 'z_mc60_his.*.nc')))
    bgc_files = [f for f in sorted(glob.glob(os.path.join(bgc_dir, 'z_mc60_bgc.*.nc')))
                 if 'dia_avg' not in f]
    his_map = {os.path.basename(f).replace('z_mc60_his.', '').replace('.nc', ''): f
               for f in his_files}
    bgc_map = {os.path.basename(f).replace('z_mc60_bgc.', '').replace('.nc', ''): f
               for f in bgc_files}
    common = sorted(set(his_map) & set(bgc_map))
    return [(his_map[s], bgc_map[s], s) for s in common]


def load_masked(path, var):
    data = np.array(Dataset(path)[var][:], dtype=np.float32)
    data[data > FILL_THRESH] = np.nan
    data = data*masknc
    return data


def compute_flux(his_dir, bgc_dir):
    pairs = get_matched_pairs(his_dir, bgc_dir)
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
    del w_all, no3_all

    # Time series: spatial mean at every record
    time_series = np.nanmean(flux, axis=(1, 2))  # (total_nt,)

    # Read ocean_time from original his files for the time axis
    orig_his_dir = os.path.dirname(his_dir)
    ocean_times  = []
    for stamp, _ in file_meta:
        orig = os.path.join(orig_his_dir, f'mc60_his.{stamp}.nc')
        ocean_times.extend(np.array(Dataset(orig)['ocean_time'][:]).tolist())

    # PDF: all instantaneous w'NO3' values across all time steps and grid points
    all_flux = flux[~np.isnan(flux)]

    plo, phi = np.nanpercentile(all_flux, [0.5, 99.5])
    counts, edges = np.histogram(all_flux, bins=200,
                                 range=(plo, phi), density=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    return dict(
        time_series = time_series,
        ocean_times = np.array(ocean_times),   # seconds since 1995-01-01
        bin_centers = bin_centers,
        pdf         = counts,
        mean_flux   = float(np.nanmean(all_flux)),
    )

def compute_env(his_dir, bgc_dir):
    pairs = get_matched_pairs(his_dir, bgc_dir)
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
    orig_his_dir = os.path.dirname(his_dir)
    ocean_times  = []
    for stamp, _ in file_meta:
        orig = os.path.join(orig_his_dir, f'mc60_his.{stamp}.nc')
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


for name, (his_dir, bgc_dir) in scenarios.items():
    print(f'Processing {name}...')
    result = compute_flux(his_dir, bgc_dir)
    resultenv = compute_env(his_dir, bgc_dir)
    outfile = f'wno3_flux_{name}.npz'
    np.savez(outfile, **result)
    outfileenv = f'wno3_env_{name}.npz'
    np.savez(outfileenv, **resultenv)
    print(f'  mean flux = {result["mean_flux"]:.4e} mmol N m-2 s-1')
    print(f'  saved -> {outfile}')
