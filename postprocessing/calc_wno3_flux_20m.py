import os
import glob
import numpy as np
from netCDF4 import Dataset

FILL_THRESH  = 0.9e33
TARGET_DEPTH = -20   # m — select this level from the zslicefull depth array

ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'

grdnc  = Dataset('../mc60_grd.nc', 'r')
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc == 0] = np.nan

# (zslicefull_scenario_dir, orig_his_dir_for_ocean_time)
scenarios = {
    'tides_wec':     (f'{ZSLICE_ROOT}/tideswec',
                      '/data/project3/minnaho/swel/tides/mc60/wec/his'),
    'tides_nowec':   (f'{ZSLICE_ROOT}/tidesnowec',
                      '/data/project3/minnaho/swel/tides/mc60/nowec/output/his'),
    'notides_nowec': (f'{ZSLICE_ROOT}/notidesnowec',
                      '/data/project3/minnaho/swel/notides/mc60/nowec/output/his'),
    'notides_wec':   (f'{ZSLICE_ROOT}/notideswec',
                      '/data/project3/minnaho/swel/notides/mc60/wec/rerun/his'),
    'ampwec':        (f'{ZSLICE_ROOT}/notidesampwec',
                      '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything'),
}


def get_matched_pairs(zslice_dir):
    """Match z_mc60_his and z_mc60_bgc files by timestamp."""
    his_files = sorted(glob.glob(os.path.join(zslice_dir, 'z_mc60_his.*.nc')))
    bgc_files = [f for f in sorted(glob.glob(os.path.join(zslice_dir, 'z_mc60_bgc.*.nc')))
                 if 'dia_avg' not in f]
    his_map = {os.path.basename(f).replace('z_mc60_his.', '').replace('.nc', ''): f
               for f in his_files}
    bgc_map = {os.path.basename(f).replace('z_mc60_bgc.', '').replace('.nc', ''): f
               for f in bgc_files}
    common = sorted(set(his_map) & set(bgc_map))
    return [(his_map[s], bgc_map[s], s) for s in common]


def get_depth_idx(nc):
    """Return the index of TARGET_DEPTH in the file's depth array."""
    depth = np.array(nc.variables['depth'][:])
    idx   = int(np.argmin(np.abs(depth - TARGET_DEPTH)))
    return idx


def load_masked(path, var):
    """Load var at TARGET_DEPTH from a zslicefull file → (time, eta, xi)."""
    with Dataset(path) as nc:
        idx  = get_depth_idx(nc)
        data = np.array(nc.variables[var][:, idx, :, :], dtype=np.float32)
    data[data > FILL_THRESH] = np.nan
    data = data * masknc[None, :, :]
    return data


def read_ocean_times(file_meta, orig_his_dir):
    ocean_times = []
    for stamp, _ in file_meta:
        orig = os.path.join(orig_his_dir, f'mc60_his.{stamp}.nc')
        ocean_times.extend(np.array(Dataset(orig)['ocean_time'][:]).tolist())
    return np.array(ocean_times)


def compute_flux(zslice_dir, orig_his_dir):
    pairs = get_matched_pairs(zslice_dir)
    print(f'  {len(pairs)} file pairs found')

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

    mean_w   = np.nanmean(w_all,   axis=0)
    mean_no3 = np.nanmean(no3_all, axis=0)
    flux     = (w_all - mean_w) * (no3_all - mean_no3)
    del w_all, no3_all

    time_series = np.nanmean(flux, axis=(1, 2))
    ocean_times = read_ocean_times(file_meta, orig_his_dir)

    all_flux = flux[~np.isnan(flux)]
    plo, phi = np.nanpercentile(all_flux, [0.5, 99.5])
    counts, edges = np.histogram(all_flux, bins=200, range=(plo, phi), density=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    return dict(
        time_series = time_series,
        ocean_times = ocean_times,
        bin_centers = bin_centers,
        pdf         = counts,
        mean_flux   = float(np.nanmean(all_flux)),
    )


def compute_env(zslice_dir, orig_his_dir):
    pairs = get_matched_pairs(zslice_dir)
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

    mean_w   = np.nanmean(w_all,   axis=0)
    mean_no3 = np.nanmean(no3_all, axis=0)
    flux     = (w_all - mean_w) * (no3_all - mean_no3)
    del w_all, no3_all

    time_series_max  = np.nanmax(flux,  axis=(1, 2))
    time_series_min  = np.nanmin(flux,  axis=(1, 2))
    time_series_mean = np.nanmean(flux, axis=(1, 2))
    ocean_times      = read_ocean_times(file_meta, orig_his_dir)

    all_flux = flux[~np.isnan(flux)]
    plo, phi = np.nanpercentile(all_flux, [0.5, 99.5])
    counts, edges = np.histogram(all_flux, bins=200, range=(plo, phi), density=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    return dict(
        ts_max      = time_series_max,
        ts_min      = time_series_min,
        ts_mean     = time_series_mean,
        ocean_times = ocean_times,
        bin_centers = bin_centers,
        pdf         = counts,
        mean_flux   = float(np.nanmean(all_flux)),
    )


for name, (zslice_dir, orig_his_dir) in scenarios.items():
    print(f'Processing {name}...')
    result    = compute_flux(zslice_dir, orig_his_dir)
    resultenv = compute_env(zslice_dir, orig_his_dir)
    outfile    = f'wno3_flux_20m_{name}.npz'
    outfileenv = f'wno3_env_20m_{name}.npz'
    np.savez(outfile,    **result)
    np.savez(outfileenv, **resultenv)
    print(f'  mean flux = {result["mean_flux"]:.4e} mmol N m-2 s-1')
    print(f'  saved -> {outfile}')
