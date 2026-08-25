"""
Daily-averaged bar(w'NO3') version of calc_wno3_flux_20m_100m.py -- same 20 m
zslice level, same h<=100m (shelf) domain restriction, but the eddy
decomposition and averaging period are both DAILY instead of using the
whole multi-day record as a single background state:

  calc_wno3_flux_20m_100m.py (raw, hourly):
    w_bar   = mean(w[all t])          -- one background field for the WHOLE record
    no3_bar = mean(NO3[all t])
    w'(t)   = w(t)   - w_bar          -- hourly anomaly, every hour
    NO3'(t) = NO3(t) - no3_bar
    flux(t) = w'(t) * NO3'(t)         -- one flux field PER HOUR

  calc_wno3_flux_20m_100m_daily.py (this script, bar(w'NO3'), daily):
    for each calendar day d:
      w_bar_d   = mean(w[t in d])     -- background RESETS every day
      no3_bar_d = mean(NO3[t in d])
      w'(t)     = w(t)   - w_bar_d    -- hourly anomaly from THAT DAY's mean
      NO3'(t)   = NO3(t) - no3_bar_d
      bar(w'NO3')_d = mean_{t in d}( w'(t) * NO3'(t) )   -- one flux field PER DAY

This is the standard Reynolds/tidal-flux block-decomposition: using a daily
background isolates sub-daily (tidal/bore-scale) eddy correlation and
filters out any multi-day/spring-neap trend from what counts as "eddy" --
unlike the whole-record-mean version, whose w'/NO3' can carry slow drift
that isn't really a bore-scale fluctuation. Chosen over the simpler
alternative (keep the whole-record decomposition, just block-average the
resulting hourly flux into daily bins) per explicit confirmation.

compute_flux/compute_env are merged into one compute_daily_flux() here
(unlike the base script, which loads w/NO3 twice, once per function) since
both statistics need the same per-day binning of the same w_all/no3_all
arrays -- no reason to read+concatenate the zslice files twice per scenario.

Days are grouped by calendar date (UTC, from ocean_time = seconds since
1995-01-01), not by a fixed 24-sample stride -- robust to the few-second
per-step timestamp jitter already documented elsewhere in this project
(ROMS output isn't exactly on the hour) and to partial boundary days at the
start/end of a scenario's record, which are just averaged over however many
hours they actually have.

Output: wno3_flux_20m_100m_daily_<scenario>.npz, wno3_env_20m_100m_daily_<scenario>.npz
  time_series/ts_mean, raw_time_series, ts_max, ts_min -- (n_days,) each,
    one value per calendar day (spatial mean/max/min of that day's
    bar(w'NO3') field)
  ocean_times -- (n_days,), mean ocean_time of each day's samples
  bin_centers, pdf -- PDF pooled over every unmasked pixel on every day
    (n_days * n_shelf_cells samples, not n_days*24x that of the hourly
    script, since each day contributes one already-time-averaged field)
  mean_flux -- scalar, mean of bar(w'NO3') over all days and pixels
"""

import os
import glob
import datetime
import numpy as np
from netCDF4 import Dataset

FILL_THRESH  = 0.9e33
TARGET_DEPTH = -20   # m — select this level from the zslicefull depth array
OCEAN_TIME_EPOCH = datetime.datetime(1995, 1, 1)

ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'

grdnc  = Dataset('../mc60_grd.nc', 'r')
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc == 0] = np.nan
h_nc = np.array(grdnc.variables['h'])
masknc[h_nc > 100] = np.nan   # restrict to h<=100m (shelf) only

# (zslicefull_scenario_dir, orig_his_dir_for_ocean_time) -- same 4 scenarios
# as offshore_flux_zslice.py / calc_wtrace_flux.py (swapped from the original
# tides_wec/notides_wec 1x-WEC pair to the 2.5x-WEC ampwec/tidesampwec pair)
scenarios = {
    'notidesnowec': (f'{ZSLICE_ROOT}/notidesnowec',
                     '/data/project3/minnaho/swel/notides/mc60/nowec/his'),
    'tidesnowec':   (f'{ZSLICE_ROOT}/tidesnowec',
                     '/data/project3/minnaho/swel/tides/mc60/nowec/output/his'),
    'ampwec':       (f'{ZSLICE_ROOT}/notidesampwec',
                     '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything'),
    'tidesampwec':  (f'{ZSLICE_ROOT}/tidesampwec',
                     '/data/project3/minnaho/swel/tides/mc60/ampwec/everything'),
}

# tidesampwec's raw source has a trailing 1-timestep file
# (...20190429110056) whose zslice output has no time dimension at all
TIDESAMPWEC_EXCLUDE = ('20190429110056',)


def get_matched_pairs(zslice_dir):
    """Match z_mc60_his and z_mc60_bgc files by timestamp."""
    # bgc files live in the scenario's bgc/ subdir, not the zslice root
    # (same bug already fixed in offshore_flux_zslice.py)
    his_files = sorted(glob.glob(os.path.join(zslice_dir, 'z_mc60_his.*.nc')))
    bgc_files = [f for f in sorted(glob.glob(os.path.join(zslice_dir, 'bgc', 'z_mc60_bgc.*.nc')))
                 if 'dia_avg' not in f]
    if os.path.basename(zslice_dir.rstrip('/')) == 'tidesampwec':
        his_files = [f for f in his_files if not any(x in f for x in TIDESAMPWEC_EXCLUDE)]
        bgc_files = [f for f in bgc_files if not any(x in f for x in TIDESAMPWEC_EXCLUDE)]
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


def day_bins(ocean_times):
    """Group time-axis indices by calendar date (UTC, from ocean_time
    seconds since 1995-01-01). Returns (sorted unique dates, {date: [idx]})."""
    dates = [(OCEAN_TIME_EPOCH + datetime.timedelta(seconds=float(t))).date()
             for t in ocean_times]
    days = sorted(set(dates))
    day_idx = {d: [i for i, dd in enumerate(dates) if dd == d] for d in days}
    return days, day_idx


def compute_daily_flux(zslice_dir, orig_his_dir):
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

    ocean_times = read_ocean_times(file_meta, orig_his_dir)
    days, day_idx = day_bins(ocean_times)
    n_days = len(days)
    print(f'  {n_days} calendar days in record')

    time_series     = np.full(n_days, np.nan)
    raw_time_series = np.full(n_days, np.nan)
    ts_max          = np.full(n_days, np.nan)
    ts_min          = np.full(n_days, np.nan)
    ts_mean         = np.full(n_days, np.nan)
    day_ocean_times = np.full(n_days, np.nan)
    all_flux_chunks = []

    for di, d in enumerate(days):
        idx = day_idx[d]
        day_ocean_times[di] = np.mean(ocean_times[idx])

        # a single-sample day has an anomaly that is identically zero by
        # construction (mean of 1 value = itself) -- that's a degenerate
        # estimate, not a real "zero flux", so leave it NaN rather than
        # report a misleading 0.0. Happens at record start/end when a raw
        # file's hour boundary doesn't line up with midnight.
        if len(idx) < 2:
            print(f'  WARNING: {d} has only {len(idx)} sample(s), skipping '
                  f'(need >=2 for a day-local anomaly)')
            continue

        w_day   = w_all[idx]      # (n_t_day, eta_rho, xi_rho)
        no3_day = no3_all[idx]

        # this day's OWN mean (background resets every day, not the
        # whole-record mean calc_wno3_flux_20m_100m.py uses)
        w_day_mean   = np.nanmean(w_day,   axis=0)   # (eta_rho, xi_rho)
        no3_day_mean = np.nanmean(no3_day, axis=0)

        flux_day = np.nanmean((w_day - w_day_mean) * (no3_day - no3_day_mean), axis=0)
        raw_day  = np.nanmean(w_day * no3_day, axis=0)

        time_series[di]     = np.nanmean(flux_day)
        raw_time_series[di] = np.nanmean(raw_day)
        ts_max[di]          = np.nanmax(flux_day)
        ts_min[di]          = np.nanmin(flux_day)
        ts_mean[di]         = time_series[di]

        all_flux_chunks.append(flux_day[~np.isnan(flux_day)])

    all_flux = np.concatenate(all_flux_chunks)
    plo, phi = np.nanpercentile(all_flux, [0.5, 99.5])
    counts, edges = np.histogram(all_flux, bins=200, range=(plo, phi), density=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    mean_flux = float(np.nanmean(all_flux))

    flux_result = dict(
        time_series     = time_series,
        raw_time_series = raw_time_series,
        ocean_times     = day_ocean_times,
        bin_centers     = bin_centers,
        pdf             = counts,
        mean_flux       = mean_flux,
    )
    env_result = dict(
        ts_max      = ts_max,
        ts_min      = ts_min,
        ts_mean     = ts_mean,
        ocean_times = day_ocean_times,
        bin_centers = bin_centers,
        pdf         = counts,
        mean_flux   = mean_flux,
    )
    return flux_result, env_result


for name, (zslice_dir, orig_his_dir) in scenarios.items():
    print(f'Processing {name}...')
    result, resultenv = compute_daily_flux(zslice_dir, orig_his_dir)
    outfile    = f'wno3_flux_20m_100m_daily_{name}.npz'
    outfileenv = f'wno3_env_20m_100m_daily_{name}.npz'
    np.savez(outfile,    **result)
    np.savez(outfileenv, **resultenv)
    print(f'  mean flux = {result["mean_flux"]:.4e} mmol N m-2 s-1')
    print(f'  saved -> {outfile}')
