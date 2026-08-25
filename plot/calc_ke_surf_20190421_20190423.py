"""
Surface KE spectra, restricted to the 2019-04-21 00:00 -- 2019-04-23 00:00
window (48 hourly samples, half-open interval so the window ends exactly at
2019-04-23 00:00 without including it). Direct structural copy of
calc_ke_surf.py -- same scenarios, masks, and FFT/PSD method -- only the
time selection differs.

calc_ke_surf.py assumes every file contributes a fixed tdim=12 time steps
and concatenates file-by-file in glob order. That doesn't hold here: the
window's start/end fall in the middle of a file, so file selection and
time-index selection are both done against each file's actual ocean_time
values (seconds since 1995-01-01, per this dataset's convention -- verified
via ncdump) rather than trusting filenames or a fixed tdim. Files with zero
overlap with the window are skipped without reading u/v at all; files that
partially overlap are read in full (pyfuncs.rho_uv_angle_surf always reads
a file's whole time dimension) and then sliced down to the in-window steps.

Usage: python -u calc_ke_surf_20190421_20190423.py

Output: ./ke_spectra_comparison_20190421_20190423.npz
  (same key schema as ke_spectra_comparison.npz: freqs, psd_<scen>_masknc,
  psd_<scen>_coastal -- written to a separate file so the full-record
  calc_ke_surf.py output is untouched)
"""

import sys
import glob
import datetime as dt
import numpy as np
from netCDF4 import Dataset
from scipy.fft import fft, fftshift, fftfreq

sys.path.append('/data/project3/minnaho/global/')
import pyfuncs as pf

# ==========================================
# Configuration & File Paths
# ==========================================
grd = '/data/project3/minnaho/project9copy/swel/mc60_grd.nc'

SCENARIOS = {
    'tideswec':     sorted(glob.glob('/data/project3/minnaho/swel/tides/mc60/wec/his/mc60_his.201904*.nc')),
    'tidesnowec':   sorted(glob.glob('/data/project3/minnaho/swel/tides/mc60/nowec/output/his/mc60_his.201904*.nc')),
    'notidesnowec': sorted(glob.glob('/data/project3/minnaho/swel/notides/mc60/nowec/his/mc60_his.201904*.nc')),
    'notideswec':   sorted(glob.glob('/data/project3/minnaho/swel/notides/mc60/wec/rerun/his/mc60_his.201904*.nc')),
    'ampwec':       sorted(glob.glob('/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything/mc60_his.201904*.nc')),
    # excludes mc60_his.20190429110056.nc: a trailing 1-timestep file (all
    # others have 12), which breaks the fixed-tdim-per-file assumption below
    'tidesampwec':  sorted(f for f in glob.glob('/data/project3/minnaho/swel/tides/mc60/ampwec/everything/mc60_his.201904*.nc')
                           if '20190429110056' not in f),
}

out_path = './ke_spectra_comparison_20190421_20190423.npz'
dt_hours = 1.0

OCEAN_TIME_EPOCH = dt.datetime(1995, 1, 1)
WINDOW_START = dt.datetime(2019, 4, 21, 0, 0)
WINDOW_END   = dt.datetime(2019, 4, 23, 0, 0)   # exclusive -- window is [START, END)
WINDOW_START_SEC = (WINDOW_START - OCEAN_TIME_EPOCH).total_seconds()
WINDOW_END_SEC   = (WINDOW_END - OCEAN_TIME_EPOCH).total_seconds()

# ==========================================
# Load Grid & Masks
# ==========================================
print('Loading grid data...')
with Dataset(grd, 'r') as grdnc:
    masknc_sub = np.array(grdnc.variables['mask_rho'][:])

with Dataset('./coastal_mask.nc', 'r') as cmask_nc:
    coastal_mask_sub = np.array(cmask_nc.variables['coastal_mask'][:])

is_water_masknc  = (np.nan_to_num(masknc_sub)       == 1)
is_water_coastal = (np.nan_to_num(coastal_mask_sub)  == 1)
union_water      = is_water_masknc | is_water_coastal

len_eta, len_xi = masknc_sub.shape

# ==========================================
# Core Calculation
# ==========================================
def calculate_dataset_spectra(file_list, name):
    kept_chunks = []
    kept_times_sec = []

    for i, f in enumerate(file_list):
        with Dataset(f, 'r') as nc:
            ocean_time = np.array(nc.variables['ocean_time'][:])
        in_window = (ocean_time >= WINDOW_START_SEC) & (ocean_time < WINDOW_END_SEC)
        if not in_window.any():
            continue

        print(f'  -> {name} {i+1}/{len(file_list)}: {f} '
              f'({in_window.sum()} steps in window)')
        urho, vrho = pf.rho_uv_angle_surf(f, grd, rotate=False)
        u_sub = urho[in_window, 0, :, :]
        v_sub = vrho[in_window, 0, :, :]
        kept_chunks.append(u_sub + 1j * v_sub)
        kept_times_sec.append(ocean_time[in_window])

    if not kept_chunks:
        raise RuntimeError(f'no files for {name} overlap the '
                            f'{WINDOW_START} -- {WINDOW_END} window')

    w_complex = np.concatenate(kept_chunks, axis=0)
    times_sec = np.concatenate(kept_times_sec)
    total_time = w_complex.shape[0]

    # sanity check: hourly to within ROMS output jitter (a few seconds --
    # verified spacings like 3598s alongside 3600s across these files), no
    # gaps/duplicates/out-of-order steps
    dt_sec = np.diff(times_sec)
    if not np.allclose(dt_sec, dt_hours * 3600.0, atol=30.0):
        raise RuntimeError(
            f'{name}: window time steps are not uniformly spaced by '
            f'{dt_hours}h (found spacings {np.unique(dt_sec)} seconds) -- '
            f'check for missing/duplicate files in the window')

    t0 = OCEAN_TIME_EPOCH + dt.timedelta(seconds=float(times_sec[0]))
    t1 = OCEAN_TIME_EPOCH + dt.timedelta(seconds=float(times_sec[-1]))
    print(f'  -> {name}: {total_time} time steps, {t0} -- {t1}')

    print(f'  -> Computing FFT for {name}...')
    w_complex[:, ~union_water] = 0.0
    w_mean    = np.mean(w_complex, axis=0)
    w_detrend = w_complex - w_mean

    w_fft  = fftshift(fft(w_detrend, axis=0), axes=0)
    psd_3d = (np.abs(w_fft) ** 2) / (total_time ** 2)

    psd_masknc  = np.nanmean(np.where(is_water_masknc,  psd_3d, np.nan), axis=(1, 2))
    psd_coastal = np.nanmean(np.where(is_water_coastal, psd_3d, np.nan), axis=(1, 2))

    del w_complex, w_detrend, w_fft, psd_3d
    return psd_masknc, psd_coastal, total_time

# ==========================================
# Process All Scenarios
# ==========================================
results = {}
n_time = None
for scen, files in SCENARIOS.items():
    if not files:
        print(f'WARNING: no files found for {scen}, skipping')
        continue
    psd_m, psd_c, nt = calculate_dataset_spectra(files, scen)
    results[scen] = (psd_m, psd_c)
    if n_time is None:
        n_time = nt
    elif nt != n_time:
        print(f'WARNING: {scen} has {nt} in-window time steps, expected '
              f'{n_time} (from the first scenario) -- freqs axis below '
              f'is computed from the first scenario only')

freqs = fftshift(fftfreq(n_time, d=dt_hours))

# ==========================================
# Save
# ==========================================
save_dict = {'freqs': freqs}
for scen, (psd_m, psd_c) in results.items():
    save_dict[f'psd_{scen}_masknc']  = psd_m
    save_dict[f'psd_{scen}_coastal'] = psd_c

np.savez(out_path, **save_dict)
print(f'\nDone. Saved to {out_path}')
