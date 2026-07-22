"""
Depth-averaged frequency spectrum of KE perturbations (u', v') from ROMS his files.

Processes one s_rho level at a time to keep peak memory ~500 MB.
Uses rfft (one-sided) — no rotary decomposition; total KE at each freq = |u'|^2 + |v'|^2.
Arithmetic mean over all 100 s_rho levels (no Hz weighting).

Output: ke_spectra_depth.npz
  freqs          -- positive-only frequency array (cph), length n_freq
  psd_<scen>_masknc   -- depth-averaged PSD, full domain mask
  psd_<scen>_coastal  -- depth-averaged PSD, coastal mask
"""
import sys
import glob
import numpy as np
from netCDF4 import Dataset
from scipy.fft import rfft, rfftfreq

sys.path.append('/data/project3/minnaho/global/')

grd = '/data/project3/minnaho/project9copy/swel/mc60_grd.nc'

SCENARIOS = {
    'tideswec':     sorted(glob.glob('/data/project3/minnaho/swel/tides/mc60/wec/his/mc60_his.201904*.nc')),
    'tidesnowec':   sorted(glob.glob('/data/project3/minnaho/swel/tides/mc60/nowec/output/his/mc60_his.201904*.nc')),
    'notidesnowec': sorted(glob.glob('/data/project3/minnaho/swel/notides/mc60/nowec/output/his/mc60_his.201904*.nc')),
    'notideswec':   sorted(glob.glob('/data/project3/minnaho/swel/notides/mc60/wec/rerun/his/mc60_his.201904*.nc')),
    'ampwec':       sorted(glob.glob('/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything/mc60_his.201904*.nc')),
}

out_path = './ke_spectra_depth.npz'
dt_hours = 1.0

# ==========================================
# Grid & Masks
# ==========================================
print('Loading grid...')
with Dataset(grd, 'r') as grdnc:
    mask_rho = np.array(grdnc.variables['mask_rho'][:])  # (eta_rho, xi_rho)

with Dataset('./coastal_mask.nc', 'r') as cm:
    coastal_mask = np.array(cm.variables['coastal_mask'][:])

is_water_full    = (np.nan_to_num(mask_rho)     == 1)  # (1202, 702)
is_water_coastal = (np.nan_to_num(coastal_mask) == 1)

ETA_RHO, XI_RHO = mask_rho.shape  # 1202, 702

_ref = next(v for v in SCENARIOS.values() if v)
with Dataset(_ref[0], 'r') as tmp:
    tdim    = tmp.dimensions['time'].size
    N_S_RHO = tmp.dimensions['s_rho'].size  # 100

# ==========================================
# Per-scenario processing
# ==========================================
def process_scenario(file_list, name):
    total_time = len(file_list) * tdim
    n_freq     = total_time // 2 + 1  # rfft output length

    print(f'\n=== {name} | {len(file_list)} files | {total_time} time steps ===')

    psd_accum_full    = np.zeros(n_freq, dtype=np.float64)
    psd_accum_coastal = np.zeros(n_freq, dtype=np.float64)

    for s_lvl in range(N_S_RHO):
        print(f'  s_rho level {s_lvl+1}/{N_S_RHO}', end='\r', flush=True)

        # Accumulate time series for this level
        u_ts = np.zeros((total_time, ETA_RHO, XI_RHO), dtype=np.float32)
        v_ts = np.zeros((total_time, ETA_RHO, XI_RHO), dtype=np.float32)

        t_idx = 0
        for f in file_list:
            with Dataset(f, 'r') as nc:
                nt = nc.dimensions['time'].size
                # u: (time, s_rho, eta_rho, xi_u=701)
                u_raw = np.array(nc.variables['u'][:, s_lvl, :, :])  # (nt,1202,701)
                # v: (time, s_rho, eta_v=1201, xi_rho)
                v_raw = np.array(nc.variables['v'][:, s_lvl, :, :])  # (nt,1201,702)

            # Interpolate u to rho points (average in xi)
            u_rho = np.empty((nt, ETA_RHO, XI_RHO), dtype=np.float32)
            u_rho[:, :, 1:-1] = 0.5 * (u_raw[:, :, :-1] + u_raw[:, :, 1:])
            u_rho[:, :, 0]    = u_raw[:, :, 0]
            u_rho[:, :, -1]   = u_raw[:, :, -1]

            # Interpolate v to rho points (average in eta)
            v_rho = np.empty((nt, ETA_RHO, XI_RHO), dtype=np.float32)
            v_rho[:, 1:-1, :] = 0.5 * (v_raw[:, :-1, :] + v_raw[:, 1:, :])
            v_rho[:, 0, :]    = v_raw[:, 0, :]
            v_rho[:, -1, :]   = v_raw[:, -1, :]

            u_ts[t_idx:t_idx + nt] = u_rho
            v_ts[t_idx:t_idx + nt] = v_rho
            t_idx += nt

        # Zero land before FFT
        u_ts[:, ~is_water_full] = 0.0
        v_ts[:, ~is_water_full] = 0.0

        # Remove time mean
        u_ts -= np.mean(u_ts, axis=0)
        v_ts -= np.mean(v_ts, axis=0)

        # rfft along time
        u_fft = rfft(u_ts, axis=0)  # (n_freq, eta, xi) complex64
        v_fft = rfft(v_ts, axis=0)

        psd_3d = (np.abs(u_fft) ** 2 + np.abs(v_fft) ** 2) / (total_time ** 2)
        # double non-DC/Nyquist bins to account for negative-frequency mirror
        psd_3d[1:-1] *= 2.0

        psd_accum_full    += np.nanmean(np.where(is_water_full,    psd_3d, np.nan), axis=(1, 2))
        psd_accum_coastal += np.nanmean(np.where(is_water_coastal, psd_3d, np.nan), axis=(1, 2))

        del u_ts, v_ts, u_fft, v_fft, psd_3d

    print(f'\n  Done {name}')
    return psd_accum_full / N_S_RHO, psd_accum_coastal / N_S_RHO, total_time

# ==========================================
# Run
# ==========================================
results = {}
n_time  = None
for scen, files in SCENARIOS.items():
    if not files:
        print(f'WARNING: no files for {scen}, skipping')
        continue
    psd_m, psd_c, nt = process_scenario(files, scen)
    results[scen] = (psd_m, psd_c)
    if n_time is None:
        n_time = nt

freqs = rfftfreq(n_time, d=dt_hours)  # positive only, cph

save_dict = {'freqs': freqs}
for scen, (psd_m, psd_c) in results.items():
    save_dict[f'psd_{scen}_masknc']  = psd_m
    save_dict[f'psd_{scen}_coastal'] = psd_c

np.savez(out_path, **save_dict)
print(f'\nSaved to {out_path}')
