"""
Vertical KE (w^2) wavenumber spectrum from zslice his files, following
Hypolite et al. (2021) -- their vertical spectrum uses w, not u/v.

Uses the near-surface uniform depth section: zslice levels 0-50 (depth 0 to -50 m,
dz = 1 m, 51 levels). At each horizontal point and time step, computes rfft of
w'(z) along the depth axis, then spatially averages to get the 1D radial-in-z
density E(k_z) [m^3 s^-2] such that integral(E(k_z) dk_z) = mean(w'^2).

Zslice his files have w already on rho-points (time, depth, eta_rho, xi_rho)
-- no horizontal interpolation needed.

6 scenarios (notidesampwec is the zslice-dir name for the raw 'ampwec' run).

Output: ke_vert_wavenumber.npz
  k_z               -- vertical wavenumber array (cycles m^-1), length n_kz
  E_<scen>_masknc   -- time-space-averaged E(k_z) [m^3 s^-2], full domain
  E_<scen>_coastal  -- same, coastal mask
"""
import sys
import glob
import numpy as np
from netCDF4 import Dataset
from scipy.fft import rfft, rfftfreq

sys.path.append('/data/project3/minnaho/global/')

grd = '/data/project3/minnaho/project9copy/swel/mc60_grd.nc'

ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'
SCENARIOS = {
    'tideswec':     sorted(glob.glob(f'{ZSLICE_ROOT}/tideswec/z_mc60_his.*.nc')),
    'tidesnowec':   sorted(glob.glob(f'{ZSLICE_ROOT}/tidesnowec/z_mc60_his.*.nc')),
    'notidesnowec': sorted(glob.glob(f'{ZSLICE_ROOT}/notidesnowec/z_mc60_his.*.nc')),
    'notideswec':   sorted(glob.glob(f'{ZSLICE_ROOT}/notideswec/z_mc60_his.*.nc')),
    # excludes z_mc60_his.20190429110056.nc: the source file has only 1
    # timestep (all others have 12) and its zslice output lacks a time
    # dimension entirely, which breaks the fixed-tdim-per-file assumption below
    'tidesampwec':  sorted(f for f in glob.glob(f'{ZSLICE_ROOT}/tidesampwec/z_mc60_his.*.nc')
                           if '20190429110056' not in f),
    'notidesampwec': sorted(glob.glob(f'{ZSLICE_ROOT}/notidesampwec/z_mc60_his.*.nc')),
}

# Use the uniform near-surface section: depth indices 0..N_Z_SURF-1 (dz=1m)
N_Z_SURF = 51   # 0 to -50 m, dz=1 m
DZ_M     = 1.0  # metres

out_path = './ke_vert_wavenumber.npz'

# ==========================================
# Grid & Masks
# ==========================================
print('Loading grid...')
with Dataset(grd, 'r') as grdnc:
    mask_rho = np.array(grdnc.variables['mask_rho'][:])  # (1202, 702)

with Dataset('./coastal_mask.nc', 'r') as cm:
    coast_mask = np.array(cm.variables['coastal_mask'][:])

is_water_full    = (np.nan_to_num(mask_rho)   == 1)
is_water_coastal = (np.nan_to_num(coast_mask) == 1)

NETA, NXI = mask_rho.shape   # 1202, 702
n_kz = N_Z_SURF // 2 + 1    # rfft output length for 51-point series

_ref = next(v for v in SCENARIOS.values() if v)
with Dataset(_ref[0], 'r') as tmp:
    tdim = tmp.dimensions['time'].size

k_z = rfftfreq(N_Z_SURF, d=DZ_M)  # cycles m^-1, 0 to 0.5

# ==========================================
# Per-scenario processing
# ==========================================
def process_scenario(file_list, name):
    total_time = len(file_list) * tdim
    print(f'\n=== {name} | {len(file_list)} files | {total_time} time steps ===')

    E_accum_full    = np.zeros(n_kz, dtype=np.float64)
    E_accum_coastal = np.zeros(n_kz, dtype=np.float64)
    n_samples_full    = 0
    n_samples_coastal = 0

    for i, f in enumerate(file_list):
        print(f'  file {i+1}/{len(file_list)}: {f}')
        with Dataset(f, 'r') as nc:
            # w: already on rho-points (time, depth, eta_rho, xi_rho)
            w_rho = np.array(nc.variables['w'][:, :N_Z_SURF, :, :])  # (tdim,51,1202,702)

        # Replace fill values and mask land
        w_rho[np.abs(w_rho) > 1e10] = 0.0
        w_rho[:, :, ~is_water_full] = 0.0

        # Remove depth-mean at each (eta, xi) point for the perturbation
        w_rho -= np.mean(w_rho, axis=1, keepdims=True)

        # rfft along depth axis (axis=1)
        w_fft = rfft(w_rho, axis=1)  # (tdim, n_kz, eta, xi)

        # Per-mode power -> 1D density E(k_z) [m^3 s^-2] such that
        # integral(E(k_z) dk_z) = mean(w'^2): divide by N (bin-width
        # normalization, Δk_z = 1/(N_Z_SURF*DZ_M)) then double non-DC/Nyquist
        # bins to fold the one-sided rfft back to full-spectrum power.
        E_3d = (np.abs(w_fft) ** 2) * DZ_M / N_Z_SURF
        E_3d[:, 1:-1, :, :] *= 2.0

        # Spatial average over wet points, then time-sum
        for t in range(tdim):
            E_t = E_3d[t]  # (n_kz, eta, xi)
            E_accum_full += np.nanmean(
                np.where(is_water_full[np.newaxis, :, :], E_t, np.nan), axis=(1, 2))
            E_accum_coastal += np.nanmean(
                np.where(is_water_coastal[np.newaxis, :, :], E_t, np.nan), axis=(1, 2))
            n_samples_full    += 1
            n_samples_coastal += 1

        del w_rho, w_fft, E_3d

    print(f'  Done {name}')
    return E_accum_full / n_samples_full, E_accum_coastal / n_samples_coastal

# ==========================================
# Run
# ==========================================
results = {}
for scen, files in SCENARIOS.items():
    if not files:
        print(f'WARNING: no files for {scen}, skipping')
        continue
    E_m, E_c = process_scenario(files, scen)
    results[scen] = (E_m, E_c)

save_dict = {'k_z': k_z}
for scen, (E_m, E_c) in results.items():
    save_dict[f'E_{scen}_masknc']  = E_m
    save_dict[f'E_{scen}_coastal'] = E_c

np.savez(out_path, **save_dict)
print(f'\nSaved to {out_path}')
