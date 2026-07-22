"""
KE spectra at 50 m depth, from zsliced his output.

Same complex-velocity FFT approach as calc_ke_surf.py / calc_ke_10m.py, but
the surface velocity there (raw s_rho top level, always valid in water) is
replaced here by u/v read off the zsliced z_mc60_his.*.nc files at the fixed
z-level closest to -50 m (index 50 on the 157-level zslice depth grid --
z_mc60_his depth[50] == -50.0 exactly).

The zsliced u/v are still on their native staggered grids (eta_rho x
xi_u, eta_v x xi_rho) and are NOT rotated to east/north -- interpolated
to rho points here the same way pf.rho_uv_angle_surf does, but pyfuncs'
helper can't be reused directly since it always grabs the top s_rho
level (index -1), not a specific zslice depth index.

Below-bathymetry fill values (cells shallower than 50 m, where the
zslice has no real data at that level) are masked to NaN before
interpolation, so they propagate to NaN in the interpolated rho point
and are naturally excluded by the final spatial nanmean, rather than
being silently averaged in as if they were real velocities.

Output: ./ke_spectra_50m_comparison.npz
"""

import glob
import numpy as np
from netCDF4 import Dataset
from scipy.fft import fft, fftshift, fftfreq

# ==========================================
# Configuration & File Paths
# ==========================================
grd = '/data/project3/minnaho/project9copy/swel/mc60_grd.nc'

ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'
ZSLICE_DIRS = {'ampwec': 'notidesampwec'}   # label -> zslice subdirectory (identity for tidesampwec)

# tidesampwec's raw source has a trailing 1-timestep file
# (...20190429110056) whose zslice output has no time dimension at all
TIDESAMPWEC_EXCLUDE = ('20190429110056',)


def zslice_his_files(scen):
    sd = ZSLICE_DIRS.get(scen, scen)
    files = sorted(glob.glob(f'{ZSLICE_ROOT}/{sd}/z_mc60_his.*.nc'))
    if scen == 'tidesampwec':
        files = [f for f in files if not any(s in f for s in TIDESAMPWEC_EXCLUDE)]
    return files


SCENARIOS = {
    'tideswec':     zslice_his_files('tideswec'),
    'tidesnowec':   zslice_his_files('tidesnowec'),
    'notidesnowec': zslice_his_files('notidesnowec'),
    'notideswec':   zslice_his_files('notideswec'),
    'ampwec':       zslice_his_files('ampwec'),
    'tidesampwec':  zslice_his_files('tidesampwec'),
}

out_path = './ke_spectra_50m_comparison.npz'
dt_hours = 1.0

# ==========================================
# Load Grid & Masks
# ==========================================
print('Loading grid data...')
with Dataset(grd, 'r') as grdnc:
    masknc_sub = np.array(grdnc.variables['mask_rho'][:])
    angle_rho  = np.array(grdnc.variables['angle'][:])

with Dataset('./coastal_mask.nc', 'r') as cmask_nc:
    coastal_mask_sub = np.array(cmask_nc.variables['coastal_mask'][:])

is_water_masknc  = (np.nan_to_num(masknc_sub)       == 1)
is_water_coastal = (np.nan_to_num(coastal_mask_sub)  == 1)
union_water      = is_water_masknc | is_water_coastal

len_eta, len_xi = masknc_sub.shape

# time steps per file, and the zslice depth index closest to -50 m, from
# the first available scenario
_ref_files = next(v for v in SCENARIOS.values() if v)
with Dataset(_ref_files[0], 'r') as tmp:
    tdim = tmp.dimensions['time'].size
    depth_1d = np.array(tmp.variables['depth'][:])
DEPTH_IDX = int(np.argmin(np.abs(depth_1d - (-50.0))))
print(f'Using zslice depth index {DEPTH_IDX} (depth = {depth_1d[DEPTH_IDX]} m)')

# ==========================================
# Helpers
# ==========================================
def uv_rho_at_depth(f, depth_idx, rotate=False):
    """Interpolate u,v from a zsliced his file's fixed-depth level onto rho
    points, optionally rotating to east/north (mirrors pf.rho_uv_angle_surf's
    interface, which can't target an arbitrary zslice depth index directly).

    Rotation does NOT change the KE spectrum: at a fixed grid cell,
    u_east + i*v_north == e^(i*angle) * (u + i*v), a time-invariant phase
    multiply, and the time-axis FFT commutes with a time-invariant scalar, so
    |FFT(e^(i*angle) z)|^2 == |FFT(z)|^2. Kept for interface parity / reuse
    where actual east/north components (not just the spectrum) are needed --
    calc_ke_50m.py itself calls with rotate=False, same as calc_ke_surf.py.

    Returns (time, eta_rho, xi_rho).
    """
    with Dataset(f, 'r') as nc:
        u = np.array(nc.variables['u'][:, depth_idx, :, :])   # (time, eta_rho, xi_u)
        v = np.array(nc.variables['v'][:, depth_idx, :, :])   # (time, eta_v, xi_rho)
    u[np.abs(u) > 1e10] = np.nan   # below-bathymetry fill value
    v[np.abs(v) > 1e10] = np.nan

    Nt, Mp, L = u.shape
    Lp = L + 1
    u_temp = 0.5 * (u[:, :, 1:L] + u[:, :, :L - 1])
    u_rho = np.full((Nt, Mp, Lp), np.nan)
    u_rho[:, :, 1:-1] = u_temp
    u_rho[:, :, 0]    = u_temp[:, :, 0]
    u_rho[:, :, -1]   = u_temp[:, :, -1]

    Nt, M, Lp2 = v.shape
    Mp = M + 1
    v_temp = 0.5 * (v[:, 1:M, :] + v[:, :M - 1, :])
    v_rho = np.full((Nt, Mp, Lp2), np.nan)
    v_rho[:, 1:-1, :] = v_temp
    v_rho[:, 0, :]    = v_temp[:, 0, :]
    v_rho[:, -1, :]   = v_temp[:, -1, :]

    if rotate:
        angle_4d = angle_rho[np.newaxis, :, :]
        cosang, sinang = np.cos(angle_4d), np.sin(angle_4d)
        u_east  = u_rho * cosang - v_rho * sinang
        v_north = u_rho * sinang + v_rho * cosang
        return u_east, v_north

    return u_rho, v_rho

# ==========================================
# Core Calculation
# ==========================================
def calculate_dataset_spectra(file_list, name):
    total_time = len(file_list) * tdim
    print(f'\nProcessing {len(file_list)} {name} files ({total_time} time steps)...')

    w_complex = np.zeros((total_time, len_eta, len_xi), dtype=complex)

    t_idx = 0
    for i, f in enumerate(file_list):
        print(f'  -> {name} {i+1}/{len(file_list)}: {f}')
        u_rho, v_rho = uv_rho_at_depth(f, DEPTH_IDX, rotate=False)
        w_complex[t_idx:t_idx + tdim] = u_rho + 1j * v_rho
        t_idx += tdim

    print(f'  -> Computing FFT for {name}...')
    w_complex[:, ~union_water] = 0.0
    # NaN (below-bathymetry-at-50m) cells propagate through mean/FFT and
    # are excluded by the final spatial nanmean below -- not zeroed here,
    # since zeroing would wrongly count them as valid zero-KE water.
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
