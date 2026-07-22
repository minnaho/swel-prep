import sys
import glob
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
    'notidesnowec': sorted(glob.glob('/data/project3/minnaho/swel/notides/mc60/nowec/output/his/mc60_his.201904*.nc')),
    'notideswec':   sorted(glob.glob('/data/project3/minnaho/swel/notides/mc60/wec/rerun/his/mc60_his.201904*.nc')),
    'ampwec':       sorted(glob.glob('/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything/mc60_his.201904*.nc')),
    # excludes mc60_his.20190429110056.nc: a trailing 1-timestep file (all
    # others have 12), which breaks the fixed-tdim-per-file assumption below
    'tidesampwec':  sorted(f for f in glob.glob('/data/project3/minnaho/swel/tides/mc60/ampwec/everything/mc60_his.201904*.nc')
                           if '20190429110056' not in f),
}

out_path = './ke_spectra_comparison.npz'
dt_hours = 1.0

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

# time steps per file from first available scenario
_ref_files = next(v for v in SCENARIOS.values() if v)
with Dataset(_ref_files[0], 'r') as tmp:
    tdim = tmp.dimensions['time'].size

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
        urho, vrho = pf.rho_uv_angle_surf(f, grd, rotate=False)
        u_sub = urho[:, 0, :, :]
        v_sub = vrho[:, 0, :, :]
        w_complex[t_idx:t_idx + tdim] = u_sub + 1j * v_sub
        t_idx += tdim

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
