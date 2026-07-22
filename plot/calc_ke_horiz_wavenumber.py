"""
Horizontal wavenumber spectrum of KE, following Hypolite et al. (2021).

For each time step, computes 2D rfft of surface u' and v' on the horizontal
grid, forms E(kx, ky) = |û|^2 + |v̂|^2, then reduces to the 1D radial
spectrum E(k_h) vs horizontal wavenumber k_h = sqrt(kx^2 + ky^2):
  1. Per-mode power E(kx,ky) is converted to a 2D spectral density by
     dividing by the (kx,ky) bin area (Δkx·Δky).
  2. The isotropic 2D density is collapsed to a 1D radial density by
     annulus-integrating: E(k) = 2*pi*k * <density>_annulus.
This normalization is required for E(k) to satisfy integral(E(k) dk) = KE,
matching Hypolite et al.'s KE [m^3 s^-2] convention (Fig. 5).

The ROMS grid is treated as approximately uniform (dx = dy = DX_M = 60 m).
This introduces small spectral leakage but is acceptable for slope comparison.

Output: ke_horiz_wavenumber.npz
  k_h              -- 1D wavenumber array (cycles m^-1)
  E_<scen>_masknc  -- time-averaged E(k_h), full-domain mask (m^3 s^-2)
  E_<scen>_coastal -- same, coastal mask
"""
import sys
import glob
import numpy as np
from netCDF4 import Dataset
from scipy.fft import rfft2, rfftfreq, fftfreq

sys.path.append('/data/project3/minnaho/global/')
import pyfuncs as pf

grd   = '/data/project3/minnaho/project9copy/swel/mc60_grd.nc'
DX_M  = 60.0  # approximate grid spacing in metres

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

N_K_BINS = 200   # number of logarithmic wavenumber bins
out_path = './ke_horiz_wavenumber.npz'

# ==========================================
# Grid & Masks
# ==========================================
print('Loading grid...')
with Dataset(grd, 'r') as grdnc:
    mask_rho  = np.array(grdnc.variables['mask_rho'][:])

with Dataset('./coastal_mask.nc', 'r') as cm:
    coast_mask = np.array(cm.variables['coastal_mask'][:])

is_water_full    = (np.nan_to_num(mask_rho)   == 1)
is_water_coastal = (np.nan_to_num(coast_mask) == 1)

NETA, NXI = mask_rho.shape  # 1202, 702

_ref = next(v for v in SCENARIOS.values() if v)
with Dataset(_ref[0], 'r') as tmp:
    tdim = tmp.dimensions['time'].size

# ==========================================
# Wavenumber grid (2D rfft output has shape NETA x NXI//2+1)
# ==========================================
kx = rfftfreq(NXI,  d=DX_M)   # cycles m^-1, positive only (NXI//2+1 values)
ky = fftfreq(NETA,  d=DX_M)   # cycles m^-1, full (NETA values)
KX, KY = np.meshgrid(kx, ky)
KH = np.sqrt(KX ** 2 + KY ** 2)   # (NETA, NXI//2+1)

k_max = kx.max()
k_min = kx[1]  # first non-zero
k_bins = np.logspace(np.log10(k_min), np.log10(k_max), N_K_BINS + 1)
k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

# 1 / (Δkx · Δky), the (kx,ky) bin-area normalization -- Δkx=1/(NXI*DX_M), Δky=1/(NETA*DX_M)
INV_DK_AREA = NETA * NXI * DX_M ** 2

def azimuthal_average(E_2d):
    """Reduce per-mode power E_2d (NETA, NXI//2+1) to the 1D radial KE
    density E(k_h) [m^3 s^-2], i.e. E(k) = 2*pi*k * <E_2d/(Δkx·Δky)>_annulus,
    so that integral(E(k) dk) = KE (Hypolite et al. 2021 convention)."""
    E_1d = np.full(N_K_BINS, np.nan)
    for i in range(N_K_BINS):
        mask = (KH >= k_bins[i]) & (KH < k_bins[i + 1])
        if mask.any():
            E_1d[i] = 2 * np.pi * k_centers[i] * np.mean(E_2d[mask]) * INV_DK_AREA
    return E_1d

# ==========================================
# Per-scenario processing
# ==========================================
def process_scenario(file_list, name):
    total_time = len(file_list) * tdim
    print(f'\n=== {name} | {total_time} time steps ===')

    # First pass: compute time-mean u and v for detrending
    u_mean = np.zeros((NETA, NXI), dtype=np.float32)
    v_mean = np.zeros((NETA, NXI), dtype=np.float32)
    for i, f in enumerate(file_list):
        print(f'  mean pass {i+1}/{len(file_list)}', end='\r', flush=True)
        urho, vrho = pf.rho_uv_angle_surf(f, grd, rotate=False)
        u_mean += np.sum(urho[:, 0, :, :], axis=0)
        v_mean += np.sum(vrho[:, 0, :, :], axis=0)
    u_mean /= total_time
    v_mean /= total_time

    # Second pass: compute 2D FFT at each time step, accumulate azimuthal average
    E_accum_full    = np.zeros(N_K_BINS, dtype=np.float64)
    E_accum_coastal = np.zeros(N_K_BINS, dtype=np.float64)
    count = 0

    for i, f in enumerate(file_list):
        print(f'  fft pass  {i+1}/{len(file_list)}', end='\r', flush=True)
        urho, vrho = pf.rho_uv_angle_surf(f, grd, rotate=False)
        u_snap = urho[:, 0, :, :]  # (tdim, eta, xi)
        v_snap = vrho[:, 0, :, :]

        for t in range(tdim):
            u_p = (u_snap[t] - u_mean) * is_water_full
            v_p = (v_snap[t] - v_mean) * is_water_full

            u_fft2 = rfft2(u_p)  # (NETA, NXI//2+1) complex
            v_fft2 = rfft2(v_p)

            N2 = float(NETA * NXI) ** 2
            E_2d = (np.abs(u_fft2) ** 2 + np.abs(v_fft2) ** 2) / N2

            E_accum_full    += azimuthal_average(E_2d)
            E_accum_coastal += azimuthal_average(E_2d * is_water_coastal[:, :NXI // 2 + 1])
            count += 1

    print(f'\n  Done {name}')
    return E_accum_full / count, E_accum_coastal / count

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

save_dict = {'k_h': k_centers}
for scen, (E_m, E_c) in results.items():
    save_dict[f'E_{scen}_masknc']  = E_m
    save_dict[f'E_{scen}_coastal'] = E_c

np.savez(out_path, **save_dict)
print(f'\nSaved to {out_path}')
