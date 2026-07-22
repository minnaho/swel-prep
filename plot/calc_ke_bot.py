import sys
import glob
import numpy as np
from netCDF4 import Dataset
from scipy.fft import fft, fftshift, fftfreq

# Append your custom path
sys.path.append('/data/project3/minnaho/global/')
import pyfuncs as pf

# ==========================================
# 1. Configuration & File Paths
# ==========================================
grd = '/data/project3/minnaho/project9copy/swel/mc60_grd.nc'

# Use glob to grab all relevant history files and sort them chronologically
tideswec_files = sorted(glob.glob('/data/project3/minnaho/swel/tides/mc60/wec/his/mc60_his.201904*.nc'))
tidesnowec_files = sorted(glob.glob('/data/project3/minnaho/swel/tides/mc60/nowec/output/his/mc60_his.201904*.nc'))
notidesnowec_files = sorted(glob.glob('/data/project3/minnaho/swel/notides/mc60/nowec/output/his/mc60_his.201904*.nc'))

# Output paths for the calculated spectra
out_spectra_path = './ke_spectra_comparison.npz'

# Define your spatial subset
eta_start, eta_end = 0, -1
xi_start, xi_end = 0, -1

# Time step in hours
dt_hours = 1.0 

# ==========================================
# 2. Load Grid & Mask (For the Subset)
# ==========================================
print('Loading grid data...')

with Dataset(grd, 'r') as grdnc:
    # Slicing with [0:-1] grabs everything except the very last index. 
    # If you want the ENTIRE domain, it is safer to use [:] or specific integer bounds.
    masknc_sub = np.array(grdnc.variables['mask_rho'][eta_start:eta_end, xi_start:xi_end])

with Dataset('../plot/coastal_mask.nc', 'r') as cmask_nc:
    coastal_mask_sub = np.array(cmask_nc.variables['coastal_mask'][eta_start:eta_end, xi_start:xi_end])

# Create boolean arrays for valid water points (True for water, False for land/NaN)
# This safely handles if the mask uses 0s or NaNs for land
is_water_masknc = (np.nan_to_num(masknc_sub) == 1)
is_water_coastal = (np.nan_to_num(coastal_mask_sub) == 1)

# Get the exact dimensions of the subset to avoid negative length errors
len_eta, len_xi = masknc_sub.shape

# Extract time dimension size per file safely
with Dataset(tideswec_files[0], 'r') as temp_nc:
    tdim = temp_nc.dimensions['time'].size

# ==========================================
# 3. Core Calculation Function
# ==========================================
def calculate_dataset_spectra(file_list, dataset_name):
    """
    Iterates through files, calculates the 3D PSD, and returns TWO 1D spectra:
    one averaged over the general mask, and one over the coastal mask.
    """
    total_time = len(file_list) * tdim
    print(f'\nProcessing {len(file_list)} {dataset_name} files (Total time steps: {total_time})...')
    
    # Initialize a continuous array for complex velocity
    w_complex = np.zeros((total_time, len_eta, len_xi), dtype=complex)
    
    t_idx = 0
    for f_idx, file in enumerate(file_list):
        print(f'  -> {dataset_name} {f_idx+1}/{len(file_list)}: {file}')
        
        urho, vrho = pf.rho_uv_angle_surf(file, grd, rotate=False)
        
        # Slice to the spatial subset and drop the singleton 's' dimension
        u_sub = urho[:, 0, eta_start:eta_end, xi_start:xi_end]
        v_sub = vrho[:, 0, eta_start:eta_end, xi_start:xi_end]
        
        # Form complex velocity
        w_complex[t_idx : t_idx + tdim, :, :] = u_sub + 1j * v_sub
        
        t_idx += tdim

    print(f'  -> Computing FFT and Spatial Averages for {dataset_name}...')
    
    # Pre-process: To avoid NaN propagation in FFT, set all absolute land to 0
    # (Points that are land in BOTH masks won't be used anyway)
    union_water = is_water_masknc | is_water_coastal
    w_complex[:, ~union_water] = 0.0
    
    # Detrend: Remove the temporal mean at each spatial point
    w_mean = np.mean(w_complex, axis=0)
    w_detrend = w_complex - w_mean
    
    # Compute FFT along the time axis
    w_fft = fftshift(fft(w_detrend, axis=0), axes=0)
    
    # Calculate complete 3D Power Spectral Density (Kinetic Energy)
    psd_3d = (np.abs(w_fft)**2) / (total_time**2)
    
    # Separate Spatial Averaging
    # np.where applies the specific mask, turning invalid points to NaN before averaging
    psd_1d_masknc = np.nanmean(np.where(is_water_masknc, psd_3d, np.nan), axis=(1, 2))
    psd_1d_coastal = np.nanmean(np.where(is_water_coastal, psd_3d, np.nan), axis=(1, 2))
    
    # Free memory
    del w_complex, w_detrend, w_fft, psd_3d
    
    return psd_1d_masknc, psd_1d_coastal, total_time

# ==========================================
# 4. Process All Datasets
# ==========================================
# Calculate spectra (returns 2 arrays per dataset now)
psd_tw_masknc, psd_tw_coastal, n_time = calculate_dataset_spectra(tideswec_files, "Tides WEC")
psd_tnw_masknc, psd_tnw_coastal, _ = calculate_dataset_spectra(tidesnowec_files, "Tides no-WEC")
psd_ntnw_masknc, psd_ntnw_coastal, _ = calculate_dataset_spectra(notidesnowec_files, "No-Tides no-WEC")

# Generate the frequency array 
freqs = fftshift(fftfreq(n_time, d=dt_hours))

# ==========================================
# 5. Save Results
# ==========================================
np.savez(out_spectra_path, 
         freqs=freqs, 
         # Masknc outputs
         psd_tideswec_masknc=psd_tw_masknc, 
         psd_tidesnowec_masknc=psd_tnw_masknc, 
         psd_notidesnowec_masknc=psd_ntnw_masknc,
         # Coastal outputs
         psd_tideswec_coastal=psd_tw_coastal,
         psd_tidesnowec_coastal=psd_tnw_coastal,
         psd_notidesnowec_coastal=psd_ntnw_coastal)

print(f'\nAll spectral calculations complete. Data saved to {out_spectra_path}')
