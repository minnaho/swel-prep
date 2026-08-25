import sys
import glob
import numpy as np
from netCDF4 import Dataset
sys.path.append('/data/project3/minnaho/global/')
import pyfuncs as pf

# ==========================================
# 1. Configuration & File Paths
# ==========================================
grd = '/data/project3/minnaho/project9copy/swel/mc60_grd.nc'

# Use glob to grab all relevant history files and sort them chronologically
tideswec_files = sorted(glob.glob('/data/project3/minnaho/swel/tides/mc60/wec/his/mc60_his.2019042[1-8]*.nc'))
tidesnowec_files = sorted(glob.glob('/data/project3/minnaho/swel/tides/mc60/nowec/output/his/mc60_his.2019042[1-8]*.nc'))
notidesnowec_files = sorted(glob.glob('/data/project3/minnaho/swel/notides/mc60/nowec/his/mc60_his.2019042[1-8]*.nc'))

# Output paths for the individual arrays
out_tideswec_path = './surfvort_tideswec.npz'
out_tidesnowec_path = './surfvort_tidesnowec.npz'
out_notidesnowec_path = './surfvort_notidesnowec.npz'

# ==========================================
# 2. Load Grid & Mask
# ==========================================
print('Loading grid data...')
with Dataset(grd, 'r') as grdnc:
    f_nc = np.array(grdnc.variables['f'])
    masknc = np.array(grdnc.variables['mask_rho'])

masknc[masknc == 0] = np.nan

with Dataset('../plot/coastal_mask.nc', 'r') as cmask_nc:
    coastal_mask = np.array(cmask_nc.variables['coastal_mask'])

coastal_mask[coastal_mask==0] = np.nan

# Extract dimensions safely
with Dataset(tideswec_files[0], 'r') as temp_nc:
    tdim = temp_nc.dimensions['time'].size
    etadim = temp_nc.dimensions['eta_rho'].size
    xidim = temp_nc.dimensions['xi_rho'].size

# ==========================================
# 3. Iteratively Process WEC Files
# ==========================================
print(f'Processing {len(tideswec_files)} WEC files...')

# Initialize ONLY WEC arrays to save memory
wecvort_mask = np.ones((len(tideswec_files), tdim, 1, etadim, xidim)) * np.nan
wecvort_cmask = np.ones((len(tideswec_files), tdim, 1, etadim, xidim)) * np.nan

for f_idx, file in enumerate(tideswec_files):
    print(f'  -> WEC {f_idx+1}/{len(tideswec_files)}: {file}')
    urho, vrho = pf.rho_uv_angle_surf(file, grd, rotate=True)
    vort = pf.vorticity(grd, urho, vrho)
    
    wecvort_mask[f_idx,:,:,:,:] = (vort / f_nc) * masknc 
    wecvort_cmask[f_idx,:,:,:,:] = (vort / f_nc) * coastal_mask 
    
np.savez(out_tideswec_path, vort_mask=wecvort_mask, vort_cmask=wecvort_cmask)
print(f'Saved WEC data to {out_tideswec_path}')

# Free up memory!
del wecvort_mask, wecvort_cmask

# ==========================================
# 4. Iteratively Process Tides no-WEC Files
# ==========================================
print(f'\nProcessing {len(tidesnowec_files)} Tides no-WEC files...')

# Initialize ONLY Tides no-WEC arrays
nowecvort_mask = np.ones((len(tidesnowec_files), tdim, 1, etadim, xidim)) * np.nan
nowecvort_cmask = np.ones((len(tidesnowec_files), tdim, 1, etadim, xidim)) * np.nan

for f_idx, file in enumerate(tidesnowec_files):
    print(f'  -> Tides no-WEC {f_idx+1}/{len(tidesnowec_files)}: {file}')
    urho, vrho = pf.rho_uv_angle_surf(file, grd, rotate=True)
    vort = pf.vorticity(grd, urho, vrho)
    
    nowecvort_mask[f_idx,:,:,:,:] = (vort / f_nc) * masknc 
    nowecvort_cmask[f_idx,:,:,:,:] = (vort / f_nc) * coastal_mask 
    
np.savez(out_tidesnowec_path, vort_mask=nowecvort_mask, vort_cmask=nowecvort_cmask)
print(f'Saved Tides no-WEC data to {out_tidesnowec_path}')

del nowecvort_mask, nowecvort_cmask

# ==========================================
# 5. Iteratively Process No-Tides no-WEC Files
# ==========================================
print(f'\nProcessing {len(notidesnowec_files)} No-Tides no-WEC files...')

# Initialize ONLY No-Tides no-WEC arrays
notidesnowecvort_mask = np.ones((len(notidesnowec_files), tdim, 1, etadim, xidim)) * np.nan
notidesnowecvort_cmask = np.ones((len(notidesnowec_files), tdim, 1, etadim, xidim)) * np.nan

for f_idx, file in enumerate(notidesnowec_files):
    print(f'  -> No-Tides no-WEC {f_idx+1}/{len(notidesnowec_files)}: {file}')
    urho, vrho = pf.rho_uv_angle_surf(file, grd, rotate=True)
    vort = pf.vorticity(grd, urho, vrho)
    
    notidesnowecvort_mask[f_idx,:,:,:,:] = (vort / f_nc) * masknc 
    notidesnowecvort_cmask[f_idx,:,:,:,:] = (vort / f_nc) * coastal_mask 
    
np.savez(out_notidesnowec_path, vort_mask=notidesnowecvort_mask, vort_cmask=notidesnowecvort_cmask)
print(f'Saved No-Tides no-WEC data to {out_notidesnowec_path}')

print('\nAll calculations complete.')
