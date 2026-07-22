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
notidesnowec_files = sorted(glob.glob('/data/project3/minnaho/swel/notides/mc60/nowec/output/his/mc60_his.2019042[1-8]*.nc'))

# Output paths for the individual arrays
out_tideswec_path = './surfvorticity_tideswec.npz'
out_tidesnowec_path = './surfvorticity_tidesnowec.npz'
out_notidesnowec_path = './surfvorticity_notidesnowec.npz'

# ==========================================
# 2. Load Grid & Mask
# ==========================================
print('Loading grid data...')
grdnc = Dataset(grd, 'r')
f_nc = np.array(grdnc.variables['f'])
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc == 0] = np.nan

coastal_mask = np.array(Dataset('../plot/coastal_mask.nc','r').variables['coastal_mask'])
coastal_mask[coastal_mask==0] = np.nan


# ==========================================
# 3. Initialize PDF (Histogram) Variables
# ==========================================
tdim = Dataset(tideswec_files[0],'r').dimensions['time'].size
etadim = Dataset(tideswec_files[0],'r').dimensions['eta_rho'].size
xidim = Dataset(tideswec_files[0],'r').dimensions['xi_rho'].size
# file, time dimension, srho (surface only), eta, xi
wecvort_mask = np.ones((len(tideswec_files),tdim,1,etadim,xidim))*np.nan
wecvort_cmask = np.ones((len(tideswec_files),tdim,1,etadim,xidim))*np.nan
nowecvort_mask = np.ones((len(tidesnowec_files),tdim,1,etadim,xidim))*np.nan
nowecvort_cmask = np.ones((len(tidesnowec_files),tdim,1,etadim,xidim))*np.nan
notidesnowecvort_mask = np.ones((len(notidesnowec_files),tdim,1,etadim,xidim))*np.nan
notidesnowecvort_cmask = np.ones((len(notidesnowec_files),tdim,1,etadim,xidim))*np.nan

# ==========================================
# 4. Iteratively Process WEC Files
# ==========================================
print(f'Processing {len(tideswec_files)} WEC files...')

for f_idx, file in enumerate(tideswec_files):
    print(f'  -> WEC {f_idx+1}/{len(tideswec_files)}: {file}')
    urho, vrho = pf.rho_uv_angle_surf(file, grd, rotate=True)
    vort = pf.vorticity(grd, urho, vrho)
    norm_vort_mask = (vort / f_nc) * masknc 
    norm_vort_cmask = (vort / f_nc) * coastal_mask 
    wecvort_mask[f_idx,:,:,:,:] = norm_vort_mask
    wecvort_cmask[f_idx,:,:,:,:] = norm_vort_cmask
    
# Normalize and Save WEC array
np.savez(out_tideswec_path, vort_mask=wecvort_mask, vort_cmask=wecvort_cmask)
print(f'Saved WEC data to {out_tideswec_path}')

# ==========================================
# 5. Iteratively Process no-WEC Files
# ==========================================
print(f'Processing {len(tidesnowec_files)} WEC files...')

for f_idx, file in enumerate(tidesnowec_files):
    print(f'  -> WEC {f_idx+1}/{len(tidesnowec_files)}: {file}')
    urho, vrho = pf.rho_uv_angle_surf(file, grd, rotate=True)
    vort = pf.vorticity(grd, urho, vrho)
    norm_vort_mask = (vort / f_nc) * masknc 
    norm_vort_cmask = (vort / f_nc) * coastal_mask 
    nowecvort_mask[f_idx,:,:,:,:] = norm_vort_mask
    nowecvort_cmask[f_idx,:,:,:,:] = norm_vort_cmask
    
# Normalize and Save WEC array
np.savez(out_tidesnowec_path, vort_mask=nowecvort_mask, vort_cmask=nowecvort_cmask)
print(f'Saved no-WEC data to {out_tidesnowec_path}')

# ==========================================
# 6. Iteratively Process no tides no-WEC Files
# ==========================================
print(f'Processing {len(notidesnowec_files)} WEC files...')

for f_idx, file in enumerate(notidesnowec_files):
    print(f'  -> WEC {f_idx+1}/{len(notidesnowec_files)}: {file}')
    urho, vrho = pf.rho_uv_angle_surf(file, grd, rotate=True)
    vort = pf.vorticity(grd, urho, vrho)
    norm_vort_mask = (vort / f_nc) * masknc 
    norm_vort_cmask = (vort / f_nc) * coastal_mask 
    notidesnowecvort_mask[f_idx,:,:,:,:] = norm_vort_mask
    notidesnowecvort_cmask[f_idx,:,:,:,:] = norm_vort_cmask
    
# Normalize and Save WEC array
np.savez(out_notidesnowec_path, vort_mask=notidesnowecvort_mask, vort_cmask=notidesnowecvort_cmask)
print(f'Saved no-WEC data to {out_notidesnowec_path}')

print('All calculations complete.')
