# smooth_wec.py
import shutil
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
plt.ion()

# -------------------------------------------------------------
# Paths and filenames
# -------------------------------------------------------------
grd = '/data/project9/minnaho/swel/mc60_grd.nc'
wec_path = '/data/project9/minnaho/swel/tides/mc60/frc/'
wec_file = 'mc60_wec.20190415.nc'

# Copy original WEC file to new smoothed version
#shutil.copy2(wec_path + wec_file, wec_path + 'mc60_wec_smooth.20190415.nc')

# Open grids and WEC files
grd_nc = Dataset(grd, 'r')
mask = np.array(grd_nc.variables['mask_rho'])
lat = np.array(grd_nc.variables['lat_rho'])
lon = np.array(grd_nc.variables['lon_rho'])

nc_in = Dataset(wec_path + wec_file, 'r')
nc_out = Dataset(wec_path + 'mc60_wec_smooth.20190415.nc', 'r+')

# List of variables
var_names = list(nc_in.variables.keys())

# -------------------------------------------------------------
# Function for masked Gaussian smoothing
# -------------------------------------------------------------
def smooth_masked(data, mask, sigma=1):
    """
    Smooth a 2D array with Gaussian filter while respecting a mask.
    Mask should be 1 for ocean, 0 for land.
    """
    # Apply mask: land=0
    data_masked = np.where(mask, data, 0)
    # Smooth data and mask separately
    smooth_data = gaussian_filter(data_masked, sigma=sigma)
    smooth_mask = gaussian_filter(mask.astype(float), sigma=sigma)
    # Normalize so land doesn't bleed into ocean
    smooth_norm = np.where(smooth_mask > 0, smooth_data / smooth_mask, 0)
    return smooth_norm

# -------------------------------------------------------------
# Loop over time and variables
# -------------------------------------------------------------
for t_i in range(nc_in.variables['wwv_time'].shape[0]):
    for d_i, var_name in enumerate(var_names):
        print(f'Time {t_i+1}, Variable {d_i+1}/{len(var_names)}: {var_name}')
        if var_name == 'wwv_time':
            continue
        
        var_data = np.array(nc_in.variables[var_name][t_i, :, :])
        # sigma = 8 is 60 m * 8 
        # roughly half a kilometer smoothing
        smoothed = smooth_masked(var_data, mask, sigma=8)
        
        nc_out.variables[var_name][t_i, :, :] = smoothed*mask


#fig,ax = plt.subplots(1,1)
##pplt = ax.pcolormesh(lonnc,latnc,lap,cmap='bwr',vcenter=0)
##pplt = ax.pcolormesh(lap, cmap='bwr', norm=norm)
#pplt = ax.pcolormesh(smoothed*mask,cmap='rainbow')
#ax.contour(mask,[0])
#ax.set_title('smoothed')
#fig.colorbar(pplt)
#
#fig,ax = plt.subplots(1,1)
##pplt = ax.pcolormesh(lonnc,latnc,lap,cmap='bwr',vcenter=0)
##pplt = ax.pcolormesh(lap, cmap='bwr', norm=norm)
#pplt = ax.pcolormesh(var_data*mask,cmap='rainbow')
#ax.contour(mask,[0])
#fig.colorbar(pplt)
#
#fig,ax = plt.subplots(1,1)
##pplt = ax.pcolormesh(lonnc,latnc,lap,cmap='bwr',vcenter=0)
##pplt = ax.pcolormesh(lap, cmap='bwr', norm=norm)
#pplt = ax.pcolormesh((smoothed-var_data)*mask,cmap='bwr',vmin=np.nanmin((smoothed-var_data)*mask),vmax=-np.nanmin((smoothed-var_data)*mask))
#ax.contour(mask,[0])
#fig.colorbar(pplt)

