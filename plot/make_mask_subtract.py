# create 10 km coastal mask

import numpy as np
from scipy.ndimage import binary_dilation
from netCDF4 import Dataset

grd = Dataset('/data/project3/minnaho/project9copy/swel/mc60_grd.nc','r')
mask_rho = np.array(grd.variables['mask_rho'])
pm = np.array(grd.variables['pm'])
pn = np.array(grd.variables['pn'])
lon_rho = np.array(grd.variables['lon_rho'])
lat_rho = np.array(grd.variables['lat_rho'])

ny, nx = mask_rho.shape

ocean = mask_rho == 1
land  = ~ocean

# coastline: ocean cells adjacent to land in any cardinal direction
coast = ocean & binary_dilation(land)

coastal_mask = np.zeros_like(mask_rho, dtype=bool)

for j in range(ny):
    row_ocean = ocean[j, :]
    coast_row = coast[j, :]
    dx_row    = 1.0 / pm[j, :]

    dist = np.full(nx, np.inf)
    dist[coast_row] = 0.0

    # left-to-right: propagate distances from west-facing coast cells eastward
    for i in range(1, nx):
        if row_ocean[i] and row_ocean[i - 1]:
            nd = dist[i - 1] + 0.5 * (dx_row[i - 1] + dx_row[i])
            if nd < dist[i]:
                dist[i] = nd

    # right-to-left: propagate distances from east-facing coast cells westward
    for i in range(nx - 2, -1, -1):
        if row_ocean[i] and row_ocean[i + 1]:
            nd = dist[i + 1] + 0.5 * (dx_row[i] + dx_row[i + 1])
            if nd < dist[i]:
                dist[i] = nd

    coastal_mask[j, :] = (dist <= 10000.0) & row_ocean


fnc = Dataset('coastal_mask_subtract.nc', 'w', format='NETCDF4')

# ---- Dimensions ----
fnc.createDimension('eta_rho', ny)
fnc.createDimension('xi_rho', nx)

# ---- Variables ----
lon = fnc.createVariable('lon_rho', 'f8', ('eta_rho','xi_rho'))
lat = fnc.createVariable('lat_rho', 'f8', ('eta_rho','xi_rho'))
mask = fnc.createVariable('coastal_mask', 'f8', ('eta_rho','xi_rho'))

# ---- Attributes ----
lon.long_name = "longitude of RHO-points"
lon.units = "degrees_east"

lat.long_name = "latitude of RHO-points"
lat.units = "degrees_north"

mask.long_name = "10 km coastal mask"
mask.description = "1 = ocean within 10 km of coast, 0 = elsewhere"
mask.coordinates = "lon_rho lat_rho"

# ---- Write data ----
lon[:] = lon_rho
lat[:] = lat_rho
mask[:] = coastal_mask.astype(np.int32)

# ---- Global attributes ----
fnc.title = "ROMS Monterey Bay Coastal Mask (10 km offshore)"
fnc.history = "Created by distance-to-coast calculation using pm/pn"
fnc.source = "ROMS grid"

fnc.close()
