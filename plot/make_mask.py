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

mask = mask_rho == 1
land = ~mask

coast = mask & binary_dilation(land)

dx = 1.0 / pm   # meters
dy = 1.0 / pn   # meters

import heapq

ny, nx = mask.shape
dist = np.full((ny, nx), np.inf)

# initialize queue with coastline points
queue = []

coast_idx = np.argwhere(coast)
for j, i in coast_idx:
    dist[j, i] = 0.0
    heapq.heappush(queue, (0.0, j, i))

# neighbor offsets (4 + diagonals)
neighbors = [(-1,0),(1,0),(0,-1),(0,1),
             (-1,-1),(-1,1),(1,-1),(1,1)]

while queue:
    d, j, i = heapq.heappop(queue)

    if d > dist[j, i]:
        continue

    for dj, di in neighbors:
        jn, in_ = j + dj, i + di

        if 0 <= jn < ny and 0 <= in_ < nx and mask[jn, in_]:
            
            # local distances
            dx_local = dx[j, i]
            dy_local = dy[j, i]

            if dj == 0:
                step = dx_local
            elif di == 0:
                step = dy_local
            else:
                step = np.sqrt(dx_local**2 + dy_local**2)

            new_d = d + step

            if new_d < dist[jn, in_]:
                dist[jn, in_] = new_d
                heapq.heappush(queue, (new_d, jn, in_))

coastal_mask = (dist <= 10000) & mask

fnc = Dataset('coastal_mask.nc', 'w', format='NETCDF4')

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
