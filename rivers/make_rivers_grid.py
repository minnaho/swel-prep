# make river locations from scratch
# Monterey Bay

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from collections import deque
plt.ion()

# rivers:
#		River 01 = "Carmel River" ;
#		River 02 = "Salinas River" ;
#		River 03 = "Elkhorn Slough" ;
#		River 04 = "Pajaro River" ;
#		River 05 = "Soquel Creek" ;
#		River 06 = "San Lorenzo River" ;
#		River 07 = "Scott Creek" ;
#		River 08 = "Waddell beach" ;

# manually looked up each on Google Maps
rlats = [36.536193,36.750960,36.807486,36.845233,36.971238,36.963483,37.041162,37.094633]
rlons = [-121.928496,-121.804904,-121.788303,-121.807558,-121.951972,-122.01286,-122.230858,-122.277816]

ch_grd_nm = '/data/project9/minnaho/swel/mc60_newlarge_grd.nc'
ch_grd = Dataset(ch_grd_nm,'r+')

clon = np.array(ch_grd.variables['lon_rho'])-360
clat = np.array(ch_grd.variables['lat_rho'])
cmask = np.array(ch_grd.variables['mask_rho'])

# had to add this because tethys down...
def calc_ij(nc_grd,lat_sites,lon_sites):

    lon_nc = nc_grd.variables['lon_rho'][:,:]-360
    lat_nc = nc_grd.variables['lat_rho'][:,:]

    nsites = len(lat_sites)
    isites = np.ones(nsites)*np.nan
    jsites = np.ones(nsites)*np.nan

    for s in range(nsites):
        ##################################
        # FIND SITE IN GRIDPOINTS
        ####################################
        min_1D = np.abs( (lat_nc - lat_sites[s])**2 + (lon_nc - lon_sites[s])**2)
        y_site, x_site = np.unravel_index(min_1D.argmin(), min_1D.shape)
        isites[s] = x_site
        jsites[s] = y_site

    return isites, jsites

ch_xi,ch_eta = calc_ij(ch_grd,rlats,rlons)

# move all river points on water to land
if 1 in cmask[ch_eta.astype(int),ch_xi.astype(int)]:
    plt.imshow(cmask,origin='lower')
    distance, indices = distance_transform_edt(cmask == 1, return_indices=True)
    for c_i in range(len(cmask[ch_eta.astype(int),ch_xi.astype(int)])):
        if cmask[ch_eta.astype(int),ch_xi.astype(int)][c_i] == 1:
            nearest_land = (indices[0][ch_eta[c_i].astype(int),ch_xi[c_i].astype(int)], indices[1][ch_eta[c_i].astype(int),ch_xi[c_i].astype(int)])
            print("Closest land to eta, xi:",ch_eta[c_i].astype(int),ch_xi[c_i].astype(int)," is at ",nearest_land)
            print('moving to that point now')
            ch_eta[c_i] = nearest_land[0]
            ch_xi[c_i] = nearest_land[1]

# check all land points are next to water
# and if not, move them to the next land point next to water that 
# doesn't already have a river flux
def find_nearest_zero_next_to_one_unique(mask, x_list, y_list, max_search_radius=10):

    rows, cols = mask.shape
    results = []
    used_coords = set()  # Track already-used coordinates

    def is_next_to_one(y, x):
        neighbors = [
            (y - 1, x),
            (y + 1, x),
            (y, x - 1),
            (y, x + 1)
        ]
        for ny, nx in neighbors:
            if 0 <= ny < rows and 0 <= nx < cols:
                if mask[ny, nx] == 1:
                    return True
        return False

    for x0, y0 in zip(x_list, y_list):
        if mask[y0, x0] != 0:
            results.append((False, None))  # not a 0 to begin with
            continue

        # Case 1: the original 0 is next to a 1 and not already used
        if is_next_to_one(y0, x0) and (x0, y0) not in used_coords:
            used_coords.add((x0, y0))
            results.append((True, (x0, y0)))
            continue

        # Case 2: BFS to find nearest unused 0 next to 1
        visited = set()
        queue = deque()
        queue.append((x0, y0, 0))
        visited.add((x0, y0))
        found = False
        nearest_coords = None
        while queue:
            x, y, dist = queue.popleft()
            if dist > max_search_radius:
                break
            if mask[y, x] == 0 and is_next_to_one(y, x) and (x, y) not in used_coords:
                found = True
                nearest_coords = (x, y)
                used_coords.add((x, y))
                break
            # Check 4 neighbors
            neighbors = [
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1)
            ]
            for ny, nx in neighbors:
                if 0 <= ny < rows and 0 <= nx < cols:
                    if (nx, ny) not in visited and mask[ny, nx] == 0:
                        visited.add((nx, ny))
                        queue.append((nx, ny, dist + 1))
        if found:
            results.append((False, nearest_coords))
        else:
            results.append((False, None))
    return results

output = find_nearest_zero_next_to_one_unique(cmask,ch_xi.astype(int),ch_eta.astype(int))
for original, result in zip(zip(ch_xi.astype(int), ch_eta.astype(int)), output):
    status, coords = result
    print(f"Original: {original}, Next to 1: {status}, Nearest Zero Next to 1: {coords}")

# Move invalid river points to the nearest valid '0' next to '1'
for i, ((orig_x, orig_y), (status, new_coords)) in enumerate(zip(zip(ch_xi.astype(int), ch_eta.astype(int)), output)):
    if not status and new_coords is not None:
        print(f"Moving river point from ({orig_x}, {orig_y}) to nearest valid point {new_coords}")
        ch_xi[i] = new_coords[0]
        ch_eta[i] = new_coords[1]
    elif not status and new_coords is None:
        print(f"Warning: No valid neighbor found within search radius for point ({orig_x}, {orig_y})")

# manually adjust rivers, and make it so each river has 3 locations
river_id = [1.3333334, 1.3333334, 1.3333334, 2.3333333, 2.3333333, 2.3333333, 3.3333333, 3.3333333, 3.3333333, 4.3333335, 4.3333335, 4.3333335, 5.3333335, 5.3333335, 5.3333335, 6.3333335, 6.3333335, 6.3333335, 7.3333335, 7.3333335, 7.3333335, 8.333333 , 8.333333 , 8.333333 ]

ch_xi = [286,287,287,621,621,622,686,686,686,691,691,691,595,596,597,505,506,507,276,277,277,259,259,259]
ch_eta = [107,108,109,392,393,394,476,479,480,564,568,563,855,855,855,882,882,882,1147,1146,1145,1201,1200,1199]


# Optionally plot moved points for verification
plt.figure()
plt.imshow(cmask, origin='lower', cmap='gray')
plt.scatter(ch_xi, ch_eta, c=river_id,cmap='tab10')
plt.colorbar(label='River ID')
plt.title("Adjusted River Points on Mask")

if 'river_flux' not in ch_grd.variables:
    river_flux_var = ch_grd.createVariable('river_flux', 'f4', ('eta_rho', 'xi_rho'))
else:
    river_flux_var = ch_grd.variables['river_flux']

river_flux_var[:, :] = 0.0

# Ensure your ch_eta and ch_xi are integers
ch_eta_int = ch_eta
ch_xi_int = ch_xi

for r_i in range(len(ch_eta_int)):
    river_flux_var[ch_eta_int[r_i], ch_xi_int[r_i]] = river_id[r_i]

# Add metadata (optional)
river_flux_var.long_name = "River volume flux partition"
print("river_flux variable created and initialized")

# Close the file after writing
ch_grd.close()

# check locations
ch_grd = Dataset(ch_grd_nm,'r')
crf2 = np.array(ch_grd.variables['river_flux'])

criv_eta = np.where(crf2>0)[0]
criv_xi = np.where(crf2>0)[1]

plt.figure()
plt.imshow(cmask, origin='lower', cmap='gray')
plt.scatter(criv_xi, criv_eta, color='red')
plt.legend()
plt.title("child grid rivers")



