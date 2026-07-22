"""
Diagonal cross-section from coast to offshore using z-sliced data.

The transect starts at (ETA0, XI0) on the coast and extends in the -xi
direction (offshore). SLOPE = deta/dxi controls the diagonal angle:
  SLOPE =  0   → purely zonal (no eta change as xi decreases)
  SLOPE =  0.5 → eta decreases as you go offshore (southward for typical coast)
  SLOPE = -0.5 → eta increases as you go offshore (northward)

Output: ./figs/cs_zslice_<SCENARIO>_<VAR>.png
"""

import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.ndimage import map_coordinates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCENARIO  = 'notidesnowec'
VAR       = 'NO3'                   # variable name in zslice file
DEPTH_LIM = -300                    # y-axis bottom (m), e.g. -300
CMAP      = 'viridis'

# Transect geometry (all in grid index space)
ETA0      = 600                     # starting eta index (on or near coast)
XI0       = 500                     # starting xi index (on or near coast)
SLOPE     = 0.3                     # deta/dxi  (eta change per xi step)
LENGTH_XI = 200                     # how far offshore in xi cells
N_PTS     = 300                     # interpolation resolution along transect

# Paths
GRD       = 'mc60_grd.nc'
ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'

# ---------------------------------------------------------------------------
# Build transect coordinates
# Transect goes in -xi direction: xi decreases from XI0 to XI0 - LENGTH_XI
# ---------------------------------------------------------------------------
xi_pts  = np.linspace(XI0,  XI0  - LENGTH_XI, N_PTS)
eta_pts = np.linspace(ETA0, ETA0 + SLOPE * (-LENGTH_XI), N_PTS)
coords  = np.array([eta_pts, xi_pts])   # shape (2, N_PTS) for map_coordinates

# ---------------------------------------------------------------------------
# Load grid — lat/lon for distance axis and land mask for validity check
# ---------------------------------------------------------------------------
grd      = Dataset(GRD)
lat      = np.array(grd['lat_rho'][:])
lon      = np.array(grd['lon_rho'][:])
mask_rho = np.array(grd['mask_rho'][:])   # 1=ocean, 0=land

# interpolate lat/lon along transect for distance axis
lat_t = map_coordinates(lat, coords, order=1, mode='nearest')
lon_t = map_coordinates(lon, coords, order=1, mode='nearest')

dlat   = np.diff(lat_t) * 111.0
dlon   = np.diff(lon_t) * 111.0 * np.cos(np.deg2rad(lat_t[:-1]))
dist_km = np.concatenate([[0], np.cumsum(np.sqrt(dlat**2 + dlon**2))])

# ocean mask along transect (True = ocean)
mask_t = map_coordinates(mask_rho.astype(float), coords, order=0, mode='nearest') > 0.5

# ---------------------------------------------------------------------------
# Load and interpolate z-sliced files
# ---------------------------------------------------------------------------
zfiles = sorted(glob.glob(f'{ZSLICE_ROOT}/{SCENARIO}/z_mc60_bgc.*.nc'))
if not zfiles:
    zfiles = sorted(glob.glob(f'{ZSLICE_ROOT}/{SCENARIO}/z_mc60_his.*.nc'))

print(f'Found {len(zfiles)} zslice files for {SCENARIO}')

sum_field = None
count     = 0
depth     = None

for f in zfiles:
    with Dataset(f) as nc:
        if VAR not in nc.variables:
            continue
        if depth is None:
            depth = np.array(nc.variables['depth'][:])   # (n_z,) negative downward
        arr = np.array(nc.variables[VAR][:])              # (t, n_z, eta, xi)
        # replace fill values with NaN
        arr = np.where(np.abs(arr) > 1e30, np.nan, arr)
        # time-mean within this file
        arr_mean = np.nanmean(arr, axis=0)                # (n_z, eta, xi)

        # interpolate each depth level along the transect
        n_z = arr_mean.shape[0]
        section = np.full((n_z, N_PTS), np.nan)
        for iz in range(n_z):
            layer = arr_mean[iz]
            # NaN cells break map_coordinates — fill with 0 temporarily, then mask
            nan_mask = np.isnan(layer)
            layer_filled = np.where(nan_mask, 0.0, layer)
            row = map_coordinates(layer_filled, coords, order=1, mode='nearest')
            # mask points that were NaN in the source or are on land
            nan_along = map_coordinates(nan_mask.astype(float), coords,
                                        order=1, mode='nearest') > 0.5
            row[nan_along | ~mask_t] = np.nan
            section[iz] = row

        if sum_field is None:
            sum_field = np.where(np.isnan(section), 0.0, section)
            count     = (~np.isnan(section)).astype(int)
        else:
            sum_field += np.where(np.isnan(section), 0.0, section)
            count     += (~np.isnan(section)).astype(int)

    print(f'  processed {f.split("/")[-1]}')

mean_section = np.where(count > 0, sum_field / np.maximum(count, 1), np.nan)

# apply depth limit
depth_mask = depth >= DEPTH_LIM
depth_plot = depth[depth_mask]
section_plot = mean_section[depth_mask, :]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))

pc = ax.pcolormesh(dist_km, depth_plot, section_plot,
                   shading='nearest', cmap=CMAP)
plt.colorbar(pc, ax=ax, label=VAR)

ax.set_xlabel('distance from coast (km)')
ax.set_ylabel('depth (m)')
ax.set_ylim([DEPTH_LIM, 0])
ax.set_title(f'{SCENARIO} — {VAR} cross-section\n'
             f'start (eta={ETA0}, xi={XI0}), slope={SLOPE}, length={LENGTH_XI} xi-cells')
ax.grid(True, alpha=0.2)

out = f'./figs/cs_zslice_{SCENARIO}_{VAR}.png'
plt.tight_layout()
plt.savefig(out, dpi=600, bbox_inches='tight')
print(f'saved -> {out}')
