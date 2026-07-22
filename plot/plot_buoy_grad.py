"""
Horizontal buoyancy gradient magnitude |∇b| maps from zsliced his files.
2×2 layout per depth: top-left = notidesnowec (absolute); other three = difference.
One PNG per depth: buoy_grad_surface.png, buoy_grad_10m.png, buoy_grad_20m.png, buoy_grad_50m.png

Gradient computed on time-mean density: |∇b| = (g/ρ₀) √( (∂ρ/∂x)² + (∂ρ/∂y)² )
pm (1/dx) and pn (1/dy) from the ROMS grid convert index differences to physical gradients.
"""

import os
import glob
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean

ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'
GRD         = 'mc60_grd.nc'
CACHE       = './figs/buoy_grad_cache.npz'
G           = 9.81
RHO0        = 1025.0
REF         = 'notidesnowec'
DEPTHS_M    = [0, 10, 20, 50]   # m below surface; 0 = surface

XLIM      = [-122.4, -121.78]
YLIM      = [36.47, 37.06]
LON_TICKS = np.array([-122.4, -122.1, -121.8])
LAT_TICKS = np.array([36.47, 36.6, 36.8, 37.0])

SCENARIOS = ['notidesnowec', 'tideswec', 'tidesnowec', 'notideswec']
LABELS = {
    'notidesnowec': 'no tides, no WEC',
    'tideswec':     'tides, WEC',
    'tidesnowec':   'tides, no WEC',
    'notideswec':   'no tides, WEC',
}
DEPTH_LABELS = {0: 'surface', 10: '10m', 20: '20m', 50: '50m'}


def _fill_to_nan(arr):
    return np.where(np.abs(arr) > 1e30, np.nan, arr)


def find_depth_idx(depth_arr, target_m):
    """Index of depth level closest to -target_m (depth_arr is negative downward)."""
    return int(np.argmin(np.abs(depth_arr + target_m)))


def time_mean_rho(scen, depth_idxs):
    """
    Time-mean rho at each depth index across all zsliced his files.
    Returns {idx: (eta, xi) mean array}.
    """
    zfiles = sorted(glob.glob(f'{ZSLICE_ROOT}/{scen}/z_mc60_his.*.nc'))
    if not zfiles:
        raise FileNotFoundError(f'No his zslice files found for {scen}')
    sums   = {i: None for i in depth_idxs}
    counts = {i: 0    for i in depth_idxs}
    for zf in zfiles:
        with Dataset(zf) as nc:
            rho_all = _fill_to_nan(np.array(nc.variables['rho'][:]))
        if rho_all.ndim == 3:
            rho_all = rho_all[np.newaxis]
        for t in range(rho_all.shape[0]):
            for i in depth_idxs:
                sl    = rho_all[t, i, :, :]
                valid = ~np.isnan(sl)
                if sums[i] is None:
                    sums[i] = np.zeros(sl.shape, dtype=np.float64)
                sums[i][valid] += sl[valid]
                counts[i] += valid.astype(np.int64)
    return {i: np.where(counts[i] > 0, sums[i] / np.maximum(counts[i], 1), np.nan)
            for i in depth_idxs}


def buoy_grad_mag(rho_2d, pm, pn):
    """Horizontal buoyancy gradient magnitude in s⁻²."""
    drho_dx = np.gradient(rho_2d, axis=1) * pm
    drho_dy = np.gradient(rho_2d, axis=0) * pn
    return (G / RHO0) * np.hypot(drho_dx, drho_dy)


# ── grid ──────────────────────────────────────────────────────────────────────
grd      = Dataset(GRD)
lon_rho  = np.array(grd.variables['lon_rho'])
lat_rho  = np.array(grd.variables['lat_rho'])
mask_rho = np.array(grd.variables['mask_rho'])
pm       = np.array(grd.variables['pm'])
pn       = np.array(grd.variables['pn'])
land     = np.where(mask_rho == 0, 1.0, np.nan)

# ── cache ─────────────────────────────────────────────────────────────────────
cache_keys = [f'{s}_{d}' for s in SCENARIOS for d in DEPTHS_M]
bg = {}

if os.path.exists(CACHE):
    cc = np.load(CACHE)
    if all(k in cc for k in cache_keys) and all(cc[k].ndim == 2 for k in cache_keys):
        for s in SCENARIOS:
            bg[s] = {d: cc[f'{s}_{d}'] for d in DEPTHS_M}
        print(f'Loaded cache from {CACHE}')
    else:
        print('Cache invalid — recomputing')

if not bg:
    zf0 = sorted(glob.glob(f'{ZSLICE_ROOT}/{REF}/z_mc60_his.*.nc'))[0]
    with Dataset(zf0) as nc:
        depth_arr = np.array(nc.variables['depth'][:])
    didx = {d: find_depth_idx(depth_arr, d) for d in DEPTHS_M}
    print(f'depth indices: { {d: (didx[d], depth_arr[didx[d]]) for d in DEPTHS_M} }')

    for scen in SCENARIOS:
        print(f'  loading {scen} ...')
        means = time_mean_rho(scen, list(didx.values()))
        bg[scen] = {}
        for d, i in didx.items():
            rho_m = np.where(mask_rho == 1, means[i], np.nan)
            bg[scen][d] = buoy_grad_mag(rho_m, pm, pn)

    np.savez(CACHE, **{f'{s}_{d}': bg[s][d] for s in SCENARIOS for d in DEPTHS_M})
    print(f'Saved cache -> {CACHE}')

# ── plot one figure per depth ─────────────────────────────────────────────────
plt.rcParams.update({'font.size': 14})
proj = ccrs.PlateCarree()

for depth_m in DEPTHS_M:
    data = {s: bg[s][depth_m] for s in SCENARIOS}

    all_vals = np.concatenate([data[s].ravel() for s in SCENARIOS])
    vmax = np.nanpercentile(all_vals[np.isfinite(all_vals)], 99)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10),
                             subplot_kw=dict(projection=proj),
                             gridspec_kw=dict(hspace=0.05, wspace=0.35))

    pc = None
    for i, (ax, scen) in enumerate(zip(axes.ravel(), SCENARIOS)):
        row, col = divmod(i, 2)

        pc = ax.pcolormesh(lon_rho, lat_rho, data[scen],
                           cmap=cmocean.cm.amp, vmin=0, vmax=vmax,
                           transform=proj, shading='nearest')
        ax.pcolormesh(lon_rho, lat_rho, land, cmap='gray', vmin=0, vmax=1,
                      transform=proj, shading='nearest', zorder=2)
        ax.contour(lon_rho, lat_rho, mask_rho, levels=[0.5],
                   colors='k', linewidths=0.5, transform=proj, zorder=3)

        ax.set_extent([XLIM[0], XLIM[1], YLIM[0], YLIM[1]], crs=proj)
        ax.set_title(LABELS[scen], fontsize=14)
        ax.set_xticks(LON_TICKS, crs=proj)
        ax.set_yticks(LAT_TICKS, crs=proj)
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(labelbottom=(row == 1), labelleft=(col == 0), labelsize=14)

    fig.canvas.draw()

    pos01 = axes[0, 1].get_position()
    pos11 = axes[1, 1].get_position()
    cax = fig.add_axes([pos01.x1 + 0.02, pos11.y0, 0.01, pos01.y1 - pos11.y0])
    fig.colorbar(pc, cax=cax, label=r'|$\nabla b$| (s$^{-2}$)')

    dlabel = DEPTH_LABELS[depth_m]
    fig.suptitle(f'Horizontal buoyancy gradient |∇b|, time mean — {dlabel}', fontsize=14)

    out = f'./figs/buoy_grad_{dlabel}.png'
    plt.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'saved -> {out}')
