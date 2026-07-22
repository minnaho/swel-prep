"""
Eddy vertical TIN flux: <w'(NO3'+NH4')> = <w(NO3+NH4)> - <w>(<NO3>+<NH4>).
2×2 layout per depth — all four scenarios absolute.
One PNG per depth: figs/wDIN_eddy_surface.png, ..._10m.png, ..._20m.png, ..._50m.png
Cache: figs/wDIN_eddy_cache.npz
"""

import os
import glob
import re
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
CACHE       = './figs/wDIN_eddy_cache.npz'
DEPTHS_M    = [0, 10, 20, 50]

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
    return int(np.argmin(np.abs(depth_arr + target_m)))


def ts_key(path):
    m = re.search(r'z_mc60_(?:his|bgc)\.(\d+)\.nc', os.path.basename(path))
    return m.group(1) if m else None


# ── grid ──────────────────────────────────────────────────────────────────────
grd      = Dataset(GRD)
lon_rho  = np.array(grd.variables['lon_rho'])
lat_rho  = np.array(grd.variables['lat_rho'])
mask_rho = np.array(grd.variables['mask_rho'])
grd.close()
land  = np.where(mask_rho == 0, 1.0, np.nan)
shape = lon_rho.shape   # (eta_rho, xi_rho)

# ── cache ─────────────────────────────────────────────────────────────────────
cache_keys = [f'{s}_{d}' for s in SCENARIOS for d in DEPTHS_M]
flux = {}

if os.path.exists(CACHE):
    cc = np.load(CACHE)
    if all(k in cc for k in cache_keys) and all(cc[k].ndim == 2 for k in cache_keys):
        for s in SCENARIOS:
            flux[s] = {d: cc[f'{s}_{d}'] for d in DEPTHS_M}
        print(f'Loaded cache from {CACHE}')
    else:
        print('Cache invalid — recomputing')

if not flux:
    ref_files = sorted(glob.glob(f'{ZSLICE_ROOT}/notidesnowec/z_mc60_his.*.nc'))
    with Dataset(ref_files[0]) as nc:
        depth_arr = np.array(nc.variables['depth'][:])
    didx = {d: find_depth_idx(depth_arr, d) for d in DEPTHS_M}
    print(f'depth indices: { {d: (didx[d], float(depth_arr[didx[d]])) for d in DEPTHS_M} }')

    for scen in SCENARIOS:
        his_files = sorted(glob.glob(f'{ZSLICE_ROOT}/{scen}/z_mc60_his.*.nc'))
        bgc_files = sorted(glob.glob(f'{ZSLICE_ROOT}/{scen}/z_mc60_bgc.*.nc'))
        if not his_files or not bgc_files:
            raise FileNotFoundError(f'Missing zslice files for {scen}')

        his_dict = {ts_key(f): f for f in his_files}
        bgc_dict = {ts_key(f): f for f in bgc_files}
        common   = sorted(set(his_dict) & set(bgc_dict))
        print(f'  {scen}: {len(common)} matched timestep files')

        sum_w   = {d: np.zeros(shape, dtype=np.float64) for d in DEPTHS_M}
        sum_tin = {d: np.zeros(shape, dtype=np.float64) for d in DEPTHS_M}
        sum_wt  = {d: np.zeros(shape, dtype=np.float64) for d in DEPTHS_M}
        cnt     = {d: np.zeros(shape, dtype=np.int64)   for d in DEPTHS_M}

        for ts in common:
            with Dataset(his_dict[ts]) as hnc:
                with Dataset(bgc_dict[ts]) as bnc:
                    for d in DEPTHS_M:
                        i    = didx[d]
                        w_t  = _fill_to_nan(np.array(hnc.variables['w'][:, i, :, :]))
                        no3  = _fill_to_nan(np.array(bnc.variables['NO3'][:, i, :, :]))
                        nh4  = _fill_to_nan(np.array(bnc.variables['NH4'][:, i, :, :]))
                        tin  = no3 + nh4
                        valid = ~(np.isnan(w_t) | np.isnan(tin))
                        sum_w[d]   += np.where(valid, w_t,       0.0).sum(axis=0)
                        sum_tin[d] += np.where(valid, tin,       0.0).sum(axis=0)
                        sum_wt[d]  += np.where(valid, w_t * tin, 0.0).sum(axis=0)
                        cnt[d]     += valid.sum(axis=0).astype(np.int64)

        flux[scen] = {}
        for d in DEPTHS_M:
            n        = np.maximum(cnt[d], 1)
            mean_w   = np.where(cnt[d] > 0, sum_w[d]  / n, np.nan)
            mean_tin = np.where(cnt[d] > 0, sum_tin[d] / n, np.nan)
            mean_wt  = np.where(cnt[d] > 0, sum_wt[d]  / n, np.nan)
            eddy     = mean_wt - mean_w * mean_tin
            flux[scen][d] = np.where(mask_rho == 1, eddy, np.nan)

    np.savez(CACHE, **{f'{s}_{d}': flux[s][d] for s in SCENARIOS for d in DEPTHS_M})
    print(f'Saved cache -> {CACHE}')

# ── plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 14})
proj = ccrs.PlateCarree()

for depth_m in DEPTHS_M:
    data = {s: flux[s][depth_m] for s in SCENARIOS}

    all_vals = np.concatenate([data[s].ravel() for s in SCENARIOS])
    finite   = all_vals[np.isfinite(all_vals)]
    vmax     = np.nanpercentile(np.abs(finite), 99)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10),
                             subplot_kw=dict(projection=proj),
                             gridspec_kw=dict(hspace=0.05, wspace=0.35))

    pc = None
    for i, (ax, scen) in enumerate(zip(axes.ravel(), SCENARIOS)):
        row, col = divmod(i, 2)
        pc = ax.pcolormesh(lon_rho, lat_rho, data[scen],
                           cmap=cmocean.cm.balance, vmin=-vmax, vmax=vmax,
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
    fig.colorbar(pc, cax=cax,
                 label=r"$\langle w'(\mathrm{NO}_3'+\mathrm{NH}_4')\rangle$"
                       r" (mmol m$^{-2}$ s$^{-1}$)")

    dlabel = DEPTH_LABELS[depth_m]
    fig.suptitle(
        r"$\langle w'(\mathrm{NO}_3'+\mathrm{NH}_4')\rangle$"
        f' z = {dlabel}',
        fontsize=14)

    out = f'./figs/wDIN_eddy_{dlabel}.png'
    plt.savefig(out, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'saved -> {out}')
