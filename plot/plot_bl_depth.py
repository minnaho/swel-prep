"""
Maps of time-mean surface and bottom boundary-layer depth, from the npz
files written by postprocessing/calc_bl_depth_sbl.py and calc_bl_depth_bbl.py
(run those first, once per scenario, before this script) -- diagnosed via
the Akt > 1e-4 m2/s criterion (see those scripts' docstrings for the
SBL/BBL definitions and why SBL uses zsliced Akt while BBL uses raw-file
Akt on native s_w levels). The two boundaries are loaded from separate npz
families, so sbl_depth.png renders fully as soon as every bl_depth_sbl_*.npz
exists, whether or not the (much slower) bl_depth_bbl_*.npz jobs are done.

Two figures, each a 3x2 cartopy grid (one panel per scenario, layout/style
matching plot_map_w_rms_pycnocline.py):

  sbl_depth.png -- time-mean surface boundary layer depth (m)
  bbl_depth.png -- time-mean bottom boundary layer depth (m)

Both fields are positive-definite (a depth extent), so a sequential
colormap is used, with a shared 0-99th-percentile color range across all
present panels within each figure -- consistent with the rest of plot/.
"""

import os
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean

import scenario_style as ss

GRD = 'mc60_grd.nc'
SAVEPATH = './figs/'
NPZ_DIR = '../postprocessing/'

SCENARIOS = ['notidesnowec', 'tideswec', 'tidesnowec',
             'notideswec', 'ampwec', 'tidesampwec']

XLIM = [-122.4, -121.78]
YLIM = [36.47, 37.06]
LON_TICKS = np.array([-122.4, -122.1, -121.8])
LAT_TICKS = np.array([36.47, 36.6, 36.8, 37.0])

# (row, col, show_lat, show_lon) for a 3x2 grid, in SCENARIOS order
GRID_POS = [(0, 0, True, False), (0, 1, False, False),
            (1, 0, True, False), (1, 1, False, False),
            (2, 0, True, True), (2, 1, False, True)]

DEPTH_CMAP = cmocean.cm.deep

# ---------------------------------------------------------------------------
# Load npz outputs -- SBL and BBL are independent npz families (see module
# docstring), each scenario loaded/warned about separately
# ---------------------------------------------------------------------------
def _load_family(prefix, calc_script):
    out = {}
    for scen in SCENARIOS:
        path = os.path.join(NPZ_DIR, f'{prefix}_{scen}.npz')
        if not os.path.exists(path):
            print(f'WARNING: missing {path} -- run {calc_script} {scen} first')
            out[scen] = None
            continue
        out[scen] = np.load(path)
    return out


data_sbl = _load_family('bl_depth_sbl', 'calc_bl_depth_sbl.py')
data_bbl = _load_family('bl_depth_bbl', 'calc_bl_depth_bbl.py')

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
grd = Dataset(GRD)
lon_rho = np.array(grd.variables['lon_rho']) - 360
lat_rho = np.array(grd.variables['lat_rho'])
mask_rho = np.array(grd.variables['mask_rho'])
land = np.where(mask_rho == 0, 1.0, np.nan)

os.makedirs(SAVEPATH, exist_ok=True)
plt.rcParams.update({'font.size': 14})
proj = ccrs.PlateCarree()


def _new_grid():
    fig, axes = plt.subplots(3, 2, figsize=(10, 13),
                             subplot_kw=dict(projection=proj),
                             gridspec_kw=dict(hspace=0.15, wspace=0.4))
    return fig, axes


def _style_ax(ax, show_lat, show_lon):
    ax.pcolormesh(lon_rho, lat_rho, land, cmap='gray', vmin=0, vmax=1,
                  transform=proj, shading='nearest', zorder=2)
    ax.contour(lon_rho, lat_rho, mask_rho, levels=[0.5],
               colors='k', linewidths=0.5, transform=proj, zorder=3)
    ax.set_extent([XLIM[0], XLIM[1], YLIM[0], YLIM[1]], crs=proj)
    ax.set_xticks(LON_TICKS, crs=proj)
    ax.set_yticks(LAT_TICKS, crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelbottom=show_lon, labelleft=show_lat, labelsize=13)


def plot_depth_grid(data, key, out_name, cbar_label):
    fig, axes = _new_grid()

    present = [scen for scen in SCENARIOS if data[scen] is not None]
    if not present:
        print(f'WARNING: no npz files found for {key} -- skipping {out_name}')
        plt.close(fig)
        return
    vmax = np.nanpercentile(
        np.concatenate([data[s][key].ravel() for s in present]), 99)

    pc = None
    for scen, (row, col, show_lat, show_lon) in zip(SCENARIOS, GRID_POS):
        ax = axes[row, col]
        ax.set_title(ss.label(scen), fontsize=13)
        if data[scen] is None:
            ax.text(0.5, 0.5, 'not yet\ncomputed', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_extent([XLIM[0], XLIM[1], YLIM[0], YLIM[1]], crs=proj)
            _style_ax(ax, show_lat, show_lon)
        else:
            pc = ax.pcolormesh(lon_rho, lat_rho, data[scen][key],
                               cmap=DEPTH_CMAP, vmin=0, vmax=vmax,
                               transform=proj, shading='nearest')
            _style_ax(ax, show_lat, show_lon)

    fig.canvas.draw()
    pos_top = axes[0, 1].get_position()
    pos_bot = axes[2, 1].get_position()
    cax = fig.add_axes([pos_top.x1 + 0.02, pos_bot.y0, 0.02, pos_top.y1 - pos_bot.y0])
    if pc is not None:
        fig.colorbar(pc, cax=cax, orientation='vertical', label=cbar_label)

    out = f'{SAVEPATH}{out_name}'
    plt.savefig(out, bbox_inches='tight', dpi=800)
    plt.close(fig)
    print(f'saved -> {out}')


plot_depth_grid(data_sbl, 'sbl_mean', 'sbl_depth.png',
                r'Surface boundary layer thickness, $K_t$ > 10$^{-4}$ m$^2$ s$^{-1}$ (m)')
plot_depth_grid(data_bbl, 'bbl_mean', 'bbl_depth.png',
                r'Bottom boundary layer thickness, $K_t$ > 10$^{-4}$ m$^2$ s$^{-1}$ (m)')
