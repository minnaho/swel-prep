"""
Spatial maps of sub-pycnocline vertical-velocity RMS and RMSE, from the npz
files written by postprocessing/calc_w_rms_pycnocline.py (run that first,
once per scenario, before this script).

Five figures, each a 3x2 cartopy grid (one panel per scenario, layout/style
matching plot_npp_depth_integrated.py):

  w_rms_pycnocline_full.png   -- RMS of w below the pycnocline, full column
  w_rms_pycnocline_300m.png   -- RMS of w, pycnocline-to-300m band only
  w_rmse_pycnocline_full.png  -- panel 1 = notidesnowec RMS (own colorbar),
                                  panels 2-6 = RMSE vs notidesnowec, full column
  w_rmse_pycnocline_300m.png  -- same, 300m band
  w_pycnocline_depth.png      -- time-mean pycnocline (1025 kg/m^3) depth,
                                  sanity-check diagnostic

RMS/RMSE are both positive-definite, so a sequential (not diverging) colormap
is used, with color limits from the 99th percentile across all panels sharing
a colorbar -- consistent with the rest of plot/.
"""

import os
import math
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
             'notideswec', 'notidesampwec', 'tidesampwec']
BASE_SCEN = 'notidesnowec'

XLIM = [-122.4, -121.78]
YLIM = [36.47, 37.06]
LON_TICKS = np.array([-122.4, -122.1, -121.8])
LAT_TICKS = np.array([36.47, 36.6, 36.8, 37.0])

# (row, col, show_lat, show_lon) for a 3x2 grid, in SCENARIOS order
GRID_POS = [(0, 0, True, False), (0, 1, False, False),
            (1, 0, True, False), (1, 1, False, False),
            (2, 0, True, True), (2, 1, False, True)]

RMS_CMAP = cmocean.cm.amp

# ---------------------------------------------------------------------------
# Load npz outputs
# ---------------------------------------------------------------------------
data = {}
for scen in SCENARIOS:
    path = os.path.join(NPZ_DIR, f'w_rms_pycnocline_{scen}.npz')
    if not os.path.exists(path):
        print(f'WARNING: missing {path} -- run calc_w_rms_pycnocline.py {scen} first')
        data[scen] = None
        continue
    data[scen] = np.load(path)

if data[BASE_SCEN] is None:
    raise RuntimeError(f'Base scenario {BASE_SCEN} npz is missing -- cannot proceed')

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


# ---------------------------------------------------------------------------
# Figure 1/2: RMS of w, all six scenarios, one shared colorbar
# ---------------------------------------------------------------------------
def plot_rms_grid(key, out_name, cbar_label):
    fig, axes = _new_grid()

    present = [scen for scen in SCENARIOS if data[scen] is not None]
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
        else:
            pc = ax.pcolormesh(lon_rho, lat_rho, data[scen][key],
                               cmap=RMS_CMAP, vmin=0, vmax=vmax,
                               transform=proj, shading='nearest')
            _style_ax(ax, show_lat, show_lon)
        if data[scen] is None:
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


plot_rms_grid('rms_w_full', 'w_rms_pycnocline_full.png',
              r'RMS $w$ below pycnocline (m s$^{-1}$)')
plot_rms_grid('rms_w_300m', 'w_rms_pycnocline_300m.png',
              r'RMS $w$, pycnocline to 300 m (m s$^{-1}$)')


# ---------------------------------------------------------------------------
# Figure 3/4: panel 1 = base RMS (own colorbar), panels 2-6 = RMSE vs base
# ---------------------------------------------------------------------------
def plot_rmse_grid(rms_key, rmse_key, out_name, raw_label, diff_label):
    diff_scens = [s for s in SCENARIOS if s != BASE_SCEN and data[s] is not None]
    n_panels = 1 + len(diff_scens)
    ncols = 2
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.3 * nrows),
                             subplot_kw=dict(projection=proj),
                             gridspec_kw=dict(hspace=0.15, wspace=0.4))
    axes_flat = np.atleast_1d(axes).flatten()
    for ax in axes_flat[n_panels:]:
        ax.axis('off')

    # panel 1: base case RMS
    ax_raw = axes_flat[0]
    base_field = data[BASE_SCEN][rms_key]
    raw_vmax = np.nanpercentile(base_field, 99)
    pc_raw = ax_raw.pcolormesh(lon_rho, lat_rho, base_field,
                               cmap=RMS_CMAP, vmin=0, vmax=raw_vmax,
                               transform=proj, shading='nearest')
    ax_raw.set_title(ss.label(BASE_SCEN), fontsize=13)
    _style_ax(ax_raw, True, False)

    # panels 2-6: RMSE vs base
    diff_vmax = np.nanpercentile(
        np.concatenate([data[s][rmse_key].ravel() for s in diff_scens]), 99)

    pc_diff = None
    for i, scen in enumerate(diff_scens, start=1):
        ax = axes_flat[i]
        row, col = divmod(i, ncols)
        pc_diff = ax.pcolormesh(lon_rho, lat_rho, data[scen][rmse_key],
                                cmap=RMS_CMAP, vmin=0, vmax=diff_vmax,
                                transform=proj, shading='nearest')
        ax.set_title(f'{ss.label(scen)}  vs  {ss.label(BASE_SCEN)}', fontsize=13)
        show_lat = (col == 0)
        show_lon = (row == nrows - 1) or (i + ncols > n_panels - 1)
        _style_ax(ax, show_lat, show_lon)

    fig.canvas.draw()

    pos_raw = ax_raw.get_position()
    cax_raw = fig.add_axes([pos_raw.x0, pos_raw.y0 - 0.05, pos_raw.width, 0.015])
    fig.colorbar(pc_raw, cax=cax_raw, orientation='horizontal', label=raw_label)

    pos_tr = axes_flat[1].get_position()
    pos_br = axes_flat[n_panels - 1].get_position()
    cax_diff = fig.add_axes([pos_tr.x1 + 0.02, pos_br.y0, 0.02, pos_tr.y1 - pos_br.y0])
    if pc_diff is not None:
        fig.colorbar(pc_diff, cax=cax_diff, orientation='vertical', label=diff_label)

    out = f'{SAVEPATH}{out_name}'
    plt.savefig(out, bbox_inches='tight', dpi=800)
    plt.close(fig)
    print(f'saved -> {out}')


plot_rmse_grid('rms_w_full', 'rmse_w_full', 'w_rmse_pycnocline_full.png',
               r'RMS $w$ (m s$^{-1}$)', r'RMSE $w$ vs base (m s$^{-1}$)')
plot_rmse_grid('rms_w_300m', 'rmse_w_300m', 'w_rmse_pycnocline_300m.png',
               r'RMS $w$ (m s$^{-1}$)', r'RMSE $w$ vs base (m s$^{-1}$)')


# ---------------------------------------------------------------------------
# Figure 5: time-mean pycnocline depth, all six scenarios -- sanity check
# ---------------------------------------------------------------------------
fig, axes = _new_grid()
present = [scen for scen in SCENARIOS if data[scen] is not None]
zpyc_all = np.concatenate([data[s]['zpyc_mean'].ravel() for s in present])
zmin = np.nanpercentile(zpyc_all, 1)
zmax = np.nanpercentile(zpyc_all, 99)

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
        pc = ax.pcolormesh(lon_rho, lat_rho, data[scen]['zpyc_mean'],
                           cmap=cmocean.cm.deep_r, vmin=zmin, vmax=zmax,
                           transform=proj, shading='nearest')
        _style_ax(ax, show_lat, show_lon)

fig.canvas.draw()
pos_top = axes[0, 1].get_position()
pos_bot = axes[2, 1].get_position()
cax = fig.add_axes([pos_top.x1 + 0.02, pos_bot.y0, 0.02, pos_top.y1 - pos_bot.y0])
if pc is not None:
    fig.colorbar(pc, cax=cax, orientation='vertical',
                 label=r'Depth of 1025 kg m$^{-3}$ isopycnal (m)')

out = f'{SAVEPATH}w_pycnocline_depth.png'
plt.savefig(out, bbox_inches='tight', dpi=800)
plt.close(fig)
print(f'saved -> {out}')
