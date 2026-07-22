"""
Cross-sections of w, sigma-t, NO3, and TOTC from the initial condition files,
comparing tides vs no-tides. One PNG per variable per transect.

Uses the same transect geometry and isopycnal overlay as plot_cs_diag*.py.
Ini files must have s-coord global attributes added by add_scoord_attrs.py.

Output: ./figs/ini_compare/cs_ini_{var}_<ts|tn>.png
"""

import os
import sys
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
import seawater
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import cmocean
from netCDF4 import Dataset
from scipy.ndimage import map_coordinates
import ROMS_depths as depths

plt.rcParams.update({'font.size': 14})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TIDES_INI   = '/data/project3/minnaho/project9copy/swel/tides/mc60/rst/mc60_ini.20190415110120.nc'
NOTIDES_INI = '/data/project3/minnaho/project9copy/swel/notides/mc60/rst/mc60_ini.20190415110120.nc'
GRD         = 'mc60_grd.nc'
SAVEPATH    = './figs/ini_compare/'

CASES = [
    ('tides',   TIDES_INI,   'tides'),
    ('notides', NOTIDES_INI, 'no tides'),
]

# ---------------------------------------------------------------------------
# Variable configs
# ---------------------------------------------------------------------------
ISO_LEVELS = [24, 24.5, 25, 25.5, 26]

VAR_CONFIGS = {
    'w': dict(
        cmap=cmocean.cm.balance,
        vmin=-1e-3, vmax=1e-3,
        label=r'w (m s$^{-1}$)',
        iso=True,
    ),
    'rho': dict(
        cmap=cmocean.cm.dense,
        vmin=23, vmax=27,
        label=r'$\sigma_t$ (kg m$^{-3}$)',
        iso=False,
    ),
    'NO3': dict(
        cmap=cmocean.cm.matter,
        vmin=0, vmax=30,
        label=r'NO$_3$ (mmol N m$^{-3}$)',
        iso=True,
    ),
    'TOTC': dict(
        cmap=cmocean.cm.algae,
        vmin=0, vmax=40,
        label=r'Total phyto C (mmol C m$^{-3}$)',
        iso=True,
    ),
}

# ---------------------------------------------------------------------------
# Transect geometry (same as plot_cs_diag*.py)
# ---------------------------------------------------------------------------
ETA0       = 271
XI0        = 543
SLOPE      = -0.8
DEPTH_LIM0 = -125
ETA1       = 832
XI1        = 646
SLOPE1     = 0.6
DEPTH_LIM1 = -90
LENGTH_XI  = 250
N_PTS      = 300

# ---------------------------------------------------------------------------
# Load grid
# ---------------------------------------------------------------------------
grdnc    = Dataset(GRD, 'r')
lat      = np.array(grdnc['lat_rho'][:])
lon      = np.array(grdnc['lon_rho'][:]) - 360
mask_rho = np.array(grdnc['mask_rho'][:])
mask_plot = mask_rho.astype(float)
mask_plot[mask_plot == 0] = np.nan

# ---------------------------------------------------------------------------
# Build transects
# ---------------------------------------------------------------------------
def build_transect(eta0, xi0, slope, depth_lim):
    xi  = np.linspace(xi0,  xi0  - LENGTH_XI, N_PTS)
    eta = np.linspace(eta0, eta0 + slope * (-LENGTH_XI), N_PTS)
    crd = np.array([eta, xi])
    return dict(
        coords    = crd,
        lon       = map_coordinates(lon, crd, order=1, mode='nearest'),
        mask      = map_coordinates(mask_rho.astype(float), crd,
                                    order=0, mode='nearest') > 0.5,
        depth_lim = depth_lim,
    )

TRANSECTS = [
    build_transect(ETA0, XI0, SLOPE,  DEPTH_LIM0),
    build_transect(ETA1, XI1, SLOPE1, DEPTH_LIM1),
]
TRANSECT_NAMES = ['ts', 'tn']   # ts = south transect (ETA0), tn = north transect (ETA1)

# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------
def interp_transect(field2d, coords, mask_t):
    nan_mask  = np.isnan(field2d)
    filled    = np.where(nan_mask, 0.0, field2d)
    row       = map_coordinates(filled, coords, order=1, mode='nearest')
    nan_along = map_coordinates(nan_mask.astype(float), coords,
                                order=1, mode='nearest') > 0.5
    row[nan_along | ~mask_t] = np.nan
    return row

def interp_section(field3d, coords, mask_t):
    n_s = field3d.shape[0]
    out = np.full((n_s, N_PTS), np.nan)
    for iz in range(n_s):
        out[iz] = interp_transect(field3d[iz], coords, mask_t)
    return out

# ---------------------------------------------------------------------------
# Load fields from both ini files; precompute transect sections
# ---------------------------------------------------------------------------
os.makedirs(SAVEPATH, exist_ok=True)

# sections[case_name][ti] = {var: section_array, 'zr': zr_section}
sections = {}

for case_name, ini_path, _ in CASES:
    nc   = Dataset(ini_path, 'r')
    zr3d = depths.get_zr_zeta_tind(nc, grdnc, 0)   # (N, eta, xi)

    # w: s_w → average to s_rho
    w_sw = np.squeeze(np.array(nc.variables['w'][0]))
    w3d  = (0.5 * (w_sw[:-1] + w_sw[1:])) * mask_plot

    # sigma-t from temp + salt
    temp = np.squeeze(np.array(nc.variables['temp'][0]))
    salt = np.squeeze(np.array(nc.variables['salt'][0]))
    rho3d = seawater.dens0(salt, temp) - 1000.0

    # NO3
    no3_3d = np.squeeze(np.array(nc.variables['NO3'][0])) * mask_plot

    # TOTC = DIATC + DIAZC + SPC
    totc_3d = sum(np.squeeze(np.array(nc.variables[v][0]))
                  for v in ['DIATC', 'DIAZC', 'SPC']) * mask_plot

    nc.close()

    fields3d = {'w': w3d, 'rho': rho3d, 'NO3': no3_3d, 'TOTC': totc_3d}

    sections[case_name] = []
    for tr in TRANSECTS:
        sec = {'zr': interp_section(zr3d, tr['coords'], tr['mask'])}
        for var, f3d in fields3d.items():
            sec[var] = interp_section(f3d, tr['coords'], tr['mask'])
        sections[case_name].append(sec)

# ---------------------------------------------------------------------------
# Plot: one figure per variable per transect (tides / no tides / difference)
# ---------------------------------------------------------------------------
n_rows = len(CASES) + 1   # +1 for difference panel

for var, cfg in VAR_CONFIGS.items():
    for ti, tr in enumerate(TRANSECTS):
        fig, axes = plt.subplots(n_rows, 1, sharex=True,
                                 figsize=(10, 3.5 * n_rows))
        pc_last = None
        kept_sections = {}   # store trimmed arrays for the diff panel

        for ax, (case_name, _, case_label) in zip(axes, CASES):
            sec    = sections[case_name][ti]
            zr_t   = sec['zr']
            var_t  = sec[var]
            rho_t  = sec['rho']

            depth_mean = np.nanmean(zr_t, axis=1)
            keep       = depth_mean >= tr['depth_lim']
            zr_plot    = zr_t[keep, :]
            var_plot   = var_t[keep, :]
            rho_plot   = rho_t[keep, :]
            kept_sections[case_name] = (zr_plot, var_plot, rho_plot)

            pc = ax.pcolormesh(tr['lon'], zr_plot, var_plot,
                               cmap=cfg['cmap'], vmin=cfg['vmin'], vmax=cfg['vmax'],
                               shading='nearest')
            pc_last = pc

            if cfg['iso']:
                lon_2d = np.tile(tr['lon'], (zr_plot.shape[0], 1))
                cs = ax.contour(lon_2d, zr_plot, rho_plot, levels=ISO_LEVELS,
                                colors='k', linewidths=0.8)
                ax.clabel(cs, fmt='%.1f', fontsize=9)

            ax.set_ylim([tr['depth_lim'], 0])
            ax.set_ylabel('Depth (m)')
            ax.set_title(case_label)

            sf = ScalarFormatter(useOffset=False)
            sf.set_scientific(False)
            ax.xaxis.set_major_formatter(sf)
            ax.xaxis.set_major_locator(MaxNLocator(5))

        # ── difference panel (tides − no tides) ──────────────────────────
        ax_diff = axes[-1]
        zr_t,   var_t,   rho_t   = kept_sections['tides']
        zr_nt,  var_nt,  rho_nt  = kept_sections['notides']
        diff = var_t - var_nt
        vlim = np.nanmax(np.abs(diff))
        vlim = vlim if vlim > 0 else 1.0

        pc_diff = ax_diff.pcolormesh(tr['lon'], zr_t, diff,
                                     cmap=cmocean.cm.balance,
                                     vmin=-vlim, vmax=vlim,
                                     shading='nearest')
        if cfg['iso']:
            lon_2d = np.tile(tr['lon'], (zr_t.shape[0], 1))
            cs_t  = ax_diff.contour(lon_2d, zr_t,  rho_t,  levels=ISO_LEVELS,
                                    colors='k', linewidths=0.8, linestyles='-')
            cs_nt = ax_diff.contour(lon_2d, zr_nt, rho_nt, levels=ISO_LEVELS,
                                    colors='k', linewidths=0.8, linestyles='--')
            ax_diff.clabel(cs_t,  fmt='%.1f', fontsize=9)
            ax_diff.clabel(cs_nt, fmt='%.1f', fontsize=9)

        ax_diff.set_ylim([tr['depth_lim'], 0])
        ax_diff.set_ylabel('Depth (m)')
        ax_diff.set_title('tides − no tides')

        sf = ScalarFormatter(useOffset=False)
        sf.set_scientific(False)
        ax_diff.xaxis.set_major_formatter(sf)
        ax_diff.xaxis.set_major_locator(MaxNLocator(5))

        axes[-1].set_xlabel('Longitude')
        fig.canvas.draw()

        # two colorbars: variable (spans top two panels) + difference
        pos_top  = axes[0].get_position()
        pos_mid  = axes[1].get_position()
        pos_diff = axes[2].get_position()
        cb_x     = pos_top.x1 + 0.015
        cb_w     = 0.012
        cax_var  = fig.add_axes([cb_x, pos_mid.y0,  cb_w, pos_top.y1 - pos_mid.y0])
        cax_diff = fig.add_axes([cb_x, pos_diff.y0, cb_w, pos_diff.height])
        fig.colorbar(pc_last, cax=cax_var,  label=cfg['label'])
        fig.colorbar(pc_diff, cax=cax_diff, label=f'Δ {cfg["label"]}')

        fname = f'{SAVEPATH}cs_ini_{var}_{TRANSECT_NAMES[ti]}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'saved -> {fname}')
