"""
Box-averaged cross-sections of Potential Vorticity (PV) at a single snapshot,
with density (sigma-t) contours overlaid. Box-average variant of
../plot_cs_pv_snap.py. Reads pre-calculated PV from a separate _pv.nc file,
while loading zeta and rho from the corresponding _his.nc file.

Includes the looped file search to correctly find the target hour across multiple files.
"""

import sys
import os
import glob
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
from netCDF4 import Dataset
import pyfuncs as pf
import ROMS_depths as depths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import cmocean
import cs_boxavg as cb

grd = cb.GRD

grdnc    = Dataset(grd, 'r')
lat      = np.array(grdnc.variables['lat_rho'])
lon      = np.array(grdnc.variables['lon_rho']) - 360
mask_rho = np.array(grdnc.variables['mask_rho'])

mask_plot = mask_rho.astype(float)
mask_plot[mask_plot == 0] = np.nan

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_GLOB_HIS = 'mc60_his.20190421*.nc'  # Broadened to check all files starting on the 21st
TARGET_GLOB_PV  = '*_pv.nc'
TARGET_HOUR     = 7                        # 07:00 AM

SCENARIOS = [
    ('notidesnowec', 'no tides, no WEC',
     '/data/project3/minnaho/swel/notides/mc60/nowec/his',
     '/data/project3/minnaho/swel/notides/mc60/nowec/his/pv'),

    ('ampwec',       'no tides, 2.5x WEC',
     '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/his',
     '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/his/pv'),

    ('tidesnowec',   'tides, no WEC',
     '/data/project3/minnaho/swel/tides/mc60/nowec/output/his',
     '/data/project3/minnaho/swel/tides/mc60/nowec/output/his/pv'),

    ('tidesampwec',  'tides, 2.5x WEC',
     '/data/project3/minnaho/swel/tides/mc60/ampwec/his',
     '/data/project3/minnaho/swel/tides/mc60/ampwec/his/pv'),
]

SAVEFIG_DIR = cb.figpath('snapshots', 'pv_box')
os.makedirs(SAVEFIG_DIR, exist_ok=True)

axfont = 16

# NOTE: You will likely need to adjust these limits based on your actual PV values
c_map  = cmocean.cm.curl
v_min  = -1e-7
v_max  =  1e-7

# Density contour settings
RHO_REF_NC  = 1027.4             # ROMS reference density
ISO_RHO_OFF = RHO_REF_NC - 1000  # = 27.4: stored rho + offset -> sigma-t
ISO_LEVELS  = [24, 24.25, 24.5, 24.75, 25, 25.25, 25.5, 25.75, 26]

# ---------------------------------------------------------------------------
# Transects Geometry (box straddles each line perpendicularly, see cs_boxavg.py)
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

ETA_MID       = 477
DEPTH_LIM_MID = -300
MID_XLIM      = [-121.96, -121.8]
MID_XTICKS    = np.linspace(MID_XLIM[0], MID_XLIM[1], 5)

TRANSECTS = [
    cb.build_box_transect(lon, lat, mask_rho, ETA0, XI0, SLOPE, DEPTH_LIM0,
                          LENGTH_XI, N_PTS, name='ts', title='south transect'),
    cb.build_box_transect(lon, lat, mask_rho, ETA1, XI1, SLOPE1, DEPTH_LIM1,
                          LENGTH_XI, N_PTS, name='tn', title='north transect'),
    cb.build_box_mid_transect(lon, lat, mask_rho, ETA_MID, DEPTH_LIM_MID,
                              xlim=MID_XLIM, xticks=MID_XTICKS, name='mid',
                              title=f'mid transect (eta={ETA_MID})'),
]

# ---------------------------------------------------------------------------
# Match snapshot and extract data
# ---------------------------------------------------------------------------
def match_time_index(nc_files, target_hour, time_var='ocean_time', epoch='seconds since 1995-01-01'):
    for nc_file in nc_files:
        with Dataset(nc_file, 'r') as nc:
            oceantime = np.array(nc.variables[time_var])

        oceandt = pf.numdate(oceantime, epoch)
        matches = [i for i, dt in enumerate(oceandt) if dt.hour == target_hour]

        if len(matches) == 1:
            return nc_file, matches[0], oceandt[matches[0]]

    raise ValueError(f'Could not find hour=={target_hour} in any of these files: {nc_files}')

print('Loading PV, zeta, and rho for 4 scenarios...')
panel_sections = {tr['name']: [] for tr in TRANSECTS}

for key, label, his_dir, pv_dir in SCENARIOS:
    # 1. Find matching HIS file
    his_files = sorted(glob.glob(os.path.join(his_dir, TARGET_GLOB_HIS)))
    assert len(his_files) >= 1, f'No HIS files found in {his_dir}'

    f_his, t_idx_his, dt_his = match_time_index(his_files, TARGET_HOUR, epoch='second since 1995-01-01')

    # 2. Find matching PV file
    pv_files = sorted(glob.glob(os.path.join(pv_dir, TARGET_GLOB_PV)))
    assert len(pv_files) >= 1, f'No PV files found in {pv_dir}'

    f_pv, t_idx_pv, dt_pv = match_time_index(pv_files, TARGET_HOUR, epoch='seconds since 1900-01-01 00:00:00')

    print(f'  {label}: matched {dt_his} (HIS) with {dt_pv} (PV)')

    # 3. Load zeta, rho, and calculate depth grid
    with Dataset(f_his, 'r') as hisnc:
        zeta_t = np.squeeze(hisnc.variables['zeta'][t_idx_his, :, :])
        zr3d   = depths.get_zr_zeta(hisnc, grdnc, zeta_t)
        rho3d  = (np.squeeze(hisnc.variables['rho'][t_idx_his, :, :, :]) + ISO_RHO_OFF) * mask_plot

    # 4. Load pre-calculated PV
    with Dataset(f_pv, 'r') as pvnc:
        pv3d = np.squeeze(pvnc.variables['pv'][t_idx_pv, :, :, :])
        pv3d = pv3d * mask_plot

    for tr in TRANSECTS:
        pv_plot  = cb.boxavg_section(pv3d,  zr3d, tr)
        rho_plot = cb.boxavg_section(rho3d, zr3d, tr)
        panel_sections[tr['name']].append((label, pv_plot, rho_plot))

# ---------------------------------------------------------------------------
# Plot: one figure per transect, 2x2 panels
# ---------------------------------------------------------------------------
for tr in TRANSECTS:
    out_path = f'{SAVEFIG_DIR}/cs_pv_snap_box_{tr["name"]}_20190422_07.png'
    if os.path.exists(out_path):
        continue

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=[14, 10])
    zgrid = tr['zgrid']

    pc = None
    for ax, (label, pv_plot, rho_plot) in zip(axes.flat, panel_sections[tr['name']]):

        pc = ax.pcolormesh(tr['lon'], zgrid, pv_plot,
                           cmap=c_map, vmin=v_min, vmax=v_max, shading='nearest')

        # Density contours
        cs = ax.contour(tr['lon'], zgrid, rho_plot, levels=ISO_LEVELS,
                        colors='k', linewidths=0.8)
        ax.clabel(cs, fmt='%.2f', fontsize=7)

        ax.set_ylim([tr['depth_lim'], 0])
        ax.set_title(label, fontsize=axfont)
        ax.tick_params(axis='both', which='major', labelsize=axfont)

        sf = ScalarFormatter(useOffset=False)
        sf.set_scientific(False)
        ax.xaxis.set_major_formatter(sf)
        if tr.get('xticks') is not None:
            ax.set_xticks(tr['xticks'])
        else:
            ax.xaxis.set_major_locator(MaxNLocator(5))

    if tr.get('xlim') is not None:
        axes[0, 0].set_xlim(tr['xlim'])
    for ax in axes[:, 0]:
        ax.set_ylabel('Depth (m)', fontsize=axfont)
    for ax in axes[1, :]:
        ax.set_xlabel('Longitude', fontsize=axfont)

    fig.tight_layout()

    pos_tr = axes[0, 1].get_position().get_points().flatten()
    pos_br = axes[1, 1].get_position().get_points().flatten()
    cb_ax = fig.add_axes([pos_tr[2] + .02, pos_br[1], .015, pos_tr[3] - pos_br[1]])

    cbar = fig.colorbar(pc, cax=cb_ax, orientation='vertical')
    cbar.set_label('PV (m/s³)', fontsize=axfont)
    cbar.ax.tick_params(axis='both', which='major', labelsize=axfont)

    plt.savefig(out_path, bbox_inches='tight', dpi=800)
    plt.close(fig)
    print(f'saved -> {out_path}')
