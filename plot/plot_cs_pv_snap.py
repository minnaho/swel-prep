"""
Cross-sections of Potential Vorticity (PV) at a single snapshot, 
with density (sigma-t) contours overlaid.
Reads pre-calculated PV from a separate _pv.nc file, while loading zeta
and rho from the corresponding _his.nc file.

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
from scipy.ndimage import map_coordinates
import cmocean

grd = '/data/project3/minnaho/project9copy/swel/mc60_grd.nc'

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

SAVEFIG_DIR = './figs/snapshots/pv'
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
# Transects Geometry
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

def build_transect(name, title, eta0, xi0, slope, depth_lim):
    xi  = np.linspace(xi0,  xi0  - LENGTH_XI, N_PTS)
    eta = np.linspace(eta0, eta0 + slope * (-LENGTH_XI), N_PTS)
    crd = np.array([eta, xi])
    return dict(
        mode      = 'diag',
        name      = name,
        title     = title,
        coords    = crd,
        lon       = map_coordinates(lon, crd, order=1, mode='nearest'),
        mask      = map_coordinates(mask_rho.astype(float), crd, order=0, mode='nearest') > 0.5,
        depth_lim = depth_lim,
    )

def build_mid_transect(name, title, eta_slice, depth_lim, xlim=None, xticks=None):
    return dict(
        mode      = 'mid',
        name      = name,
        title     = title,
        eta_slice = eta_slice,
        lon       = lon[eta_slice, :],
        depth_lim = depth_lim,
        xlim      = xlim,
        xticks    = xticks,
    )

TRANSECTS = [
    build_transect('ts', 'south transect', ETA0, XI0, SLOPE,  DEPTH_LIM0),
    build_transect('tn', 'north transect', ETA1, XI1, SLOPE1, DEPTH_LIM1),
    build_mid_transect('mid', f'mid transect (eta={ETA_MID})', ETA_MID, DEPTH_LIM_MID,
                       xlim=MID_XLIM, xticks=MID_XTICKS),
]

def interp_transect(field2d, coords, mask_t):
    nan_mask  = np.isnan(field2d)
    filled    = np.where(nan_mask, 0.0, field2d)
    row       = map_coordinates(filled, coords, order=1, mode='nearest')
    nan_along = map_coordinates(nan_mask.astype(float), coords, order=1, mode='nearest') > 0.5
    row[nan_along | ~mask_t] = np.nan
    return row

def interp_section(field3d, coords, mask_t):
    n_s = field3d.shape[0]
    out = np.full((n_s, N_PTS), np.nan)
    for iz in range(n_s):
        out[iz] = interp_transect(field3d[iz], coords, mask_t)
    return out

def extract_section(field3d, tr):
    if tr['mode'] == 'diag':
        return interp_section(field3d, tr['coords'], tr['mask'])
    else:
        return field3d[:, tr['eta_slice'], :]

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
        pv_t  = extract_section(pv3d, tr)
        zr_t  = extract_section(zr3d, tr)
        rho_t = extract_section(rho3d, tr)
        panel_sections[tr['name']].append((label, pv_t, zr_t, rho_t))

# ---------------------------------------------------------------------------
# Plot: one figure per transect, 2x2 panels
# ---------------------------------------------------------------------------
for tr in TRANSECTS:
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=[14, 10])

    pc = None
    for ax, (label, pv_t, zr_t, rho_t) in zip(axes.flat, panel_sections[tr['name']]):
        
        # Plotting the full depth array and letting the axis limits crop the visual
        pc = ax.pcolormesh(tr['lon'], zr_t, pv_t,
                           cmap=c_map, vmin=v_min, vmax=v_max, shading='nearest')
        
        # Density contours
        lon_2d = np.tile(tr['lon'], (zr_t.shape[0], 1))
        cs = ax.contour(lon_2d, zr_t, rho_t, levels=ISO_LEVELS,
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
    
    cb = fig.colorbar(pc, cax=cb_ax, orientation='vertical')
    cb.set_label('PV (m/s³)', fontsize=axfont)
    cb.ax.tick_params(axis='both', which='major', labelsize=axfont)

    out_path = f'{SAVEFIG_DIR}/cs_pv_snap_{tr["name"]}_20190422_07.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'saved -> {out_path}')
