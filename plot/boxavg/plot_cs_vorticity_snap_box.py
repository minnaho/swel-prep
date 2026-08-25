"""
Box-averaged cross-sections of normalized vorticity (zeta/f) at the same
single instant as plot_vorticity_snap.py (2019-04-22 07:01, matched by
hour). Box-average variant of ../plot_cs_vorticity_snap.py, along the same
3 transects as plot_cs_diag_avg_diff_box.py (south/north diagonal + the
fixed-eta "mid" geographical section).

Computed from raw native s_rho history output (same as plot_vorticity_snap.py
and the plot_cs_diag*.py family), not the zsliced product -- s_rho has no
fixed depth grid (it moves with zeta each timestep), so each transect's
zr is reconstructed per scenario via ROMS_depths.get_zr_zeta and passed to
cs_boxavg.boxavg_section, which interpolates every offset column onto the
transect's shared tr['zgrid'] before averaging across the box.

Layout: one figure per transect, 2x2 panels (top-left = no tides, no WEC;
top-right = no tides, 2.5x WEC; bottom-left = tides, no WEC; bottom-right =
tides, 2.5x WEC) -- same corner order as plot_vorticity_snap.py.

Output: ./figs/snapshots/vorticity_box/cs_vort_snap_box_<ts|tn|mid>_20190422_07.png
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
f_nc     = np.array(grdnc.variables['f'])
mask_rho = np.array(grdnc.variables['mask_rho'])

mask_plot = mask_rho.astype(float)
mask_plot[mask_plot == 0] = np.nan

TARGET_GLOB = 'mc60_his.2019042123*.nc'
TARGET_HOUR = 7

# (key, label, raw directory) in the requested panel order: top-left,
# top-right, bottom-left, bottom-right
SCENARIOS = [
    ('notidesnowec', 'no tides, no WEC',   '/data/project3/minnaho/swel/notides/mc60/nowec/his'),
    ('ampwec',       'no tides, 2.5x WEC', '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything'),
    ('tidesnowec',   'tides, no WEC',      '/data/project3/minnaho/swel/tides/mc60/nowec/output/his'),
    ('tidesampwec',  'tides, 2.5x WEC',    '/data/project3/minnaho/swel/tides/mc60/ampwec/everything'),
]

SAVEFIG_DIR = cb.figpath('snapshots', 'vorticity_box')
os.makedirs(SAVEFIG_DIR, exist_ok=True)

axfont = 16
c_map  = cmocean.cm.balance
v_min  = -5
v_max  =  5

# Density contour settings
RHO_REF_NC  = 1027.4             # ROMS reference density
ISO_RHO_OFF = RHO_REF_NC - 1000  # = 27.4: stored rho + offset → sigma-t
ISO_LEVELS  = [24, 24.25, 24.5, 24.75, 25, 25.25, 25.5, 25.75, 26]

# ---------------------------------------------------------------------------
# Transects -- same geometry as plot_cs_diag_avg_diff_box.py / plot_cs_diag_rho_box.py
# / plot_cs_mid_o2.py (box straddles each line perpendicularly, see cs_boxavg.py)
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
# Match the snapshot instant, then compute vorticity + reconstructed depth
# ---------------------------------------------------------------------------
def matched_file_and_index(raw_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, TARGET_GLOB)))
    assert len(files) == 1, f'expected 1 file matching {TARGET_GLOB} in {raw_dir}, found {len(files)}'
    f = files[0]

    oceantime = np.array(Dataset(f, 'r').variables['ocean_time'])
    oceandt   = pf.numdate(oceantime, 'second since 1995-01-01')
    matches   = [i for i, dt in enumerate(oceandt) if dt.hour == TARGET_HOUR]
    assert len(matches) == 1, f'expected 1 record with hour=={TARGET_HOUR} in {f}, found {len(matches)}'
    t_idx = matches[0]
    return f, t_idx, oceandt[t_idx]


print('Computing native s_rho vorticity for 4 scenarios...')
panel_sections = {tr['name']: [] for tr in TRANSECTS}
for key, label, raw_dir in SCENARIOS:
    f, t_idx, dt = matched_file_and_index(raw_dir)
    print(f'  {label}: matched {dt} ({f})')

    with Dataset(f, 'r') as hisnc:
        zeta_t = np.squeeze(hisnc.variables['zeta'][t_idx, :, :])
        zr3d   = depths.get_zr_zeta(hisnc, grdnc, zeta_t)   # (s_rho, eta_rho, xi_rho)
        # Load rho, convert to sigma-t, apply mask
        rho3d  = (np.squeeze(hisnc.variables['rho'][t_idx, :, :, :]) + ISO_RHO_OFF) * mask_plot

    # single-timestep variant (rho_uv_angle_tind) instead of rho_uv_angle --
    # the latter loads/rotates u,v across every timestep in the file (full
    # s_rho depth x full grid), which OOM-killed this script at ~42GB RSS
    # for a single scenario when only one timestep is ever used downstream
    urho, vrho = pf.rho_uv_angle_tind(f, grd, t_idx, rotate=True)   # (s_rho, eta_rho, xi_rho)
    vort3d = pf.vorticity(grd, urho[np.newaxis], vrho[np.newaxis])[0] / f_nc
    vort3d = vort3d * mask_plot                                     # (s_rho, eta_rho, xi_rho)

    for tr in TRANSECTS:
        vort_plot = cb.boxavg_section(vort3d, zr3d, tr)
        rho_plot  = cb.boxavg_section(rho3d,  zr3d, tr)
        panel_sections[tr['name']].append((label, vort_plot, rho_plot))

# ---------------------------------------------------------------------------
# Plot: one figure per transect, 2x2 panels
# ---------------------------------------------------------------------------
for tr in TRANSECTS:
    out_path = f'{SAVEFIG_DIR}/cs_vort_snap_box_{tr["name"]}_20190422_07.png'
    if os.path.exists(out_path):
        continue

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=[14, 10])
    zgrid = tr['zgrid']

    pc = None
    for ax, (label, vort_plot, rho_plot) in zip(axes.flat, panel_sections[tr['name']]):
        pc = ax.pcolormesh(tr['lon'], zgrid, vort_plot,
                           cmap=c_map, vmin=v_min, vmax=v_max, shading='nearest')

        # Overlay density contours
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
    cbar.set_label(r'$\zeta/f$', fontsize=axfont)
    cbar.ax.tick_params(axis='both', which='major', labelsize=axfont)

    plt.savefig(out_path, bbox_inches='tight', dpi=800)
    plt.close(fig)
    print(f'saved -> {out_path}')
