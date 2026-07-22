"""
Geographical cross-section at a fixed eta index (a constant-latitude slice
running from inside the bay to offshore), using native s_rho ROMS output.
Saves one PNG per time step.

O2-only version (previously plotted w and NO3 side-by-side for 4 scenarios
in a 2-column layout; now O2 for all 6 scenarios in a single column).
Plotting style matches plot_cs_diag_o2.py: one stacked panel per scenario,
a shared colorbar, and an isopycnal contour overlay.

Output: ./figs/snapshots/cs_mid_o2/cs_mid_o2-YYYY-MM-DD-HH.png
"""

import os
import sys
sys.path.append('/data/project3/minnaho/global/')
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import cmocean

plt.rcParams.update({'font.size': 14})
from netCDF4 import Dataset, num2date
import ROMS_depths as depths

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCENARIOS = {
    'tideswec':     '/data/project3/minnaho/swel/tides/mc60/wec',
    'tidesnowec':   '/data/project3/minnaho/swel/tides/mc60/nowec/output',
    'notidesnowec': '/data/project3/minnaho/swel/notides/mc60/nowec/output',
    'notideswec':   '/data/project3/minnaho/swel/notides/mc60/wec/rerun',
    'ampwec':       '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything',
    'tidesampwec':  '/data/project3/minnaho/swel/tides/mc60/ampwec/everything',
}

LABELS = {
    'notidesnowec': 'no tides, no WEC',
    'tideswec':     'tides, WEC',
    'tidesnowec':   'tides, no WEC',
    'notideswec':   'no tides, WEC',
    'ampwec':       'no tides, 2.5x WEC',
    'tidesampwec':  'tides, 2.5x WEC',
}

VAR       = 'O2'      # variable name — bgc
VAR_SRC   = 'bgc'
VAR_CMAP  = cmocean.cm.haline
VMIN      = 100
VMAX      = 300
VAR_LABEL = r'O$_2$ (mmol m$^{-3}$)'

# Isopycnal contour overlay (sigma-t levels) -- same as plot_cs_diag_o2.py
RHO_REF_NC  = 1027.4             # ROMS reference density
ISO_RHO_OFF = RHO_REF_NC - 1000  # = 27.4: stored rho + offset → sigma-t
ISO_LEVELS  = [24, 24.5, 25, 25.5, 26]

ETA_SLICE = 477       # eta index — slice from inside the bay to outside
DEPTH_LIM = -300       # y-axis bottom (m)

GRD      = 'mc60_grd.nc'
SAVEPATH = './figs/snapshots/cs_mid_o2/'

# ---------------------------------------------------------------------------
# Load grid
# ---------------------------------------------------------------------------
grdnc    = Dataset(GRD, 'r')
lon      = np.array(grdnc['lon_rho'][:]) - 360
mask_rho = np.array(grdnc['mask_rho'][:])

# land mask for variable fields
mask_plot = mask_rho.astype(float)
mask_plot[mask_plot == 0] = np.nan

lon_slice = lon[ETA_SLICE, :]   # degrees east, fixed eta row

# ---------------------------------------------------------------------------
# Build per-scenario file lists (handles flat vs his/ subdir layout)
# ---------------------------------------------------------------------------
def src_glob(root, kind):
    sub = os.path.join(root, kind)
    base = sub if os.path.isdir(sub) else root
    stem = 'mc60_his' if kind == 'his' else 'mc60_bgc'
    return os.path.join(base, f'{stem}.*.nc')

his_files = {name: sorted(glob.glob(src_glob(root, 'his')))
             for name, root in SCENARIOS.items()}
var_files = {name: sorted(glob.glob(src_glob(root, 'bgc')))
             for name, root in SCENARIOS.items()}

n_files = len(list(his_files.values())[0])
n_scen  = len(SCENARIOS)
for name, files in his_files.items():
    print(f'  {name}: {len(files)} his files')

# ---------------------------------------------------------------------------
# Main loop — one PNG per time step
# ---------------------------------------------------------------------------
os.makedirs(SAVEPATH, exist_ok=True)

x_ticks = np.linspace(-121.96, -121.8, 5)   # start, end, and how many ticks total

for hf in range(n_files):
    ncs = {}
    for name in SCENARIOS:
        hisnc = Dataset(his_files[name][hf], 'r')
        varnc = Dataset(var_files[name][hf], 'r')
        ncs[name] = (hisnc, varnc)

    ref_hisnc = ncs[list(SCENARIOS)[0]][0]
    his_time  = np.array(ref_hisnc['ocean_time'][:])

    for t_i in range(his_time.shape[0]):
        dt0 = num2date(his_time, 'seconds since 1995-01-01')[t_i]
        dstr = (f'{dt0.year}-{dt0.month:02d}-{dt0.day:02d}-{dt0.hour:02d}')

        fig, axes = plt.subplots(n_scen, 1, sharex=True,
                                 figsize=(10, 3 * n_scen))

        for ax, name in zip(axes, SCENARIOS):
            hisnc, varnc = ncs[name]
            zeta  = np.squeeze(hisnc['zeta'][t_i, :, :])
            zr3d  = depths.get_zr_zeta(hisnc, grdnc, zeta)
            var2d = (np.squeeze(np.array(varnc[VAR][t_i])) * mask_plot)[:, ETA_SLICE, :]
            rho2d = ((np.squeeze(np.array(hisnc['rho'][t_i])) + ISO_RHO_OFF)
                     * mask_plot)[:, ETA_SLICE, :]
            zr2d  = zr3d[:, ETA_SLICE, :]

            pc = ax.pcolormesh(lon_slice, zr2d, var2d,
                               cmap=VAR_CMAP, vmin=VMIN, vmax=VMAX,
                               shading='nearest')
            lon_2d = np.tile(lon_slice, (zr2d.shape[0], 1))
            cs = ax.contour(lon_2d, zr2d, rho2d, levels=ISO_LEVELS,
                            colors='k', linewidths=0.8)
            ax.clabel(cs, fmt='%.1f', fontsize=9)
            ax.set_ylim([DEPTH_LIM, 0])
            ax.set_xlim([x_ticks[0], x_ticks[-1]])
            ax.set_xticks(x_ticks)
            ax.set_ylabel('Depth (m)')
            ax.set_title(LABELS[name])

            sf = ScalarFormatter(useOffset=False)
            sf.set_scientific(False)
            ax.xaxis.set_major_formatter(sf)

        axes[-1].set_xlabel('Longitude')
        fig.canvas.draw()
        pos_top = axes[0].get_position()
        pos_bot = axes[-1].get_position()
        cax = fig.add_axes([pos_top.x1 + 0.015, pos_bot.y0,
                            0.012, pos_top.y1 - pos_bot.y0])
        fig.colorbar(pc, cax=cax, label=VAR_LABEL)
        fig.suptitle(f'{dt0.year}-{dt0.month:02d}-{dt0.day:02d} '
                     f'{dt0.hour:02d}:{dt0.minute:02d} UTC', y=0.92)

        fname = f'{SAVEPATH}cs_mid_o2-{dstr}.png'
        plt.savefig(fname, dpi=800, bbox_inches='tight')
        plt.close()
        print(f'  saved -> {fname}')

    for hisnc, varnc in ncs.values():
        hisnc.close()
        varnc.close()
