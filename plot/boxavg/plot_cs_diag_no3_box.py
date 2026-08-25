"""
Box-averaged diagonal cross-section from coast to offshore, using native
s_rho ROMS output. Box-average variant of ../plot_cs_diag_no3.py.

The transect starts at (ETA0, XI0) and extends in the -xi direction (offshore).
SLOPE = deta/dxi controls the diagonal angle:
  SLOPE =  0   → purely zonal
  SLOPE =  0.5 → eta decreases as xi decreases (southward offshore)
  SLOPE = -0.5 → eta increases as xi decreases (northward offshore)

NO3 version.
Output: ./figs/snapshots/cs_diag_no3_box_allscen-<ts|tn>-YYYY-MM-DD-HH.png
"""

import os
import sys
sys.path.append('/data/project3/minnaho/global/')
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import cmocean

plt.rcParams.update({'font.size': 14})
from netCDF4 import Dataset, num2date
import ROMS_depths as depths
import cs_boxavg as cb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCENARIOS = {
    'tideswec':     '/data/project3/minnaho/swel/tides/mc60/wec',
    'tidesnowec':   '/data/project3/minnaho/swel/tides/mc60/nowec/output',
    'notidesnowec': '/data/project3/minnaho/swel/notides/mc60/nowec',
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

VAR      = 'NO3'     # variable name — HIS or BGC
VAR_SRC  = 'bgc'       # 'his' or 'bgc'
VAR_CMAP = cmocean.cm.matter
VMIN     = 0
VMAX     = 25
VAR_LABEL = r'NO$_3$ (mmol m$^{-3}$)'

RHO_OFFSET = 0.0

# Isopycnal contour overlay (sigma-t levels)
RHO_REF_NC  = 1027.4             # ROMS reference density
ISO_RHO_OFF = RHO_REF_NC - 1000  # = 27.4: stored rho + offset → sigma-t
ISO_LEVELS  = [24, 24.25, 24.5, 24.75, 25, 25.25, 25.5, 25.75, 26]

# Transect geometry (grid index space)
ETA0       = 271       # transect 0 — starting eta index (coast end)
XI0        = 543       # transect 0 — starting xi index (coast end)
SLOPE      = -0.8      # transect 0 — deta/dxi
DEPTH_LIM0 = -125      # transect 0 — y-axis bottom (m)
ETA1       = 832       # transect 1 — starting eta index (coast end)
XI1        = 646       # transect 1 — starting xi index (coast end)
SLOPE1     = 0.6       # transect 1 — deta/dxi
DEPTH_LIM1 = -90      # transect 1 — y-axis bottom (m)
LENGTH_XI = 250        # transect length in xi cells (both)
N_PTS     = 300        # interpolation resolution along transect (both)

GRD      = cb.GRD
SAVEPATH = cb.figpath('snapshots') + '/'

# ---------------------------------------------------------------------------
# Load grid
# ---------------------------------------------------------------------------
grdnc    = Dataset(GRD, 'r')
lat      = np.array(grdnc['lat_rho'][:])
lon      = np.array(grdnc['lon_rho'][:]) - 360
mask_rho = np.array(grdnc['mask_rho'][:])

# land mask for variable fields
mask_plot = mask_rho.astype(float)
mask_plot[mask_plot == 0] = np.nan

# ---------------------------------------------------------------------------
# Build box transects (goes in -xi direction; box straddles it perpendicularly)
# ---------------------------------------------------------------------------
TRANSECTS = [
    cb.build_box_transect(lon, lat, mask_rho, ETA0, XI0, SLOPE,  DEPTH_LIM0,
                          LENGTH_XI, N_PTS, name='ts'),
    cb.build_box_transect(lon, lat, mask_rho, ETA1, XI1, SLOPE1, DEPTH_LIM1,
                          LENGTH_XI, N_PTS, name='tn'),
]
TRANSECT_NAMES = ['ts', 'tn']   # ts = south transect (ETA0), tn = north transect (ETA1)

# ---------------------------------------------------------------------------
# Transect preview — press Enter to proceed, Esc to cancel
# ---------------------------------------------------------------------------
def _preview_transect():
    confirmed = [False]
    colors = ['red', 'blue']

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.pcolormesh(lon, lat, mask_rho, cmap='gray_r', vmin=0, vmax=1)
    for i, tr in enumerate(TRANSECTS):
        lbl = f't{i}  (η={tr["eta0"]}, ξ={tr["xi0"]}, slope={tr["slope"]})'
        ax.plot(tr['lon'], tr['lat'], '-', color=colors[i], linewidth=2, label=lbl)
        ax.plot(tr['lon'][0], tr['lat'][0], 'o', color=colors[i], markersize=7)
        (elo, ela), (ehi, eha) = cb.box_edges(tr)
        ax.plot(elo, ela, '--', color=colors[i], linewidth=1)
        ax.plot(ehi, eha, '--', color=colors[i], linewidth=1)
    ax.set_title(
        f'Box transect preview — L={LENGTH_XI} cells, '
        f'half-width={cb.HALF_WIDTH_PTS} cells\n'
        'Press  Enter  to proceed  |  Esc  to cancel'
    )
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='upper left')
    plt.tight_layout()

    def on_key(event):
        if event.key == 'enter':
            confirmed[0] = True
            plt.close(fig)
        elif event.key == 'escape':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if not confirmed[0]:
        print('Cancelled.')
        sys.exit(0)

if matplotlib.get_backend().lower() != 'agg':
    _preview_transect()

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
var_files = {name: sorted(glob.glob(src_glob(root, 'bgc'))) if VAR_SRC == 'bgc'
             else his_files[name]
             for name, root in SCENARIOS.items()}

n_files = len(list(his_files.values())[0])
n_scen  = len(SCENARIOS)
for name, files in his_files.items():
    print(f'  {name}: {len(files)} his files')

# ---------------------------------------------------------------------------
# Main loop — one PNG per time step
# ---------------------------------------------------------------------------
os.makedirs(SAVEPATH, exist_ok=True)

for hf in range(n_files):
    ncs = {}
    for name in SCENARIOS:
        hisnc = Dataset(his_files[name][hf], 'r')
        varnc = Dataset(var_files[name][hf], 'r') if VAR_SRC == 'bgc' else hisnc
        ncs[name] = (hisnc, varnc)

    ref_hisnc  = ncs[list(SCENARIOS)[0]][0]
    his_time   = np.array(ref_hisnc['ocean_time'][:])

    for t_i in range(his_time.shape[0]):
        dt0 = num2date(his_time, 'seconds since 1995-01-01')[t_i]
        dstr = (f'{dt0.year}-{dt0.month:02d}-{dt0.day:02d}-{dt0.hour:02d}')

        # read all scenario data once for this time step
        scen_data = {}
        for name, (hisnc, varnc) in ncs.items():
            zeta  = np.squeeze(hisnc['zeta'][t_i, :, :])
            zr3d  = depths.get_zr_zeta(hisnc, grdnc, zeta)
            var3d = (np.squeeze(np.array(varnc[VAR][t_i])) + RHO_OFFSET) * mask_plot
            rho3d = (np.squeeze(np.array(hisnc['rho'][t_i])) + ISO_RHO_OFF) * mask_plot
            scen_data[name] = (zr3d, var3d, rho3d)

        # one figure per transect
        for ti, tr in enumerate(TRANSECTS):
            fname = f'{SAVEPATH}cs_diag_no3_box_allscen-{TRANSECT_NAMES[ti]}-{dstr}.png'
            if os.path.exists(fname):
                continue

            fig, axes = plt.subplots(n_scen, 1, sharex=True,
                                     figsize=(10, 3 * n_scen))
            zgrid = tr['zgrid']

            for ax, name in zip(axes, SCENARIOS):
                zr3d, var3d, rho3d = scen_data[name]
                var_plot = cb.boxavg_section(var3d, zr3d, tr)
                rho_plot = cb.boxavg_section(rho3d, zr3d, tr)

                pc = ax.pcolormesh(tr['lon'], zgrid, var_plot,
                                   cmap=VAR_CMAP, vmin=VMIN, vmax=VMAX,
                                   shading='nearest')
                cs = ax.contour(tr['lon'], zgrid, rho_plot, levels=ISO_LEVELS,
                                colors='k', linewidths=0.8)
                ax.clabel(cs, fmt='%.2f', fontsize=7)
                ax.set_ylim([tr['depth_lim'], 0])
                ax.set_ylabel('Depth (m)')
                ax.set_title(LABELS[name])

                sf = ScalarFormatter(useOffset=False)
                sf.set_scientific(False)
                ax.xaxis.set_major_formatter(sf)
                ax.xaxis.set_major_locator(MaxNLocator(5))

            axes[-1].set_xlabel('Longitude')
            fig.canvas.draw()
            pos_top = axes[0].get_position()
            pos_bot = axes[-1].get_position()
            cax = fig.add_axes([pos_top.x1 + 0.015, pos_bot.y0,
                                0.012, pos_top.y1 - pos_bot.y0])
            fig.colorbar(pc, cax=cax, label=VAR_LABEL)
            fig.suptitle(f'{dt0.year}-{dt0.month:02d}-{dt0.day:02d} '
                         f'{dt0.hour:02d}:{dt0.minute:02d} UTC', y=0.92)

            plt.savefig(fname, dpi=800, bbox_inches='tight')
            plt.close()
            print(f'  saved -> {fname}')

    for hisnc, varnc in ncs.values():
        hisnc.close()
        if varnc is not hisnc:
            varnc.close()
