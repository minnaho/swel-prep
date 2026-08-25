"""
Raw-sigma-top-surface version of plot_surf_rtrace.py / plot_surf_ptrace.py,
combined into one script since both tracers live in the same raw his file
-- one read per timestep serves both instead of opening/reading it twice.

**Naming note**: this file was originally a zsliced (fixed depth=0m) version,
but that was switched back to raw sigma-top after realizing a fixed z=0m
level isn't actually "the surface" -- sigma levels are free-surface-following
by construction, so the top sigma level IS the instantaneous ocean surface
regardless of tide/wave phase, while a fixed z=0m slice is a static
reference depth the true (undulating) surface moves around -- during a
low-tide or wave-trough moment the real surface can sit below z=0, so the
zsliced sample there isn't actually surface water. This matters more here
than in most other places in this codebase, since WEC-driven surface
elevation is the whole point of this study. The filename keeps the
"_zslice" suffix for output-path/run_plots.py stability even though the
data source is now raw his, not zsliced -- a deliberate, flagged
inconsistency, not an oversight.

Restricted to 4 scenarios: notidesnowec, tidesnowec, notidesampwec,
tidesampwec (dict keys kept as the zslice-style names already used
elsewhere in this pair of scripts; 'notidesampwec' maps to the same raw
'ampwec' root plot_surf_rtrace.py's original hisfolder4 pointed at).

No skip-if-exists restart logic (always overwrites); same log10 color
scale, per-tracer colormap (c_mapr for rtrace, c_mapp for ptrace), and 2x2
panel layout as the original raw versions.

Output: ./figs/snapshots/rtrace_zslice/surf_rtrace_zslice-YYYY-MM-DD-HH.png
        ./figs/snapshots/ptrace_zslice/surf_ptrace_zslice-YYYY-MM-DD-HH.png
"""

import os
import glob
import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
import cmocean

GRD = 'mc60_grd.nc'

SCENARIOS = ['notidesnowec', 'tidesnowec', 'notidesampwec', 'tidesampwec']
SCEN_ROOTS = {
    'notidesnowec':  '/data/project3/minnaho/swel/notides/mc60/nowec',
    'tidesnowec':    '/data/project3/minnaho/swel/tides/mc60/nowec/output',
    'notidesampwec': '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything',
    'tidesampwec':   '/data/project3/minnaho/swel/tides/mc60/ampwec/everything',
}
LABELS = {
    'notidesnowec': 'no tides, no WEC',
    'tidesnowec':   'tides, no WEC',
    'notidesampwec': 'no tides, 2.5x WEC',
    'tidesampwec':  'tides, 2.5x WEC',
}
# 2x2 panel positions, matching plot_surf_rtrace.py's layout
PANEL_ORDER = ['notidesnowec', 'notidesampwec', 'tidesnowec', 'tidesampwec']

TRACERS = {
    'rtrace': dict(savepath='./figs/snapshots/rtrace_zslice/',
                    prefix='surf_rtrace_zslice'),
    'ptrace': dict(savepath='./figs/snapshots/ptrace_zslice/',
                    prefix='surf_ptrace_zslice'),
}

grdnc = Dataset(GRD, 'r')
lat_nc = np.array(grdnc.variables['lat_rho'])
lon_nc = np.array(grdnc.variables['lon_rho']) - 360
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc == 0] = np.nan
maskc = np.array(grdnc.variables['mask_rho'])

figw, figh = 12, 15
c_mapr = cmocean.cm.rain
c_mapp = cmocean.cm.matter
c_mapr.set_bad(color='w')
c_mapp.set_bad(color='w')
CMAPS = {'rtrace': c_mapr, 'ptrace': c_mapp}
v_min, v_max = -7, -1
axfont = 16

for cfg in TRACERS.values():
    os.makedirs(cfg['savepath'], exist_ok=True)


def src_glob(root):
    """Locate raw his files, handling flat vs his/ subdir scenario layouts
    (notidesampwec/tidesampwec are flat; the other two use a his/ subdir)."""
    sub  = os.path.join(root, 'his')
    base = sub if os.path.isdir(sub) else root
    return os.path.join(base, 'mc60_his.*.nc')


his_files = {scen: sorted(glob.glob(src_glob(SCEN_ROOTS[scen]))) for scen in SCENARIOS}
n_files = min(len(his_files[s]) for s in SCENARIOS)
print(f'{n_files} matched file(s) per scenario', flush=True)


def to_log10_panel(arr):
    arr = arr.copy()
    arr[arr < 0] = 0
    arr = np.log10(arr)         # log makes 0 -> -inf
    arr[np.isinf(arr)] = -38    # put -inf back to a small value
    return arr


def plot_tracer(tracer, panels, time_str, out_file):
    cfg = TRACERS[tracer]
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[figw, figh])
    fig.suptitle(time_str, fontsize=axfont + 2, fontweight='bold')

    p_plot = None
    for i, scen in enumerate(PANEL_ORDER):
        p_plot = ax.flat[i].pcolormesh(lon_nc, lat_nc, panels[scen],
                                        cmap=CMAPS[tracer], vmin=v_min, vmax=v_max)
        ax.flat[i].contour(lon_nc, lat_nc, maskc, colors='k', linewidths=1)
        ax.flat[i].set_title(LABELS[scen], fontsize=axfont)

    for i in range(4):
        ax.flat[i].set_ylim([36.47, 37.05])
        ax.flat[i].set_xlim([-122.4, -121.75])
        ax.flat[i].tick_params(axis='both', which='major', labelsize=axfont - 2)

    ax.flat[2].set_xlabel('Longitude', fontsize=axfont)
    ax.flat[3].set_xlabel('Longitude', fontsize=axfont)
    ax.flat[0].set_ylabel('Latitude', fontsize=axfont)
    ax.flat[2].set_ylabel('Latitude', fontsize=axfont)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    p_top = ax.flat[1].get_position().get_points().flatten()
    p_bot = ax.flat[3].get_position().get_points().flatten()
    cb_ax = fig.add_axes([p_top[2] + .02, p_bot[1], .015, p_top[3] - p_bot[1]])
    cb = fig.colorbar(p_plot, cax=cb_ax, orientation='vertical')
    cb.set_label(rf'log$_{{10}}$({tracer})', fontsize=axfont)
    cb.ax.tick_params(axis='both', which='major', labelsize=axfont)

    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


for hf in range(n_files):
    print(his_files[SCENARIOS[0]][hf], flush=True)
    ncs = {scen: Dataset(his_files[scen][hf], 'r') for scen in SCENARIOS}
    his_time = np.array(ncs[SCENARIOS[0]].variables['ocean_time'])

    # read the top sigma level for every timestep/tracer at once -- avoids
    # re-reading the full (time, s_rho, eta, xi) variable on every timestep
    # iteration (12x/file), which is what made the naive per-timestep read
    # this replaced far slower than it needed to be
    surf = {scen: {tracer: np.array(ncs[scen].variables[tracer][:, -1, :, :])
                   for tracer in TRACERS}
            for scen in SCENARIOS}

    for t_i in range(his_time.shape[0]):
        dt0 = num2date(his_time, 'seconds since 1995-01-01')[t_i]
        time_str = f"{dt0.year}-{dt0.month:02d}-{dt0.day:02d} {dt0.hour:02d}:{dt0.minute:02d}"
        date_str = f"{dt0.year}-{dt0.month:02d}-{dt0.day:02d}-{dt0.hour:02d}"

        out_files = {t: f"{TRACERS[t]['savepath']}{TRACERS[t]['prefix']}-{date_str}.png"
                     for t in TRACERS}

        for tracer in TRACERS:
            panels = {}
            for scen in SCENARIOS:
                arr = surf[scen][tracer][t_i] * masknc
                panels[scen] = to_log10_panel(arr)
            plot_tracer(tracer, panels, time_str, out_files[tracer])

    for scen in SCENARIOS:
        ncs[scen].close()
