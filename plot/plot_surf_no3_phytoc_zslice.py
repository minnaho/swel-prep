"""
Raw-sigma-top-surface NO3 and phyto C snapshot maps, combined into one
script since both come from the same raw bgc file -- one read per timestep
serves both. phytoC = total phytoplankton biomass = DIATC + DIAZC + SPC
(diatom + diazotroph + small phytoplankton carbon), same convention
plot_hov_transect_raw.py uses.

**Naming note**: this file was originally a zsliced (fixed depth=0m)
version, but that was switched back to raw sigma-top after realizing a
fixed z=0m level isn't actually "the surface" -- sigma levels are
free-surface-following by construction, so the top sigma level IS the
instantaneous ocean surface regardless of tide/wave phase, while a fixed
z=0m slice is a static reference depth the true (undulating) surface moves
around -- during a low-tide or wave-trough moment the real surface can sit
below z=0, so the zsliced sample there isn't actually surface water. This
matters more here than in most other places in this codebase, since
WEC-driven surface elevation is the whole point of this study. The filename
keeps the "_zslice" suffix for output-path/run_plots.py stability even
though the data source is now raw bgc, not zsliced -- a deliberate, flagged
inconsistency, not an oversight.

Restricted to 4 scenarios: notidesnowec, tidesnowec, notidesampwec,
tidesampwec.

NO3/phytoC use a linear color scale (not log10 -- that's for
plot_surf_rtrace_ptrace_zslice.py's trace tracers spanning many orders of
magnitude), matching every other NO3/phytoC plot in this codebase.

No skip-if-exists restart logic (always overwrites); same 2x2 panel layout
as plot_surf_rtrace.py / plot_surf_ptrace.py.

Output: ./figs/snapshots/no3_zslice/surf_no3_zslice-YYYY-MM-DD-HH.png
        ./figs/snapshots/phytoc_zslice/surf_phytoc_zslice-YYYY-MM-DD-HH.png
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
PANEL_ORDER = ['notidesnowec', 'notidesampwec', 'tidesnowec', 'tidesampwec']

PHYTOC_VARS = ['DIATC', 'DIAZC', 'SPC']

VAR_CONFIGS = {
    'NO3': dict(savepath='./figs/snapshots/no3_zslice/',
                prefix='surf_no3_zslice',
                cmap=cmocean.cm.turbid, vmin=0, vmax=30,
                label=r'NO$_3$ (mmol N m$^{-3}$)'),
    'phytoC': dict(savepath='./figs/snapshots/phytoc_zslice/',
                   prefix='surf_phytoc_zslice',
                   cmap=cmocean.cm.algae, vmin=0, vmax=100,
                   label=r'phyto C (mmol C m$^{-3}$)'),
}

grdnc = Dataset(GRD, 'r')
lat_nc = np.array(grdnc.variables['lat_rho'])
lon_nc = np.array(grdnc.variables['lon_rho']) - 360
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc == 0] = np.nan
maskc = np.array(grdnc.variables['mask_rho'])

figw, figh = 12, 15
axfont = 16

for cfg in VAR_CONFIGS.values():
    os.makedirs(cfg['savepath'], exist_ok=True)


def src_glob(root, kind):
    """Locate raw his/ or bgc/ files, handling flat vs subdir scenario
    layouts (notidesampwec/tidesampwec are flat; the other two use
    his//bgc/ subdirs)."""
    sub  = os.path.join(root, kind)
    base = sub if os.path.isdir(sub) else root
    stem = 'mc60_his' if kind == 'his' else 'mc60_bgc'
    return os.path.join(base, f'{stem}.*.nc')


bgc_files = {scen: sorted(glob.glob(src_glob(SCEN_ROOTS[scen], 'bgc'))) for scen in SCENARIOS}
n_files = min(len(bgc_files[s]) for s in SCENARIOS)
print(f'{n_files} matched file(s) per scenario', flush=True)


def clean(arr):
    arr = np.array(arr)
    return np.where(np.abs(arr) > 1e30, np.nan, arr)


def plot_var(var, panels, time_str, out_file):
    cfg = VAR_CONFIGS[var]
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[figw, figh])
    fig.suptitle(time_str, fontsize=axfont + 2, fontweight='bold')

    p_plot = None
    for i, scen in enumerate(PANEL_ORDER):
        p_plot = ax.flat[i].pcolormesh(lon_nc, lat_nc, panels[scen],
                                        cmap=cfg['cmap'], vmin=cfg['vmin'], vmax=cfg['vmax'])
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
    cb.set_label(cfg['label'], fontsize=axfont)
    cb.ax.tick_params(axis='both', which='major', labelsize=axfont)

    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


for hf in range(n_files):
    print(bgc_files[SCENARIOS[0]][hf], flush=True)
    ncs = {scen: Dataset(bgc_files[scen][hf], 'r') for scen in SCENARIOS}
    bgc_time = np.array(ncs[SCENARIOS[0]].variables['ocean_time'])

    # read the top sigma level for every timestep/var at once -- avoids
    # re-reading the full (time, s_rho, eta, xi) variable on every timestep
    # iteration (12x/file)
    surf = {scen: {v: clean(np.array(ncs[scen].variables[v][:, -1, :, :]))
                   for v in ['NO3'] + PHYTOC_VARS}
            for scen in SCENARIOS}

    for t_i in range(bgc_time.shape[0]):
        dt0 = num2date(bgc_time, 'seconds since 1995-01-01')[t_i]
        time_str = f"{dt0.year}-{dt0.month:02d}-{dt0.day:02d} {dt0.hour:02d}:{dt0.minute:02d}"
        date_str = f"{dt0.year}-{dt0.month:02d}-{dt0.day:02d}-{dt0.hour:02d}"

        out_files = {v: f"{VAR_CONFIGS[v]['savepath']}{VAR_CONFIGS[v]['prefix']}-{date_str}.png"
                     for v in VAR_CONFIGS}

        panels = {'NO3': {}, 'phytoC': {}}
        for scen in SCENARIOS:
            no3 = surf[scen]['NO3'][t_i] * masknc
            phytoc = sum(surf[scen][v][t_i] for v in PHYTOC_VARS) * masknc
            panels['NO3'][scen] = no3
            panels['phytoC'][scen] = phytoc

        for var in VAR_CONFIGS:
            plot_var(var, panels[var], time_str, out_files[var])

    for scen in SCENARIOS:
        ncs[scen].close()
