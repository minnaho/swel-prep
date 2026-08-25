"""
Time Hovmöller of offshore flux along the western edge of the coastal band.
4x1 layout (one row per scenario).  Uses z-sliced NPZ output which already
carries data on a fixed uniform z-grid — no depth-binning needed.

_norm version: restricts the alongshore collapse to columns where
h_edge >= max_depth (the bottom of DEPTH_YLIM for the chosen tracer), keeping
the spatial sample constant at every depth level and removing the ~100 m
artifact caused by ~30 % of columns having h_edge ≈ 90–105 m and dropping
out of the mean partway through the plotted range.  vmin/vmax is also
restricted to within DEPTH_YLIM so deep cells do not compress the colour scale.

offshore_flux in the npz is already per-unit-area (mmol/m2/s, from -u*C at
the band edge) -- it is NOT divided by dz*dy_face here (that conversion only
applies to the total-flux npz written by the _old.py postprocessing scripts,
which are not used).
"""

import os
import sys
sys.path.append('/data/project3/minnaho/global/')
import pyfuncs as pf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import cmocean

TRACER     = 'ptrace'   # 'rtrace', 'ptrace', or 'NO3'
print(TRACER)
NPZ_DIR = '../postprocessing'
CACHE   = f'./figs/offshore_flux_hov_time_zslice_norm_cache_{TRACER}.npz'

def npz_path(name):
    if TRACER in ('ptrace', 'rtrace'):
        return f'{NPZ_DIR}/offshore_flux_{TRACER}_zslice_{name}.npz'
    return f'{NPZ_DIR}/offshore_flux_zslice_{name}_{TRACER}.npz'

TRACER_UNITS = {
    'rtrace': r'mmol m$^{-2}$ s$^{-1}$',
    'ptrace': r'mmol m$^{-2}$ s$^{-1}$',
    'NO3':    r'mmol N m$^{-2}$ s$^{-1}$',
}
DEPTH_YLIM = {
    'rtrace': [-75, 0],
    'ptrace': [-75, 0],
    'NO3':    [-500, 0],
}


def nice_round_up(x):
    if x <= 0:
        return x
    exp = np.floor(np.log10(x))
    return float(np.ceil(x / 10**exp)) * 10**exp


scenario_rows = [
    ('tidesampwec',   'tides, 2.5x WEC'),
    ('ampwec',        'no tides, 2.5x WEC'),
    ('tidesnowec',    'tides, no WEC'),
    ('notidesnowec',  'no tides, no WEC'),
]

axfont = 16

if os.path.exists(CACHE):
    print(f'Loading cached arrays from {CACHE}')
    c         = np.load(CACHE)
    data      = {name: dict(F=c[f'F_{name}'], t_num=c[f't_num_{name}'])
                 for name, _ in scenario_rows}
    depth     = c['depth']
    vmin      = float(c['vmin'])
    vmax      = float(c['vmax'])
else:
    max_depth = abs(DEPTH_YLIM[TRACER][0])   # e.g. 500 for NO3, 150 for ptrace

    data  = {}
    depth = None
    for name, _ in scenario_rows:
        d       = np.load(npz_path(name))
        flux    = d['offshore_flux']              # (time, n_z, n_valid)  mmol/m2/s
        depth   = d['depth']                      # (n_z,) negative downward
        h_edge  = d['h_edge']                     # (n_valid,) positive bathymetry at band edge

        # Restrict to columns wet throughout the full plotted depth range so
        # the spatial sample is constant at every depth level, removing the
        # ~100 m artifact from columns with h_edge ≈ 90–105 m dropping out.
        deep    = h_edge >= max_depth

        # Collapse alongshore with consistent sample — nanmean is now safe since
        # all selected columns are wet throughout the depth range.
        F_td    = np.nanmean(flux[:, :, deep], axis=2)           # (time, n_z)

        times   = pf.numdate(d['ocean_time'], 'seconds since 1995-01-01')
        data[name] = dict(F=F_td, t_num=mdates.date2num(times))

    # Restrict colour range to within the plotted depth window.
    depth_lim = DEPTH_YLIM[TRACER]
    in_range  = (depth >= depth_lim[0]) & (depth <= depth_lim[1])
    all_F     = np.concatenate([data[n]['F'][:, in_range].ravel()
                                for n, _ in scenario_rows])
    finite_F  = all_F[np.isfinite(all_F)]
    vmax      = nice_round_up(np.nanpercentile(np.abs(finite_F), 99))
    vmin      = -vmax

    np.savez(CACHE,
             depth=depth,
             vmin=np.float64(vmin), vmax=np.float64(vmax),
             **{f'F_{name}':     data[name]['F']     for name, _ in scenario_rows},
             **{f't_num_{name}': data[name]['t_num'] for name, _ in scenario_rows})
    print(f'Saved cache -> {CACHE}')

cmap = cmocean.cm.balance
cmap.set_bad(color='w')

fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=[14, 14])

pc = None
for row_idx, (name, label) in enumerate(scenario_rows):
    F     = data[name]['F']       # (time, n_z)
    t_num = data[name]['t_num']   # (time,)
    pc = ax[row_idx].pcolormesh(t_num, depth, F.T,
                                cmap=cmap, vmin=vmin, vmax=vmax,
                                shading='nearest')
    ax[row_idx].set_ylabel(f'{label}\nDepth (m)', fontsize=axfont)
    ax[row_idx].tick_params(axis='both', which='major', labelsize=axfont - 2)
    ax[row_idx].set_ylim(DEPTH_YLIM[TRACER])

ax[0].set_title(f'Offshore flux ({TRACER})', fontsize=axfont + 2)
ax[3].xaxis_date()
ax[3].xaxis.set_major_locator(mdates.AutoDateLocator())
ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
fig.autofmt_xdate()

fig.tight_layout(rect=[0, 0.08, 1, 0.96])

pos   = ax[3].get_position().get_points().flatten()
cb_ax = fig.add_axes([pos[0], pos[1] - 0.06, pos[2] - pos[0], 0.015])
cb    = fig.colorbar(pc, cax=cb_ax, orientation='horizontal')
cb.set_label(f'alongshore-mean offshore flux ({TRACER}) [{TRACER_UNITS[TRACER]}]',
             fontsize=axfont)
cb.set_ticks(np.linspace(vmin, vmax, 5))
cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
cb.ax.tick_params(axis='both', which='major', labelsize=axfont - 2)

out = f'./figs/offshore_flux_hov_time_zslice_norm_{TRACER}.png'
plt.savefig(out, bbox_inches='tight', dpi=600)
print(f'saved -> {out}')
