"""
Time Hovmöller of offshore flux along the western edge of the coastal band.
4x1 layout (one row per scenario).  Uses z-sliced NPZ output which already
carries data on a fixed uniform z-grid — no depth-binning needed.
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

TRACER     = 'NO3'   # 'rtrace', 'ptrace', or 'NO3'
print(TRACER)
NPZ_DIR = '../postprocessing'
CACHE   = f'./figs/offshore_flux_hov_time_zslice_cache_{TRACER}.npz'

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
    'rtrace': [-150, 0],
    'ptrace': [-150, 0],
    'NO3':    [-500, 0],
}


def nice_round_up(x):
    if x <= 0:
        return x
    exp = np.floor(np.log10(x))
    return float(np.ceil(x / 10**exp)) * 10**exp


scenario_rows = [
    ('tides_wec',     'tides, WEC'),
    ('notides_wec',   'no tides, WEC'),
    ('tides_nowec',   'tides, no WEC'),
    ('notides_nowec', 'no tides, no WEC'),
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
    data  = {}
    depth = None
    for name, _ in scenario_rows:
        d     = np.load(npz_path(name))
        flux  = d['offshore_flux']              # (time, n_z, n_valid)
        depth = d['depth']                      # (n_z,) negative downward
        F_td  = np.nanmean(flux, axis=2)        # (time, n_z) — collapse alongshore
        times = pf.numdate(d['ocean_time'], 'seconds since 1995-01-01')
        data[name] = dict(F=F_td, t_num=mdates.date2num(times))

    all_F = np.concatenate([data[n]['F'].ravel() for n, _ in scenario_rows])
    vmax  = nice_round_up(np.nanpercentile(np.abs(all_F), 99))
    vmin  = -vmax

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
ax[3].set_xlabel('Time', fontsize=axfont)
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

out = f'./figs/offshore_flux_hov_time_zslice_{TRACER}.png'
plt.savefig(out, bbox_inches='tight')
print(f'saved -> {out}')
