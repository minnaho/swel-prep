"""
Vertical profile of offshore flux averaged over time and alongshore.
Uses z-sliced NPZ output (offshore_flux_*_zslice_<scenario>.npz) which already
carries data on a fixed uniform z-grid — no depth-binning needed.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRACER     = 'ptrace'   # 'rtrace', 'ptrace', or 'NO3'
print(TRACER)
NPZ_DIR = '../postprocessing'
CACHE   = f'./figs/offshore_flux_profile_zslice_cache_{TRACER}.npz'

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

scenarios = ['tides_wec', 'tides_nowec', 'notides_nowec', 'notides_wec']
labels    = {'tides_wec':     'tides + WEC',
             'tides_nowec':   'tides, no WEC',
             'notides_nowec': 'no tides, no WEC',
             'notides_wec':   'no tides + WEC'}
colors    = {'tides_wec':     'C0',
             'tides_nowec':   'C1',
             'notides_nowec': 'C2',
             'notides_wec':   'C3'}

if os.path.exists(CACHE):
    print(f'Loading cached arrays from {CACHE}')
    c         = np.load(CACHE)
    depth     = c['depth']
    plot_data = {name: dict(F_mean=c[f'F_mean_{name}']) for name in scenarios}
else:
    plot_data = {}
    depth     = None
    for name in scenarios:
        d     = np.load(npz_path(name))
        flux  = d['offshore_flux']   # (time, n_z, n_valid)
        depth = d['depth']           # (n_z,) negative downward
        F_mean = np.nanmean(flux, axis=(0, 2))   # (n_z,)
        plot_data[name] = dict(F_mean=F_mean)

    np.savez(CACHE,
             depth=depth,
             **{f'F_mean_{name}': plot_data[name]['F_mean'] for name in scenarios})
    print(f'Saved cache -> {CACHE}')

fig, ax = plt.subplots(figsize=(6, 8))

for name in scenarios:
    F_mean = plot_data[name]['F_mean']
    ax.plot(F_mean, depth, color=colors[name], label=labels[name], linewidth=1.5)

ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)
ax.set_xlabel(f'offshore flux ({TRACER}) [{TRACER_UNITS[TRACER]}]')
ax.set_ylabel('depth (m)')
ax.set_ylim(DEPTH_YLIM[TRACER])
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

out = f'./figs/offshore_flux_profile_zslice_{TRACER}.png'
plt.savefig(out, bbox_inches='tight', dpi=600)
print(f'saved -> {out}')
