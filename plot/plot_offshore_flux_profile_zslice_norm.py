"""
Vertical profile of offshore flux averaged over time and alongshore.
Uses z-sliced NPZ output (offshore_flux_*_zslice_<scenario>.npz) which already
carries data on a fixed uniform z-grid — no depth-binning needed.

_norm version: restricts the alongshore average to columns where
h_edge >= max_depth (the bottom of DEPTH_YLIM for the chosen tracer).  This
keeps the spatial sample consistent at every depth level, removing the artifact
at ~100 m caused by ~30 % of coastal-band columns having their seafloor at
90–105 m and dropping out of the mean partway through the plotted depth range.

For ptrace/rtrace (DEPTH_YLIM bottom = 50 m) all 1202 columns qualify.
For NO3 (DEPTH_YLIM bottom = 500 m) ~38 % of columns (the deep shelf/canyon
ones) qualify — these are the physically relevant locations for deep NO3 flux.

offshore_flux in the npz is already per-unit-area (mmol/m2/s, from -u*C at
the band edge) -- it is NOT divided by dz*dy_face here (that conversion only
applies to the total-flux npz written by the _old.py postprocessing scripts,
which are not used).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRACER     = 'ptrace'   # 'rtrace', 'ptrace', or 'NO3'
print(TRACER)
NPZ_DIR = '../postprocessing'
CACHE   = f'./figs/offshore_flux_profile_zslice_norm_cache_{TRACER}.npz'

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
    'rtrace': [-50, 0],
    'ptrace': [-50, 0],
    'NO3':    [-500, 0],
}

scenarios = ['tidesampwec', 'tidesnowec', 'notidesnowec', 'ampwec']
labels    = {'tidesampwec':   'tides + 2.5x WEC',
             'tidesnowec':    'tides, no WEC',
             'notidesnowec':  'no tides, no WEC',
             'ampwec':        'no tides + 2.5x WEC'}
colors    = {'tidesampwec':   'C0',
             'tidesnowec':    'C1',
             'notidesnowec':  'C2',
             'ampwec':        'C3'}

if os.path.exists(CACHE):
    print(f'Loading cached arrays from {CACHE}')
    c         = np.load(CACHE)
    depth     = c['depth']
    plot_data = {name: dict(F_mean=c[f'F_mean_{name}']) for name in scenarios}
else:
    plot_data = {}
    depth     = None
    max_depth = abs(DEPTH_YLIM[TRACER][0])   # e.g. 500 for NO3, 50 for ptrace

    for name in scenarios:
        d       = np.load(npz_path(name))
        flux    = d['offshore_flux']   # (time, n_z, n_valid)  mmol/m2/s
        depth   = d['depth']           # (n_z,) negative downward
        h_edge  = d['h_edge']          # (n_valid,) positive bathymetry at band edge

        # Restrict to columns that are wet throughout the full plotted depth
        # range so the spatial sample is constant at every depth level.
        deep   = h_edge >= max_depth  # (n_valid,) boolean mask
        F_tm   = np.nanmean(flux[:, :, deep], axis=0)   # (n_z, n_deep)
        F_mean = np.nanmean(F_tm,    axis=1)    # (n_z,) — sample now constant in depth

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

out = f'./figs/offshore_flux_profile_zslice_norm_{TRACER}.png'
plt.savefig(out, bbox_inches='tight', dpi=600)
print(f'saved -> {out}')
