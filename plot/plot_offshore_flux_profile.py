"""
Vertical profile of offshore flux averaged over time and alongshore.
For each scenario, collapses the (time, alongshore) dimensions of the band-edge
flux to a single F(depth) curve and overlays all four scenarios on one panel.

Reads ../postprocessing/offshore_flux_<TRACER>_<scenario>.npz produced by
postprocessing/offshore_flux_{rtrace,ptrace}.py (or offshore_flux.py for NO3).

Sign convention (see offshore_flux*.py):
    F > 0 -> offshore export (band -> open ocean)
    F < 0 -> onshore import  (open ocean -> band)

Depth binning: because terrain-following s_rho levels do not align in z across
columns, each (time, alongshore) column is binned into uniform depth bins before
averaging — following the approach in sftracer/plot/plot_transport.py.
After binning, each profile is reversed so index 0 = near-surface, NaNs moved
to the end, then nanmean collapses all (time, alongshore) profiles at each depth.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

TRACER = 'ptrace'
print(TRACER)
NPZ_DIR = '../postprocessing'
NPZ_PREFIX = f'offshore_flux_{TRACER}' if TRACER != 'NO3' else 'offshore_flux'
BIN_SZ = 2  # depth bin size in metres
CACHE  = f'./figs/offshore_flux_profile_cache_{TRACER}.npz'

# Per-cell flux units = tracer * m^3 / s (see postprocessing/offshore_flux*.py).
TRACER_UNITS = {
    'rtrace': r'mmol s$^{-1}$',
    'ptrace': r'mmol s$^{-1}$',
    'NO3':    r'mmol N s$^{-1}$',
}

# Tracer-specific depth window — passive tracers are confined above 150 m.
DEPTH_YLIM = {
    'rtrace': [-150, 0],
    'ptrace': [-150, 0],
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
    c          = np.load(CACHE)
    depth_axis = c['depth_axis']
    plot_data  = {name: dict(F_mean=c[f'F_mean_{name}']) for name in scenarios}
else:
    # Pass 1: find global z minimum across all scenarios (deepest s_rho level only)
    z_global_min = 0.0
    for name in scenarios:
        d_tmp = np.load(f'{NPZ_DIR}/{NPZ_PREFIX}_{name}.npz')
        z_global_min = min(z_global_min, float(np.nanmin(d_tmp['z_r'][:, 0, :])))

    shared_bins = np.arange(np.floor(z_global_min), 2 * BIN_SZ, BIN_SZ)
    depth_axis  = shared_bins[:-1][::-1]   # surface-first: [0, -BIN_SZ, ...]
    n_bins      = len(shared_bins) - 1

    plot_data = {}
    for name in scenarios:
        d     = np.load(f'{NPZ_DIR}/{NPZ_PREFIX}_{name}.npz')
        F_all = d['offshore_flux']   # (time, s_rho, alongshore)
        z_all = d['z_r']

        bidx = np.searchsorted(shared_bins, z_all.ravel(), side='right') - 1
        ok   = (bidx >= 0) & (bidx < n_bins)
        rsum = np.bincount(bidx[ok], weights=F_all.ravel()[ok], minlength=n_bins)
        rcnt = np.bincount(bidx[ok], minlength=n_bins)
        F_mean = np.where(rcnt > 0, rsum / rcnt, np.nan)[::-1]   # surface-first
        plot_data[name] = dict(F_mean=F_mean)

    np.savez(CACHE,
             depth_axis=depth_axis,
             **{f'F_mean_{name}': plot_data[name]['F_mean'] for name in scenarios})
    print(f'Saved cache -> {CACHE}')

fig, ax = plt.subplots(figsize=(6, 8))

for name in scenarios:
    F_mean = plot_data[name]['F_mean']
    n = min(len(depth_axis), len(F_mean))
    ax.plot(F_mean[:n], depth_axis[:n], color=colors[name], label=labels[name], linewidth=1.5)

ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)
ax.set_xlabel(f'offshore flux ({TRACER}) [{TRACER_UNITS[TRACER]}]')
ax.set_ylabel('depth (m)')
ax.set_ylim(DEPTH_YLIM[TRACER])
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.savefig(f'./figs/offshore_flux_profile_{TRACER}.png',
            bbox_inches='tight', dpi=600)
print(f'saved -> ./figs/offshore_flux_profile_{TRACER}.png')
