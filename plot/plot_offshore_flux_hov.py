"""
Time-mean offshore-flux section along the western edge of the coastal band.
4x1 layout (one row per scenario), styled to match plot_cs_w_NO3.py.

Reads ../postprocessing/offshore_flux_<TRACER>_<scenario>.npz produced by
postprocessing/offshore_flux_{rtrace,ptrace}.py (or offshore_flux.py for NO3).

Sign convention (see offshore_flux*.py):
    F > 0 -> offshore export (band -> open ocean)
    F < 0 -> onshore import  (open ocean -> band)

Depth binning: because terrain-following s_rho levels do not align in z across
time steps, each (time, alongshore) column is binned into uniform depth bins
before averaging — following the approach in plot_offshore_flux_profile.py.
After binning, each profile is reversed so index 0 = near-surface, then
nanmean collapses all time steps at each depth bin (alongshore is preserved).
"""

import os
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import cmocean

TRACER     = 'ptrace'   # 'rtrace', 'ptrace', or 'NO3'
print(TRACER)
NPZ_DIR    = '../postprocessing'
NPZ_PREFIX = f'offshore_flux_{TRACER}' if TRACER != 'NO3' else 'offshore_flux'
GRD        = 'mc60_grd.nc'
BIN_SZ     = 2  # depth bin size in metres
CACHE      = f'./figs/offshore_flux_hov_cache_{TRACER}.npz'

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



def nice_round_up(x):
    """Round x up to the nearest integer * 10^n (n = floor(log10(x)))."""
    if x <= 0:
        return x
    exp = np.floor(np.log10(x))
    return float(np.ceil(x / 10**exp)) * 10**exp


# Row order matches plot_cs_w_NO3.py
scenario_rows = [
    ('tidesampwec',   'tides, 2.5x WEC'),
    ('ampwec',        'no tides, 2.5x WEC'),
    ('tidesnowec',    'tides, no WEC'),
    ('notidesnowec',  'no tides, no WEC'),
]

axfont = 16

if os.path.exists(CACHE):
    print(f'Loading cached arrays from {CACHE}')
    c          = np.load(CACHE)
    data       = {name: dict(F=c[f'F_{name}']) for name, _ in scenario_rows}
    lat_edge   = c['lat_edge']
    depth_axis = c['depth_axis']
    vmin       = float(c['vmin'])
    vmax       = float(c['vmax'])
else:
    # Pass 1: find global z minimum across all scenarios (deepest s_rho level only)
    z_global_min = 0.0
    for name, _ in scenario_rows:
        d_tmp = np.load(f'{NPZ_DIR}/{NPZ_PREFIX}_{name}.npz')
        z_global_min = min(z_global_min, float(np.nanmin(d_tmp['z_r'][:, 0, :])))

    shared_bins = np.arange(np.floor(z_global_min), 2 * BIN_SZ, BIN_SZ)
    # bin centers, surface-first; shading='auto' on mpl ≥ 3.5 treats Y as
    # centers, so passing bin edges would shift each cell up by BIN_SZ/2.
    depth_axis  = (shared_bins[:-1] + BIN_SZ / 2)[::-1]
    n_bins      = len(shared_bins) - 1

    # --- Load + depth-bin + time-average ---
    data = {}
    for name, _ in scenario_rows:
        d     = np.load(f'{NPZ_DIR}/{NPZ_PREFIX}_{name}.npz')
        F_all = d['offshore_flux']   # (time, s_rho, alongshore)
        z_all = d['z_r']
        n_time, n_srho, n_along = F_all.shape

        # Collapse (time, s_rho) → keep along; vectorized over time*srho per column
        F_2d = F_all.reshape(-1, n_along)   # (n_time * n_srho, n_along)
        z_2d = z_all.reshape(-1, n_along)

        rsum = np.zeros((n_along, n_bins))
        rcnt = np.zeros((n_along, n_bins), dtype=np.int32)
        for h in range(n_along):
            bidx     = np.searchsorted(shared_bins, z_2d[:, h], side='right') - 1
            ok       = (bidx >= 0) & (bidx < n_bins)
            rsum[h]  = np.bincount(bidx[ok], weights=F_2d[ok, h], minlength=n_bins)
            rcnt[h]  = np.bincount(bidx[ok], minlength=n_bins)

        F_hd = np.where(rcnt > 0, rsum / rcnt, np.nan)[:, ::-1]   # (n_along, n_bins) surface-first
        data[name] = dict(F=F_hd, eta=d['eta_idx'])

    # Latitude along the band-edge cells
    grdnc = Dataset(GRD)
    lat_rho = np.array(grdnc.variables['lat_rho'])
    ref = data[scenario_rows[0][0]]
    lat_edge = lat_rho[ref['eta'], np.load(
        f'{NPZ_DIR}/{NPZ_PREFIX}_{scenario_rows[0][0]}.npz')['xi_idx']]

    # Symmetric color limits from the 99th percentile across all scenarios,
    # rounded up to integer * 10^n so the colorbar shows clean numbers.
    all_F = np.concatenate([data[n]['F'].ravel() for n, _ in scenario_rows])
    vmax  = nice_round_up(np.nanpercentile(np.abs(all_F), 99))
    vmin  = -vmax

    np.savez(CACHE,
             lat_edge=lat_edge, depth_axis=depth_axis,
             vmin=np.float64(vmin), vmax=np.float64(vmax),
             **{f'F_{name}': data[name]['F'] for name, _ in scenario_rows})
    print(f'Saved cache -> {CACHE}')

cmap = cmocean.cm.balance
cmap.set_bad(color='w')

# --- Plot ---
figw = 10
figh = 14
fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=[figw, figh])

pc = None
for row_idx, (name, label) in enumerate(scenario_rows):
    F = data[name]['F']                          # (n_along, n_bins)
    n = min(len(depth_axis), F.shape[1])

    pc = ax[row_idx].pcolormesh(lat_edge, depth_axis[:n], F[:, :n].T,
                                cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    ax[row_idx].set_ylabel(f'{label}\nDepth (m)', fontsize=axfont)
    ax[row_idx].tick_params(axis='both', which='major', labelsize=axfont-2)
    ax[row_idx].set_ylim(DEPTH_YLIM[TRACER])

ax[0].set_title(f'Offshore flux ({TRACER})',
                fontsize=axfont+2)
ax[3].set_xlabel('Latitude', fontsize=axfont)

fig.tight_layout(rect=[0, 0.08, 1, 0.96])

# Horizontal colorbar at bottom, spanning the column
pos = ax[3].get_position().get_points().flatten()
cb_ax = fig.add_axes([pos[0], pos[1]-0.06, pos[2]-pos[0], 0.015])
cb = fig.colorbar(pc, cax=cb_ax, orientation='horizontal')
cb.set_label(f'time-mean offshore flux ({TRACER}) [{TRACER_UNITS[TRACER]}]',
             fontsize=axfont)
cb.set_ticks(np.linspace(vmin, vmax, 5))
cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
cb.ax.tick_params(axis='both', which='major', labelsize=axfont-2)

out = f'./figs/offshore_flux_hov_{TRACER}.png'
plt.savefig(out, bbox_inches='tight')
print(f'saved -> {out}')
