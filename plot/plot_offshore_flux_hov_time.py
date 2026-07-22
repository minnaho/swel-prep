"""
Time Hovmöller of offshore flux along the western edge of the coastal band.
4x1 layout (one row per scenario), styled to match plot_offshore_flux_hov.py
but with the alongshore axis collapsed and time on the horizontal axis instead.

Reads ../postprocessing/offshore_flux_<TRACER>_<scenario>.npz produced by
postprocessing/offshore_flux_{rtrace,ptrace}.py (or offshore_flux.py for NO3).

Sign convention (see offshore_flux*.py):
    F > 0 -> offshore export (band -> open ocean)
    F < 0 -> onshore import  (open ocean -> band)

Depth binning: because terrain-following s_rho levels do not align in z across
columns, each (time, alongshore) column is binned into uniform depth bins before
averaging — following the approach in plot_offshore_flux_profile.py.
After binning, each profile is reversed so index 0 = near-surface, then
nanmean collapses all alongshore profiles at each depth bin (time is preserved).
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
NPZ_DIR    = '../postprocessing'
NPZ_PREFIX = f'offshore_flux_{TRACER}' if TRACER != 'NO3' else 'offshore_flux'
BIN_SZ     = 2  # depth bin size in metres
CACHE      = f'./figs/offshore_flux_hov_time_cache_{TRACER}.npz'

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

# Row order matches plot_offshore_flux_hov.py
scenario_rows = [
    ('tides_wec',     'tides, WEC'),
    ('notides_wec',   'no tides, WEC'),
    ('tides_nowec',   'tides, no WEC'),
    ('notides_nowec', 'no tides, no WEC'),
]

axfont = 16



if os.path.exists(CACHE):
    print(f'Loading cached arrays from {CACHE}')
    c          = np.load(CACHE)
    data       = {name: dict(F=c[f'F_{name}'], t_num=c[f't_num_{name}'])
                  for name, _ in scenario_rows}
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

    # --- Load + depth-bin + alongshore-average ---
    data = {}
    for name, _ in scenario_rows:
        d     = np.load(f'{NPZ_DIR}/{NPZ_PREFIX}_{name}.npz')
        F_all = d['offshore_flux']   # (time, s_rho, alongshore)
        z_all = d['z_r']
        n_time, n_srho, n_along = F_all.shape

        # Collapse (s_rho, along) → keep time; vectorized over along*srho per time step
        F_2d = F_all.transpose(0, 2, 1).reshape(n_time, -1)   # (n_time, n_along * n_srho)
        z_2d = z_all.transpose(0, 2, 1).reshape(n_time, -1)

        rsum = np.zeros((n_time, n_bins))
        rcnt = np.zeros((n_time, n_bins), dtype=np.int32)
        for t in range(n_time):
            bidx     = np.searchsorted(shared_bins, z_2d[t], side='right') - 1
            ok       = (bidx >= 0) & (bidx < n_bins)
            rsum[t]  = np.bincount(bidx[ok], weights=F_2d[t][ok], minlength=n_bins)
            rcnt[t]  = np.bincount(bidx[ok], minlength=n_bins)

        F_td  = np.where(rcnt > 0, rsum / rcnt, np.nan)[:, ::-1]   # (n_time, n_bins) surface-first
        times = pf.numdate(d['ocean_time'], 'seconds since 1995-01-01')
        data[name] = dict(F=F_td, t_num=mdates.date2num(times))

    # Symmetric color limits from the 99th percentile across all scenarios,
    # rounded up to integer * 10^n so the colorbar shows clean numbers.
    all_F = np.concatenate([data[n]['F'].ravel() for n, _ in scenario_rows])
    vmax  = nice_round_up(np.nanpercentile(np.abs(all_F), 99))
    vmin  = -vmax

    np.savez(CACHE,
             depth_axis=depth_axis,
             vmin=np.float64(vmin), vmax=np.float64(vmax),
             **{f'F_{name}':     data[name]['F']     for name, _ in scenario_rows},
             **{f't_num_{name}': data[name]['t_num'] for name, _ in scenario_rows})
    print(f'Saved cache -> {CACHE}')

cmap = cmocean.cm.balance
cmap.set_bad(color='w')

# --- Plot ---
figw = 14
figh = 14
fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=[figw, figh])

pc = None
for row_idx, (name, label) in enumerate(scenario_rows):
    F     = data[name]['F']                              # (time, n_bins)
    t_num = data[name]['t_num']                          # (time,)
    n     = min(len(depth_axis), F.shape[1])

    pc = ax[row_idx].pcolormesh(t_num, depth_axis[:n], F[:, :n].T,
                                cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    ax[row_idx].set_ylabel(f'{label}\nDepth (m)', fontsize=axfont)
    ax[row_idx].tick_params(axis='both', which='major', labelsize=axfont-2)
    ax[row_idx].set_ylim(DEPTH_YLIM[TRACER])

ax[0].set_title(f'Offshore flux ({TRACER}) — positive = offshore',
                fontsize=axfont+2)
ax[3].set_xlabel('Time', fontsize=axfont)
ax[3].xaxis_date()
ax[3].xaxis.set_major_locator(mdates.AutoDateLocator())
ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

fig.tight_layout(rect=[0, 0.08, 1, 0.96])

# Horizontal colorbar at bottom, spanning the column
pos = ax[3].get_position().get_points().flatten()
cb_ax = fig.add_axes([pos[0], pos[1]-0.06, pos[2]-pos[0], 0.015])
cb = fig.colorbar(pc, cax=cb_ax, orientation='horizontal')
cb.set_label(f'alongshore-mean offshore flux ({TRACER}) [{TRACER_UNITS[TRACER]}]',
             fontsize=axfont)
cb.set_ticks(np.linspace(vmin, vmax, 5))
cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
cb.ax.tick_params(axis='both', which='major', labelsize=axfont-2)

out = f'./figs/offshore_flux_hov_time_{TRACER}.png'
plt.savefig(out, bbox_inches='tight')
print(f'saved -> {out}')
