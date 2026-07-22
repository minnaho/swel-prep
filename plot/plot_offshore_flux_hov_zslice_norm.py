"""
Time-mean offshore-flux section along the western edge of the coastal band.
4x1 layout (one row per scenario).  Uses z-sliced NPZ output which already
carries data on a fixed uniform z-grid — no depth-binning needed.

_norm version: the lat×depth spatial display is unchanged (NaN below seafloor
shows as white, which is correct), but vmin/vmax is computed only from cells
within DEPTH_YLIM so that deep cells with potentially stronger flux do not
compress the colour scale for the displayed depth range.
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
NPZ_DIR = '../postprocessing'
GRD     = 'mc60_grd.nc'
CACHE   = f'./figs/offshore_flux_hov_zslice_norm_cache_{TRACER}.npz'

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
    c        = np.load(CACHE)
    data     = {name: dict(F=c[f'F_{name}']) for name, _ in scenario_rows}
    depth    = c['depth']
    lat_edge = c['lat_edge']
    vmin     = float(c['vmin'])
    vmax     = float(c['vmax'])
else:
    data = {}
    depth = None
    for name, _ in scenario_rows:
        d       = np.load(npz_path(name))
        flux    = d['offshore_flux']              # (time, n_z, n_valid)  mmol/s
        dz      = d['dz']                         # (n_z, n_valid)
        dy      = d['dy_face']                    # (n_valid,)
        depth   = d['depth']                      # (n_z,) negative downward
        flux_pa = flux / (dz[None, :, :] * dy[None, None, :])   # mmol/m2/s
        F_mean  = np.nanmean(flux_pa, axis=0)     # (n_z, n_valid) — time mean only
        data[name] = dict(F=F_mean, eta=d['eta_idx'], xi=d['xi_idx'])

    grdnc    = Dataset(GRD)
    lat_rho  = np.array(grdnc.variables['lat_rho'])
    ref      = data[scenario_rows[0][0]]
    lat_edge = lat_rho[ref['eta'], ref['xi']]

    # Restrict colour range to within the plotted depth window so that deeper
    # cells with potentially stronger flux do not compress the near-surface scale.
    depth_lim = DEPTH_YLIM[TRACER]
    in_range  = (depth >= depth_lim[0]) & (depth <= depth_lim[1])
    all_F     = np.concatenate([data[n]['F'][in_range, :].ravel()
                                for n, _ in scenario_rows])
    finite_F  = all_F[np.isfinite(all_F)]
    vmax      = nice_round_up(np.nanpercentile(np.abs(finite_F), 99))
    vmin      = -vmax

    np.savez(CACHE,
             depth=depth, lat_edge=lat_edge,
             vmin=np.float64(vmin), vmax=np.float64(vmax),
             **{f'F_{name}': data[name]['F'] for name, _ in scenario_rows})
    print(f'Saved cache -> {CACHE}')

cmap = cmocean.cm.balance
cmap.set_bad(color='w')

fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=[10, 14])

pc = None
for row_idx, (name, label) in enumerate(scenario_rows):
    F = data[name]['F']   # (n_z, n_valid)
    pc = ax[row_idx].pcolormesh(lat_edge, depth, F,
                                cmap=cmap, vmin=vmin, vmax=vmax,
                                shading='nearest')
    ax[row_idx].set_ylabel(f'{label}\nDepth (m)', fontsize=axfont)
    ax[row_idx].tick_params(axis='both', which='major', labelsize=axfont - 2)
    ax[row_idx].set_ylim(DEPTH_YLIM[TRACER])

ax[0].set_title(f'Offshore flux ({TRACER})', fontsize=axfont + 2)
ax[3].set_xlabel('Latitude', fontsize=axfont)

fig.tight_layout(rect=[0, 0.08, 1, 0.96])

pos  = ax[3].get_position().get_points().flatten()
cb_ax = fig.add_axes([pos[0], pos[1] - 0.06, pos[2] - pos[0], 0.015])
cb   = fig.colorbar(pc, cax=cb_ax, orientation='horizontal')
cb.set_label(f'time-mean offshore flux ({TRACER}) [{TRACER_UNITS[TRACER]}]', fontsize=axfont)
cb.set_ticks(np.linspace(vmin, vmax, 5))
cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
cb.ax.tick_params(axis='both', which='major', labelsize=axfont - 2)

out = f'./figs/offshore_flux_hov_zslice_norm_{TRACER}.png'
plt.savefig(out, bbox_inches='tight', dpi=600)
print(f'saved -> {out}')
