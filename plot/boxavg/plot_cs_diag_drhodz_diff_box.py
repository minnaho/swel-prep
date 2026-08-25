"""
Box-averaged cross-section differences of time-averaged drho/dz, using
zsliced his output. Box-average variant of ../plot_cs_diag_drhodz_diff.py.

For each scenario, time-averages 'rho' from the zsliced z_mc60_his.*.nc files
onto the same two coast-to-offshore diagonal transects used by
plot_cs_diag_box.py, plus the fixed-eta "mid" geographical cross-section used
by plot_cs_mid_o2.py / plot_cs_w_NO3.py, box-averaged perpendicular to each
line (see cs_boxavg.py -- since the zslice output is already on a fixed
depth grid at every time step, boxavg_section_fixedz needs no vertical
interpolation), then takes d(rho)/dz on that fixed depth grid.

drho/dz here is a finite difference on the zslice output's FIXED depth grid --
the same z-levels at every time step (unlike raw s_rho, which moves with zeta
each time step). Differentiation, box-averaging, and time-averaging are all
linear operators; on a fixed grid they commute exactly, so
mean(box(drho/dz)) == box(d(mean(rho))/dz). We average rho first -- one
gradient at the end instead of one per time step -- for an identical,
cheaper result.

Layout differs from the single-scenario-diff original: panel 1 shows the
base case (notidesnowec) RAW time-mean drho/dz; panels 2-6 show each other
scenario's time-mean drho/dz DIFFERENCED against the base case, as before.
drho/dz has no single-variable sibling script to copy a fixed raw range
from, so the raw panel's range is derived from the base case itself
(2nd/98th percentile over the plotted depth window).

Output: ./figs/cs_diag_drhodz_diff_box_<ts|tn|mid>.png (one per transect)
Cache:  ./figs/cs_diag_drhodz_box_cache_<scen>_<ts|tn|mid>.npz
"""

import os
import sys
sys.path.append('/data/project3/minnaho/global/')
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import cmocean

plt.rcParams.update({'font.size': 14})
from netCDF4 import Dataset
import cs_boxavg as cb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'
SCENARIOS_ZSLICE = {
    'tideswec':      f'{ZSLICE_ROOT}/tideswec',
    'tidesnowec':    f'{ZSLICE_ROOT}/tidesnowec',
    'notidesnowec':  f'{ZSLICE_ROOT}/notidesnowec',
    'notideswec':    f'{ZSLICE_ROOT}/notideswec',
    'tidesampwec':   f'{ZSLICE_ROOT}/tidesampwec',
    'notidesampwec': f'{ZSLICE_ROOT}/notidesampwec',
}
BASE_SCEN = 'notidesnowec'

LABELS = {
    'notidesnowec':  'no tides, no WEC',
    'tideswec':      'tides, WEC',
    'tidesnowec':    'tides, no WEC',
    'notideswec':    'no tides, WEC',
    'notidesampwec': 'no tides, 2.5x WEC',
    'tidesampwec':   'tides, 2.5x WEC',
}

# stored `rho` is a deviation from RHO_REF; add (RHO_REF - 1000) to convert to
# density anomaly relative to 1000 kg m^-3
RHO_REF    = 1027.4
RHO_OFFSET = RHO_REF - 1000.0

DIFF_CMAP  = cmocean.cm.balance
DIFF_LABEL = r'$\Delta\,(\partial\rho/\partial z)$ (kg m$^{-4}$)'
RAW_CMAP   = cmocean.cm.dense_r
RAW_LABEL  = r'$\partial\rho/\partial z$ (kg m$^{-4}$)'

# Require at least half the box's offsets to have valid data at a given
# (depth, transect position) before including it in the box-averaged rho
# profile. Without this, a transect point's deepest reachable level can be
# supported by just 1-2 of the 21 offsets (the rest already below their own
# local bathymetry) instead of the ~21 backing the level just above it --
# and that thin, unrepresentative remainder can differ sharply from the
# well-supported mean one level up, producing a spurious drho/dz spike right
# before the profile goes NaN (seen concretely: 21/21 offsets agreeing on
# sigma-t 25.22 at -28 m, then a single surviving offset giving 13.39 at
# -29 m -- a fake ~12 kg/m^3 jump). Masking those thin levels out here
# removes the spike instead of differentiating through it.
RHO_MIN_FRAC = 0.5

# Diagonal transect geometry (grid index space) -- same as plot_cs_diag_box.py / plot_cs_diag_rho_box.py
ETA0       = 271       # south transect — starting eta index (coast end)
XI0        = 543       # south transect — starting xi index (coast end)
SLOPE      = -0.8      # south transect — deta/dxi
DEPTH_LIM0 = -125      # south transect — y-axis bottom (m)
ETA1       = 832       # north transect — starting eta index (coast end)
XI1        = 646       # north transect — starting xi index (coast end)
SLOPE1     = 0.6       # north transect — deta/dxi
DEPTH_LIM1 = -90       # north transect — y-axis bottom (m)
LENGTH_XI  = 250       # transect length in xi cells (both)
N_PTS      = 300       # interpolation resolution along transect (both)

# Mid transect geometry -- same as plot_cs_mid_o2.py / plot_cs_w_NO3.py
ETA_MID       = 477                 # fixed eta index — inside the bay to offshore
DEPTH_LIM_MID = -300                # y-axis bottom (m)
MID_XLIM      = [-121.96, -121.8]   # longitude window
MID_XTICKS    = np.linspace(MID_XLIM[0], MID_XLIM[1], 5)

GRD      = cb.GRD
SAVEPATH = cb.figpath() + '/'

# ---------------------------------------------------------------------------
# Load grid
# ---------------------------------------------------------------------------
grdnc    = Dataset(GRD, 'r')
lat      = np.array(grdnc['lat_rho'][:])
lon      = np.array(grdnc['lon_rho'][:]) - 360
mask_rho = np.array(grdnc['mask_rho'][:])

mask_plot = mask_rho.astype(float)
mask_plot[mask_plot == 0] = np.nan

# ---------------------------------------------------------------------------
# Build box transect coordinates
# ---------------------------------------------------------------------------
TRANSECTS = [
    cb.build_box_transect(lon, lat, mask_rho, ETA0, XI0, SLOPE,  DEPTH_LIM0,
                          LENGTH_XI, N_PTS, name='ts', title='south transect'),
    cb.build_box_transect(lon, lat, mask_rho, ETA1, XI1, SLOPE1, DEPTH_LIM1,
                          LENGTH_XI, N_PTS, name='tn', title='north transect'),
    cb.build_box_mid_transect(lon, lat, mask_rho, ETA_MID, DEPTH_LIM_MID,
                              xlim=MID_XLIM, xticks=MID_XTICKS, name='mid',
                              title=f'mid transect (eta={ETA_MID})'),
]

# ---------------------------------------------------------------------------
# Per-scenario zsliced file lists
# ---------------------------------------------------------------------------
def zslice_files(scen_dir, exclude_stamps=()):
    files = sorted(glob.glob(os.path.join(scen_dir, 'z_mc60_his.*.nc')))
    return [f for f in files if not any(s in f for s in exclude_stamps)]

SCENARIO_FILES = {
    'tideswec':      zslice_files(SCENARIOS_ZSLICE['tideswec']),
    'tidesnowec':    zslice_files(SCENARIOS_ZSLICE['tidesnowec']),
    'notidesnowec':  zslice_files(SCENARIOS_ZSLICE['notidesnowec']),
    'notideswec':    zslice_files(SCENARIOS_ZSLICE['notideswec']),
    # 20190418230114 was previously corrupted on disk (Akv/Akt content
    # instead of u/v/rho/w) but has since been regenerated correctly -- no
    # longer excluded. 20190429110056 is still excluded: its 1-timestep
    # source file's zslice output has no time dimension at all
    'tidesampwec':   zslice_files(SCENARIOS_ZSLICE['tidesampwec'],
                                  exclude_stamps=('20190429110056',)),
    'notidesampwec': zslice_files(SCENARIOS_ZSLICE['notidesampwec']),
}

for name, files in SCENARIO_FILES.items():
    print(f'  {name}: {len(files)} zslice files')

with Dataset(SCENARIO_FILES[BASE_SCEN][0], 'r') as _tmp:
    DEPTH_1D = np.array(_tmp.variables['depth'][:])   # (157,) metres, 0 to -1980, surface-first

# ---------------------------------------------------------------------------
# Time-averaged, box-averaged rho -> drho/dz per scenario, per transect (cached)
# ---------------------------------------------------------------------------
os.makedirs(SAVEPATH, exist_ok=True)

def drhodz_cache_path(scen, tr_name):
    return f'{SAVEPATH}cs_diag_drhodz_box_cache_{scen}_{tr_name}.npz'


def compute_rho_mean(scen, files):
    """Time-averaged, box-averaged rho(depth, transect position) per
    transect, from zsliced 'rho'. Box-averages each time step's field
    (boxavg_section_fixedz) before accumulating the time mean. Returns a
    list of (n_depth, n_pts) arrays, one per TRANSECTS entry."""
    n_depth = DEPTH_1D.size
    accum_sum   = [np.zeros((n_depth, tr['n_pts'])) for tr in TRANSECTS]
    accum_count = [np.zeros((n_depth, tr['n_pts'])) for tr in TRANSECTS]

    for fi, f in enumerate(files):
        print(f'  {scen}: file {fi + 1}/{len(files)}')
        with Dataset(f, 'r') as nc:
            # Read per timestep, not the whole (tdim, n_depth, eta_rho,
            # xi_rho) variable at once -- each zslice 'rho' is 6.36 GB
            # (12, 157, 1202, 702 float32, uncompressed). A single timestep
            # is 530 MB.
            n_t = nc.variables['rho'].shape[0]
            for t in range(n_t):
                rho3d = np.array(nc.variables['rho'][t]) + RHO_OFFSET
                rho3d[np.abs(rho3d) > 1e10] = np.nan   # below-bathymetry fill value
                rho3d = rho3d * mask_plot

                for ti, tr in enumerate(TRANSECTS):
                    rho_box = cb.boxavg_section_fixedz(rho3d, tr, min_frac=RHO_MIN_FRAC)
                    valid = ~np.isnan(rho_box)
                    accum_sum[ti][valid]   += rho_box[valid]
                    accum_count[ti][valid] += 1

    with np.errstate(invalid='ignore'):
        return [accum_sum[ti] / accum_count[ti] for ti in range(len(TRANSECTS))]


print('\nComputing box-averaged time-mean drho/dz per scenario...')
drhodz_mean = {}   # scen -> [ (n_depth, n_pts) per transect ]
for scen, files in SCENARIO_FILES.items():
    cached = [drhodz_cache_path(scen, tr['name']) for tr in TRANSECTS]
    if all(os.path.exists(c) for c in cached):
        print(f'  {scen}: loading cache...')
        drhodz_mean[scen] = [np.load(c)['drhodz'] for c in cached]
        continue

    if not files:
        print(f'  WARNING: no zslice files for {scen}, skipping')
        continue

    rho_mean_list = compute_rho_mean(scen, files)
    drhodz_list = [np.gradient(rho_mean_list[ti], DEPTH_1D, axis=0)
                   for ti in range(len(TRANSECTS))]
    for ti, drhodz in enumerate(drhodz_list):
        np.savez(cached[ti], drhodz=drhodz)
    drhodz_mean[scen] = drhodz_list

# ---------------------------------------------------------------------------
# Plot: panel 1 = base case raw, panels 2-6 = diff from base case, one
# figure per transect
# ---------------------------------------------------------------------------
diff_scens = [s for s in SCENARIOS_ZSLICE if s != BASE_SCEN and s in drhodz_mean]

for ti, tr in enumerate(TRANSECTS):
    fname = f'{SAVEPATH}cs_diag_drhodz_diff_box_{tr["name"]}.png'
    if os.path.exists(fname):
        continue

    n_rows = 1 + len(diff_scens)
    fig, axes = plt.subplots(n_rows, 1, sharex=True,
                             figsize=(10, 3 * n_rows))

    keep = DEPTH_1D >= tr['depth_lim']

    def _format_ax(ax):
        if tr.get('xlim') is not None:
            ax.set_xlim(tr['xlim'])
        if tr.get('xticks') is not None:
            ax.set_xticks(tr['xticks'])
        ax.set_ylim([tr['depth_lim'], 0])
        ax.set_ylabel('Depth (m)')
        sf = ScalarFormatter(useOffset=False)
        sf.set_scientific(False)
        ax.xaxis.set_major_formatter(sf)
        if tr.get('xticks') is None:
            ax.xaxis.set_major_locator(MaxNLocator(5))

    # panel 1: base case, raw time-mean drho/dz -- range from the base
    # case's own 2nd/98th percentile over the plotted depth window (no
    # fixed-range single-variable sibling script to copy from)
    ax_raw = axes[0]
    base_field = drhodz_mean[BASE_SCEN][ti][keep, :]
    raw_vmin, raw_vmax = np.nanpercentile(base_field, [2, 98])
    pc_raw = ax_raw.pcolormesh(tr['lon'], DEPTH_1D[keep], base_field,
                               cmap=RAW_CMAP, vmin=raw_vmin, vmax=raw_vmax,
                               shading='nearest')
    ax_raw.set_title(LABELS[BASE_SCEN])
    _format_ax(ax_raw)

    # panels 2-6: diff from base case
    diffs = {s: drhodz_mean[s][ti] - drhodz_mean[BASE_SCEN][ti] for s in diff_scens}
    vmax = np.nanpercentile(
        np.abs(np.concatenate([d.ravel() for d in diffs.values()])), 98)

    pc_diff = None
    for ax, scen in zip(axes[1:], diff_scens):
        pc_diff = ax.pcolormesh(tr['lon'], DEPTH_1D[keep], diffs[scen][keep, :],
                                cmap=DIFF_CMAP, vmin=-vmax, vmax=vmax, shading='nearest')
        ax.set_title(f'{LABELS[scen]}  −  {LABELS[BASE_SCEN]}')
        _format_ax(ax)

    axes[-1].set_xlabel('Longitude')
    fig.canvas.draw()

    # two colorbars: raw (panel 1 only) + diff (spans panels 2-6)
    pos_raw      = axes[0].get_position()
    pos_top_diff = axes[1].get_position()
    pos_bot      = axes[-1].get_position()
    cb_w = 0.012
    cax_raw  = fig.add_axes([pos_raw.x1 + 0.015, pos_raw.y0, cb_w, pos_raw.height])
    cax_diff = fig.add_axes([pos_top_diff.x1 + 0.015, pos_bot.y0, cb_w,
                             pos_top_diff.y1 - pos_bot.y0])
    fig.colorbar(pc_raw,  cax=cax_raw,  label=RAW_LABEL)
    fig.colorbar(pc_diff, cax=cax_diff, label=DIFF_LABEL)

    fig.suptitle('Box-averaged time-mean $\\partial\\rho/\\partial z$ — '
                 f'{tr["title"]}', y=0.92)

    plt.savefig(fname, dpi=800, bbox_inches='tight')
    plt.close()
    print(f'  saved -> {fname}')
