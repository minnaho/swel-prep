"""
Box-averaged cross-section differences of time-averaged fields, using
zsliced output, laid out as a 3x2 grid. Same box-averaged computation as
plot_cs_diag_avg_diff_box.py (identical cache files, reused here directly --
this script only changes the plot layout), grid/colorbar layout copied from
../plot_cs_diag_avg_diff.py.

For each variable and scenario, time-averages the field from the zsliced
z_mc60_his.*.nc / z_mc60_bgc.*.nc files onto three cross-sections -- the same
two coast-to-offshore diagonal transects used by plot_cs_diag_box.py, plus the
fixed-eta "mid" geographical cross-section used by plot_cs_mid_o2.py /
plot_cs_w_NO3.py -- box-averaged perpendicular to each line (see
cs_boxavg.py). The zslice output already sits on a FIXED depth grid at every
time step, so the box average here uses boxavg_section_fixedz: no vertical
interpolation is needed, just an average across box offsets at each existing
depth level.

Layout: 3x2 grid of panels -- panel 1 (top-left) is the base case
(notidesnowec) RAW time-mean, on the same cmap/range as the matching
plot_cs_diag_*_box.py single-variable script; the remaining 5 panels are each
other scenario's time-mean DIFFERENCED against the base case. Two colorbars:
one for the raw panel (horizontal, ticks+label above so they don't run into
the panel below), one shared across the 5 diff panels.

'u' and 'dudz' variables:
  'u' is the geographic EASTWARD velocity (not the raw grid-relative
  zsliced `u`, which lives on the curvilinear xi_u grid and points along
  the local grid direction, not true east). Each timestep, u and v are
  interpolated to the rho grid (interior-average + edge-copy) and rotated
  by the grid angle -- u_east = u_rho*cos(angle) - v_rho*sin(angle) -- same
  convention as calc_vort_rmse_wec_shelf.py / calc_vort_std.py in
  postprocessing/. This needs both u and v read every timestep, so 'u'
  costs ~2x a single-field entry like 'w' (same order as the vorticity
  scripts).

  'dudz' is d(u_east)/dz on the zslice's fixed depth grid. Box-averaging,
  time-averaging, and the vertical finite difference are all linear
  operators that commute exactly on this fixed grid, so
  mean(box(du/dz)) == box(d(mean(u_east))/dz) -- one gradient at the end
  instead of one per time step, same reasoning as
  plot_cs_diag_drhodz_diff_box_3x2.py's drho/dz. Like that script, 'dudz'
  recomputes its own box-averaged u_east with DUDZ_MIN_FRAC=0.5 (require at
  least half the box's offsets valid before including a point) instead of
  reusing the 'u' entry's cache computed at the default MIN_FRAC=0.0 --
  differentiating through a thin-support cell (as few as 1-2 of 21 offsets
  right before a transect point goes fully NaN) produces a spurious,
  unrepresentative shear spike, the same failure mode documented in that
  script. This means 'dudz' pays for its own full u/v pass rather than
  reusing 'u's -- ~2x 'u's cost, ~4x 'w's, for this one entry.

  Neither has a fixed-range single-variable sibling script to copy a raw
  colorbar range from, so both panels use a range derived from the base
  case's own data (symmetric about zero, 98th percentile of |value|) --
  same fallback drho/dz uses for its own raw panel.

Output: ./figs/cs_diag_avg_diff_box_3x2_<var>_<transect>.png (one per variable per transect)
Cache:  ./figs/cs_diag_avg_diff_box_cache_<var>_<scen>_<transect>.npz (shared with
        plot_cs_diag_avg_diff_box.py -- same underlying field, different plot only)
"""

import os
import sys
sys.path.append('/data/project3/minnaho/global/')
import glob
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.colors import LogNorm
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

DIFF_CMAP = cmocean.cm.balance

# Per-variable config: subdir (None = scenario root), file stem, the netCDF
# variable name(s) to sum (TOTC sums three), an additive offset, the diff
# colorbar label, and the RAW base-case panel's cmap/range/label -- taken
# verbatim from the matching single-variable plot_cs_diag_*_box.py script so
# panel 1 is directly comparable to those figures.
VAR_CONFIGS = {
    'w':    dict(subdir=None,  stem='z_mc60_his', vars=['w'],    offset=0.0,
                 label=r'$\Delta w$ (m s$^{-1}$)', diff_cmap=DIFF_CMAP,
                 raw_cmap=cmocean.cm.balance, raw_vmin=-1.5e-4, raw_vmax=1.5e-4,
                 raw_norm=None, raw_label=r'w (m s$^{-1}$)'),
    'rho':  dict(subdir=None,  stem='z_mc60_his', vars=['rho'],  offset=RHO_OFFSET,
                 label=r'$\Delta\,(\rho - 1000)$ (kg m$^{-3}$)', diff_cmap=DIFF_CMAP,
                 raw_cmap=cmocean.cm.dense, raw_vmin=24, raw_vmax=26.0,
                 raw_norm=None, raw_label=r'$\rho - 1000$ (kg m$^{-3}$)'),
    'NO3':  dict(subdir='bgc', stem='z_mc60_bgc', vars=['NO3'],  offset=0.0,
                 label=r'$\Delta$ NO$_3$ (mmol m$^{-3}$)', diff_cmap=cmocean.cm.diff,
                 raw_cmap=cmocean.cm.matter, raw_vmin=0, raw_vmax=25,
                 raw_norm=None, raw_label=r'NO$_3$ (mmol m$^{-3}$)'),
    'O2':   dict(subdir='bgc', stem='z_mc60_bgc', vars=['O2'],   offset=0.0,
                 label=r'$\Delta$ O$_2$ (mmol m$^{-3}$)', diff_cmap=cmocean.cm.tarn,
                 raw_cmap=cmocean.cm.haline, raw_vmin=100, raw_vmax=300,
                 raw_norm=None, raw_label=r'O$_2$ (mmol m$^{-3}$)'),
    'TOTC': dict(subdir='bgc', stem='z_mc60_bgc', vars=['DIATC', 'DIAZC', 'SPC'], offset=0.0,
                 label=r'$\Delta$ Total phyto C (mmol C m$^{-3}$)', diff_cmap=DIFF_CMAP,
                 raw_cmap=cmocean.cm.algae, raw_vmin=0, raw_vmax=40,
                 raw_norm=None, raw_label=r'Total phyto C (mmol C m$^{-3}$)'),
    'Akt':  dict(subdir='ak',  stem='z_mc60_his', vars=['Akt'],  offset=0.0,
                 label=r'$\Delta K_t$ (m$^2$ s$^{-1}$)', diff_cmap=DIFF_CMAP,
                 raw_cmap='viridis', raw_vmin=None, raw_vmax=None,
                 raw_norm=LogNorm(vmin=1e-5, vmax=1e-1),
                 raw_label=r'$K_t$ (m$^2$ s$^{-1}$)'),
    'Akv':  dict(subdir='ak',  stem='z_mc60_his', vars=['Akv'],  offset=0.0,
                 label=r'$\Delta K_v$ (m$^2$ s$^{-1}$)', diff_cmap=DIFF_CMAP,
                 raw_cmap='viridis', raw_vmin=None, raw_vmax=None,
                 raw_norm=LogNorm(vmin=1e-4, vmax=1e-1),
                 raw_label=r'$K_v$ (m$^2$ s$^{-1}$)'),
    # geographic eastward velocity (rotated from grid u/v) and its vertical
    # shear -- see module docstring for the rotation convention, the
    # commuting-derivative reasoning behind 'dudz', and why 'dudz' pays for
    # its own u/v pass at DUDZ_MIN_FRAC instead of reusing 'u's cache
    'u':    dict(kind='u_east', subdir=None, stem='z_mc60_his', vars=['u', 'v'], offset=0.0,
                 label=r'$\Delta u$ (m s$^{-1}$)', diff_cmap=DIFF_CMAP,
                 raw_cmap=cmocean.cm.balance, raw_vmin=None, raw_vmax=None,
                 raw_norm=None, raw_label=r'$u$ (m s$^{-1}$, eastward)'),
    'dudz': dict(kind='dudz', subdir=None, stem='z_mc60_his', vars=['u', 'v'], offset=0.0,
                 label=r'$\Delta\,(\partial u/\partial z)$ (s$^{-1}$)', diff_cmap=DIFF_CMAP,
                 raw_cmap=cmocean.cm.balance, raw_vmin=None, raw_vmax=None,
                 raw_norm=None, raw_label=r'$\partial u/\partial z$ (s$^{-1}$)'),
}

# require at least half the box's offsets valid before including a point in
# 'dudz's box-averaged u_east -- see module docstring
DUDZ_MIN_FRAC = 0.5

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

# grid angle for rotating u/v to geographic east/north -- 'u' and 'dudz' only
angle  = np.array(grdnc['angle'][:])
cosang = np.cos(angle)
sinang = np.sin(angle)

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
# Per-(scenario, subdir) zsliced file lists
# ---------------------------------------------------------------------------
def zslice_files(scen, subdir, stem, exclude_stamps=()):
    root = SCENARIOS_ZSLICE[scen]
    d = os.path.join(root, subdir) if subdir else root
    files = sorted(glob.glob(os.path.join(d, f'{stem}.*.nc')))
    return [f for f in files if not any(s in f for s in exclude_stamps)]

# tidesampwec's raw source has a trailing 1-timestep file
# (...20190429110056) whose zslice output has no time dimension at all --
# excluded in every subdir (root/bgc/ak) it appears in
TIDESAMPWEC_EXCLUDE = ('20190429110056',)

def scenario_files_for(subdir, stem):
    files = {}
    for scen in SCENARIOS_ZSLICE:
        exclude = TIDESAMPWEC_EXCLUDE if scen == 'tidesampwec' else ()
        files[scen] = zslice_files(scen, subdir, stem, exclude_stamps=exclude)
    return files

with Dataset(zslice_files(BASE_SCEN, None, 'z_mc60_his')[0], 'r') as _tmp:
    DEPTH_1D = np.array(_tmp.variables['depth'][:])   # (157,) metres, 0 to -1980, surface-first

# ---------------------------------------------------------------------------
# Time-averaged, box-averaged field per (variable, scenario, transect) -- cached
# ---------------------------------------------------------------------------
os.makedirs(SAVEPATH, exist_ok=True)

def cache_path(var_key, scen, tr_name):
    # same cache naming as plot_cs_diag_avg_diff_box.py -- the underlying
    # box-averaged, time-averaged field doesn't depend on plot layout, so
    # caches computed by that script (or this one) are shared/reused.
    return f'{SAVEPATH}cs_diag_avg_diff_box_cache_{var_key}_{scen}_{tr_name}.npz'


def compute_var_mean(var_key, cfg, scen, files):
    """Time-averaged, box-averaged field(depth, transect position) per
    transect. Box-averages each time step's field (boxavg_section_fixedz,
    on the zslice's fixed depth grid) before accumulating the time mean --
    equivalent to box-averaging the time mean itself, since both operations
    are linear and the depth grid never changes between time steps. Returns
    a list of (n_depth, n_pts) arrays, one per TRANSECTS entry."""
    n_depth = DEPTH_1D.size
    accum_sum   = [np.zeros((n_depth, tr['n_pts'])) for tr in TRANSECTS]
    accum_count = [np.zeros((n_depth, tr['n_pts'])) for tr in TRANSECTS]

    for fi, f in enumerate(files):
        print(f'  {var_key} | {scen}: file {fi + 1}/{len(files)}')
        with Dataset(f, 'r') as nc:
            # Read per timestep, not the whole (tdim, n_depth, eta_rho,
            # xi_rho) variable at once -- each zslice variable is 6.36 GB
            # (12, 157, 1202, 702 float32, uncompressed), and TOTC sums 3 of
            # them, so a whole-file read peaks at ~19 GB. A single timestep
            # is 530 MB regardless of how many vars[] are summed.
            n_t = nc.variables[cfg['vars'][0]].shape[0]
            for t in range(n_t):
                var_t = None
                for v in cfg['vars']:
                    arr = np.array(nc.variables[v][t])   # (n_depth, eta_rho, xi_rho)
                    arr[np.abs(arr) > 1e10] = np.nan      # below-bathymetry fill value
                    var_t = arr if var_t is None else var_t + arr

                field3d = (var_t + cfg['offset']) * mask_plot

                for ti, tr in enumerate(TRANSECTS):
                    field_box = cb.boxavg_section_fixedz(field3d, tr)
                    valid = ~np.isnan(field_box)
                    accum_sum[ti][valid]   += field_box[valid]
                    accum_count[ti][valid] += 1

    with np.errstate(invalid='ignore'):
        return [accum_sum[ti] / accum_count[ti] for ti in range(len(TRANSECTS))]


def uv_to_rho(u, v):
    """u (depth,eta_rho,xi_u), v (depth,eta_v,xi_rho) -> u_rho, v_rho
    (depth,eta_rho,xi_rho); same interior-average / edge-copy scheme as
    pyfuncs.rho_uv_angle -- see calc_vort_std.py in postprocessing/."""
    u_temp = 0.5 * (u[:, :, 1:] + u[:, :, :-1])
    u_rho = np.empty((u.shape[0], u.shape[1], u.shape[2] + 1), dtype=np.float32)
    u_rho[:, :, 1:-1] = u_temp
    u_rho[:, :, 0] = u_temp[:, :, 0]
    u_rho[:, :, -1] = u_temp[:, :, -1]

    v_temp = 0.5 * (v[:, 1:, :] + v[:, :-1, :])
    v_rho = np.empty((v.shape[0], v.shape[1] + 1, v.shape[2]), dtype=np.float32)
    v_rho[:, 1:-1, :] = v_temp
    v_rho[:, 0, :] = v_temp[:, 0, :]
    v_rho[:, -1, :] = v_temp[:, -1, :]
    return u_rho, v_rho


def compute_u_east_mean(var_key, cfg, scen, files, min_frac=None):
    """Time-averaged, box-averaged geographic eastward velocity
    (depth, transect position) per transect. Reads u AND v every time step
    (unlike compute_var_mean's single field), interpolates both to the rho
    grid, rotates by the grid angle, then box-averages u_east only -- see
    module docstring for the rotation convention. min_frac is forwarded to
    boxavg_section_fixedz ('dudz' passes DUDZ_MIN_FRAC; 'u' leaves it at the
    module default). Returns a list of (n_depth, n_pts) arrays, one per
    TRANSECTS entry."""
    n_depth = DEPTH_1D.size
    accum_sum   = [np.zeros((n_depth, tr['n_pts'])) for tr in TRANSECTS]
    accum_count = [np.zeros((n_depth, tr['n_pts'])) for tr in TRANSECTS]

    for fi, f in enumerate(files):
        print(f'  {var_key} | {scen}: file {fi + 1}/{len(files)}')
        with Dataset(f, 'r') as nc:
            # per timestep, not the whole file -- see compute_var_mean's
            # comment; u and v are each 6.36 GB per file, ~1.06 GB/timestep
            # combined
            n_t = nc.variables['u'].shape[0]
            for t in range(n_t):
                u = np.array(nc.variables['u'][t], dtype=np.float32)
                v = np.array(nc.variables['v'][t], dtype=np.float32)
                u[np.abs(u) > 1e10] = np.nan
                v[np.abs(v) > 1e10] = np.nan
                u_rho, v_rho = uv_to_rho(u, v)
                u_east = u_rho * cosang - v_rho * sinang

                field3d = (u_east + cfg['offset']) * mask_plot

                for ti, tr in enumerate(TRANSECTS):
                    field_box = cb.boxavg_section_fixedz(field3d, tr, min_frac=min_frac)
                    valid = ~np.isnan(field_box)
                    accum_sum[ti][valid]   += field_box[valid]
                    accum_count[ti][valid] += 1

    with np.errstate(invalid='ignore'):
        return [accum_sum[ti] / accum_count[ti] for ti in range(len(TRANSECTS))]


# ---------------------------------------------------------------------------
# Run: for each variable, compute box-averaged time-mean per scenario, then plot
# ---------------------------------------------------------------------------
for var_key, cfg in VAR_CONFIGS.items():
    print(f'\n=== {var_key} ===')
    scen_files = scenario_files_for(cfg['subdir'], cfg['stem'])
    for name, files in scen_files.items():
        print(f'  {name}: {len(files)} zslice files')

    var_mean = {}   # scen -> [ (n_depth, n_pts) per transect ]
    for scen, files in scen_files.items():
        cached = [cache_path(var_key, scen, tr['name']) for tr in TRANSECTS]
        if all(os.path.exists(c) for c in cached):
            print(f'  {scen}: loading cache...')
            var_mean[scen] = [np.load(c)['mean'] for c in cached]
            continue

        if not files:
            print(f'  WARNING: no zslice files for {var_key} | {scen}, skipping')
            continue

        kind = cfg.get('kind', 'sum')
        if kind == 'u_east':
            mean_list = compute_u_east_mean(var_key, cfg, scen, files)
        elif kind == 'dudz':
            u_mean_list = compute_u_east_mean(var_key, cfg, scen, files, min_frac=DUDZ_MIN_FRAC)
            mean_list = [np.gradient(m, DEPTH_1D, axis=0) for m in u_mean_list]
        else:
            mean_list = compute_var_mean(var_key, cfg, scen, files)
        for ti, arr in enumerate(mean_list):
            np.savez(cached[ti], mean=arr)
        var_mean[scen] = mean_list

    # -----------------------------------------------------------------------
    # Plot: panel 1 = base case raw, panels 2-6 = diff from base case, one
    # figure per transect, laid out as a 3x2 grid
    # -----------------------------------------------------------------------
    diff_scens = [s for s in SCENARIOS_ZSLICE if s != BASE_SCEN and s in var_mean]
    if BASE_SCEN not in var_mean or not diff_scens:
        print(f'  WARNING: missing base case or no scenarios to diff for {var_key}, skipping plot')
        continue

    for ti, tr in enumerate(TRANSECTS):
        fname = f'{SAVEPATH}cs_diag_avg_diff_box_3x2_{var_key}_{tr["name"]}.png'

        n_panels = 1 + len(diff_scens)
        ncols = 2
        nrows = math.ceil(n_panels / ncols)
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                                 figsize=(14, 4 * nrows))
        axes_flat = np.atleast_1d(axes).flatten()
        for ax in axes_flat[n_panels:]:
            ax.axis('off')

        keep = DEPTH_1D >= tr['depth_lim']

        def _format_ax(ax, row, col):
            if tr.get('xlim') is not None:
                ax.set_xlim(tr['xlim'])
            if tr.get('xticks') is not None:
                ax.set_xticks(tr['xticks'])
            ax.set_ylim([tr['depth_lim'], 0])
            sf = ScalarFormatter(useOffset=False)
            sf.set_scientific(False)
            ax.xaxis.set_major_formatter(sf)
            if tr.get('xticks') is None:
                ax.xaxis.set_major_locator(MaxNLocator(5))
            if col == 0:
                ax.set_ylabel('Depth (m)')
            if row == nrows - 1:
                ax.set_xlabel('Longitude')
            ax.label_outer()

        # panel 1: base case, raw time-mean
        ax_raw = axes_flat[0]
        base_field = var_mean[BASE_SCEN][ti][keep, :]
        if cfg['raw_norm'] is not None:
            raw_kwargs = dict(norm=cfg['raw_norm'])
        elif cfg['raw_vmin'] is None and cfg['raw_vmax'] is None:
            # no fixed-range single-variable sibling script to copy from
            # ('u', 'dudz') -- derive a symmetric-about-zero range from the
            # base case's own data, same fallback drho/dz uses
            rv = np.nanpercentile(np.abs(base_field), 98)
            raw_kwargs = dict(vmin=-rv, vmax=rv)
        else:
            raw_kwargs = dict(vmin=cfg['raw_vmin'], vmax=cfg['raw_vmax'])
        pc_raw = ax_raw.pcolormesh(tr['lon'], DEPTH_1D[keep], base_field,
                                   cmap=cfg['raw_cmap'], shading='nearest', **raw_kwargs)
        ax_raw.set_title(LABELS[BASE_SCEN])
        _format_ax(ax_raw, 0, 0)

        # panels 2-6: diff from base case
        diffs = {s: var_mean[s][ti] - var_mean[BASE_SCEN][ti] for s in diff_scens}
        vmax = np.nanpercentile(
            np.abs(np.concatenate([d.ravel() for d in diffs.values()])), 98)

        pc_diff = None
        for i, scen in enumerate(diff_scens, start=1):
            ax = axes_flat[i]
            row, col = divmod(i, ncols)
            pc_diff = ax.pcolormesh(tr['lon'], DEPTH_1D[keep], diffs[scen][keep, :],
                                    cmap=cfg['diff_cmap'], vmin=-vmax, vmax=vmax, shading='nearest')
            ax.set_title(f'{LABELS[scen]}  −  {LABELS[BASE_SCEN]}')
            _format_ax(ax, row, col)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.3)
        fig.canvas.draw()

        # two colorbars: raw (below panel 1 only) + diff (right of the grid).
        # The raw colorbar sits centered in the actual gap between panel 1
        # and the panel below it (not a fixed offset), so it can't intrude
        # into that panel -- a fixed -0.06 offset overlapped it because
        # tight_layout's natural row gap is only ~0.024 fig-fraction.
        pos_raw = ax_raw.get_position()
        below = axes_flat[ncols] if n_panels > ncols else None
        if below is not None:
            gap_top, gap_bottom = pos_raw.y0, below.get_position().y1
            cb_h = min(0.02, (gap_top - gap_bottom) * 0.4)
            cb_y = gap_bottom + (gap_top - gap_bottom - cb_h) / 2
        else:
            cb_h, cb_y = 0.02, pos_raw.y0 - 0.06
        cax_raw = fig.add_axes([pos_raw.x0, cb_y, pos_raw.width, cb_h])
        cbar_raw = fig.colorbar(pc_raw, cax=cax_raw, orientation='horizontal', label=cfg['raw_label'])
        cbar_raw.ax.xaxis.set_label_position('top')
        cbar_raw.ax.xaxis.set_ticks_position('top')

        pos_tr  = axes_flat[1].get_position()
        pos_br  = axes_flat[n_panels - 1].get_position()
        cax_diff = fig.add_axes([pos_tr.x1 + 0.015, pos_br.y0, 0.015, pos_tr.y1 - pos_br.y0])
        fig.colorbar(pc_diff, cax=cax_diff, label=cfg['label'])

        plt.savefig(fname, dpi=800, bbox_inches='tight')
        plt.close()
        print(f'  saved -> {fname}')
