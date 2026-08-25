"""
Cross-section differences of time-averaged fields, using zsliced output.
Generalizes plot_cs_diag_drhodz_diff.py's average-then-difference approach
to every variable plotted by the plot_cs_diag_*.py family (w, rho, NO3, O2,
TOTC, Akt, Akv) instead of just d(rho)/dz.

For each variable and scenario, time-averages the field from the zsliced
z_mc60_his.*.nc / z_mc60_bgc.*.nc files onto three cross-sections -- the same
two coast-to-offshore diagonal transects used by plot_cs_diag.py, plus the
fixed-eta "mid" geographical cross-section used by plot_cs_mid_o2.py /
plot_cs_w_NO3.py -- on the zslice output's FIXED depth grid (same z-levels at
every time step -- unlike raw s_rho, which moves with zeta each time step).
Each scenario's time-mean is then differenced against the base case
(notidesnowec).

Unlike the drhodz script, no derivative is taken here, so there is no
average/differentiate ordering question -- straight time-averaging is used.

Variable sources (all on the same 157-level zslice depth grid):
  w, rho            -- zslicefull/<scen>/z_mc60_his.*.nc      (root)
  NO3, O2, TOTC      -- zslicefull/<scen>/bgc/z_mc60_bgc.*.nc  (TOTC = DIATC+DIAZC+SPC)
  Akt, Akv           -- zslicefull/<scen>/ak/z_mc60_his.*.nc

The diagonal transects (t0, t1) are extracted with map_coordinates
interpolation along a diagonal grid path. The mid transect (mid) is a direct
slice at a fixed eta index -- already regular in longitude, so no
interpolation is needed there.

Layout: 3x2 grid of panels -- panel 1 (top-left) is the base case
(notidesnowec) RAW time-mean, on the same cmap/range as the matching
plot_cs_diag_*.py single-variable script; the remaining 5 panels are each
other scenario's time-mean DIFFERENCED against the base case, as before.
Two colorbars: one for the raw panel, one shared across the 5 diff panels.

Output: ./figs/cs_diag_avg_diff_<var>_<transect>.png (one per variable per transect)
Cache:  ./figs/cs_diag_avg_diff_cache_<var>_<scen>_<transect>.npz
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
from scipy.ndimage import map_coordinates

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
# verbatim from the matching single-variable plot_cs_diag_*.py script so
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
                 raw_norm=LogNorm(vmin=1e-5, vmax=1e-1),
                 raw_label=r'$K_v$ (m$^2$ s$^{-1}$)'),
}

# Diagonal transect geometry (grid index space) -- same as plot_cs_diag.py / plot_cs_diag_rho.py
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

GRD      = 'mc60_grd.nc'
SAVEPATH = './figs/'

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
# Build transect coordinates
# ---------------------------------------------------------------------------
def build_transect(name, title, eta0, xi0, slope, depth_lim):
    """Diagonal transect (goes in -xi direction), extracted via map_coordinates."""
    xi  = np.linspace(xi0,  xi0  - LENGTH_XI, N_PTS)
    eta = np.linspace(eta0, eta0 + slope * (-LENGTH_XI), N_PTS)
    crd = np.array([eta, xi])
    return dict(
        mode      = 'diag',
        name      = name,
        title     = title,
        coords    = crd,
        lon       = map_coordinates(lon, crd, order=1, mode='nearest'),
        lat       = map_coordinates(lat, crd, order=1, mode='nearest'),
        mask      = map_coordinates(mask_rho.astype(float), crd,
                                    order=0, mode='nearest') > 0.5,
        depth_lim = depth_lim,
        n_pts     = N_PTS,
        eta0=eta0, xi0=xi0, slope=slope,
    )


def build_mid_transect(name, title, eta_slice, depth_lim, xlim=None, xticks=None):
    """Geographical cross-section at a fixed eta index -- a direct row slice,
    already regular in longitude, so no interpolation is needed."""
    lon_row = lon[eta_slice, :]
    return dict(
        mode      = 'mid',
        name      = name,
        title     = title,
        eta_slice = eta_slice,
        lon       = lon_row,
        depth_lim = depth_lim,
        n_pts     = lon_row.shape[0],
        xlim      = xlim,
        xticks    = xticks,
    )

TRANSECTS = [
    build_transect('ts', 'south transect', ETA0, XI0, SLOPE,  DEPTH_LIM0),
    build_transect('tn', 'north transect', ETA1, XI1, SLOPE1, DEPTH_LIM1),
    build_mid_transect('mid', f'mid transect (eta={ETA_MID})', ETA_MID, DEPTH_LIM_MID,
                       xlim=MID_XLIM, xticks=MID_XTICKS),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def interp_transect(field2d, coords, mask_t):
    nan_mask  = np.isnan(field2d)
    filled    = np.where(nan_mask, 0.0, field2d)
    row       = map_coordinates(filled, coords, order=1, mode='nearest')
    nan_along = map_coordinates(nan_mask.astype(float), coords,
                                order=1, mode='nearest') > 0.5
    row[nan_along | ~mask_t] = np.nan
    return row


def interp_section(field3d, coords, mask_t, n_pts):
    """Interpolate a 3D (depth, eta, xi) field along a diagonal transect.
    Returns (depth, n_pts)."""
    n_z = field3d.shape[0]
    out = np.full((n_z, n_pts), np.nan)
    for iz in range(n_z):
        out[iz] = interp_transect(field3d[iz], coords, mask_t)
    return out


def extract_section(field3d, tr):
    """Extract a (depth, n_pts) cross-section for either transect mode.
    field3d is already land-masked (NaN on land), so the mid transect's
    direct row slice needs no additional masking."""
    if tr['mode'] == 'diag':
        return interp_section(field3d, tr['coords'], tr['mask'], tr['n_pts'])
    else:   # 'mid'
        return field3d[:, tr['eta_slice'], :]

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
# Time-averaged field per (variable, scenario, transect) -- cached
# ---------------------------------------------------------------------------
os.makedirs(SAVEPATH, exist_ok=True)

def cache_path(var_key, scen, tr_name):
    return f'{SAVEPATH}cs_diag_avg_diff_cache_{var_key}_{scen}_{tr_name}.npz'


def compute_var_mean(var_key, cfg, scen, files):
    """Time-averaged field(depth, transect position) per transect. Returns a
    list of (n_depth, n_pts) arrays, one per TRANSECTS entry."""
    n_depth = DEPTH_1D.size
    accum_sum   = [np.zeros((n_depth, tr['n_pts'])) for tr in TRANSECTS]
    accum_count = [np.zeros((n_depth, tr['n_pts'])) for tr in TRANSECTS]

    for fi, f in enumerate(files):
        print(f'  {var_key} | {scen}: file {fi + 1}/{len(files)}')
        with Dataset(f, 'r') as nc:
            var_all = None
            for v in cfg['vars']:
                arr = np.array(nc.variables[v][:])   # (tdim, n_depth, eta_rho, xi_rho)
                arr[np.abs(arr) > 1e10] = np.nan      # below-bathymetry fill value
                var_all = arr if var_all is None else var_all + arr

        for t in range(var_all.shape[0]):
            field3d = (var_all[t] + cfg['offset']) * mask_plot

            for ti, tr in enumerate(TRANSECTS):
                field_t = extract_section(field3d, tr)
                valid = ~np.isnan(field_t)
                accum_sum[ti][valid]   += field_t[valid]
                accum_count[ti][valid] += 1

    with np.errstate(invalid='ignore'):
        return [accum_sum[ti] / accum_count[ti] for ti in range(len(TRANSECTS))]


# ---------------------------------------------------------------------------
# Run: for each variable, compute time-mean per scenario, then diff-plot
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

        mean_list = compute_var_mean(var_key, cfg, scen, files)
        for ti, arr in enumerate(mean_list):
            np.savez(cached[ti], mean=arr)
        var_mean[scen] = mean_list

    # -----------------------------------------------------------------------
    # Plot: panel 1 = base case raw, panels 2-6 = diff from base case, one
    # figure per transect, laid out as a 3x2 grid (or nrows x 2 for however
    # many scenarios are actually present)
    # -----------------------------------------------------------------------
    diff_scens = [s for s in SCENARIOS_ZSLICE if s != BASE_SCEN and s in var_mean]
    if BASE_SCEN not in var_mean or not diff_scens:
        print(f'  WARNING: missing base case or no scenarios to diff for {var_key}, skipping plot')
        continue

    for ti, tr in enumerate(TRANSECTS):
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
        raw_kwargs = (dict(norm=cfg['raw_norm']) if cfg['raw_norm'] is not None
                     else dict(vmin=cfg['raw_vmin'], vmax=cfg['raw_vmax']))
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

        fname = f'{SAVEPATH}cs_diag_avg_diff_{var_key}_{tr["name"]}.png'
        plt.savefig(fname, dpi=800, bbox_inches='tight')
        plt.close()
        print(f'  saved -> {fname}')
