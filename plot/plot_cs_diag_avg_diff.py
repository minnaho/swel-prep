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

Output: ./figs/cs_diag_avg_diff_<var>_<transect>.png (one per variable per transect)
Cache:  ./figs/cs_diag_avg_diff_cache_<var>_<scen>_<transect>.npz
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
# variable name(s) to sum (TOTC sums three), an additive offset, and the
# colorbar label for the difference field.
VAR_CONFIGS = {
    'w':    dict(subdir=None,  stem='z_mc60_his', vars=['w'],    offset=0.0,
                 label=r'$\Delta w$ (m s$^{-1}$)'),
    'rho':  dict(subdir=None,  stem='z_mc60_his', vars=['rho'],  offset=RHO_OFFSET,
                 label=r'$\Delta\,(\rho - 1000)$ (kg m$^{-3}$)'),
    'NO3':  dict(subdir='bgc', stem='z_mc60_bgc', vars=['NO3'],  offset=0.0,
                 label=r'$\Delta$ NO$_3$ (mmol m$^{-3}$)'),
    'O2':   dict(subdir='bgc', stem='z_mc60_bgc', vars=['O2'],   offset=0.0,
                 label=r'$\Delta$ O$_2$ (mmol m$^{-3}$)'),
    'TOTC': dict(subdir='bgc', stem='z_mc60_bgc', vars=['DIATC', 'DIAZC', 'SPC'], offset=0.0,
                 label=r'$\Delta$ Total phyto C (mmol C m$^{-3}$)'),
    'Akt':  dict(subdir='ak',  stem='z_mc60_his', vars=['Akt'],  offset=0.0,
                 label=r'$\Delta A_{Kt}$ (m$^2$ s$^{-1}$)'),
    'Akv':  dict(subdir='ak',  stem='z_mc60_his', vars=['Akv'],  offset=0.0,
                 label=r'$\Delta A_{Kv}$ (m$^2$ s$^{-1}$)'),
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
    # Plot: difference from the base case (notidesnowec), one figure per transect
    # -----------------------------------------------------------------------
    diff_scens = [s for s in SCENARIOS_ZSLICE if s != BASE_SCEN and s in var_mean]
    if BASE_SCEN not in var_mean or not diff_scens:
        print(f'  WARNING: missing base case or no scenarios to diff for {var_key}, skipping plot')
        continue

    for ti, tr in enumerate(TRANSECTS):
        fig, axes = plt.subplots(len(diff_scens), 1, sharex=True,
                                 figsize=(10, 3 * len(diff_scens)))
        if len(diff_scens) == 1:
            axes = [axes]

        diffs = {s: var_mean[s][ti] - var_mean[BASE_SCEN][ti] for s in diff_scens}
        vmax = np.nanpercentile(
            np.abs(np.concatenate([d.ravel() for d in diffs.values()])), 98)

        keep = DEPTH_1D >= tr['depth_lim']
        for ax, scen in zip(axes, diff_scens):
            pc = ax.pcolormesh(tr['lon'], DEPTH_1D[keep], diffs[scen][keep, :],
                               cmap=DIFF_CMAP, vmin=-vmax, vmax=vmax, shading='nearest')
            ax.set_ylim([tr['depth_lim'], 0])
            if tr.get('xlim') is not None:
                ax.set_xlim(tr['xlim'])
            if tr.get('xticks') is not None:
                ax.set_xticks(tr['xticks'])
            ax.set_ylabel('Depth (m)')
            ax.set_title(f'{LABELS[scen]}  −  {LABELS[BASE_SCEN]}')

            sf = ScalarFormatter(useOffset=False)
            sf.set_scientific(False)
            ax.xaxis.set_major_formatter(sf)
            if tr.get('xticks') is None:
                ax.xaxis.set_major_locator(MaxNLocator(5))

        axes[-1].set_xlabel('Longitude')
        fig.canvas.draw()
        pos_top = axes[0].get_position()
        pos_bot = axes[-1].get_position()
        cax = fig.add_axes([pos_top.x1 + 0.015, pos_bot.y0,
                            0.012, pos_top.y1 - pos_bot.y0])
        fig.colorbar(pc, cax=cax, label=cfg['label'])
        fig.suptitle(f'Time-averaged {var_key} difference from '
                     f'{LABELS[BASE_SCEN]} — {tr["title"]}', y=0.92)

        fname = f'{SAVEPATH}cs_diag_avg_diff_{var_key}_{tr["name"]}.png'
        plt.savefig(fname, dpi=800, bbox_inches='tight')
        plt.close()
        print(f'  saved -> {fname}')
