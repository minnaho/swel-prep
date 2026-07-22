"""
Cross-section differences of time-averaged drho/dz, using zsliced his output.

For each scenario, time-averages 'rho' from the zsliced z_mc60_his.*.nc files
onto the same two coast-to-offshore diagonal transects used by plot_cs_diag.py,
plus the fixed-eta "mid" geographical cross-section used by plot_cs_mid_o2.py /
plot_cs_w_NO3.py, then takes d(rho)/dz on the zslice output's fixed depth grid.
Each scenario's drho/dz is differenced against the base case (notidesnowec).

drho/dz here is a finite difference on the zslice output's FIXED depth grid --
the same z-levels at every time step (unlike raw s_rho, which moves with zeta
each time step). Differentiation and time-averaging are both linear operators;
on a fixed grid they commute exactly, so mean(drho/dz) == d(mean(rho))/dz.
(That equivalence would NOT hold on raw s_rho data, since the vertical sample
points themselves shift between time steps there.) We average rho first --
one gradient at the end instead of one per time step -- for an identical,
cheaper result.

The diagonal transects (ts, tn) are extracted with map_coordinates
interpolation along a diagonal grid path. The mid transect (mid) is a direct
slice at a fixed eta index -- already regular in longitude, so no
interpolation is needed there.

Output: ./figs/cs_diag_drhodz_diff_<ts|tn|mid>.png (one per transect)
Cache:  ./figs/cs_diag_drhodz_cache_<scen>_<ts|tn|mid>.npz
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

DIFF_CMAP  = cmocean.cm.balance
DIFF_LABEL = r'$\Delta\,(\partial\rho/\partial z)$ (kg m$^{-4}$)'

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
# Time-averaged rho -> drho/dz per scenario, per transect (cached)
# ---------------------------------------------------------------------------
os.makedirs(SAVEPATH, exist_ok=True)

def drhodz_cache_path(scen, tr_name):
    return f'{SAVEPATH}cs_diag_drhodz_cache_{scen}_{tr_name}.npz'


def compute_rho_mean(scen, files):
    """Time-averaged rho(depth, transect position) per transect, from zsliced
    'rho'. Returns a list of (n_depth, n_pts) arrays, one per TRANSECTS entry."""
    n_depth = DEPTH_1D.size
    accum_sum   = [np.zeros((n_depth, tr['n_pts'])) for tr in TRANSECTS]
    accum_count = [np.zeros((n_depth, tr['n_pts'])) for tr in TRANSECTS]

    for fi, f in enumerate(files):
        print(f'  {scen}: file {fi + 1}/{len(files)}')
        with Dataset(f, 'r') as nc:
            rho_all = np.array(nc.variables['rho'][:])  # (tdim, n_depth, eta_rho, xi_rho)

        for t in range(rho_all.shape[0]):
            rho3d = rho_all[t] + RHO_OFFSET
            rho3d[np.abs(rho3d) > 1e10] = np.nan   # below-bathymetry fill value
            rho3d = rho3d * mask_plot

            for ti, tr in enumerate(TRANSECTS):
                rho_t = extract_section(rho3d, tr)
                valid = ~np.isnan(rho_t)
                accum_sum[ti][valid]   += rho_t[valid]
                accum_count[ti][valid] += 1

    with np.errstate(invalid='ignore'):
        return [accum_sum[ti] / accum_count[ti] for ti in range(len(TRANSECTS))]


print('\nComputing time-averaged drho/dz per scenario...')
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
# Plot: difference from the base case (notidesnowec), one figure per transect
# ---------------------------------------------------------------------------
diff_scens = [s for s in SCENARIOS_ZSLICE if s != BASE_SCEN and s in drhodz_mean]

for ti, tr in enumerate(TRANSECTS):
    fig, axes = plt.subplots(len(diff_scens), 1, sharex=True,
                             figsize=(10, 3 * len(diff_scens)))
    if len(diff_scens) == 1:
        axes = [axes]

    diffs = {s: drhodz_mean[s][ti] - drhodz_mean[BASE_SCEN][ti] for s in diff_scens}
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
    fig.colorbar(pc, cax=cax, label=DIFF_LABEL)
    fig.suptitle('Time-averaged $\\partial\\rho/\\partial z$ difference from '
                 f'{LABELS[BASE_SCEN]} — {tr["title"]}', y=0.92)

    fname = f'{SAVEPATH}cs_diag_drhodz_diff_{tr["name"]}.png'
    plt.savefig(fname, dpi=800, bbox_inches='tight')
    plt.close()
    print(f'  saved -> {fname}')
