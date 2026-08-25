"""
Single-instant cross section of the resolved eddy NO3 flux (w'NO3') and the
parameterized diffusive NO3 flux (Akt*dNO3/dz), at 2019-04-21 11:01 UTC, for
both the south (ts) and north (tn) transects, all 6 scenarios -- computed
directly from raw native s_rho his/bgc output (same transect geometry and
raw-file conventions as plot_hov_transect_raw.py).

w'NO3': w'=w-w̄, NO3'=NO3-N̄. w̄/N̄ are computed on a FIXED PHYSICAL DEPTH grid
from the zsliced product (zslicefull/<scenario>/z_mc60_his.*.nc for w,
.../bgc/z_mc60_bgc.*.nc for NO3 -- same 157-level depth grid used elsewhere
in this codebase), not at fixed sigma index -- fixed early on using sigma
levels, then corrected: a fixed sigma index maps to a different actual depth
at every timestep (sigma levels move with the tide/free surface), so a
"mean at fixed sigma index" isn't a mean at fixed depth, most notably near
the surface. **The zslice depth grid is NOT uniformly spaced** (confirmed
directly): 1m spacing 0 to -50m, 5m spacing -50 to -300m, 30m spacing -300
to -1980m -- interp_mean_to_zr() below interpolates against the actual
depth_z values via np.interp (not an assumed fixed spacing), so this is
handled correctly regardless. At the target timestep, the raw (sigma-level,
actual-depth-reconstructed) w/NO3 anomaly is taken against this fixed-depth
mean by vertically interpolating the mean (157 fixed levels) onto each
transect column's actual sigma-level depths (zr, from the raw ROMS_depths
reconstruction) via a per-column np.interp -- points deeper than the
zsliced grid's -1980m floor get NaN (no extrapolation) rather than a
misleading flat fill.

**That said, this script is still expensive, not fast** -- eliminating
ROMS_depths turned out not to be the dominant cost. Every timestep across the
full record still needs a full (157, 1202, 702) read of both zsliced `w` and
`NO3` (~525MB each -- slightly larger than the raw 100-level sigma grid this
used to read) to build the time mean, and on this filesystem that alone ran
~30-40s/timestep in testing -- i.e. multiple hours per scenario for the full
~252-timestep record, not the "a few minutes" the ROMS_depths savings alone
would suggest. Run this in `screen`, not interactively.

**Much faster alternative: precompute the means with NCO's `ncra`** instead
of letting compute_means() do the Python full-record scan -- ncra streams
and accumulates the record-dimension average natively in C in one pass,
measured ~5-7x faster than the equivalent Python-loop work (71s for ncra on
24 timesteps of `w` alone vs. an estimated 6-8 min in Python). Run all 6
scenarios x 2 variables (12 jobs) in parallel:

  mkdir -p figs/cs_wno3_aktdno3dz_snap/ncra_means
  declare -A ZDIR=( [tideswec]=tideswec [tidesnowec]=tidesnowec
                    [notidesnowec]=notidesnowec [notideswec]=notideswec
                    [ampwec]=notidesampwec [tidesampwec]=tidesampwec )
  for scen in tideswec tidesnowec notidesnowec notideswec ampwec tidesampwec; do
    zdir=${ZDIR[$scen]}
    his=$(ls /data/project1/minnaho/swel/zslicefull/$zdir/z_mc60_his.*.nc | grep -v 20190429110056)
    bgc=$(ls /data/project1/minnaho/swel/zslicefull/$zdir/bgc/z_mc60_bgc.*.nc | grep -v dia_avg | grep -v 20190429110056)
    ncra -O -v w,depth   $his figs/cs_wno3_aktdno3dz_snap/ncra_means/w_mean_${scen}.nc &
    ncra -O -v NO3,depth $bgc figs/cs_wno3_aktdno3dz_snap/ncra_means/no3_mean_${scen}.nc &
  done
  wait

load_ncra_mean() checks for these files first (before falling back to
compute_means()'s Python scan) and, once found, promotes them into the
regular npz mean cache too -- so a later run without the ncra files present
still hits the fast cached path.

Akt*dNO3/dz: no mean-removal / Reynolds decomposition (per the conversation
that motivated this script -- Akt already parameterizes unresolved/subgrid
turbulent transport via a closure scheme, so unlike w and NO3 there's no
resolved advective-eddy analog to isolate by removing a time mean). Just the
raw instantaneous product at the target timestep. Akt lives on s_w (101
levels); averaged to s_rho (100 levels, matching NO3/w) before multiplying.
dNO3/dz via np.gradient on the s_rho grid against actual depth (zr), same
convention plot_hov_transect_raw.py already uses for N². Sign: standard
downgradient-diffusion convention, flux = -Akt*dNO3/dz (z positive up) --
positive = upward nutrient supply, matching w'NO3''s sign convention and
plot_wno3_flux_100m.py's AktdNO3dz entries (also flipped from the literal
Akt*dNO3/dz that calc_akt_dno3dz_20m_100m.py/_30m_100m.py store).

Two-tier cache: the expensive part -- each scenario's full-record w_mean/
no3_mean per transect -- is cached separately from the (cheap) per-time
snapshot, and is NOT keyed by TARGET_TIME, since the mean doesn't depend on
which instant you're plotting. So changing TARGET_TIME and rerunning skips
the full-record scan entirely and only pays for the cheap targeted read
(one file open, one ROMS_depths call) -- this is the main lever for making a
second/third cross section at a different time fast. The final per-time
snapshot cache IS keyed by scenario+TARGET_TIME (so switching times can't
silently reuse a stale snapshot computed for a different time).

Output (one figure per variable per transect, single-column 6-scenario layout
matching plot_cs_diag_no3.py's size/colorbar style):
  ./figs/cs_wno3_snap_<ts|tn>.png
  ./figs/cs_aktdno3dz_snap_<ts|tn>.png
"""

import os
import sys
sys.path.append('/data/project3/minnaho/global/')
import glob
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cmocean
from netCDF4 import Dataset, num2date
from scipy.ndimage import map_coordinates
import ROMS_depths as depths

plt.rcParams.update({'font.size': 12})

# ---------------------------------------------------------------------------
# Paths / scenarios (raw ROMS output — same roots as plot_hov_transect_raw.py)
# ---------------------------------------------------------------------------
GRD         = 'mc60_grd.nc'
SAVEPATH    = './figs/'
CACHE_DIR   = './figs/cs_wno3_aktdno3dz_snap/'
NCRA_DIR    = './figs/cs_wno3_aktdno3dz_snap/ncra_means/'
ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'

TARGET_TIME   = datetime.datetime(2019, 4, 21, 12, 1)
TOLERANCE_SEC = 1800   # 30 min -- guards against silently picking the wrong timestep

SCENARIOS = {
    'tideswec':     '/data/project3/minnaho/swel/tides/mc60/wec',
    'tidesnowec':   '/data/project3/minnaho/swel/tides/mc60/nowec/output',
    'notidesnowec': '/data/project3/minnaho/swel/notides/mc60/nowec',
    'notideswec':   '/data/project3/minnaho/swel/notides/mc60/wec/rerun',
    'ampwec':       '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything',
    'tidesampwec':  '/data/project3/minnaho/swel/tides/mc60/ampwec/everything',
}
# raw scenario name -> zslicefull subdirectory name (same naming gotcha as
# every other zsliced script in this codebase: ampwec's zslice dir is
# 'notidesampwec', everything else matches the raw scenario key directly)
SCEN_ZSLICE_DIRS = {'ampwec': 'notidesampwec'}
# tidesampwec's raw source has a trailing 1-timestep file
# (...20190429110056) whose zslice output has no time dimension at all
TIDESAMPWEC_EXCLUDE = ('20190429110056',)
LABELS = {
    'tideswec':     'tides, WEC',
    'tidesnowec':   'tides, no WEC',
    'notidesnowec': 'no tides, no WEC',
    'notideswec':   'no tides, WEC',
    'ampwec':       'no tides, 2.5× WEC',
    'tidesampwec':  'tides, 2.5× WEC',
}

WNO3_LABEL = r"$w'NO_3^-'$"
AKT_LABEL  = r"$-A_{kt}\,\partial NO_3^-/\partial z$"
FLUX_UNITS = r'mmol N m$^{-2}$ s$^{-1}$'

# ---------------------------------------------------------------------------
# Transect geometry (same as plot_cs_diag*.py / plot_hov_transect_raw.py)
# ---------------------------------------------------------------------------
ETA0, XI0, SLOPE,  DEPTH_LIM0 = 271, 543, -0.8, -125
ETA1, XI1, SLOPE1, DEPTH_LIM1 = 832, 646,  0.6,  -90
LENGTH_XI = 250
N_PTS     = 300

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
grdnc     = Dataset(GRD, 'r')
lon       = np.array(grdnc['lon_rho'][:]) - 360
mask_rho  = np.array(grdnc['mask_rho'][:])
mask_plot = mask_rho.astype(float)
mask_plot[mask_plot == 0] = np.nan


def build_transect(eta0, xi0, slope, depth_lim):
    xi  = np.linspace(xi0,  xi0  - LENGTH_XI, N_PTS)
    eta = np.linspace(eta0, eta0 + slope * (-LENGTH_XI), N_PTS)
    crd = np.array([eta, xi])
    return dict(
        coords    = crd,
        lon       = map_coordinates(lon, crd, order=1, mode='nearest'),
        mask      = map_coordinates(mask_rho.astype(float), crd,
                                    order=0, mode='nearest') > 0.5,
        depth_lim = depth_lim,
    )


TRANSECTS = [
    build_transect(ETA0, XI0, SLOPE,  DEPTH_LIM0),
    build_transect(ETA1, XI1, SLOPE1, DEPTH_LIM1),
]
TRANSECT_NAMES = ['ts', 'tn']   # ts = south transect, tn = north transect

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def src_glob(root, kind):
    """Locate his/ or bgc/ files, handling flat vs subdir scenario layouts
    (ampwec/tidesampwec are flat; the other four use his/ + bgc/ subdirs)."""
    sub  = os.path.join(root, kind)
    base = sub if os.path.isdir(sub) else root
    stem = 'mc60_his' if kind == 'his' else 'mc60_bgc'
    return os.path.join(base, f'{stem}.*.nc')


def zslice_glob(scen, kind):
    """Locate zsliced z_mc60_his (kind='his') or z_mc60_bgc (kind='bgc')
    files for a scenario, on the fixed physical-depth grid -- used for the
    w'NO3' time-mean (see module docstring). Excludes tidesampwec's known
    trailing partial file."""
    zdir = SCEN_ZSLICE_DIRS.get(scen, scen)
    if kind == 'his':
        files = sorted(glob.glob(os.path.join(ZSLICE_ROOT, zdir, 'z_mc60_his.*.nc')))
    else:
        files = sorted(glob.glob(os.path.join(ZSLICE_ROOT, zdir, 'bgc', 'z_mc60_bgc.*.nc')))
        files = [f for f in files if 'dia_avg' not in f]
    if zdir == 'tidesampwec':
        files = [f for f in files if not any(x in f for x in TIDESAMPWEC_EXCLUDE)]
    return files


def clean(arr):
    arr = np.array(arr)
    return np.where(np.abs(arr) > 1e30, np.nan, arr)


def interp_section(field3d, coords, mask_t):
    """(n_z, eta, xi) → (n_z, N_PTS) in one map_coordinates call."""
    n_z    = field3d.shape[0]
    nan3d  = np.isnan(field3d)
    filled = np.where(nan3d, 0.0, field3d)
    z_idx  = np.repeat(np.arange(n_z), N_PTS)
    eta_c  = np.tile(coords[0], n_z)
    xi_c   = np.tile(coords[1], n_z)
    full_c = np.array([z_idx, eta_c, xi_c])
    vals   = map_coordinates(filled, full_c, order=1, mode='nearest').reshape(n_z, N_PTS)
    nans   = map_coordinates(nan3d.astype(float), full_c,
                             order=1, mode='nearest').reshape(n_z, N_PTS) > 0.5
    vals[nans | ~mask_t[None, :]] = np.nan
    return vals


# ---------------------------------------------------------------------------
# Per-scenario computation
# ---------------------------------------------------------------------------
def find_target_timestep(his_files):
    """Cheap metadata-only scan (ocean_time only, no 3D reads) for the
    (file_idx, local_idx) closest to TARGET_TIME."""
    best = None
    for hf, f in enumerate(his_files):
        with Dataset(f) as hnc:
            ocean_time = np.array(hnc.variables['ocean_time'][:])
        dts = num2date(ocean_time, 'seconds since 1995-01-01',
                       only_use_cftime_datetimes=False)
        for li, dt0 in enumerate(dts):
            diff = abs((dt0 - TARGET_TIME).total_seconds())
            if best is None or diff < best[0]:
                best = (diff, hf, li, dt0)
    return best


def compute_means(scen):
    """Full-record pass over the ZSLICED product (fixed physical-depth grid,
    not sigma level -- see module docstring for why) -- the expensive part,
    but time-independent (doesn't depend on TARGET_TIME), so it's cached
    separately and reused by any future run of this script regardless of
    which instant is requested. Returns (means, depth_z): means[transect] =
    (w_mean, no3_mean), each (157, N_PTS) on the fixed depth_z grid."""
    his_files = zslice_glob(scen, 'his')
    bgc_files = zslice_glob(scen, 'bgc')
    n_pairs   = min(len(his_files), len(bgc_files))
    print(f'  [{scen}] {len(his_files)} zsliced his, {len(bgc_files)} zsliced bgc, '
          f'pairing {n_pairs}', flush=True)

    with Dataset(his_files[0]) as hnc0:
        depth_z = np.array(hnc0.variables['depth'][:])

    n_tr = len(TRANSECTS)
    w_full   = [[] for _ in range(n_tr)]
    no3_full = [[] for _ in range(n_tr)]
    for hf in range(n_pairs):
        with Dataset(his_files[hf]) as hnc, Dataset(bgc_files[hf]) as bnc:
            n_t = hnc.variables['w'].shape[0]
            for t_i in range(n_t):
                w3d    = clean(hnc.variables['w'][t_i])   * mask_plot
                no3_3d = clean(bnc.variables['NO3'][t_i]) * mask_plot
                for ti, tr in enumerate(TRANSECTS):
                    w_full[ti].append(interp_section(w3d, tr['coords'], tr['mask']))
                    no3_full[ti].append(interp_section(no3_3d, tr['coords'], tr['mask']))
        print(f'  [{scen}] mean scan: {hf + 1}/{n_pairs} file pairs done', flush=True)

    means = {}
    for ti in range(n_tr):
        w_arr   = np.array(w_full[ti])    # (n_t, 157, N_PTS)
        no3_arr = np.array(no3_full[ti])
        means[TRANSECT_NAMES[ti]] = (np.nanmean(w_arr, axis=0), np.nanmean(no3_arr, axis=0))
    return means, depth_z


def load_ncra_mean(scen):
    """Load a user-precomputed ncra full-domain time-mean (see module
    docstring for the shell command), if present. Faster than compute_means()
    for the same reason ncra beats a Python per-timestep loop generally:
    NCO streams/accumulates the record-dimension average natively in C, in
    one pass, vs. a separate netCDF4-python read + map_coordinates call per
    timestep -- measured ~5-7x faster in testing (71s for ncra on 24
    timesteps of `w` alone vs. an estimated 6-8 min for the equivalent
    Python-loop work). Returns (means, depth_z) in the same shape
    compute_means() returns, or None if the expected files aren't there."""
    w_path   = f'{NCRA_DIR}w_mean_{scen}.nc'
    no3_path = f'{NCRA_DIR}no3_mean_{scen}.nc'
    if not (os.path.exists(w_path) and os.path.exists(no3_path)):
        return None

    with Dataset(w_path) as nc:
        depth_z     = np.array(nc.variables['depth'][:])
        w_mean_full = clean(np.squeeze(np.array(nc.variables['w']))) * mask_plot     # (157, eta, xi)
    with Dataset(no3_path) as nc:
        no3_mean_full = clean(np.squeeze(np.array(nc.variables['NO3']))) * mask_plot

    means = {}
    for ti, tr in enumerate(TRANSECTS):
        means[TRANSECT_NAMES[ti]] = (
            interp_section(w_mean_full,   tr['coords'], tr['mask']),
            interp_section(no3_mean_full, tr['coords'], tr['mask']),
        )
    return means, depth_z


def interp_mean_to_zr(mean_z, depth_z, zr_t):
    """Vertically interpolate a (157, N_PTS) fixed-depth-grid mean field
    onto (100, N_PTS) actual sigma-level depths, one transect column at a
    time (each column has different sigma-level depths, since bathymetry/
    free-surface vary along the transect). Uses np.interp against the
    actual depth_z values, so this is correct regardless of the zslice
    grid's non-uniform spacing (see module docstring). No extrapolation:
    points deeper than the zsliced grid's floor (-1980m) become NaN rather
    than a misleading flat fill."""
    order     = np.argsort(depth_z)   # depth_z is descending, NOT uniformly spaced
    depth_asc = depth_z[order]
    out = np.full(zr_t.shape, np.nan)
    for j in range(zr_t.shape[1]):
        col   = mean_z[order, j]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        out[:, j] = np.interp(zr_t[:, j], depth_asc[valid], col[valid],
                               left=np.nan, right=np.nan)
    return out


def compute_target_snapshot(his_files, bgc_files, target_hf, target_li):
    """Cheap targeted read at just the matched timestep: w, NO3, Akt, and
    the one z-reconstruction (ROMS_depths.get_zw_zeta) this script needs."""
    with Dataset(his_files[target_hf]) as hnc, Dataset(bgc_files[target_hf]) as bnc:
        zeta   = np.squeeze(hnc.variables['zeta'][target_li, :, :])
        z_w    = depths.get_zw_zeta(hnc, grdnc, zeta)
        zr3d   = 0.5 * (z_w[1:] + z_w[:-1]) * mask_plot
        w3d    = clean(hnc.variables['w'][target_li])   * mask_plot
        akt3d  = clean(hnc.variables['Akt'][target_li]) * mask_plot   # (101, eta, xi)
        no3_3d = clean(bnc.variables['NO3'][target_li]) * mask_plot

    snap = {}
    for ti, tr in enumerate(TRANSECTS):
        snap[TRANSECT_NAMES[ti]] = dict(
            lon   = tr['lon'],
            zr    = interp_section(zr3d,   tr['coords'], tr['mask']),   # (100, N_PTS)
            w     = interp_section(w3d,    tr['coords'], tr['mask']),
            no3   = interp_section(no3_3d, tr['coords'], tr['mask']),
            akt_w = interp_section(akt3d,  tr['coords'], tr['mask']),   # (101, N_PTS)
        )
    return snap


def compute_scenario(scen):
    root      = SCENARIOS[scen]
    his_files = sorted(glob.glob(src_glob(root, 'his')))
    bgc_files = sorted(glob.glob(src_glob(root, 'bgc')))
    n_pairs   = min(len(his_files), len(bgc_files))
    print(f'  [{scen}] {len(his_files)} his, {len(bgc_files)} bgc, pairing {n_pairs}',
          flush=True)

    diff, target_hf, target_li, matched_dt = find_target_timestep(his_files[:n_pairs])
    assert diff <= TOLERANCE_SEC, (
        f'[{scen}] no timestep within {TOLERANCE_SEC}s of {TARGET_TIME} '
        f'(closest: {matched_dt}, off by {diff:.0f}s)')
    print(f'  [{scen}] matched {matched_dt} (target {TARGET_TIME}, off by {diff:.0f}s) '
          f'in {os.path.basename(his_files[target_hf])}[{target_li}]', flush=True)

    cached_means = load_mean_cache(scen)
    if cached_means is not None:
        print(f'  [{scen}] loaded cached zsliced w/NO3 means -- skipping full-record scan',
              flush=True)
        means, depth_z = cached_means
    else:
        ncra_means = load_ncra_mean(scen)
        if ncra_means is not None:
            print(f'  [{scen}] loaded ncra-precomputed means -- skipping full-record scan',
                  flush=True)
            means, depth_z = ncra_means
        else:
            print(f'  [{scen}] no mean cache / ncra output -- scanning full zsliced record '
                  f'(this is the slow part; see module docstring for a much faster ncra '
                  f'alternative)...', flush=True)
            means, depth_z = compute_means(scen)
        save_mean_cache(scen, means, depth_z)

    snap = compute_target_snapshot(his_files, bgc_files, target_hf, target_li)

    result = {}
    for name in TRANSECT_NAMES:
        w_mean_z, no3_mean_z = means[name]
        w_t, no3_t, zr_t = snap[name]['w'], snap[name]['no3'], snap[name]['zr']
        akt_rho_t = 0.5 * (snap[name]['akt_w'][:-1] + snap[name]['akt_w'][1:])   # (100, N_PTS)

        # interpolate the fixed-depth mean onto this column's actual
        # sigma-level depths before subtracting -- see module docstring
        w_mean_at_zr   = interp_mean_to_zr(w_mean_z,   depth_z, zr_t)
        no3_mean_at_zr = interp_mean_to_zr(no3_mean_z, depth_z, zr_t)

        wno3_flux  = (w_t - w_mean_at_zr) * (no3_t - no3_mean_at_zr)
        dno3dz     = np.gradient(no3_t, axis=0) / np.gradient(zr_t, axis=0)
        # standard downgradient-diffusion convention (F = -Akt*dNO3/dz, z
        # positive up: positive = upward nutrient supply), matching the
        # sign flip applied in plot_wno3_flux_100m.py's AktdNO3dz entries
        akt_dno3dz = -akt_rho_t * dno3dz

        result[name] = dict(lon=snap[name]['lon'], zr=zr_t,
                             wno3=wno3_flux, aktdno3dz=akt_dno3dz)
    return result


# ---------------------------------------------------------------------------
# Cache -- two tiers, see module docstring
# ---------------------------------------------------------------------------
os.makedirs(CACHE_DIR, exist_ok=True)

def mean_cache_path(scen):
    # NOT keyed by TARGET_TIME -- the full-record mean doesn't depend on it
    return f'{CACHE_DIR}cs_wno3_aktdno3dz_mean_{scen}.npz'

def load_mean_cache(scen):
    cp = mean_cache_path(scen)
    if not os.path.exists(cp):
        return None
    d = np.load(cp, allow_pickle=False)
    means = {name: (d[f'{name}_w_mean'], d[f'{name}_no3_mean']) for name in TRANSECT_NAMES}
    return means, d['depth_z']

def save_mean_cache(scen, means, depth_z):
    out = {'depth_z': depth_z}
    for name in TRANSECT_NAMES:
        w_mean, no3_mean = means[name]
        out[f'{name}_w_mean']   = w_mean
        out[f'{name}_no3_mean'] = no3_mean
    np.savez(mean_cache_path(scen), **out)
    print(f'  saved mean cache -> {mean_cache_path(scen)}', flush=True)


def cache_path(scen):
    # keyed by TARGET_TIME too -- otherwise switching TARGET_TIME and
    # rerunning would silently reuse a stale snapshot from a different time
    tstamp = TARGET_TIME.strftime('%Y%m%d%H%M')
    return f'{CACHE_DIR}cs_wno3_aktdno3dz_snap_{scen}_{tstamp}.npz'

def load_cache(scen):
    cp = cache_path(scen)
    if not os.path.exists(cp):
        return None
    d = np.load(cp, allow_pickle=False)
    result = {}
    for name in TRANSECT_NAMES:
        result[name] = dict(lon=d[f'{name}_lon'], zr=d[f'{name}_zr'],
                             wno3=d[f'{name}_wno3'], aktdno3dz=d[f'{name}_aktdno3dz'])
    return result

def save_cache(scen, result):
    out = {}
    for name in TRANSECT_NAMES:
        out[f'{name}_lon']       = result[name]['lon']
        out[f'{name}_zr']        = result[name]['zr']
        out[f'{name}_wno3']      = result[name]['wno3']
        out[f'{name}_aktdno3dz'] = result[name]['aktdno3dz']
    np.savez(cache_path(scen), **out)
    print(f'  saved cache -> {cache_path(scen)}', flush=True)


all_results = {}
for scen in SCENARIOS:
    cached = load_cache(scen)
    if cached is not None:
        print(f'[{scen}] loading cache...', flush=True)
        all_results[scen] = cached
    else:
        print(f'[{scen}] computing...', flush=True)
        result = compute_scenario(scen)
        all_results[scen] = result
        save_cache(scen, result)

# ---------------------------------------------------------------------------
# Plot — one figure per variable per transect, 6 scenario rows x 1 column
# (same figsize/colorbar-placement convention as plot_cs_diag_no3.py)
# ---------------------------------------------------------------------------
os.makedirs(SAVEPATH, exist_ok=True)
scen_list = list(SCENARIOS)

VMAX_WNO3 = 0.0065
VMAX_AKT  = 0.00035


def plot_variable(var_key, label, vmax, fname_stem):
    for ti, tr_name in enumerate(TRANSECT_NAMES):
        depth_lim = TRANSECTS[ti]['depth_lim']

        fig, axes = plt.subplots(len(scen_list), 1, sharex=True,
                                  figsize=(10, 3 * len(scen_list)))
        pc = None
        for ax, scen in zip(axes, scen_list):
            d = all_results[scen][tr_name]
            pc = ax.pcolormesh(d['lon'], d['zr'], d[var_key], cmap=cmocean.cm.diff,
                                vmin=-vmax, vmax=vmax, shading='nearest')
            ax.set_ylim([depth_lim, 0])
            ax.set_ylabel('Depth (m)')
            ax.set_title(LABELS[scen])
            ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        axes[-1].set_xlabel('Longitude')
        fig.canvas.draw()
        pos_top = axes[0].get_position()
        pos_bot = axes[-1].get_position()
        cax = fig.add_axes([pos_top.x1 + 0.015, pos_bot.y0,
                             0.012, pos_top.y1 - pos_bot.y0])
        fig.colorbar(pc, cax=cax, label=f'{label} ({FLUX_UNITS})')
        fig.suptitle(f'{TARGET_TIME.isoformat()} UTC — {tr_name} transect', y=0.92)

        out = f'{SAVEPATH}{fname_stem}_{tr_name}.png'
        plt.savefig(out, dpi=800, bbox_inches='tight')
        plt.close(fig)
        print(f'saved -> {out}')


plot_variable('wno3',      WNO3_LABEL, VMAX_WNO3, 'cs_wno3_snap')
plot_variable('aktdno3dz', AKT_LABEL,  VMAX_AKT,  'cs_aktdno3dz_snap')
