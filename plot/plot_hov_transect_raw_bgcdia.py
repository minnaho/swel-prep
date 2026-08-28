"""
Hovmöller of DIAT/SP nutrient-limitation and uptake diagnostics along the two
cross-shore transects defined in plot_cs_diag*.py, computed directly from raw
mc60_bgc_dia_avg files (no z-slicing). Sibling of plot_hov_transect_raw.py,
restricted to 4 scenarios and the bgc_dia_avg diagnostic variables.

Scenarios: tidesampwec (regular dia/ output — a full continuous run, not a
rerun subset); tidesnowec, notidesnowec, and ampwec (rerun bgc_dia_avg
output only, at dia/rerun_bgcdia/, covering a limited rerun date window --
ampwec's continuous everything/ dia_avg output lacks the LIM/UPTAKE
variables entirely, same reason tidesnowec/notidesnowec needed a targeted
rerun).

mc60_bgc_dia_avg files have no zeta variable, so depths are reconstructed by
pairing each dia_avg file with the regular his/ file sharing the same
YYYYMMDDHHMM filename prefix and reconstructing z_w via ROMS_depths using
that his file's time-mean zeta (matching the averaging window of the
dia_avg diagnostic). If no matching his file is found, 'avg' is left NaN for
that file's rows (surf/bot need no zeta and are still computed).

s_rho convention: index 0 = seafloor, index -1 (last) = surface, same as the
raw his/bgc files (confirmed via ROMS_depths.get_zw_zeta in the sibling
script). "surf"/"bot" are trivial single-level slices; "avg" is the
Hz-weighted mean over the full water column.

Variables (DIAT + SP only, per BEC2_DIAG output):
  LIM:    DIAT_N_LIM, DIAT_FE_LIM, DIAT_PO4_LIM, DIAT_SIO3_LIM,
          DIAT_LIGHT_LIM, DIAT_P_LIM,
          SP_N_LIM, SP_FE_LIM, SP_PO4_LIM, SP_LIGHT_LIM, SP_P_LIM
  UPTAKE: DIAT_NO3_UPTAKE, DIAT_NH4_UPTAKE, DIAT_NO2_UPTAKE,
          SP_NO3_UPTAKE, SP_NH4_UPTAKE, SP_NO2_UPTAKE

Output: ./figs/hov_transect_bgcdia/hov_{var}_{avg,surf,bot}_<ts|tn>.png
"""

import os
import sys
sys.path.append('/data/project3/minnaho/global/')
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cmocean
from netCDF4 import Dataset, num2date
from scipy.ndimage import map_coordinates
import ROMS_depths as depths

plt.rcParams.update({'font.size': 12})

# ---------------------------------------------------------------------------
# Paths / scenarios
# ---------------------------------------------------------------------------
GRD      = 'mc60_grd.nc'
SAVEPATH = './figs/hov_transect_bgcdia/'

SCENARIOS = {
    'tidesampwec':  '/data/project3/minnaho/swel/tides/mc60/ampwec',
    'tidesnowec':   '/data/project3/minnaho/swel/tides/mc60/nowec/output',
    'notidesnowec': '/data/project3/minnaho/swel/notides/mc60/nowec',
    'ampwec':       '/data/project3/minnaho/swel/notides/mc60/wec/ampwec',
}
LABELS = {
    'tidesampwec':  'tides, amplified WEC',
    'tidesnowec':   'tides, no WEC',
    'notidesnowec': 'no tides, no WEC',
    'ampwec':       'no tides, amplified WEC',
}
# tidesnowec/notidesnowec/ampwec only reran this date window — rerun
# bgc_dia_avg output lands in dia/rerun_bgcdia/, not dia/ itself. tidesampwec
# is a full continuous run, so it uses the default 'dia' (see
# DIA_SRC_SUBDIR.get below). ampwec's SCENARIOS path is the base root
# (his/ + dia/rerun_bgcdia/ live directly under it) -- NOT
# .../ampwec/everything, a separate flat-layout continuous run whose
# dia_avg files lack the LIM/UPTAKE variables.
DIA_SRC_SUBDIR = {
    'tidesnowec':   'dia/rerun_bgcdia',
    'notidesnowec': 'dia/rerun_bgcdia',
    'ampwec':       'dia/rerun_bgcdia',
}

# ---------------------------------------------------------------------------
# Transect geometry (same as plot_cs_diag*.py / plot_hov_transect_raw.py)
# ---------------------------------------------------------------------------
ETA0, XI0, SLOPE,  DEPTH_LIM0 = 271, 543, -0.8, -125
ETA1, XI1, SLOPE1, DEPTH_LIM1 = 832, 646,  0.6,  -90
LENGTH_XI = 250
N_PTS     = 300

METRICS      = ['avg', 'surf', 'bot']
METRIC_TITLE = {'avg': 'depth-avg', 'surf': 'surface', 'bot': 'bottom'}

DIAT_LIM_VARS    = ['DIAT_N_LIM', 'DIAT_FE_LIM', 'DIAT_PO4_LIM',
                     'DIAT_SIO3_LIM', 'DIAT_LIGHT_LIM', 'DIAT_P_LIM']
SP_LIM_VARS      = ['SP_N_LIM', 'SP_FE_LIM', 'SP_PO4_LIM',
                     'SP_LIGHT_LIM', 'SP_P_LIM']
DIAT_UPTAKE_VARS = ['DIAT_NO3_UPTAKE', 'DIAT_NH4_UPTAKE', 'DIAT_NO2_UPTAKE',
                     'DIAT_SI_UPTAKE']
SP_UPTAKE_VARS   = ['SP_NO3_UPTAKE', 'SP_NH4_UPTAKE', 'SP_NO2_UPTAKE']

LIM_VARS    = DIAT_LIM_VARS + SP_LIM_VARS
UPTAKE_VARS = DIAT_UPTAKE_VARS + SP_UPTAKE_VARS
PLOT_VARS   = LIM_VARS + UPTAKE_VARS

VAR_LONG_NAME = {
    'DIAT_N_LIM':        'Diatom N limitation',
    'DIAT_FE_LIM':       'Diatom Fe limitation',
    'DIAT_PO4_LIM':      'Diatom PO4 limitation',
    'DIAT_SIO3_LIM':     'Diatom SiO3 limitation',
    'DIAT_LIGHT_LIM':    'Diatom light limitation',
    'DIAT_P_LIM':        'Diatom P limitation',
    'SP_N_LIM':          'Small phyto N limitation',
    'SP_FE_LIM':         'Small phyto Fe limitation',
    'SP_PO4_LIM':        'Small phyto PO4 limitation',
    'SP_LIGHT_LIM':      'Small phyto light limitation',
    'SP_P_LIM':          'Small phyto P limitation',
    'DIAT_NO3_UPTAKE':   'Diatom NO3 uptake',
    'DIAT_NH4_UPTAKE':   'Diatom NH4 uptake',
    'DIAT_NO2_UPTAKE':   'Diatom NO2 uptake',
    'DIAT_SI_UPTAKE':    'Diatom Si uptake',
    'SP_NO3_UPTAKE':     'Small phyto NO3 uptake',
    'SP_NH4_UPTAKE':     'Small phyto NH4 uptake',
    'SP_NO2_UPTAKE':     'Small phyto NO2 uptake',
}

VAR_CONFIGS = {}
for _v in LIM_VARS:
    VAR_CONFIGS[_v] = dict(
        cmap=cmocean.cm.tempo, sym=False, vmin=0, vmax=1,
        label={m: f'{VAR_LONG_NAME[_v]} ({METRIC_TITLE[m]})' for m in METRICS},
    )
for _v in UPTAKE_VARS:
    VAR_CONFIGS[_v] = dict(
        cmap=cmocean.cm.algae, sym=False, vmin=0,
        label={m: f'{VAR_LONG_NAME[_v]} (mmol m$^{{-2}}$ s$^{{-1}}$, {METRIC_TITLE[m]})'
               for m in METRICS},
    )

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
        depth_lim = depth_lim,   # unused (full-column avg); kept for parity
    )


TRANSECTS = [
    build_transect(ETA0, XI0, SLOPE,  DEPTH_LIM0),
    build_transect(ETA1, XI1, SLOPE1, DEPTH_LIM1),
]
# ts = south transect (ETA0), tn = north transect (ETA1) -- output filenames only;
# internal cache dict keys stay hov_t0/hov_t1 to avoid invalidating existing caches
TRANSECT_NAMES = ['ts', 'tn']

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean(arr):
    """Convert to ndarray, fill netCDF sentinel fill values with nan."""
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


def vavg2d(section, Hz):
    """Thickness-weighted full-water-column average.
    section, Hz: (n_z, N_PTS) — Hz varies per column (not a fixed 1-D dz,
    since layer thickness depends on local bathymetry + free surface)."""
    numer = np.nansum(section * Hz, axis=0)
    denom = np.nansum(np.where(np.isnan(section), 0.0, Hz), axis=0)
    return np.where(denom > 0, numer / denom, np.nan)


# ---------------------------------------------------------------------------
# Compute all Hovmöller arrays for one scenario from raw bgc_dia_avg output.
# Returns {var: {metric: (times_mpl, [hov_t0, hov_t1])}}  metric in METRICS
# ---------------------------------------------------------------------------
def compute_hovmollers(scen):
    root    = SCENARIOS[scen]
    dia_dir = os.path.join(root, DIA_SRC_SUBDIR.get(scen, 'dia'))
    his_dir = os.path.join(root, 'his')

    dia_files = sorted(glob.glob(os.path.join(dia_dir, 'mc60_bgc_dia_avg.*.nc')))
    his_files = sorted(glob.glob(os.path.join(his_dir, 'mc60_his.*.nc')))
    # match by YYYYMMDDHH (10 chars) only, not full YYYYMMDDHHMM — the minute
    # digit sometimes shifts by 1 between dia_avg and his filenames for files
    # that were regenerated/repaired independently (e.g. tidesampwec's HDF-
    # corrupted-and-rejoined 04-25..04-27 files: dia stamp ...1100..., his
    # stamp ...1101...). Hour-level prefixes are confirmed unique per scenario.
    his_by_prefix = {os.path.basename(f)[len('mc60_his.'):len('mc60_his.') + 10]: f
                      for f in his_files}
    print(f'  [{scen}] {len(dia_files)} dia_avg files, '
          f'{len(his_files)} his files available for zeta', flush=True)

    n_tr       = len(TRANSECTS)
    times_list = []
    acc = {var: [{'avg': [], 'surf': [], 'bot': []} for _ in range(n_tr)]
           for var in PLOT_VARS}

    for df in dia_files:
        with Dataset(df) as dnc:
            ocean_time = np.array(dnc.variables['ocean_time'][:])
            n_t = ocean_time.shape[0]
            times_list.extend(mdates.date2num(
                num2date(ocean_time, 'seconds since 1995-01-01',
                          only_use_cftime_datetimes=False)))

            prefix = os.path.basename(df)[len('mc60_bgc_dia_avg.'):
                                           len('mc60_bgc_dia_avg.') + 10]
            his_f = his_by_prefix.get(prefix)
            Hz_t_by_tr = None
            if his_f is not None:
                with Dataset(his_f) as hnc:
                    zeta_mean = np.nanmean(clean(hnc.variables['zeta'][:]), axis=0)
                z_w  = depths.get_zw_zeta(dnc, grdnc, zeta_mean)   # (n_z+1, eta, xi)
                Hz3d = np.diff(z_w, axis=0)                         # (n_z, eta, xi)
                Hz_t_by_tr = [interp_section(Hz3d, tr['coords'], tr['mask'])
                              for tr in TRANSECTS]
            else:
                print(f'  WARNING: no his match for {os.path.basename(df)} '
                      f'(prefix {prefix}) — avg metric will be NaN for this file',
                      flush=True)

            for t_i in range(n_t):
                for var in PLOT_VARS:
                    if var not in dnc.variables:
                        continue
                    arr3d = clean(dnc.variables[var][t_i]) * mask_plot
                    for ti, tr in enumerate(TRANSECTS):
                        sec = interp_section(arr3d, tr['coords'], tr['mask'])
                        if Hz_t_by_tr is not None:
                            acc[var][ti]['avg'].append(vavg2d(sec, Hz_t_by_tr[ti]))
                        else:
                            acc[var][ti]['avg'].append(np.full(N_PTS, np.nan))
                        acc[var][ti]['surf'].append(sec[-1])   # top level = surface
                        acc[var][ti]['bot'].append(sec[0])     # bottom level = seafloor

    if not times_list:
        return {}

    times = np.array(times_list)
    idx   = np.argsort(times)
    times = times[idx]

    hovs = {}
    for var, per_tr in acc.items():
        # var may be entirely (or partially) absent from this scenario's files
        # (e.g. DIAT_SI_UPTAKE isn't in tidesampwec's dia_avg files) — skip it
        # rather than indexing a length-mismatched/empty accumulator
        if any(len(per_tr[ti]['avg']) != len(times_list) for ti in range(n_tr)):
            print(f'  {var}: not present in all files for this scenario — skipping', flush=True)
            continue
        metrics = {}
        for metric in METRICS:
            hov_list = [np.array(per_tr[ti][metric])[idx] for ti in range(n_tr)]
            metrics[metric] = (times, hov_list)
        hovs[var] = metrics

    return hovs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
os.makedirs(SAVEPATH, exist_ok=True)

def cache_path(scen, var, metric):
    return f'{SAVEPATH}hov_cache_{scen}_{var}_{metric}.npz'

def load_cache_var(scen, var, metric):
    """Returns (times, hov_list) or None if this cache file doesn't exist."""
    cp = cache_path(scen, var, metric)
    if not os.path.exists(cp):
        return None
    cache = np.load(cp, allow_pickle=False)
    times = cache['times']
    hov_list = [cache[f'hov_t{ti}'] for ti in range(len(TRANSECTS))]
    return times, hov_list

def save_cache_var(scen, var, metric, times, hov_list):
    cp = cache_path(scen, var, metric)
    cache_data = {'times': times}
    for ti, hov in enumerate(hov_list):
        cache_data[f'hov_t{ti}'] = hov
    np.savez(cp, **cache_data)
    print(f'  saved cache -> {cp}', flush=True)

all_hovs = {}
for scen in SCENARIOS:
    cached = {}
    for var in PLOT_VARS:
        metrics = {}
        for metric in METRICS:
            result = load_cache_var(scen, var, metric)
            if result is not None:
                metrics[metric] = result
        if len(metrics) == len(METRICS):
            cached[var] = metrics

    missing = [v for v in PLOT_VARS if v not in cached]
    if not missing:
        print(f'[{scen}] loading cache...', flush=True)
        all_hovs[scen] = cached
    else:
        print(f'[{scen}] computing (missing: {missing})...', flush=True)
        computed = compute_hovmollers(scen)
        all_hovs[scen] = computed
        for var, metrics in computed.items():
            for metric, (times, hov_list) in metrics.items():
                save_cache_var(scen, var, metric, times, hov_list)

for var in PLOT_VARS:
    cfg = VAR_CONFIGS[var]

    for metric in METRICS:
        print(f'\nPlotting {var} ({metric})...', flush=True)

        # colormap limits
        present_hovs = [all_hovs[s][var][metric][1][ti]
                        for s in SCENARIOS if var in all_hovs[s]
                        for ti in range(len(TRANSECTS))]
        if not present_hovs:
            print(f'  no data for {var} ({metric}), skipping')
            continue

        if cfg.get('sym'):
            vmax = max(np.nanpercentile(np.abs(h), 99) for h in present_hovs)
            vmin = -vmax
        else:
            vmin = cfg.get('vmin', np.nanpercentile(np.concatenate([h.ravel() for h in present_hovs]), 1))
            vmax = cfg.get('vmax', np.nanpercentile(np.concatenate([h.ravel() for h in present_hovs]), 99))

        for ti, tr in enumerate(TRANSECTS):
            fig, axes = plt.subplots(len(SCENARIOS), 1,
                                      figsize=(12, 3 * len(SCENARIOS)),
                                      sharex=True, sharey=True)
            pc_last = None

            for ax, scen in zip(axes, SCENARIOS):
                if var not in all_hovs[scen]:
                    ax.set_title(LABELS[scen], fontsize=11)
                    continue
                t, hov_list = all_hovs[scen][var][metric]

                pc = ax.pcolormesh(t, tr['lon'], hov_list[ti].T,
                                   cmap=cfg['cmap'], vmin=vmin, vmax=vmax,
                                   shading='nearest')
                pc_last = pc
                ax.set_ylabel('Longitude')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax.set_title(LABELS[scen], fontsize=11)

            axes[-1].set_xlabel('Date')
            plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
            fig.suptitle(METRIC_TITLE[metric], y=0.995)

            if pc_last is not None:
                fig.subplots_adjust(right=0.87, hspace=0.45)
                cax = fig.add_axes([0.89, 0.15, 0.015, 0.7])
                fig.colorbar(pc_last, cax=cax, label=cfg['label'][metric])

            fname = f'{SAVEPATH}hov_{var}_{metric}_{TRANSECT_NAMES[ti]}.png'
            plt.savefig(fname, dpi=800, bbox_inches='tight')
            plt.close(fig)
            print(f'saved -> {fname}')
