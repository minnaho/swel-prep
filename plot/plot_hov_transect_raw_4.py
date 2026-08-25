"""
4-scenario version of plot_hov_transect_raw.py -- same Hovmoller
computation/layout, only SCENARIOS restricted to notidesnowec / tidesnowec /
ampwec (the "notidesampwec" -- no tides, 2.5x WEC -- scenario's dict key) /
tidesampwec.

Cache is SHARED with plot_hov_transect_raw.py (same CACHE_DIR,
hov_cache_<scen>_<var>_<metric>.npz, keyed by scenario so there's no
collision) -- running this script reuses whichever of these 4 scenarios
that script already computed/cached, and vice versa. Final PNGs go to a
separate SAVEPATH so they don't overwrite the 5/6-scenario figures (the
stacked-subplot figures depend on len(SCENARIOS), so the same output
filename would otherwise mean two different things).

See plot_hov_transect_raw.py for the full methodology docstring (transect
geometry, N²/w'NO3' derivation, raw vs. z-sliced sibling).

Output: ./figs/hov_transect_raw_4/hov_{var}_{avg,surf,bot}_<ts|tn>.png
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

G       = 9.81
RHO_REF = 1025.0

# ---------------------------------------------------------------------------
# Paths / scenarios (raw ROMS output — same roots as plot_cs_diag*.py)
# ---------------------------------------------------------------------------
GRD       = 'mc60_grd.nc'
CACHE_DIR = './figs/hov_transect_raw/'      # shared with plot_hov_transect_raw.py
SAVEPATH  = './figs/hov_transect_raw_4/'    # this script's own final PNGs

SCENARIOS = {
    'notidesnowec': '/data/project3/minnaho/swel/notides/mc60/nowec',
    'tidesnowec':   '/data/project3/minnaho/swel/tides/mc60/nowec/output',
    'ampwec':       '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything',
    'tidesampwec':  '/data/project3/minnaho/swel/tides/mc60/ampwec/everything',
}
LABELS = {
    'notidesnowec': 'no tides, no WEC',
    'tidesnowec':   'tides, no WEC',
    'ampwec':       'no tides, 2.5× WEC',
    'tidesampwec':  'tides, 2.5× WEC',
}

# ---------------------------------------------------------------------------
# Transect geometry (same as plot_cs_diag*.py)
# ---------------------------------------------------------------------------
ETA0, XI0, SLOPE,  DEPTH_LIM0 = 271, 543, -0.8, -125
ETA1, XI1, SLOPE1, DEPTH_LIM1 = 832, 646,  0.6,  -90
LENGTH_XI = 250
N_PTS     = 300

PLOT_VARS = ['N2', 'w', 'NO3', 'O2', "w'NO3'", 'phytoC']
METRICS   = ['avg', 'surf', 'bot']
METRIC_TITLE = {'avg': 'depth-avg', 'surf': 'surface', 'bot': 'bottom'}

VAR_CONFIGS = {
    'N2': dict(
        cmap  = cmocean.cm.thermal,
        sym   = False,
        label = {
            'avg':  r'$\langle N^2 \rangle$ (s$^{-2}$)',
            'surf': r'$N^2$ at surface (s$^{-2}$)',
            'bot':  r'$N^2$ at bottom (s$^{-2}$)',
        },
    ),
    'w': dict(
        cmap  = cmocean.cm.balance,
        sym   = True,
        label = {
            'avg':  r'$\langle w \rangle$ (m s$^{-1}$)',
            'surf': r'$w$ at surface (m s$^{-1}$)',
            'bot':  r'$w$ at bottom (m s$^{-1}$)',
        },
    ),
    'NO3': dict(
        cmap  = cmocean.cm.matter,
        vmin  = 0,
        vmax  = 20,
        sym   = False,
        label = {
            'avg':  r'$\langle\mathrm{NO}_3\rangle$ (mmol N m$^{-3}$)',
            'surf': r'$\mathrm{NO}_3$ at surface (mmol N m$^{-3}$)',
            'bot':  r'$\mathrm{NO}_3$ at bottom (mmol N m$^{-3}$)',
        },
    ),
    'O2': dict(
        cmap  = cmocean.cm.oxy,
        vmin  = 100,
        vmax  = 300,
        sym   = False,
        label = {
            'avg':  r'$\langle\mathrm{O}_2\rangle$ (mmol O$_2$ m$^{-3}$)',
            'surf': r'$\mathrm{O}_2$ at surface (mmol O$_2$ m$^{-3}$)',
            'bot':  r'$\mathrm{O}_2$ at bottom (mmol O$_2$ m$^{-3}$)',
        },
    ),
    "w'NO3'": dict(
        cmap  = cmocean.cm.balance,
        sym   = True,
        label = {
            'avg':  r"$\langle w'\mathrm{NO}_3'\rangle$ (m s$^{-1}$ mmol N m$^{-3}$)",
            'surf': r"$w'\mathrm{NO}_3'$ at surface (m s$^{-1}$ mmol N m$^{-3}$)",
            'bot':  r"$w'\mathrm{NO}_3'$ at bottom (m s$^{-1}$ mmol N m$^{-3}$)",
        },
    ),
    'phytoC': dict(
        cmap  = cmocean.cm.algae,
        vmin  = 0,
        vmax  = 30,
        sym   = False,
        label = {
            'avg':  r'$\langle$phyto C$\rangle$ (mmol C m$^{-3}$)',
            'surf': r'phyto C at surface (mmol C m$^{-3}$)',
            'bot':  r'phyto C at bottom (mmol C m$^{-3}$)',
        },
    ),
}

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
        depth_lim = depth_lim,   # unused by raw metrics (full-column avg); kept for parity
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
def src_glob(root, kind):
    """Locate his/ or bgc/ files, handling flat vs subdir scenario layouts
    (ampwec/tidesampwec are flat; the other scenarios use his/ + bgc/ subdirs)."""
    sub  = os.path.join(root, kind)
    base = sub if os.path.isdir(sub) else root
    stem = 'mc60_his' if kind == 'his' else 'mc60_bgc'
    return os.path.join(base, f'{stem}.*.nc')


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
# Compute all Hovmöller arrays for one scenario from raw output.
# Returns {var: {metric: (times_mpl, [hov_t0, hov_t1])}}  metric in METRICS
# ---------------------------------------------------------------------------
def compute_hovmollers(scen):
    root      = SCENARIOS[scen]
    his_files = sorted(glob.glob(src_glob(root, 'his')))
    bgc_files = sorted(glob.glob(src_glob(root, 'bgc')))
    n_pairs   = min(len(his_files), len(bgc_files))
    print(f'  [{scen}] {len(his_files)} his, {len(bgc_files)} bgc, '
          f'pairing {n_pairs}', flush=True)

    n_tr       = len(TRANSECTS)
    times_list = []

    # true model restarts (a SLURM job resubmitted from a .rst file, not just
    # a new 12h history-write chunk within one continuous run) leave a small
    # numerical transient at that file's first timestep -- the leapfrog/AB3
    # time-stepping loses its prior time levels and has to bootstrap with a
    # lower-order step. Flagged here via the :init_file global attr (changes
    # only at a true restart, confirmed against every his file's attr) so
    # that one timestep can be NaN'd out below instead of rendering as a
    # spurious full-transect vertical line in the Hovmoller
    restart_flags_list = []
    prev_init_file = None

    # running per-(var,transect,metric) accumulators (lists of (N_PTS,) rows)
    acc = {var: [{'avg': [], 'surf': [], 'bot': []} for _ in range(n_tr)]
           for var in ['w', 'N2', 'NO3', 'O2', 'phytoC']}
    # full sections retained for the w'NO3' anomaly flux — need the time
    # mean before computing anomalies, plus the matching Hz for weighting
    w_full  = [[] for _ in range(n_tr)]
    no3_full = [[] for _ in range(n_tr)]
    hz_full  = [[] for _ in range(n_tr)]

    for hf in range(n_pairs):
        with Dataset(his_files[hf]) as hnc, Dataset(bgc_files[hf]) as bnc:
            ocean_time = np.array(hnc.variables['ocean_time'][:])
            n_t = ocean_time.shape[0]
            times_list.extend(mdates.date2num(
                num2date(ocean_time, 'seconds since 1995-01-01',
                          only_use_cftime_datetimes=False)))

            init_file = getattr(hnc, 'init_file', None)
            is_restart = prev_init_file is not None and init_file != prev_init_file
            restart_flags_list.extend([is_restart] + [False] * (n_t - 1))
            prev_init_file = init_file

            for t_i in range(n_t):
                zeta = np.squeeze(hnc.variables['zeta'][t_i, :, :])
                z_w  = depths.get_zw_zeta(hnc, grdnc, zeta)   # (n_z+1, eta, xi)
                # masked with mask_plot like every field variable below -- h on
                # land cells isn't NaN (clamped to a small placeholder, ~2m), so
                # without this Hz3d/zr3d would silently leak thin fake land layers
                # into any transect point whose bilinear neighbors include land
                Hz3d = np.diff(z_w, axis=0)      * mask_plot   # (n_z, eta, xi)
                zr3d = 0.5 * (z_w[1:] + z_w[:-1]) * mask_plot   # (n_z, eta, xi)

                w3d       = clean(hnc.variables['w'][t_i])   * mask_plot
                rho3d     = clean(hnc.variables['rho'][t_i]) * mask_plot
                no3_3d    = clean(bnc.variables['NO3'][t_i]) * mask_plot
                o2_3d     = clean(bnc.variables['O2'][t_i])  * mask_plot
                phytoC_3d = (clean(bnc.variables['DIATC'][t_i]) +
                             clean(bnc.variables['DIAZC'][t_i]) +
                             clean(bnc.variables['SPC'][t_i])) * mask_plot

                for ti, tr in enumerate(TRANSECTS):
                    Hz_t = interp_section(Hz3d, tr['coords'], tr['mask'])
                    zr_t = interp_section(zr3d, tr['coords'], tr['mask'])

                    w_t      = interp_section(w3d,       tr['coords'], tr['mask'])
                    rho_t    = interp_section(rho3d,     tr['coords'], tr['mask'])
                    no3_t    = interp_section(no3_3d,    tr['coords'], tr['mask'])
                    o2_t     = interp_section(o2_3d,     tr['coords'], tr['mask'])
                    phytoC_t = interp_section(phytoC_3d, tr['coords'], tr['mask'])

                    # N² = -(g/ρ₀) ∂ρ/∂z via chain rule: (∂ρ/∂idx) / (∂z/∂idx)
                    n2_t = (-(G / RHO_REF) * np.gradient(rho_t, axis=0) /
                            np.gradient(zr_t, axis=0))

                    for var, sec in (('w', w_t), ('N2', n2_t), ('NO3', no3_t),
                                      ('O2', o2_t), ('phytoC', phytoC_t)):
                        acc[var][ti]['avg'].append(vavg2d(sec, Hz_t))
                        acc[var][ti]['surf'].append(sec[-1])   # top level = surface
                        acc[var][ti]['bot'].append(sec[0])     # bottom level = seafloor

                    w_full[ti].append(w_t)
                    no3_full[ti].append(no3_t)
                    hz_full[ti].append(Hz_t)

    if not times_list:
        return {}

    times = np.array(times_list)
    idx   = np.argsort(times)
    times = times[idx]
    restart_flags = np.array(restart_flags_list)[idx]

    hovs = {}
    for var, per_tr in acc.items():
        metrics = {}
        for metric in METRICS:
            hov_list = []
            for ti in range(n_tr):
                arr = np.array(per_tr[ti][metric])[idx]
                arr[restart_flags] = np.nan
                hov_list.append(arr)
            metrics[metric] = (times, hov_list)
        hovs[var] = metrics

    # ── w'NO3': time-mean anomaly flux, then avg/surf/bot ──────────────────
    avg_l, surf_l, bot_l = [], [], []
    for ti in range(n_tr):
        w_sec    = np.array(w_full[ti])[idx]     # (n_t, n_z, N_PTS)
        no3_sec  = np.array(no3_full[ti])[idx]
        hz_sec   = np.array(hz_full[ti])[idx]
        w_mean   = np.nanmean(w_sec,   axis=0)   # (n_z, N_PTS)
        no3_mean = np.nanmean(no3_sec, axis=0)
        flux_sec = (w_sec - w_mean[None]) * (no3_sec - no3_mean[None])  # (n_t, n_z, N_PTS)

        avg_arr  = np.array([vavg2d(flux_sec[it], hz_sec[it])
                              for it in range(flux_sec.shape[0])])
        surf_arr = flux_sec[:, -1, :].copy()
        bot_arr  = flux_sec[:, 0, :].copy()
        avg_arr[restart_flags]  = np.nan
        surf_arr[restart_flags] = np.nan
        bot_arr[restart_flags]  = np.nan
        avg_l.append(avg_arr)
        surf_l.append(surf_arr)
        bot_l.append(bot_arr)
    hovs["w'NO3'"] = {'avg': (times, avg_l), 'surf': (times, surf_l), 'bot': (times, bot_l)}

    return hovs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SAVEPATH, exist_ok=True)

def cache_path(scen, var, metric):
    vk = var.replace("'", 'p')
    return f'{CACHE_DIR}hov_cache_{scen}_{vk}_{metric}.npz'

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
                ax.ticklabel_format(axis='y', style='plain', useOffset=False)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                ax.set_title(LABELS[scen], fontsize=11)

            axes[-1].set_xlabel('Date')
            plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
            #fig.suptitle(METRIC_TITLE[metric], y=0.995)

            if pc_last is not None:
                fig.subplots_adjust(right=0.87, hspace=0.45)
                cax = fig.add_axes([0.89, 0.15, 0.015, 0.7])
                fig.colorbar(pc_last, cax=cax, label=cfg['label'][metric])

            fname_var = var.replace("'", 'p')
            fname = f'{SAVEPATH}hov_{fname_var}_{metric}_{TRANSECT_NAMES[ti]}.png'
            plt.savefig(fname, dpi=800, bbox_inches='tight')
            plt.close(fig)
            print(f'saved -> {fname}')
