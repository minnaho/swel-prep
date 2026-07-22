"""
Hovmöller of fields along the two cross-shore transects defined in
plot_cs_diag*.py. For each variable, three Hovmöllers are produced:
  avg  — depth-weighted vertical average (0 to transect depth_lim)
  surf — value at the surface (depth = 0)
  bot  — value at the deepest kept z-level (depth = transect depth_lim)

Variables:
  N²      — Brunt-Väisälä frequency squared from zslice rho
  w       — vertical velocity
  NO3     — nitrate
  O2      — dissolved oxygen
  w'NO3'  — eddy NO3 flux: (w−<w>)(NO3−<NO3>)
  phytoC  — total phytoplankton C = DIATC + DIAZC + SPC

Time is parsed from zslice filenames (12 steps/file, 1 hr/step,
filename timestamp = last step). Sections stored in memory before
computing derived quantities.

Output: ./figs/hov_transect/hov_{var}_{avg,surf,bot}_<ts|tn>.png
"""

import os
import sys
sys.path.append('/data/project3/minnaho/global/')
import datetime
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cmocean
from netCDF4 import Dataset
from scipy.ndimage import map_coordinates

plt.rcParams.update({'font.size': 12})

G       = 9.81
RHO_REF = 1025.0

# ---------------------------------------------------------------------------
# Paths / scenarios
# ---------------------------------------------------------------------------
ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'
GRD         = 'mc60_grd.nc'
SAVEPATH    = './figs/hov_transect/'

SCENARIOS = ['tideswec', 'tidesnowec', 'notidesnowec', 'notideswec', 'ampwec']
SCEN_DIRS = {'ampwec': 'notidesampwec'}
LABELS = {
    'tideswec':     'tides, WEC',
    'tidesnowec':   'tides, no WEC',
    'notidesnowec': 'no tides, no WEC',
    'notideswec':   'no tides, WEC',
    'ampwec':       'no tides, 2.5× WEC',
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
        vmin  = 0,
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
        vmax  = 5,
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
grdnc    = Dataset(GRD, 'r')
lon      = np.array(grdnc['lon_rho'][:]) - 360
mask_rho = np.array(grdnc['mask_rho'][:])


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
# ts = south transect (ETA0), tn = north transect (ETA1) -- output filenames only;
# internal cache dict keys stay hov_t0/hov_t1 to avoid invalidating existing caches
TRANSECT_NAMES = ['ts', 'tn']

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def interp_section_3d(field3d, coords, mask_t):
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


def vavg(section, dz_v):
    """Depth-weighted vertical average. section: (n_z, N_PTS), dz_v: (n_z,)."""
    numer = np.nansum(section * dz_v[:, None], axis=0)
    denom = np.nansum(np.where(np.isnan(section), 0.0, dz_v[:, None]), axis=0)
    return np.where(denom > 0, numer / denom, np.nan)


def bottom_index(depth_arr, depth_lim):
    """Index of the deepest z-level within [depth_lim, 0]. depth[0] == 0 == surface."""
    return np.where(depth_arr >= depth_lim)[0][-1]


def extract_metrics(sec_full, valid, dz_valid, bot_i):
    """sec_full: (n_time, n_z, N_PTS) raw section. Returns (avg, surf, bot),
    each (n_time, N_PTS)."""
    n_time = sec_full.shape[0]
    avg  = np.array([vavg(sec_full[it][valid], dz_valid) for it in range(n_time)])
    surf = sec_full[:, 0, :]
    bot  = sec_full[:, bot_i, :]
    return avg, surf, bot


def file_stamp(fpath):
    return os.path.basename(fpath).split('.')[1]   # 'YYYYMMDDHHMMSS'


def file_times(fpath, n_steps):
    t_end = datetime.datetime.strptime(file_stamp(fpath), '%Y%m%d%H%M%S')
    return [t_end - datetime.timedelta(hours=n_steps - 1 - i)
            for i in range(n_steps)]


def read_arr(nc, v):
    """Read variable, fill >1e30 with nan, ensure 4-D. Returns None if wrong shape."""
    arr = np.array(nc.variables[v][:])
    arr = np.where(np.abs(arr) > 1e30, np.nan, arr)
    if arr.ndim == 3:
        arr = arr[np.newaxis]
    return arr if arr.ndim == 4 else None


# ---------------------------------------------------------------------------
# Core loader: returns full-depth raw sections for a list of variables.
# sections[var][ti] shape: (n_time, n_z, N_PTS)  (sorted by time)
# ---------------------------------------------------------------------------
def load_raw_sections(files, var_list):
    depth      = None
    times_list = []
    raw        = {v: [[], []] for v in var_list}   # raw[v][ti] = list of (n_z, N_PTS)

    for f in files:
        with Dataset(f) as nc:
            present = [v for v in var_list if v in nc.variables]
            if not present:
                continue
            if depth is None:
                depth = np.array(nc.variables['depth'][:])

            arr0 = read_arr(nc, present[0])
            if arr0 is None:
                continue
            n_t = arr0.shape[0]
            times_list.extend(mdates.date2num(file_times(f, n_t)))

            for v in present:
                arr = read_arr(nc, v)
                if arr is None:
                    continue
                for it in range(arr.shape[0]):
                    for ti, tr in enumerate(TRANSECTS):
                        raw[v][ti].append(
                            interp_section_3d(arr[it], tr['coords'], tr['mask']))

    if not times_list or depth is None:
        return None, None, None, {}

    idx   = np.argsort(times_list)
    times = np.array(times_list)[idx]
    dz    = np.abs(np.gradient(depth))
    out   = {v: [np.array(raw[v][ti])[idx] for ti in range(len(TRANSECTS))]
             for v in var_list if raw[v][0]}
    return times, depth, dz, out


# ---------------------------------------------------------------------------
# Compute all Hovmöller arrays for one scenario.
# Returns {var: {metric: (times_mpl, [hov_t0, hov_t1])}}  metric in METRICS
# ---------------------------------------------------------------------------
def compute_hovmollers(scen):
    sd        = SCEN_DIRS.get(scen, scen)
    his_files = sorted(glob.glob(f'{ZSLICE_ROOT}/{sd}/z_mc60_his.*.nc'))
    bgc_files = sorted(glob.glob(f'{ZSLICE_ROOT}/{sd}/bgc/z_mc60_bgc.*.nc'))
    print(f'  [{scen}] {len(his_files)} his, {len(bgc_files)} bgc', flush=True)

    his_t, his_d, his_dz, his_sec = load_raw_sections(his_files, ['w', 'rho'])
    bgc_t, bgc_d, bgc_dz, bgc_sec = load_raw_sections(
        bgc_files, ['NO3', 'O2', 'DIATC', 'DIAZC', 'SPC'])

    hovs = {}

    def add_var(var, times, d, dz, sec_by_ti):
        """sec_by_ti[ti]: (n_time, n_z, N_PTS) raw section for that transect."""
        avg_l, surf_l, bot_l = [], [], []
        for ti, tr in enumerate(TRANSECTS):
            valid = d >= tr['depth_lim']
            bot_i = bottom_index(d, tr['depth_lim'])
            avg, surf, bot = extract_metrics(sec_by_ti[ti], valid, dz[valid], bot_i)
            avg_l.append(avg); surf_l.append(surf); bot_l.append(bot)
        hovs[var] = {'avg': (times, avg_l), 'surf': (times, surf_l), 'bot': (times, bot_l)}

    # ── w ────────────────────────────────────────────────────────────────────
    if his_t is not None and 'w' in his_sec:
        add_var('w', his_t, his_d, his_dz, his_sec['w'])

    # ── N² = -(g/ρ₀) ∂ρ/∂z  (gradient over full depth, then trim & avg) ────
    if his_t is not None and 'rho' in his_sec:
        n2_by_ti = []
        for ti in range(len(TRANSECTS)):
            n2_full = np.array([
                -(G / RHO_REF) * np.gradient(his_sec['rho'][ti][it], his_d, axis=0)
                for it in range(len(his_t))])
            n2_by_ti.append(n2_full)
        add_var('N2', his_t, his_d, his_dz, n2_by_ti)

    # ── NO3 ──────────────────────────────────────────────────────────────────
    if bgc_t is not None and 'NO3' in bgc_sec:
        add_var('NO3', bgc_t, bgc_d, bgc_dz, bgc_sec['NO3'])

    # ── O2 ───────────────────────────────────────────────────────────────────
    if bgc_t is not None and 'O2' in bgc_sec:
        add_var('O2', bgc_t, bgc_d, bgc_dz, bgc_sec['O2'])

    # ── phytoC = DIATC + DIAZC + SPC ─────────────────────────────────────────
    if bgc_t is not None and all(v in bgc_sec for v in ['DIATC', 'DIAZC', 'SPC']):
        phytoC_by_ti = [bgc_sec['DIATC'][ti] + bgc_sec['DIAZC'][ti] + bgc_sec['SPC'][ti]
                         for ti in range(len(TRANSECTS))]
        add_var('phytoC', bgc_t, bgc_d, bgc_dz, phytoC_by_ti)

    # ── w'NO3': match files by filename timestamp, compute anomaly flux ───────
    if his_t is not None and bgc_t is not None and 'w' in his_sec and 'NO3' in bgc_sec:
        his_by_stamp = {file_stamp(f): f for f in his_files}
        bgc_by_stamp = {file_stamp(f): f for f in bgc_files}
        common = sorted(set(his_by_stamp) & set(bgc_by_stamp))

        if common:
            w_raw   = [[] for _ in TRANSECTS]   # matched, sorted below
            no3_raw = [[] for _ in TRANSECTS]
            t_raw   = []

            for stamp in common:
                with Dataset(his_by_stamp[stamp]) as hnc, \
                     Dataset(bgc_by_stamp[stamp]) as bnc:
                    w_arr   = read_arr(hnc, 'w')
                    no3_arr = read_arr(bnc, 'NO3')
                    if w_arr is None or no3_arr is None:
                        continue
                    n_t = min(w_arr.shape[0], no3_arr.shape[0])
                    t_raw.extend(mdates.date2num(
                        file_times(his_by_stamp[stamp], n_t)))
                    for it in range(n_t):
                        for ti, tr in enumerate(TRANSECTS):
                            w_raw[ti].append(
                                interp_section_3d(w_arr[it],   tr['coords'], tr['mask']))
                            no3_raw[ti].append(
                                interp_section_3d(no3_arr[it], tr['coords'], tr['mask']))

            if t_raw:
                idx_t = np.argsort(t_raw)
                t_arr = np.array(t_raw)[idx_t]
                flux_by_ti = []
                for ti in range(len(TRANSECTS)):
                    w_sec   = np.array(w_raw[ti])[idx_t]    # (n_t, n_z, N_PTS)
                    no3_sec = np.array(no3_raw[ti])[idx_t]
                    w_mean   = np.nanmean(w_sec,   axis=0)  # (n_z, N_PTS)
                    no3_mean = np.nanmean(no3_sec, axis=0)
                    flux_by_ti.append((w_sec - w_mean[None]) * (no3_sec - no3_mean[None]))
                add_var("w'NO3'", t_arr, his_d, his_dz, flux_by_ti)

    return hovs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
os.makedirs(SAVEPATH, exist_ok=True)

def cache_path(scen, var, metric):
    vk = var.replace("'", 'p')
    return f'{SAVEPATH}hov_cache_{scen}_{vk}_{metric}.npz'

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
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                ax.set_title(LABELS[scen], fontsize=11)

            axes[-1].set_xlabel('Date')
            plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
            fig.suptitle(METRIC_TITLE[metric], y=0.995)

            if pc_last is not None:
                fig.subplots_adjust(right=0.87, hspace=0.45)
                cax = fig.add_axes([0.89, 0.15, 0.015, 0.7])
                fig.colorbar(pc_last, cax=cax, label=cfg['label'][metric])

            fname_var = var.replace("'", 'p')
            fname = f'{SAVEPATH}hov_{fname_var}_{metric}_{TRANSECT_NAMES[ti]}.png'
            plt.savefig(fname, dpi=800, bbox_inches='tight')
            plt.close(fig)
            print(f'saved -> {fname}')
