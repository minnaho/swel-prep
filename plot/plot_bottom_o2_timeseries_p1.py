"""
p1/p2-only version of plot_bottom_o2_timeseries.py -- same computation/cache
(the shared bottom_o2_timeseries_cache_<scen>.npz files), only the final
plot is restricted to two panels (p1, p2) instead of all four points.

POINTS is kept as the full original 4-point dict, not trimmed to just
p1/p2, because the cache format stores each point positionally
(series_0..series_3 in POINTS-iteration-order, not by name) -- reusing the
same cache with a reduced POINTS dict would silently read back the wrong
point's data. Only the plotting stage below picks out PLOT_POINTS.

Output: ./figs/bottom_o2_timeseries_p1.png
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
from netCDF4 import Dataset, num2date
from scipy.ndimage import map_coordinates
import scenario_style as ss

plt.rcParams.update({'font.size': 12})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRD      = 'mc60_grd.nc'
SAVEPATH = './figs/'   # shared cache dir with plot_bottom_o2_timeseries.py

SCENARIOS = {
    #'tideswec':     '/data/project3/minnaho/swel/tides/mc60/wec',
    'tidesnowec':   '/data/project3/minnaho/swel/tides/mc60/nowec/output',
    'notidesnowec': '/data/project3/minnaho/swel/notides/mc60/nowec',
    #'notideswec':   '/data/project3/minnaho/swel/notides/mc60/wec/rerun',
    'ampwec':       '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything',
    'tidesampwec':  '/data/project3/minnaho/swel/tides/mc60/ampwec/everything',
}
LABELS  = ss.LABELS
COLORS  = ss.COLORS
LSTYLES = ss.LSTYLES

# Transect geometry — identical to plot_cs_diag.py
ETA0, XI0, SLOPE0 = 271, 543, -0.8   # transect 0
ETA1, XI1, SLOPE1 = 832, 646,  0.6   # transect 1
LENGTH_XI = 250
N_PTS     = 300
ISOBATH   = 50.0

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
grdnc = Dataset(GRD, 'r')
lat   = np.array(grdnc['lat_rho'][:])
h     = np.array(grdnc['h'][:])


def find_isobath_point(eta0, xi0, slope, target_depth=ISOBATH):
    """Walk the transect line from the coast offshore (same parametrization
    as build_transect() in plot_cs_diag.py) and return the nearest grid
    index [eta, xi] where depth first crosses target_depth."""
    xi  = np.linspace(xi0,  xi0  - LENGTH_XI, N_PTS)
    eta = np.linspace(eta0, eta0 + slope * (-LENGTH_XI), N_PTS)
    crd = np.array([eta, xi])
    h_t = map_coordinates(h, crd, order=1, mode='nearest')

    idx = np.where((h_t[:-1] < target_depth) & (h_t[1:] >= target_depth))[0]
    if len(idx) == 0:
        raise ValueError(
            f'transect from (eta0={eta0}, xi0={xi0}) never crosses {target_depth} m'
        )
    i0   = idx[0]
    frac = (target_depth - h_t[i0]) / (h_t[i0 + 1] - h_t[i0])
    eta_c = eta[i0] + frac * (eta[i0 + 1] - eta[i0])
    xi_c  = xi[i0]  + frac * (xi[i0 + 1]  - xi[i0])
    return int(round(eta_c)), int(round(xi_c))


# south = smaller latitude at the coastal end of the transect
if lat[ETA0, XI0] < lat[ETA1, XI1]:
    south_tr, north_tr = (ETA0, XI0, SLOPE0), (ETA1, XI1, SLOPE1)
else:
    south_tr, north_tr = (ETA1, XI1, SLOPE1), (ETA0, XI0, SLOPE0)

south_pt = find_isobath_point(*south_tr)
north_pt = find_isobath_point(*north_tr)

# full 4-point dict, kept in the same order as plot_bottom_o2_timeseries.py
# for cache-index compatibility -- see module docstring
POINTS = {
    'p0':                       (948, 307),
    'p1':                       (478,676),
    'p2 (south transect, 50m)': south_pt,
    'p3 (north transect, 50m)': north_pt,
}
PLOT_POINTS = ['p1', 'p2 (south transect, 50m)']

for name, (j, i) in POINTS.items():
    print(f'{name}: [eta={j}, xi={i}]  h={h[j, i]:.1f} m')

# ---------------------------------------------------------------------------
# File listing (handles flat vs bgc/ subdir layout, per plot_cs_diag*.py)
# ---------------------------------------------------------------------------
def src_glob(root):
    sub  = os.path.join(root, 'bgc')
    base = sub if os.path.isdir(sub) else root
    return os.path.join(base, 'mc60_bgc.*.nc')


def clean(arr):
    arr = np.array(arr)
    return np.where(np.abs(arr) > 1e30, np.nan, arr)


# ---------------------------------------------------------------------------
# Extract bottom O2 time series for one scenario
# ---------------------------------------------------------------------------
def compute_series(scen):
    root  = SCENARIOS[scen]
    files = sorted(glob.glob(src_glob(root)))
    print(f'  [{scen}] {len(files)} bgc files', flush=True)

    times_list = []
    series = {name: [] for name in POINTS}

    for f in files:
        with Dataset(f) as nc:
            ocean_time = np.array(nc.variables['ocean_time'][:])
            times_list.extend(mdates.date2num(
                num2date(ocean_time, 'seconds since 1995-01-01',
                          only_use_cftime_datetimes=False)))
            for name, (j, i) in POINTS.items():
                series[name].extend(clean(nc.variables['O2'][:, 0, j, i]))

    times = np.array(times_list)
    idx   = np.argsort(times)
    times = times[idx]
    for name in POINTS:
        series[name] = np.array(series[name])[idx]

    return times, series


# ---------------------------------------------------------------------------
# Cache + main
# ---------------------------------------------------------------------------
os.makedirs(SAVEPATH, exist_ok=True)

def cache_path(scen):
    return f'{SAVEPATH}bottom_o2_timeseries_cache_{scen}.npz'

def load_cache(scen):
    cp = cache_path(scen)
    if not os.path.exists(cp):
        return None
    cache = np.load(cp, allow_pickle=False)
    times = cache['times']
    series = {name: cache[f'series_{ii}'] for ii, name in enumerate(POINTS)}
    return times, series

def save_cache(scen, times, series):
    cp = cache_path(scen)
    cache_data = {'times': times}
    for ii, name in enumerate(POINTS):
        cache_data[f'series_{ii}'] = series[name]
    np.savez(cp, **cache_data)
    print(f'  saved cache -> {cp}', flush=True)

all_series = {}
for scen in SCENARIOS:
    cached = load_cache(scen)
    if cached is not None:
        print(f'[{scen}] loading cache...', flush=True)
        all_series[scen] = cached
    else:
        print(f'[{scen}] computing...', flush=True)
        times, series = compute_series(scen)
        all_series[scen] = (times, series)
        save_cache(scen, times, series)

# ---------------------------------------------------------------------------
# Constrain every scenario to the shared time window all scenarios have data
# for -- same fix/reasoning as plot_bottom_o2_timeseries.py.
# ---------------------------------------------------------------------------
common_start = max(times[0]  for times, _ in all_series.values())
common_end   = min(times[-1] for times, _ in all_series.values())
print(f'common time window: {mdates.num2date(common_start)} to {mdates.num2date(common_end)}')

for scen in SCENARIOS:
    times, series = all_series[scen]
    keep = (times >= common_start) & (times <= common_end)
    if not keep.all():
        print(f'  [{scen}] trimming {(~keep).sum()}/{len(times)} timestep(s) '
              f'outside the common window', flush=True)
    all_series[scen] = (times[keep], {name: arr[keep] for name, arr in series.items()})

# ---------------------------------------------------------------------------
# Plot — one panel per point (p1, p2), all scenarios overlaid
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(PLOT_POINTS), 1, figsize=(12, 3 * len(PLOT_POINTS)), sharex=True)

for ax, name in zip(axes, PLOT_POINTS):
    j, i = POINTS[name]
    for scen in SCENARIOS:
        times, series = all_series[scen]
        ax.plot(times, series[name], color=COLORS[scen], linestyle=LSTYLES[scen],
                 label=LABELS[scen], linewidth=ss.lw(scen, base_lw=1.2))
    ax.set_ylabel(r'bottom O$_2$ (mmol m$^{-3}$)')
    ax.set_title(f'{name}  [eta={j}, xi={i}]  h={h[j, i]:.1f} m')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))

axes[0].legend(loc='best', fontsize=9, ncol=len(SCENARIOS))
axes[-1].set_xlabel('Date')
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.tight_layout()
out = f'{SAVEPATH}bottom_o2_timeseries_p1.png'
plt.savefig(out, dpi=800, bbox_inches='tight')
plt.close(fig)
print(f'saved -> {out}')
