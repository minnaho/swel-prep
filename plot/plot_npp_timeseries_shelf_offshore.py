"""
Time series of vertically-integrated NPP (TOT_PROD), horizontally averaged
over the h<=100m (shelf) and h>100m (offshore) regions, for all 6 scenarios.
Sibling of plot_npp_depth_integrated.py (same zslice dia source, same
vertical-integration method: sum over the uniform 2m z-grid * DZ * 86400 to
convert mmol/m3/s -> mmol/m2/d) -- that script collapses to one time-mean
map per scenario; this one keeps the time axis and collapses space instead
(h<=100m mean vs. h>100m mean).

Each zsliced dia_avg file has no time dimension of its own (already
time-averaged over its write window). notideswec/ampwec/tidesampwec write
two 12h-window files/day; tideswec/tidesnowec/notidesnowec write one
24h-window file/day. NPP is strongly diurnal (photosynthesis only in
daylight) -- plotting each file's raw value at its own timestamp would give
the 12h-window scenarios a spurious sawtooth (a documented ~4x swing between
a day-half and night-half window in this same zslice product, see
plot_npp_depth_integrated.py's diel-averaging-bug fix) that looks like real
day-to-day variability but is actually just an artifact of window width, not
comparable to the 24h-window scenarios' already-smoothed daily values.

**Fix**: files are grouped by their own filename's calendar date and
averaged (equal-weighted, since same-day 12h windows are equal duration and
non-overlapping so their mean is a true unweighted 24h daily mean) before
plotting, so every scenario ends up on the same one-point-per-day cadence.
This is per-scenario date-grouping only (no cross-scenario date matching --
that was the fragile approach explicitly rejected for
plot_npp_depth_integrated.py, and isn't needed here since each scenario's
own files are grouped independently).

Output: ./figs/npp_timeseries_shelf_offshore.png
"""

import os
import re
import glob
import datetime as dt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from netCDF4 import Dataset
import scenario_style as ss

ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'
GRD         = 'mc60_grd.nc'
CACHE       = './figs/npp_timeseries_shelf_offshore_cache.npz'
DZ          = 2.0   # m — uniform dia z-grid (0 to -200 m every 2 m)

SCENARIOS = ['tideswec', 'tidesnowec', 'notidesnowec', 'notideswec', 'ampwec', 'tidesampwec']
SCEN_DIRS = {'ampwec': 'notidesampwec'}   # label -> zslice subdirectory
LABELS  = ss.LABELS
COLORS  = ss.COLORS
LSTYLES = ss.LSTYLES

FNAME_RE = re.compile(r'z_mc60_bgc_dia_avg\.(\d{14})\.nc$')


def _fill_to_nan(arr):
    return np.where(np.abs(arr) > 1e30, np.nan, arr)


def file_time(fname):
    m = FNAME_RE.search(fname)
    return dt.datetime.strptime(m.group(1), '%Y%m%d%H%M%S')


def list_zfiles(scen):
    sd = SCEN_DIRS.get(scen, scen)
    return sorted(glob.glob(f'{ZSLICE_ROOT}/{sd}/dia/z_mc60_bgc_dia_avg.*.nc'))


# ---------------------------------------------------------------------------
# Shelf / offshore masks — same h<=100m / h>100m convention as
# profile_zslice_par_100m.py / profile_zslice_par_offshore.py
# ---------------------------------------------------------------------------
grd_nc   = Dataset(GRD)
mask_rho = np.array(grd_nc['mask_rho']).astype(float)
h        = np.array(grd_nc['h'])
mask_rho[mask_rho == 0] = np.nan

mask_shelf = mask_rho.copy()
mask_shelf[h > 100] = np.nan

mask_offshore = mask_rho.copy()
mask_offshore[h <= 100] = np.nan


def compute_series(scen):
    """One point per calendar date -- files are grouped by their own
    filename's date and averaged, so a 2-file/day scenario's day-half +
    night-half 12h windows collapse to the same true 24h daily mean a
    1-file/day scenario already has natively (equal-weighted average is
    correct here since both windows are the same duration and partition
    the day). See module docstring for why this matters."""
    zfiles = list_zfiles(scen)
    day_shelf, day_offshore = {}, {}
    for zf in zfiles:
        t = file_time(os.path.basename(zf))
        date = t.date()
        with Dataset(zf) as nc:
            arr = _fill_to_nan(np.array(nc.variables['TOT_PROD'][:]))  # (z, eta, xi)
        di = np.nansum(arr, axis=0) * DZ * 86400.0   # (eta, xi), mmol/m3/s -> mmol/m2/d
        day_shelf.setdefault(date, []).append(np.nanmean(di * mask_shelf))
        day_offshore.setdefault(date, []).append(np.nanmean(di * mask_offshore))
        print(f'  [{scen}] {os.path.basename(zf)} -> {date}', flush=True)

    dates = sorted(day_shelf)
    # drop a trailing partial day (fewer files than this scenario's typical
    # per-day count) -- e.g. tidesampwec's last file is a lone 11:01 window
    # with no matching 23:01 file, so that date's "daily mean" is really just
    # a half-day value, not comparable to every other date's true 24h mean
    if len(dates) > 1:
        counts  = [len(day_shelf[d]) for d in dates]
        typical = max(set(counts), key=counts.count)
        if len(day_shelf[dates[-1]]) < typical:
            print(f'  [{scen}] dropping trailing partial day {dates[-1]} '
                  f'({len(day_shelf[dates[-1]])}/{typical} files)', flush=True)
            dates = dates[:-1]

    times         = np.array([dt.datetime.combine(d, dt.time(12, 0)) for d in dates])
    shelf_vals    = np.array([np.mean(day_shelf[d]) for d in dates])
    offshore_vals = np.array([np.mean(day_offshore[d]) for d in dates])
    return times, shelf_vals, offshore_vals


# ---------------------------------------------------------------------------
# Load / cache
# ---------------------------------------------------------------------------
cc = {}
have_all = False
if os.path.exists(CACHE):
    cc = dict(np.load(CACHE, allow_pickle=True))
    have_all = all(f'{scen}_times' in cc for scen in SCENARIOS)

if not have_all:
    cc = {}
    for scen in SCENARIOS:
        print(f'[{scen}] computing...', flush=True)
        times, shelf_vals, offshore_vals = compute_series(scen)
        cc[f'{scen}_times']    = np.array([t.isoformat() for t in times])
        cc[f'{scen}_shelf']    = shelf_vals
        cc[f'{scen}_offshore'] = offshore_vals
    np.savez(CACHE, **cc)
    print(f'saved cache -> {CACHE}')
else:
    print(f'loaded cache -> {CACHE}')

# ---------------------------------------------------------------------------
# Trim every scenario to the shared common end date. Some scenarios' zslice
# dia records extend a day past where the others stop -- e.g. tidesnowec and
# tidesampwec both have a trailing 2019-04-29 date the other 4 scenarios
# never reach (2019-04-28) -- so without this, one or two lines would dangle
# an extra point past the rest of the group instead of a fair, fully-
# overlapping comparison. This runs on `cc` regardless of whether it was
# just computed or loaded from cache, so it applies even to an
# already-cached run without forcing a recompute.
# ---------------------------------------------------------------------------
last_dates = [np.array([dt.datetime.fromisoformat(t) for t in cc[f'{scen}_times']])[-1]
              for scen in SCENARIOS if len(cc.get(f'{scen}_times', [])) > 0]
common_end = min(last_dates)
print(f'common end date across all scenarios: {common_end.date()}')

for scen in SCENARIOS:
    times = np.array([dt.datetime.fromisoformat(t) for t in cc[f'{scen}_times']])
    keep = times <= common_end
    if not keep.all():
        dropped = [t.date().isoformat() for t in times[~keep]]
        print(f'  [{scen}] trimming date(s) past common end: {dropped}')
    cc[f'{scen}_times']    = np.array([t.isoformat() for t in times[keep]])
    cc[f'{scen}_shelf']    = cc[f'{scen}_shelf'][keep]
    cc[f'{scen}_offshore'] = cc[f'{scen}_offshore'][keep]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.rcParams.update({'font.size': 12})
fig, (ax_shelf, ax_offshore) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for scen in SCENARIOS:
    times         = np.array([dt.datetime.fromisoformat(t) for t in cc[f'{scen}_times']])
    shelf_vals    = cc[f'{scen}_shelf']
    offshore_vals = cc[f'{scen}_offshore']
    lw = ss.lw(scen, base_lw=1.5)
    ax_shelf.plot(times, shelf_vals, marker='o', markersize=4,
                  color=COLORS[scen], linestyle=LSTYLES[scen], linewidth=lw,
                  label=LABELS[scen])
    ax_offshore.plot(times, offshore_vals, marker='o', markersize=4,
                      color=COLORS[scen], linestyle=LSTYLES[scen], linewidth=lw)

ax_shelf.set_ylabel(r'$\int$NPP (mmol m$^{-2}$ d$^{-1}$)')
ax_offshore.set_ylabel(r'$\int$NPP (mmol m$^{-2}$ d$^{-1}$)')
ax_shelf.set_title('h $\\leq$ 100 m (shelf)')
ax_offshore.set_title('h > 100 m (offshore)')
ax_shelf.grid(True, alpha=0.3)
ax_offshore.grid(True, alpha=0.3)
ax_shelf.legend(loc='best', fontsize=10)

ax_offshore.set_xlabel('Date')
ax_offshore.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax_offshore.xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.tight_layout()
out = './figs/npp_timeseries_shelf_offshore.png'
plt.savefig(out, bbox_inches='tight', dpi=800)
plt.close(fig)
print(f'saved -> {out}')
