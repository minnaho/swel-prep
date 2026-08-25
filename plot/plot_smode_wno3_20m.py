"""
Parent (smode200) vs child (mc60) w'NO3' time series at 20 m.

The smode200 parent solution (tides / notides, no WEC) is nested around the
mc60 domain. This overlays the parent's raw w*NO3 and anomaly w'NO3' on the
same axes as the four mc60 child scenarios, restricted to the mc60 footprint
so the two solutions describe the same area.

Averaging-window note: the child's w'NO3' (calc_wno3_flux_20m.py) removes a
single time mean over its ~10-day record. The parent covers a full month, so
a single 31-day mean would leave event-scale (3-10 day) variability inside
w'/NO3' that the child's mean absorbs -- inflating the parent's w'NO3' for
window-length reasons, not physics. Instead the parent's anomaly is taken
about a centered MEAN_WINDOW-record running mean (clamped, not shrunk, at
the series ends) so the averaging length is always ~10 days, matching the
child's definition throughout the month.

Both quantities are computed at native hourly resolution (the running mean
needs the full hourly record to define its 10-day window) and only
daily-averaged at plot time, via daily_mean() below.
"""
import sys
sys.path.append('/data/project3/minnaho/global/')
import os
import glob
import datetime
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.path import Path
import pyfuncs as pf
import scenario_style as ss

SMODE_ROOT  = '/data/project3/minnaho/swel/smode'
SMODE_GRD   = '../smode200_grd.nc'
MC60_GRD    = '../mc60_grd.nc'

DEPTH_IDX   = 1        # depth[1] == -20.0 m
DEPTH_LABEL = '20 m'
MEAN_WINDOW = 241       # centered running-mean length, hourly records (~10 d)
MIN_H       = None      # e.g. 20. to drop cells shallower than the level
FORCE       = False      # recompute the parent cache even if the npz exists

CACHE   = 'smode_wno3_20m_cache.npz'
OUTFIG  = './figs/smode_wno3_20m.png'

PARENT_SCENS = {'tides': 'parent_tides', 'notides': 'parent_notides'}
CHILD_SCENS  = ['notidesnowec', 'ampwec', 'tidesnowec', 'tidesampwec']
CHILD_NPZ    = lambda n: f'../postprocessing/wno3_flux_20m_{n}.npz'

REF_TIME = 'seconds since 1995-01-01'


def build_selection():
    """mc60-footprint polygon test against the smode200 grid -> boolean
    selection + enclosing index window, restricted to water cells."""
    mc = Dataset(MC60_GRD)
    lon_c = np.array(mc.variables['lon_rho'][:])
    lat_c = np.array(mc.variables['lat_rho'][:])
    perim = np.concatenate([
        np.stack([lon_c[0, :],    lat_c[0, :]],    axis=1),
        np.stack([lon_c[:, -1],   lat_c[:, -1]],   axis=1),
        np.stack([lon_c[-1, ::-1], lat_c[-1, ::-1]], axis=1),
        np.stack([lon_c[::-1, 0], lat_c[::-1, 0]], axis=1),
    ])
    poly = Path(perim)

    sm = Dataset(SMODE_GRD)
    lon_p = np.array(sm.variables['lon_rho'][:])
    lat_p = np.array(sm.variables['lat_rho'][:])
    mask_p = np.array(sm.variables['mask_rho'][:])
    h_p = np.array(sm.variables['h'][:]) if MIN_H is not None else None

    inside = poly.contains_points(
        np.stack([lon_p.ravel(), lat_p.ravel()], axis=1)
    ).reshape(lon_p.shape)

    idx = np.where(inside)
    e0, e1 = idx[0].min(), idx[0].max() + 1
    x0, x1 = idx[1].min(), idx[1].max() + 1

    sel = inside[e0:e1, x0:x1] & (mask_p[e0:e1, x0:x1] == 1)
    if MIN_H is not None:
        sel &= (h_p[e0:e1, x0:x1] >= MIN_H)

    print(f'  selection window: eta {e0}:{e1}, xi {x0}:{x1}  '
          f'({e1 - e0} x {x1 - x0})')
    print(f'  water cells selected: {sel.sum()}')

    return e0, e1, x0, x1, sel


def load_parent(scen, e0, e1, x0, x1, sel):
    files = sorted(glob.glob(os.path.join(
        SMODE_ROOT, scen, 'smode_zsc_w_no3_10_20m.*.nc')))
    print(f'  {len(files)} files found for {scen}')

    w_list, no3_list, ocean_times = [], [], []
    for f in files:
        nc = Dataset(f)
        w   = np.array(nc.variables['w'][:, DEPTH_IDX, e0:e1, x0:x1],
                        dtype=np.float32)[:, sel]
        no3 = np.array(nc.variables['NO3'][:, DEPTH_IDX, e0:e1, x0:x1],
                        dtype=np.float32)[:, sel]
        w_list.append(w)
        no3_list.append(no3)
        ocean_times.extend(np.array(nc.variables['ocean_time'][:]).tolist())

    w_all   = np.concatenate(w_list,   axis=0)  # (nt, npts)
    no3_all = np.concatenate(no3_list, axis=0)
    del w_list, no3_list

    return w_all, no3_all, np.array(ocean_times)


def running_mean(a, win):
    """Centered running mean along axis 0, clamped (not shrunk) at the ends
    so every record is averaged over exactly `win` records."""
    nt = a.shape[0]
    half = win // 2
    csum = np.concatenate([np.zeros((1,) + a.shape[1:], dtype=np.float64),
                            np.cumsum(a, axis=0, dtype=np.float64)], axis=0)
    out = np.empty(a.shape, dtype=np.float64)
    for t in range(nt):
        lo = t - half
        hi = t + half + 1
        if lo < 0:
            lo, hi = 0, win
        elif hi > nt:
            lo, hi = nt - win, nt
        out[t] = (csum[hi] - csum[lo]) / win
    return out


def compute_parent(scen, e0, e1, x0, x1, sel):
    print(f'Processing parent/{scen}...')
    w_all, no3_all, ocean_times = load_parent(scen, e0, e1, x0, x1, sel)

    raw_ts = (w_all * no3_all).mean(axis=1)

    w_run   = running_mean(w_all,   MEAN_WINDOW)
    no3_run = running_mean(no3_all, MEAN_WINDOW)
    anom_ts = ((w_all - w_run) * (no3_all - no3_run)).mean(axis=1)
    del w_all, no3_all, w_run, no3_run

    return dict(raw_ts=raw_ts, anom_ts=anom_ts, ocean_times=ocean_times)


def get_or_compute_parent():
    if os.path.exists(CACHE) and not FORCE:
        print(f'Loading cached parent results from {CACHE}')
        d = np.load(CACHE, allow_pickle=True)
        return {scen: dict(raw_ts=d[f'{scen}_raw_ts'],
                            anom_ts=d[f'{scen}_anom_ts'],
                            ocean_times=d[f'{scen}_ocean_times'])
                for scen in PARENT_SCENS}

    e0, e1, x0, x1, sel = build_selection()
    results = {scen: compute_parent(scen, e0, e1, x0, x1, sel)
               for scen in PARENT_SCENS}

    save = {}
    for scen, r in results.items():
        save[f'{scen}_raw_ts']      = r['raw_ts']
        save[f'{scen}_anom_ts']     = r['anom_ts']
        save[f'{scen}_ocean_times'] = r['ocean_times']
    np.savez(CACHE, **save)
    print(f'  saved -> {CACHE}')
    return results


def daily_mean(times_dt, values):
    """Group hourly records into calendar-day (UTC) bins and average.
    Partial first/last days (the child series does not start/end on a day
    boundary) are averaged over however many hourly records fall in them.
    Returns (day_centers, daily_values); day_centers are placed at noon."""
    dates = np.array([d.date() for d in times_dt])
    uniq = np.unique(dates)
    centers = np.array([datetime.datetime(d.year, d.month, d.day, 12)
                         for d in uniq])
    means = np.array([values[dates == d].mean() for d in uniq])
    return centers, means


def load_child(name):
    path = CHILD_NPZ(name)
    if not os.path.exists(path):
        print(f'  WARNING: {path} not found -- run calc_wno3_flux_20m.py first. '
              f'Skipping {name}.')
        return None
    d = np.load(path, allow_pickle=True)
    return d


parent = get_or_compute_parent()

fig, (ax_raw, ax_anom) = plt.subplots(2, 1, figsize=[14, 12])

# --- parent --- (daily mean of the hourly raw/anomaly series)
for scen, style_key in PARENT_SCENS.items():
    r = parent[scen]
    t = pf.numdate(r['ocean_times'], REF_TIME)
    t_d, raw_d  = daily_mean(t, r['raw_ts'])
    _,   anom_d = daily_mean(t, r['anom_ts'])
    kw = ss.line_kwargs(style_key, base_lw=1.5)
    ax_raw.plot(t_d, raw_d, **kw)
    ax_anom.plot(t_d, anom_d, **kw)

# --- child --- (daily mean of the hourly raw/anomaly series)
child_t0, child_t1 = None, None
for name in CHILD_SCENS:
    d = load_child(name)
    if d is None:
        continue
    t = pf.numdate(d['ocean_times'][12:], REF_TIME)
    t_d, raw_d  = daily_mean(t, d['raw_time_series'][12:])
    _,   anom_d = daily_mean(t, d['time_series'][12:])
    kw = ss.line_kwargs(name, base_lw=1.5)
    ax_raw.plot(t_d, raw_d, **kw)
    ax_anom.plot(t_d, anom_d, **kw)
    if child_t0 is None:
        child_t0, child_t1 = t_d[0], t_d[-1]
    else:
        child_t0, child_t1 = min(child_t0, t_d[0]), max(child_t1, t_d[-1])

axfont = 14
if child_t0 is not None:
    for ax in (ax_raw, ax_anom):
        ax.axvspan(child_t0, child_t1, color='k', alpha=0.05, zorder=0)

ax_raw.axhline(0, color='k', linewidth=0.5)
ax_raw.set_ylabel(f"$w \\cdot NO_3$ ({DEPTH_LABEL})\n"
                   r'(mmol N m$^{-2}$ s$^{-1}$)', fontsize=axfont)
ax_raw.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax_raw.tick_params(axis='both', which='major', labelsize=axfont)
ax_raw.legend(fontsize=16, loc='best', ncol=2)

ax_anom.axhline(0, color='k', linewidth=0.5)
ax_anom.set_ylabel(f"$w'NO_3'$ ({DEPTH_LABEL})\n"
                    r'(mmol N m$^{-2}$ s$^{-1}$)', fontsize=axfont)
ax_anom.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax_anom.tick_params(axis='both', which='major', labelsize=axfont)

plt.tight_layout()
plt.savefig(OUTFIG, bbox_inches='tight', dpi=800)
plt.close(fig)
print(f'saved {OUTFIG}')
