"""
Diagonal cross-section(s) across a user-defined front, using native s_rho
ROMS output, at a single fixed instant: 2019-04-22 07:01, the same instant
as plot_cs_vorticity_snap.py / plot_vorticity_snap.py (matched the same
way -- by filename date-hour stamp, then by internal record hour==7; see
matched_file_and_index() below, copied from plot_cs_vorticity_snap.py's
function of the same name). Sibling of plot_cs_diag.py, reusing its
build_transect() (copied verbatim below -- these scripts are standalone,
not imported modules, matching this codebase's convention).

Restricted to the 4 main scenarios: notidesnowec, tidesnowec, ampwec (=
"notidesampwec" -- no tides, amplified WEC), tidesampwec.

Transects are defined PER SCENARIO, not shared across scenarios -- the
front's physical location/shape differs by scenario (this is why an
alongfront/alongshore average isn't meaningful for it: the front moves and
curves differently per run), so each scenario's transect(s) need their own
placement to actually cross that scenario's front. Edit TRANSECT_PARAMS
below: one list of (eta0, xi0, slope, depth_lim, length_xi) tuples per
scenario -- 3-4 per scenario is the expected case, but any number
(including zero) is supported. Same first-4-field meaning as plot_cs_diag.py's
ETA0/XI0/SLOPE/DEPTH_LIM0, plus an explicit length_xi (plot_cs_diag.py
instead shares one fixed LENGTH_XI across every transect):
    eta0, xi0 : starting grid index (coast end) -- (j, i) in ROMS
                convention, j=eta, i=xi
    slope     : deta/dxi -- 0 = purely zonal, + = eta decreases offshore,
                - = eta increases offshore
    depth_lim : y-axis bottom (m) for that transect's panel
    length_xi : transect length in xi cells (build_transect always
                traverses in the -xi direction from xi0)
Given a (j,i) start/end pair instead: length_xi = i_start - i_end, slope =
(j_start - j_end) / length_xi (see the ampwec entries below for an example).

Two stages, both per scenario, run for every scenario that has transects:

  Stage 1 -- a locator map: surface temperature with every one of that
  scenario's transect lines drawn on top (same line/label convention the
  old interactive preview used), so transect placement can be checked
  against the actual front without needing a display. Written first, before
  any cross-section is computed.
      Output: ./figs/snapshots/front_vars/cs_front_map_<scenario>.png

  Stage 2 -- for each of 7 variables (temperature, normalized vorticity
  zeta/f -- correctly rotated into true east/north before differencing,
  see compute_vorticity() --, NO3, w, Akt, Akv, and total phytoplankton
  carbon SPC+DIATC+DIAZC), one figure per (scenario, variable) stacking
  ALL of that scenario's transects as rows, sharing one colorbar.
      Output: ./figs/snapshots/front_vars/cs_front_vars_<scenario>_<var>.png

Every extracted cross-section (and the locator-map SST field) is cached to
npz under ./figs/snapshots/front_vars/cache/ -- vorticity in particular
requires a full 3D u/v load + rotation + curl, so caching makes replotting
(colormap/range/contour tweaks) near-instant. Section caches store the
transect geometry they were built from ([eta0, xi0, slope, length_xi]); a
mismatch against the live TRANSECT_PARAMS entry forces a recompute, since
transect coordinates are still being tuned scenario by scenario.

Headless by default (matplotlib.use('Agg') at import) so this script can run
in screen / via run_plots.py. Pass --preview to instead pop up the old
interactive transect-placement window (Enter to proceed, Esc to cancel)
before plotting -- needs a working DISPLAY.
"""

import os
import sys
sys.path.append('/data/project3/minnaho/global/')
import argparse
import glob
import numpy as np
import matplotlib


def parse_args():
    p = argparse.ArgumentParser(
        description='Front-transect locator maps + 7-variable cross-section '
                     'panels, all transects per scenario, one fixed instant.')
    p.add_argument('--preview', action='store_true',
                    help='Show an interactive transect-placement preview '
                         '(needs DISPLAY) before plotting. Off by default.')
    return p.parse_args()


ARGS = parse_args()

if not ARGS.preview:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.colors import LogNorm
import cmocean

plt.rcParams.update({'font.size': 14})
from netCDF4 import Dataset, num2date
from scipy.ndimage import map_coordinates
import ROMS_depths as depths
import pyfuncs as pf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCENARIOS = {
    'notidesnowec': '/data/project3/minnaho/swel/notides/mc60/nowec',
    'tidesnowec':   '/data/project3/minnaho/swel/tides/mc60/nowec/output',
    'ampwec':       '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything',
    'tidesampwec':  '/data/project3/minnaho/swel/tides/mc60/ampwec/everything',
}

LABELS = {
    'notidesnowec': 'no tides, no WEC',
    'tidesnowec':   'tides, no WEC',
    'ampwec':       'no tides, amplified WEC',
    'tidesampwec':  'tides, amplified WEC',
}

# Surface temperature range for the stage-2 'temp' cross-sections -- no
# existing cross-section script plots temp with a fixed range (it's
# normally only read to build sigma-t), so this brackets the 1st/99th
# percentile of the upper water column at the target instant.
TEMP_VMIN = 10
TEMP_VMAX = 16

# Surface temperature range for the stage-1 locator maps (separate from the
# cross-section range above), plus the map's axis limits and legend corner.
MAP_TEMP_VMIN = 12
MAP_TEMP_VMAX = 16
MAP_YMAX = 37.06
MAP_XMAX = -121.77

# Per-variable config for the stage-2 cross-sections. 'src' is 'his', 'bgc',
# or 'vort' (derived, see compute_vorticity()); 'varname' is the netCDF
# variable name (a list for phytoC, summed); 'vert' is 'rho' (s_rho, paired
# with zr3d) or 'w' (s_w -- Akt/Akv live on cell interfaces, paired with
# zw3d instead -- see plot_cs_diag_akt_box.py's docstring for why
# interp_section needs no change for this: it keys off field3d.shape[0]).
# cmap/range values are copied from the matching single-variable snapshot
# scripts (plot_cs_diag_{no3,akt,akv,totc}_box.py, plot_cs_vorticity_snap.py)
# rather than any time-mean diff script, since these are single-instant
# figures.
VAR_CONFIGS = {
    'temp': dict(
        src='his', varname='temp', vert='rho',
        cmap=cmocean.cm.thermal, vmin=TEMP_VMIN, vmax=TEMP_VMAX,
        label='Temperature (°C)',
    ),
    'vort': dict(
        src='vort', varname=None, vert='rho',
        cmap=cmocean.cm.balance, vmin=-5, vmax=5,
        label=r'$\zeta/f$',
    ),
    'NO3': dict(
        src='bgc', varname='NO3', vert='rho',
        cmap=cmocean.cm.matter, vmin=0, vmax=25,
        label=r'NO$_3$ (mmol m$^{-3}$)',
    ),
    'w': dict(
        src='his', varname='w', vert='rho',
        cmap=cmocean.cm.balance, vmin=-1e-2, vmax=1e-2,
        label=r'w (m s$^{-1}$)',
    ),
    'Akt': dict(
        src='his', varname='Akt', vert='w',
        cmap='viridis', norm=LogNorm(vmin=1e-5, vmax=1e-1),
        label=r'$K_t$ (m$^2$ s$^{-1}$)',
    ),
    'Akv': dict(
        src='his', varname='Akv', vert='w',
        cmap='viridis', norm=LogNorm(vmin=1e-5, vmax=1e-1),
        label=r'$K_v$ (m$^2$ s$^{-1}$)',
    ),
    'phytoC': dict(
        src='bgc', varname=['DIATC', 'DIAZC', 'SPC'], vert='rho',
        cmap=cmocean.cm.algae, vmin=0, vmax=40,
        label=r'Total phyto C (mmol C m$^{-3}$)',
    ),
}

# Isopycnal contour overlay (sigma-t levels)
RHO_REF_NC  = 1027.4             # ROMS reference density
ISO_RHO_OFF = RHO_REF_NC - 1000  # = 27.4: stored rho + offset → sigma-t
ISO_LEVELS  = list(np.arange(24, 28 + 0.001, 0.25))

# Transect geometry (grid index space), PER SCENARIO -- (eta0, xi0, slope,
# depth_lim, length_xi). eta0/xi0 = start; slope = deta/dxi; depth_lim =
# y-axis bottom (m); length_xi = transect length in xi cells (build_transect
# always traverses in the -xi direction from xi0 -- for a start/end pair
# where i increases from start to end, slope and length_xi both come out
# negative, which build_transect handles fine, it just walks toward
# increasing xi). All four scenarios below now have user-specified
# (j=eta, i=xi) start/end pairs.
N_PTS = 300   # interpolation resolution along transect (all)

TRANSECT_PARAMS = {
    # User-specified (j=eta, i=xi) start/end pairs -- slope and length_xi
    # derived from them (slope = (j_start-j_end)/(i_start-i_end),
    # length_xi = i_start-i_end); depth_lim defaults to -125 since none was
    # given -- adjust per transect if -125 doesn't suit its local bathymetry.
    'notidesnowec': [
        # t0: (j=852,i=394) -> (j=852,i=200)
        (852, 394, (852 - 852) / (394 - 200), -125, 394 - 200),
        # t1: (j=765,i=499) -> (j=765,i=200)
        (765, 499, (765 - 765) / (499 - 200), -125, 499 - 200),
        # t2: (j=672,i=255) -> (j=706,i=161)
        (672, 255, (672 - 706) / (255 - 161), -125, 255 - 161),
        # t3: (j=544,i=261) -> (j=554,i=165)
        (544, 261, (544 - 554) / (261 - 165), -125, 261 - 165),
    ],
    # User-specified (j=eta, i=xi) start/end pairs -- slope and length_xi
    # derived from them (slope = (j_start-j_end)/(i_start-i_end),
    # length_xi = i_start-i_end, both negative here since i increases from
    # start to end -- build_transect's -xi traversal handles this fine, it
    # just walks toward increasing xi); depth_lim defaults to -125 since
    # none was given -- adjust per transect if -125 doesn't suit its local
    # bathymetry.
    'tidesnowec': [
        # t0: (j=932,i=211) -> (j=808,i=407)
        (932, 211, (932 - 808) / (211 - 407), -125, 211 - 407),
        # t1: (j=824,i=79) -> (j=724,i=179)
        (824, 79, (824 - 724) / (79 - 179), -125, 79 - 179),
        # t2: (j=536,i=39) -> (j=500,i=119)
        (536, 39, (536 - 500) / (39 - 119), -125, 39 - 119),
    ],
    # ampwec = "notidesampwec" (no tides, amplified WEC) raw-file key.
    # User-specified (j=eta, i=xi) start/end pairs -- slope and length_xi
    # derived from them (slope = (j_start-j_end)/(i_start-i_end),
    # length_xi = i_start-i_end); depth_lim defaults to -125 (same as the
    # other scenarios' t0) since none was given -- adjust per transect if
    # -125 doesn't suit its local bathymetry.
    'ampwec': [
        # t0: (j=705,i=454) -> (j=705,i=286)
        (705, 454, (705 - 705) / (454 - 286), -125, 454 - 286),
        # t1: (j=525,i=364) -> (j=669,i=238)
        (525, 364, (525 - 669) / (364 - 238), -125, 364 - 238),
        # t2: (j=450,i=259) -> (j=579,i=130)
        (450, 259, (450 - 579) / (259 - 130), -125, 259 - 130),
    ],
    # User-specified (j=eta, i=xi) start/end pairs -- slope and length_xi
    # derived from them (slope = (j_start-j_end)/(i_start-i_end),
    # length_xi = i_start-i_end); depth_lim defaults to -125 since none was
    # given -- adjust per transect if -125 doesn't suit its local bathymetry.
    'tidesampwec': [
        # t0: (j=812,i=233) -> (j=812,i=497)
        (812, 233, (812 - 812) / (233 - 497), -125, 233 - 497),
        # t1: (j=740,i=211) -> (j=604,i=405)
        (740, 211, (740 - 604) / (211 - 405), -125, 211 - 405),
        # t2: (j=660,i=119) -> (j=520,i=169)
        (660, 119, (660 - 520) / (119 - 169), -125, 119 - 169),
    ],
}

GRD      = 'mc60_grd.nc'
SAVEPATH = './figs/snapshots/'
FRONTVARS_DIR = f'{SAVEPATH}front_vars/'
CACHE_DIR     = f'{FRONTVARS_DIR}cache/'

# ---------------------------------------------------------------------------
# Load grid
# ---------------------------------------------------------------------------
grdnc    = Dataset(GRD, 'r')
lat      = np.array(grdnc['lat_rho'][:])
lon      = np.array(grdnc['lon_rho'][:]) - 360
mask_rho = np.array(grdnc['mask_rho'][:])
f_nc     = np.array(grdnc['f'][:])

# land mask for variable fields
mask_plot = mask_rho.astype(float)
mask_plot[mask_plot == 0] = np.nan


def clean(arr):
    arr = np.array(arr)
    return np.where(np.abs(arr) > 1e30, np.nan, arr)


# ---------------------------------------------------------------------------
# Build transect coordinates (goes in -xi direction) -- based on
# plot_cs_diag.py's build_transect(), generalized to take a per-transect
# length_xi (plot_cs_diag.py's version always used one shared module-level
# LENGTH_XI) since user-specified start/end points don't all span the same
# xi distance.
# ---------------------------------------------------------------------------
def build_transect(eta0, xi0, slope, depth_lim, length_xi):
    xi  = np.linspace(xi0,  xi0  - length_xi, N_PTS)
    eta = np.linspace(eta0, eta0 + slope * (-length_xi), N_PTS)
    crd = np.array([eta, xi])
    return dict(
        mode      = 'diag',
        coords    = crd,
        lon       = map_coordinates(lon, crd, order=1, mode='nearest'),
        lat       = map_coordinates(lat, crd, order=1, mode='nearest'),
        mask      = map_coordinates(mask_rho.astype(float), crd,
                                    order=0, mode='nearest') > 0.5,
        depth_lim = depth_lim,
        eta0=eta0, xi0=xi0, slope=slope, length_xi=length_xi,
    )

# name -> list of transect dicts
TRANSECTS = {name: [build_transect(*p) for p in TRANSECT_PARAMS.get(name, [])]
             for name in SCENARIOS}
# name -> list of transect names ('t0', 't1', ...)
TRANSECT_NAMES = {name: [f't{i}' for i in range(len(trs))]
                   for name, trs in TRANSECTS.items()}


def transect_geom(name, ti):
    eta0, xi0, slope, depth_lim, length_xi = TRANSECT_PARAMS[name][ti]
    return np.array([eta0, xi0, slope, length_xi], dtype=float)

# ---------------------------------------------------------------------------
# Transect preview (--preview only) — one 2x2 panel per scenario, press
# Enter to proceed, Esc to cancel.
# ---------------------------------------------------------------------------
def _preview_transects():
    confirmed = [False]
    cmap = plt.get_cmap('tab10')

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, name in zip(axes.flat, SCENARIOS):
        ax.pcolormesh(lon, lat, mask_rho, cmap='gray_r', vmin=0, vmax=1)
        trs = TRANSECTS[name]
        colors = [cmap(i % 10) for i in range(len(trs))]
        for i, tr in enumerate(trs):
            lbl = (f'{TRANSECT_NAMES[name][i]}  (η={tr["eta0"]}, '
                   f'ξ={tr["xi0"]}, slope={tr["slope"]})')
            ax.plot(tr['lon'], tr['lat'], '-', color=colors[i], linewidth=2, label=lbl)
            ax.plot(tr['lon'][0], tr['lat'][0], 'o', color=colors[i], markersize=7)
        ax.set_title(f'{name}  ({len(trs)} transect(s))')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        if trs:
            ax.legend(loc='upper left', fontsize=8)

    fig.suptitle(
        'Transect preview — length_xi varies per transect (see legend)\n'
        'Press  Enter  to proceed  |  Esc  to cancel', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    def on_key(event):
        if event.key == 'enter':
            confirmed[0] = True
            plt.close(fig)
        elif event.key == 'escape':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if not confirmed[0]:
        print('Cancelled.')
        sys.exit(0)

if ARGS.preview:
    _preview_transects()
    matplotlib.use('Agg')

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


def interp_section(field3d, coords, mask_t):
    """Interpolate a 3D (s_rho or s_w, eta, xi) field along a transect.
    Returns (n_z, N_PTS)."""
    n_z = field3d.shape[0]
    out = np.full((n_z, N_PTS), np.nan)
    for iz in range(n_z):
        out[iz] = interp_transect(field3d[iz], coords, mask_t)
    return out


def compute_vorticity(his_f, t_idx):
    """Relative vorticity normalized by f, correctly rotated: u/v are
    averaged to rho points then rotated into true east/north using the
    grid's angle (mc60 is rotated ~25 deg) before differencing -- see
    plot_cs_vorticity_snap.py. Uses the _tind variant (single timestep);
    the full-record rho_uv_angle loads/rotates every timestep in the file
    and OOM-killed at ~42GB RSS for a single scenario."""
    urho, vrho = pf.rho_uv_angle_tind(his_f, GRD, t_idx, rotate=True)   # (s_rho, eta, xi)
    vort3d = pf.vorticity(GRD, urho[np.newaxis], vrho[np.newaxis])[0] / f_nc
    return vort3d * mask_plot

# ---------------------------------------------------------------------------
# Match the single target instant -- same convention as
# plot_cs_vorticity_snap.py's matched_file_and_index(): filename date-hour
# stamp narrows to one file, then the internal record with hour==TARGET_HOUR
# picks the exact timestep within it.
# ---------------------------------------------------------------------------
TARGET_DATE_HOUR = '2019042123'   # filename date-hour prefix
TARGET_HOUR      = 7              # internal record's hour-of-day to select

def src_dir(root, kind):
    """Handles flat vs his//bgc/ subdir layout (same as plot_cs_diag.py)."""
    sub = os.path.join(root, kind)
    return sub if os.path.isdir(sub) else root

def matched_file_and_index(directory, stem):
    pattern = os.path.join(directory, f'{stem}.{TARGET_DATE_HOUR}*.nc')
    files = sorted(glob.glob(pattern))
    assert len(files) == 1, f'expected 1 file matching {pattern}, found {len(files)}'
    f = files[0]
    with Dataset(f, 'r') as nc:
        ocean_time = np.array(nc['ocean_time'][:])
    oceandt = num2date(ocean_time, 'seconds since 1995-01-01',
                       only_use_cftime_datetimes=False)
    matches = [i for i, dt in enumerate(oceandt) if dt.hour == TARGET_HOUR]
    assert len(matches) == 1, (f'expected 1 record with hour=={TARGET_HOUR} '
                               f'in {f}, found {len(matches)}')
    t_idx = matches[0]
    return f, t_idx, oceandt[t_idx]

# ---------------------------------------------------------------------------
# Cache helpers -- section caches store the transect geometry they were
# built from; a mismatch against the live TRANSECT_PARAMS entry (checked by
# load_if_fresh) forces a recompute, since transect coordinates are still
# being tuned.
# ---------------------------------------------------------------------------
def map_cache_path(name):
    return f'{CACHE_DIR}cs_front_map_{name}.npz'

def sec_cache_path(name, var, ti):
    return f'{CACHE_DIR}cs_front_vars_{name}_{var}_t{ti}.npz'

def load_if_fresh(path, geom):
    if not os.path.exists(path):
        return None
    d = np.load(path)
    if 'geom' not in d or 'rho_zr' not in d or not np.allclose(d['geom'], geom):
        return None
    return d

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_locator_map(name, trs, sst, dt0):
    fig, ax = plt.subplots(figsize=(9, 9))
    pc = ax.pcolormesh(lon, lat, sst, cmap=cmocean.cm.thermal,
                       vmin=MAP_TEMP_VMIN, vmax=MAP_TEMP_VMAX, shading='auto')
    fig.colorbar(pc, ax=ax, label='Surface temperature (°C)')

    cmap_lines = plt.get_cmap('tab10')
    for i, tr in enumerate(trs):
        color = cmap_lines(i % 10)
        lbl = TRANSECT_NAMES[name][i]
        ax.plot(tr['lon'], tr['lat'], '-', color=color, linewidth=2, label=lbl)
        ax.plot(tr['lon'][0], tr['lat'][0], 'o', color=color, markersize=7)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_ylim(top=MAP_YMAX)
    ax.set_xlim(right=MAP_XMAX)
    ax.set_title(
        f'{LABELS[name]} — transect locations\n'
        f'{dt0.year}-{dt0.month:02d}-{dt0.day:02d} {dt0.hour:02d}:{dt0.minute:02d} UTC'
    )
    ax.legend(loc='upper right', fontsize=8)

    fname = f'{FRONTVARS_DIR}cs_front_map_{name}.png'
    plt.savefig(fname, dpi=800, bbox_inches='tight')
    plt.close()
    print(f'  saved -> {fname}')


def plot_var_figure(name, var, cfg, trs, panels, dt0):
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.6 * n),
                             gridspec_kw={'hspace': 0.6})
    if n == 1:
        axes = [axes]

    pc = None
    for ax, (sec, zr_t, rho_t, rho_zr_t, lon_t), tr, tname in zip(
            axes, panels, trs, TRANSECT_NAMES[name]):
        kwargs = dict(cmap=cfg['cmap'], shading='nearest')
        if 'norm' in cfg:
            kwargs['norm'] = cfg['norm']
        else:
            kwargs['vmin'] = cfg['vmin']
            kwargs['vmax'] = cfg['vmax']
        pc = ax.pcolormesh(lon_t, zr_t, sec, **kwargs)

        # rho_t is always on the s_rho grid (rho3d), which differs from
        # zr_t for Akt/Akv (s_w, one more level) -- use rho_zr_t (always
        # s_rho-based) as the contour's own y-axis so shapes match.
        lon_2d = np.tile(lon_t, (rho_zr_t.shape[0], 1))
        cs = ax.contour(lon_2d, rho_zr_t, rho_t, levels=ISO_LEVELS,
                        colors='k', linewidths=0.8)
        ax.clabel(cs, fmt='%.2f', fontsize=7)
        ax.set_ylim([tr['depth_lim'], 0])
        ax.set_ylabel('Depth (m)')
        ax.set_title(tname)

        sf = ScalarFormatter(useOffset=False)
        sf.set_scientific(False)
        ax.xaxis.set_major_formatter(sf)
        ax.xaxis.set_major_locator(MaxNLocator(5))

    axes[-1].set_xlabel('Longitude')
    fig.suptitle(
        f'{LABELS[name]} — {cfg["label"]}\n'
        f'{dt0.year}-{dt0.month:02d}-{dt0.day:02d} {dt0.hour:02d}:{dt0.minute:02d} UTC'
    )
    fig.colorbar(pc, ax=axes, label=cfg['label'], shrink=0.8)

    fname = f'{FRONTVARS_DIR}cs_front_vars_{name}_{var}.png'
    plt.savefig(fname, dpi=800, bbox_inches='tight')
    plt.close()
    print(f'  saved -> {fname}')

# ---------------------------------------------------------------------------
# Main loop — per scenario: locator map, then all 7 variables x all
# transects for that scenario.
# ---------------------------------------------------------------------------
os.makedirs(CACHE_DIR, exist_ok=True)

for name in SCENARIOS:
    print(f'  {name}: {len(TRANSECTS[name])} transect(s)')

for name, root in SCENARIOS.items():
    trs = TRANSECTS[name]
    if not trs:
        continue

    print(f'\n=== {name} ===')
    his_f, t_idx, dt0 = matched_file_and_index(src_dir(root, 'his'), 'mc60_his')
    print(f'  matched {dt0} ({his_f})')

    geoms = [transect_geom(name, ti) for ti in range(len(trs))]

    # --- Stage 1: locator map -------------------------------------------
    mc = map_cache_path(name)
    if os.path.exists(mc):
        sst = np.load(mc)['sst']
    else:
        with Dataset(his_f, 'r') as hisnc:
            sst = clean(np.squeeze(np.array(hisnc['temp'][t_idx, -1, :, :]))) * mask_plot
        np.savez(mc, sst=sst)
    plot_locator_map(name, trs, sst, dt0)

    # --- Stage 2: 7-variable cross-sections -------------------------------
    panels_by_var = {var: [None] * len(trs) for var in VAR_CONFIGS}
    need_recompute = {}
    for var in VAR_CONFIGS:
        need = False
        for ti in range(len(trs)):
            d = load_if_fresh(sec_cache_path(name, var, ti), geoms[ti])
            if d is None:
                need = True
            else:
                panels_by_var[var][ti] = (d['sec'], d['zr'], d['rho'], d['rho_zr'], d['lon'])
        need_recompute[var] = need

    if any(need_recompute.values()):
        need_akt_akv = need_recompute['Akt'] or need_recompute['Akv']
        need_bgc     = need_recompute['NO3'] or need_recompute['phytoC']

        fields = {}
        with Dataset(his_f, 'r') as hisnc:
            zeta  = np.squeeze(np.array(hisnc['zeta'][t_idx, :, :]))
            zr3d  = depths.get_zr_zeta(hisnc, grdnc, zeta)
            rho3d = (clean(np.squeeze(np.array(hisnc['rho'][t_idx]))) + ISO_RHO_OFF) * mask_plot
            zw3d  = depths.get_zw_zeta(hisnc, grdnc, zeta) if need_akt_akv else None

            if need_recompute['temp']:
                fields['temp'] = clean(np.squeeze(np.array(hisnc['temp'][t_idx]))) * mask_plot
            if need_recompute['w']:
                fields['w'] = clean(np.squeeze(np.array(hisnc['w'][t_idx]))) * mask_plot
            if need_recompute['Akt']:
                fields['Akt'] = clean(np.squeeze(np.array(hisnc['Akt'][t_idx]))) * mask_plot
            if need_recompute['Akv']:
                fields['Akv'] = clean(np.squeeze(np.array(hisnc['Akv'][t_idx]))) * mask_plot

        if need_recompute['vort']:
            fields['vort'] = compute_vorticity(his_f, t_idx)

        if need_bgc:
            bgc_f, bgc_idx, _ = matched_file_and_index(src_dir(root, 'bgc'), 'mc60_bgc')
            with Dataset(bgc_f, 'r') as bgcnc:
                if need_recompute['NO3']:
                    fields['NO3'] = clean(np.squeeze(np.array(bgcnc['NO3'][bgc_idx]))) * mask_plot
                if need_recompute['phytoC']:
                    fields['phytoC'] = sum(
                        clean(np.squeeze(np.array(bgcnc[v][bgc_idx])))
                        for v in ['DIATC', 'DIAZC', 'SPC']
                    ) * mask_plot

        for var, cfg in VAR_CONFIGS.items():
            if not need_recompute[var]:
                continue
            zsrc = zw3d if cfg['vert'] == 'w' else zr3d
            for ti, tr in enumerate(trs):
                sec  = interp_section(fields[var], tr['coords'], tr['mask'])
                zr_t = interp_section(zsrc,        tr['coords'], tr['mask'])
                rho_t = interp_section(rho3d,      tr['coords'], tr['mask'])
                # rho_zr_t is always on the s_rho grid, matching rho_t --
                # separate from zr_t (which is s_w for Akt/Akv) so the
                # contour overlay's x/y/z shapes always agree
                rho_zr_t = zr_t if cfg['vert'] == 'rho' else \
                    interp_section(zr3d, tr['coords'], tr['mask'])
                np.savez(sec_cache_path(name, var, ti),
                         sec=sec, zr=zr_t, rho=rho_t, rho_zr=rho_zr_t,
                         lon=tr['lon'], geom=geoms[ti])
                panels_by_var[var][ti] = (sec, zr_t, rho_t, rho_zr_t, tr['lon'])
            print(f'  {var}: computed {len(trs)} transect(s)')

    for var, cfg in VAR_CONFIGS.items():
        plot_var_figure(name, var, cfg, trs, panels_by_var[var], dt0)
