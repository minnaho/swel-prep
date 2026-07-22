"""
Diagonal cross-section from coast to offshore using native s_rho ROMS output.
Sibling of plot_cs_diag_o2.py, restricted to 3 scenarios and reading DIAT/SP
nutrient-limitation and uptake diagnostics from raw mc60_bgc_dia_avg files
(same variable set as plot_hov_transect_raw_bgcdia.py) instead of a single
his/bgc variable.

Scenarios: tidesampwec (regular dia/ output), tidesnowec and notidesnowec
(rerun bgc_dia_avg output only, at dia/rerun_bgcdia/, covering a limited
rerun date window). Since tidesnowec/notidesnowec only have 4 dia_avg files
(the rerun window), the plotted time steps are restricted to the filename
timestamps (YYYYMMDDHHMM prefix) common to all 3 scenarios' dia directories,
so every panel in a figure shows the same simulated time.

mc60_bgc_dia_avg files have no zeta/rho, so depths (for the y-axis) and
sigma-t (for the isopycnal contour overlay) are reconstructed by pairing
each dia_avg file with the regular his/ file sharing the same
YYYYMMDDHHMM filename prefix, and time-averaging that his file's zeta/rho
across its own internal time steps (matching the averaging window of the
dia_avg diagnostic). A scenario/time with no matching his file is skipped.

Saves one PNG per (variable, transect, time step) — same pattern as
plot_cs_diag_o2.py, just looped over all 18 DIAT/SP LIM + uptake variables.

Output: ./figs/snapshots/cs_diag_bgcdia_{var}-<ts|tn>-YYYY-MM-DD-HH.png
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
from netCDF4 import Dataset, num2date
from scipy.ndimage import map_coordinates
import ROMS_depths as depths

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCENARIOS = {
    'tidesampwec':  '/data/project3/minnaho/swel/tides/mc60/ampwec',
    'tidesnowec':   '/data/project3/minnaho/swel/tides/mc60/nowec/output',
    'notidesnowec': '/data/project3/minnaho/swel/notides/mc60/nowec/output',
}
LABELS = {
    'tidesampwec':  'tides, amplified WEC',
    'tidesnowec':   'tides, no WEC',
    'notidesnowec': 'no tides, no WEC',
}
# tidesnowec/notidesnowec only reran this date window — rerun bgc_dia_avg
# output lands in dia/rerun_bgcdia/, not dia/ itself. tidesampwec is a full
# continuous run, so it uses the default 'dia'.
DIA_SRC_SUBDIR = {
    'tidesnowec':   'dia/rerun_bgcdia',
    'notidesnowec': 'dia/rerun_bgcdia',
}

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
        cmap=cmocean.cm.tempo, vmin=None, vmax=None,   # resolved per-file from plotted data
        label=f'{VAR_LONG_NAME[_v]}',
    )
for _v in UPTAKE_VARS:
    VAR_CONFIGS[_v] = dict(
        cmap=cmocean.cm.algae, vmin=None, vmax=None,   # resolved per-file from plotted data
        label=f'{VAR_LONG_NAME[_v]} (mmol m$^{{-2}}$ s$^{{-1}}$)',
    )

RHO_OFFSET = 0.0

# Isopycnal contour overlay (sigma-t levels)
RHO_REF_NC  = 1027.4             # ROMS reference density
ISO_RHO_OFF = RHO_REF_NC - 1000  # = 27.4: stored rho + offset → sigma-t
ISO_LEVELS  = [24, 24.5, 25, 25.5, 26]

# Transect geometry (grid index space)
ETA0       = 271       # transect 0 — starting eta index (coast end)
XI0        = 543       # transect 0 — starting xi index (coast end)
SLOPE      = -0.8      # transect 0 — deta/dxi
DEPTH_LIM0 = -125      # transect 0 — y-axis bottom (m)
ETA1       = 832       # transect 1 — starting eta index (coast end)
XI1        = 646       # transect 1 — starting xi index (coast end)
SLOPE1     = 0.6       # transect 1 — deta/dxi
DEPTH_LIM1 = -90      # transect 1 — y-axis bottom (m)
LENGTH_XI = 250        # transect length in xi cells (both)
N_PTS     = 300        # interpolation resolution along transect (both)

GRD      = 'mc60_grd.nc'
SAVEPATH = './figs/snapshots/'

# ---------------------------------------------------------------------------
# Load grid
# ---------------------------------------------------------------------------
grdnc    = Dataset(GRD, 'r')
lat      = np.array(grdnc['lat_rho'][:])
lon      = np.array(grdnc['lon_rho'][:]) - 360
mask_rho = np.array(grdnc['mask_rho'][:])

# land mask for variable fields
mask_plot = mask_rho.astype(float)
mask_plot[mask_plot == 0] = np.nan

# ---------------------------------------------------------------------------
# Build transect coordinates (goes in -xi direction)
# ---------------------------------------------------------------------------
def build_transect(eta0, xi0, slope, depth_lim):
    xi  = np.linspace(xi0,  xi0  - LENGTH_XI, N_PTS)
    eta = np.linspace(eta0, eta0 + slope * (-LENGTH_XI), N_PTS)
    crd = np.array([eta, xi])
    return dict(
        coords    = crd,
        lon       = map_coordinates(lon, crd, order=1, mode='nearest'),
        lat       = map_coordinates(lat, crd, order=1, mode='nearest'),
        mask      = map_coordinates(mask_rho.astype(float), crd,
                                    order=0, mode='nearest') > 0.5,
        depth_lim = depth_lim,
        eta0=eta0, xi0=xi0, slope=slope,
    )

TRANSECTS = [
    build_transect(ETA0, XI0, SLOPE,  DEPTH_LIM0),
    build_transect(ETA1, XI1, SLOPE1, DEPTH_LIM1),
]
TRANSECT_NAMES = ['ts', 'tn']   # ts = south transect (ETA0), tn = north transect (ETA1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean(arr):
    """Convert to ndarray, fill netCDF sentinel fill values with nan."""
    arr = np.array(arr)
    return np.where(np.abs(arr) > 1e30, np.nan, arr)


def interp_transect(field2d, coords, mask_t):
    nan_mask  = np.isnan(field2d)
    filled    = np.where(nan_mask, 0.0, field2d)
    row       = map_coordinates(filled, coords, order=1, mode='nearest')
    nan_along = map_coordinates(nan_mask.astype(float), coords,
                                order=1, mode='nearest') > 0.5
    row[nan_along | ~mask_t] = np.nan
    return row


def interp_section(field3d, coords, mask_t):
    """Interpolate a 3D (s_rho, eta, xi) field along a transect. Returns (s_rho, N_PTS)."""
    n_s = field3d.shape[0]
    out = np.full((n_s, N_PTS), np.nan)
    for iz in range(n_s):
        out[iz] = interp_transect(field3d[iz], coords, mask_t)
    return out

# ---------------------------------------------------------------------------
# Build per-scenario dia_avg / his file lookups, keyed by YYYYMMDDHHMM prefix
# ---------------------------------------------------------------------------
def build_prefix_map(files, stem):
    # match by YYYYMMDDHH (10 chars) only, not full YYYYMMDDHHMM — the minute
    # digit sometimes shifts by 1 between dia_avg and his filenames for files
    # that were regenerated/repaired independently (e.g. tidesampwec's HDF-
    # corrupted-and-rejoined 04-25..04-27 files). Hour-level prefixes are
    # confirmed unique per scenario/file-kind.
    return {os.path.basename(f)[len(stem) + 1: len(stem) + 11]: f for f in files}

dia_by_prefix = {}
his_by_prefix = {}
for name, root in SCENARIOS.items():
    dia_dir = os.path.join(root, DIA_SRC_SUBDIR.get(name, 'dia'))
    his_dir = os.path.join(root, 'his')
    dia_files = sorted(glob.glob(os.path.join(dia_dir, 'mc60_bgc_dia_avg.*.nc')))
    his_files = sorted(glob.glob(os.path.join(his_dir, 'mc60_his.*.nc')))
    dia_by_prefix[name] = build_prefix_map(dia_files, 'mc60_bgc_dia_avg')
    his_by_prefix[name] = build_prefix_map(his_files, 'mc60_his')
    print(f'  {name}: {len(dia_files)} dia_avg files, {len(his_files)} his files')

common_prefixes = sorted(set.intersection(*(set(m) for m in dia_by_prefix.values())))
print(f'{len(common_prefixes)} time steps common to all scenarios: {common_prefixes}')

# ---------------------------------------------------------------------------
# Main loop — one set of PNGs per common time step
# ---------------------------------------------------------------------------
os.makedirs(SAVEPATH, exist_ok=True)

for prefix in common_prefixes:
    scen_data = {}
    dt0 = None

    for name in SCENARIOS:
        dia_f = dia_by_prefix[name][prefix]
        his_f = his_by_prefix[name].get(prefix)

        with Dataset(dia_f, 'r') as dnc:
            ocean_time = np.array(dnc['ocean_time'][:])
            if dt0 is None:
                dt0 = num2date(ocean_time, 'seconds since 1995-01-01',
                               only_use_cftime_datetimes=False)[0]

            zr3d = None
            rho3d = None
            if his_f is not None:
                with Dataset(his_f, 'r') as hnc:
                    zeta_mean = np.nanmean(clean(hnc['zeta'][:]), axis=0)
                    rho3d = (np.nanmean(clean(hnc['rho'][:]), axis=0)
                              + ISO_RHO_OFF) * mask_plot
                zr3d = depths.get_zr_zeta(dnc, grdnc, zeta_mean)
            else:
                print(f'  WARNING: no his match for {os.path.basename(dia_f)} '
                      f'({name}, prefix {prefix}) — skipping this scenario/time')

            var3d = {}
            for var in PLOT_VARS:
                if var in dnc.variables:
                    var3d[var] = clean(dnc[var][0]) * mask_plot

        scen_data[name] = dict(zr3d=zr3d, rho3d=rho3d, var3d=var3d)

    dstr = f'{dt0.year}-{dt0.month:02d}-{dt0.day:02d}-{dt0.hour:02d}'

    # precompute zr_t / rho_t once per (scenario, transect) — reused across all vars
    depth_cache = {}
    for name in SCENARIOS:
        zr3d, rho3d = scen_data[name]['zr3d'], scen_data[name]['rho3d']
        for ti, tr in enumerate(TRANSECTS):
            if zr3d is None:
                depth_cache[(name, ti)] = (None, None)
                continue
            zr_t  = interp_section(zr3d, tr['coords'], tr['mask'])
            rho_t = interp_section(rho3d, tr['coords'], tr['mask'])
            depth_cache[(name, ti)] = (zr_t, rho_t)

    for var in PLOT_VARS:
        cfg = VAR_CONFIGS[var]

        # precompute this var's plotted (transect-interpolated, depth-limited)
        # section for every (scenario, transect) — reused for both the
        # vmin/vmax calc below and the actual plotting loop, so
        # interp_section only runs once per (scenario, transect)
        plotted = {}   # (name, ti) -> (zr_plot, var_plot, rho_plot)
        for name in SCENARIOS:
            if var not in scen_data[name]['var3d']:
                continue
            for ti, tr in enumerate(TRANSECTS):
                zr_t, rho_t = depth_cache[(name, ti)]
                if zr_t is None:
                    continue
                var_t = interp_section(scen_data[name]['var3d'][var],
                                       tr['coords'], tr['mask'])
                depth_mean = np.nanmean(zr_t, axis=1)
                keep       = depth_mean >= tr['depth_lim']
                plotted[(name, ti)] = (zr_t[keep, :], var_t[keep, :], rho_t[keep, :])

        # resolve vmin/vmax for uptake vars from the actual plotted cross-
        # section data (not the full 3D domain, which can be dominated by
        # values far from the transect) — LIM vars keep their fixed 0-1 scale
        sections = [v[1] for v in plotted.values()]
        if cfg['vmin'] is None:
            vmin = min(np.nanmin(s) for s in sections) if sections else 0.0
        else:
            vmin = cfg['vmin']
        if cfg['vmax'] is None:
            vmax = max(np.nanmax(s) for s in sections) if sections else 1.0
        else:
            vmax = cfg['vmax']

        for ti, tr in enumerate(TRANSECTS):
            fig, axes = plt.subplots(len(SCENARIOS), 1, sharex=True,
                                     figsize=(10, 3 * len(SCENARIOS)))
            pc = None

            for ax, name in zip(axes, SCENARIOS):
                entry = plotted.get((name, ti))
                if entry is None:
                    ax.set_title(LABELS[name])
                    continue
                zr_plot, var_plot, rho_plot = entry

                pc = ax.pcolormesh(tr['lon'], zr_plot, var_plot,
                                   cmap=cfg['cmap'], vmin=vmin, vmax=vmax,
                                   shading='nearest')
                lon_2d = np.tile(tr['lon'], (zr_plot.shape[0], 1))
                cs = ax.contour(lon_2d, zr_plot, rho_plot, levels=ISO_LEVELS,
                                colors='k', linewidths=0.8)
                ax.clabel(cs, fmt='%.1f', fontsize=9)
                ax.set_ylim([tr['depth_lim'], 0])
                ax.set_ylabel('Depth (m)')
                ax.set_title(LABELS[name])

                sf = ScalarFormatter(useOffset=False)
                sf.set_scientific(False)
                ax.xaxis.set_major_formatter(sf)
                ax.xaxis.set_major_locator(MaxNLocator(5))

            if pc is None:
                plt.close(fig)
                continue

            axes[-1].set_xlabel('Longitude')
            fig.canvas.draw()
            pos_top = axes[0].get_position()
            pos_bot = axes[-1].get_position()
            cax = fig.add_axes([pos_top.x1 + 0.015, pos_bot.y0,
                                0.012, pos_top.y1 - pos_bot.y0])
            fig.colorbar(pc, cax=cax, label=cfg['label'])
            fig.suptitle(f'{dt0.year}-{dt0.month:02d}-{dt0.day:02d} '
                         f'{dt0.hour:02d}:{dt0.minute:02d} UTC', y=0.96)

            fname = f'{SAVEPATH}cs_diag_bgcdia_{var}-{TRANSECT_NAMES[ti]}-{dstr}.png'
            plt.savefig(fname, dpi=800, bbox_inches='tight')
            plt.close()
            print(f'  saved -> {fname}')
