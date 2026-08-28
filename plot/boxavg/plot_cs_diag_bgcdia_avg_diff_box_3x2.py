"""
Box-averaged, time-averaged DIAT/SP + PAR/TOT_PROD cross-section DIFFERENCES,
laid out as a 3x2-style grid -- same panel-1-raw / panels-2-N-diff layout and
plotting code as ../plot_cs_diag_avg_diff_box_3x2.py, applied to the native
s_rho bgc_dia_avg fields read by plot_cs_diag_bgcdia_box_alltime.py instead
of the fixed-depth-grid zsliced fields the *_3x2.py sibling reads.

Base case: notidesnowec (BASE_SCEN below), the only scenario present in
every group, same as plot_cs_diag_avg_diff_box_3x2.py's BASE_SCEN. The
other 3 scenarios (tidesampwec, tidesnowec, ampwec) are each differenced
against it -- 4 panels total (1 raw + 3 diff), so every figure here comes
out 2x2 rather than 3x2; the panel grid is sized dynamically from the
scenario count exactly as in the *_3x2.py sibling, so this isn't hardcoded.

Two variable groups, exactly as in plot_cs_diag_bgcdia_box_alltime.py:
  - LIM_VARS + UPTAKE_VARS: only 4 dia_avg files exist per scenario (the
    targeted rerun window -- tidesnowec/notidesnowec/ampwec via
    dia/rerun_bgcdia/, tidesampwec via its regular continuous dia/), so the
    "time mean" here is simply the mean of those (up to) 4 common-prefix
    snapshots -- restricted to prefixes common to ALL scenarios so every
    scenario's mean covers the same real dates.
  - PROD_VARS + PAR_VARS (TOT_PROD, PAR): present in every scenario's full
    continuous dia_avg record, so each scenario's time mean is taken over
    its OWN full available file list (FULL_DIA_SUBDIR below), same source
    as plot_cs_diag_bgcdia_box_alltime.py's full-record pass. Native output
    cadence differs by scenario (12-hourly for tidesampwec/ampwec, ~daily
    for tidesnowec/notidesnowec) and the scenarios' available date ranges
    are not identical, so this is a mean-over-each-scenario's-own-record
    comparison, not a mean over a shared synchronized window -- acceptable
    for a climatological comparison, but keep that caveat in mind before
    reading small differences as significant.

Each dia_avg file has no zeta/rho of its own, so depths (for the box
average's z grid) and sigma-t (isopycnal overlay) are reconstructed by
pairing every dia_avg file with the his/ file sharing its YYYYMMDDHH
filename prefix and time-averaging that his file's zeta/rho internally --
same his-pairing convention as plot_cs_diag_bgcdia_box_alltime.py. That
his-file window is a fixed ~11h regardless of the true dia_avg averaging
period, so (as in the sibling scripts) a console WARNING is printed
whenever a file's actual averaging period (from consecutive-file
ocean_time spacing) and its paired his window diverge by >50%. A
dia_avg file with no matching his file is skipped (not just for that
panel -- it drops out of that scenario's time mean entirely).

boxavg_section (native, s_rho-varying-with-zeta) always interpolates onto
each transect's fixed tr['zgrid'] (build_box_transect/make_zgrid), so
accumulating a running sum/count of each file's box-averaged section across
files is equivalent to box-averaging the time mean itself -- same reasoning
plot_cs_diag_avg_diff_box_3x2.py's compute_var_mean uses for the zsliced
fixed-depth-grid case, just with the box-average itself doing the
per-file vertical interpolation here instead of the source data already
sitting on a fixed grid.

Layout: panel 1 (top-left) = base case (notidesnowec) RAW time-mean, on a
range taken directly from that panel's own data (min/max, matching the
convention in plot_cs_diag_bgcdia_box_alltime.py where vmin/vmax are
resolved from the actual plotted data rather than fixed per-variable); the
remaining panels are each other scenario's time-mean DIFFERENCED against
the base case, on a shared symmetric-about-zero range (98th percentile of
|diff|). Two colorbars: one for the raw panel, one shared across the diff
panels -- identical scheme to plot_cs_diag_avg_diff_box_3x2.py.

Only 2 transects here (ts, tn) -- plot_cs_diag_bgcdia_box_alltime.py has no
'mid' transect.

Output: ./figs/cs_diag_bgcdia_avg_diff_box_3x2_<var>_<ts|tn>.png
Cache:  ./figs/cs_diag_bgcdia_avg_diff_box_cache_<var>_<scen>_<ts|tn>.npz
        (own cache namespace -- not shared with plot_cs_diag_bgcdia_box_
        alltime.py's per-instant snapshot PNGs, which live under
        figs/snapshots/ and use their own filename stems)
"""

import os
import sys
sys.path.append('/data/project3/minnaho/global/')
import glob
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import cmocean

plt.rcParams.update({'font.size': 14})
from netCDF4 import Dataset
import ROMS_depths as depths
import cs_boxavg as cb

# ---------------------------------------------------------------------------
# Configuration -- scenarios, variable groups, geometry: same as
# ../plot_cs_diag_bgcdia_box_alltime.py
# ---------------------------------------------------------------------------
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
BASE_SCEN = 'notidesnowec'

DIA_SRC_SUBDIR = {
    'tidesnowec':   'dia/rerun_bgcdia',
    'notidesnowec': 'dia/rerun_bgcdia',
    'ampwec':       'dia/rerun_bgcdia',
}
FULL_DIA_SUBDIR = {'ampwec': 'everything/dia'}

DIAT_LIM_VARS    = ['DIAT_N_LIM', 'DIAT_FE_LIM', 'DIAT_PO4_LIM',
                     'DIAT_SIO3_LIM', 'DIAT_LIGHT_LIM', 'DIAT_P_LIM']
SP_LIM_VARS      = ['SP_N_LIM', 'SP_FE_LIM', 'SP_PO4_LIM',
                     'SP_LIGHT_LIM', 'SP_P_LIM']
DIAT_UPTAKE_VARS = ['DIAT_NO3_UPTAKE', 'DIAT_NH4_UPTAKE', 'DIAT_NO2_UPTAKE',
                     'DIAT_SI_UPTAKE']
SP_UPTAKE_VARS   = ['SP_NO3_UPTAKE', 'SP_NH4_UPTAKE', 'SP_NO2_UPTAKE']
PROD_VARS        = ['TOT_PROD']
PAR_VARS         = ['PAR']

LIM_VARS         = DIAT_LIM_VARS + SP_LIM_VARS
UPTAKE_VARS      = DIAT_UPTAKE_VARS + SP_UPTAKE_VARS
RERUN_VARS       = LIM_VARS + UPTAKE_VARS   # rerun-window (common-prefix) mean
FULL_RECORD_VARS = PROD_VARS + PAR_VARS     # each scenario's own full record

VAR_SCALE = {'TOT_PROD': 86400.0}   # mmol C m^-3 s^-1 -> d^-1

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
    'TOT_PROD':          'NPP',
    'PAR':               'PAR',
}

DIFF_CMAP = cmocean.cm.balance

VAR_CONFIGS = {}
for _v in LIM_VARS:
    VAR_CONFIGS[_v] = dict(
        raw_cmap=cmocean.cm.tempo, raw_label=f'{VAR_LONG_NAME[_v]}',
        label=fr'$\Delta$ {VAR_LONG_NAME[_v]}',
    )
for _v in UPTAKE_VARS:
    VAR_CONFIGS[_v] = dict(
        raw_cmap=cmocean.cm.algae,
        raw_label=f'{VAR_LONG_NAME[_v]} (mmol m$^{{-2}}$ s$^{{-1}}$)',
        label=fr'$\Delta$ {VAR_LONG_NAME[_v]} (mmol m$^{{-2}}$ s$^{{-1}}$)',
    )
for _v in PROD_VARS:
    VAR_CONFIGS[_v] = dict(
        raw_cmap=cmocean.cm.algae,
        raw_label=f'{VAR_LONG_NAME[_v]} (mmol C m$^{{-3}}$ d$^{{-1}}$)',
        label=fr'$\Delta$ {VAR_LONG_NAME[_v]} (mmol C m$^{{-3}}$ d$^{{-1}}$)',
    )
for _v in PAR_VARS:
    VAR_CONFIGS[_v] = dict(
        raw_cmap=cmocean.cm.solar, raw_label='PAR (W m$^{-2}$)',
        label=r'$\Delta$ PAR (W m$^{-2}$)',
    )

# Isopycnal contour overlay (sigma-t levels) -- kept for parity with the
# other cs_diag_bgcdia*.py scripts even though this diff layout doesn't
# overlay contours (no single rho field represents a "difference" cleanly)
RHO_REF_NC  = 1027.4
ISO_RHO_OFF = RHO_REF_NC - 1000

# Transect geometry (grid index space) -- same as
# ../plot_cs_diag_bgcdia_box_alltime.py (no 'mid' transect in this family)
ETA0       = 271
XI0        = 543
SLOPE      = -0.8
DEPTH_LIM0 = -125
ETA1       = 832
XI1        = 646
SLOPE1     = 0.6
DEPTH_LIM1 = -90
LENGTH_XI  = 250
N_PTS      = 300

GRD      = cb.GRD
SAVEPATH = cb.figpath() + '/'

# ---------------------------------------------------------------------------
# Load grid
# ---------------------------------------------------------------------------
grdnc    = Dataset(GRD, 'r')
lat      = np.array(grdnc['lat_rho'][:])
lon      = np.array(grdnc['lon_rho'][:]) - 360
mask_rho = np.array(grdnc['mask_rho'][:])

mask_plot = mask_rho.astype(float)
mask_plot[mask_plot == 0] = np.nan

TRANSECTS = [
    cb.build_box_transect(lon, lat, mask_rho, ETA0, XI0, SLOPE,  DEPTH_LIM0,
                          LENGTH_XI, N_PTS, name='ts', title='south transect'),
    cb.build_box_transect(lon, lat, mask_rho, ETA1, XI1, SLOPE1, DEPTH_LIM1,
                          LENGTH_XI, N_PTS, name='tn', title='north transect'),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean(arr):
    arr = np.array(arr)
    return np.where(np.abs(arr) > 1e30, np.nan, arr)


def build_prefix_map(files, stem):
    # hour-level (YYYYMMDDHH, 10 chars) prefix match, not full
    # YYYYMMDDHHMM -- same convention/reasoning as
    # plot_cs_diag_bgcdia_box_alltime.py's build_prefix_map
    return {os.path.basename(f)[len(stem) + 1: len(stem) + 11]: f for f in files}


def cache_path(var_key, scen, tr_name):
    return f'{SAVEPATH}cs_diag_bgcdia_avg_diff_box_cache_{var_key}_{scen}_{tr_name}.npz'

# ---------------------------------------------------------------------------
# Per-scenario file lookups
# ---------------------------------------------------------------------------
his_by_prefix    = {}
dia_by_prefix     = {}   # rerun-window (LIM/UPTAKE) source
full_files_by_scen = {}  # full-record (PAR/TOT_PROD) source, sorted file list
avg_period_by_file  = {} # dia_f -> hours since the previous file for that
                         # scenario's full-record list (actual dia_avg Δt)

for name, root in SCENARIOS.items():
    his_files = sorted(glob.glob(os.path.join(root, 'his', 'mc60_his.*.nc')))
    his_by_prefix[name] = build_prefix_map(his_files, 'mc60_his')

    dia_dir = os.path.join(root, DIA_SRC_SUBDIR.get(name, 'dia'))
    dia_files = sorted(glob.glob(os.path.join(dia_dir, 'mc60_bgc_dia_avg.*.nc')))
    dia_by_prefix[name] = build_prefix_map(dia_files, 'mc60_bgc_dia_avg')

    full_dir = os.path.join(root, FULL_DIA_SUBDIR.get(name, 'dia'))
    full_files = sorted(glob.glob(os.path.join(full_dir, 'mc60_bgc_dia_avg.*.nc')))
    full_files_by_scen[name] = full_files
    prev_t = None
    for f in full_files:
        with Dataset(f, 'r') as nc:
            t = float(np.array(nc['ocean_time'][:])[0])
        avg_period_by_file[f] = (t - prev_t) / 3600.0 if prev_t is not None else np.nan
        prev_t = t

    print(f'  {name}: {len(dia_files)} rerun-window dia_avg files, '
          f'{len(full_files)} full-record dia_avg files, {len(his_files)} his files')

common_prefixes = sorted(set.intersection(*(set(m) for m in dia_by_prefix.values())))
print(f'{len(common_prefixes)} rerun-window time steps common to all scenarios')

# ---------------------------------------------------------------------------
# Time-averaged, box-averaged native field per (variable, scenario, transect)
# ---------------------------------------------------------------------------
os.makedirs(SAVEPATH, exist_ok=True)


def compute_var_mean_native(var_key, scale, scen, dia_files, his_prefix_map,
                            check_avg_period):
    """Time-averaged, box-averaged field(zgrid, transect position) per
    transect, for one (variable, scenario). Box-averages each file's
    reconstructed 3D field (cb.boxavg_section, onto the transect's fixed
    zgrid) before accumulating the running mean -- see module docstring.
    Returns (list of (n_z, n_pts) arrays or None, n_files_used)."""
    accum_sum = None
    accum_count = None
    n_used = 0

    for dia_f in dia_files:
        prefix = os.path.basename(dia_f)[len('mc60_bgc_dia_avg') + 1:
                                         len('mc60_bgc_dia_avg') + 11]
        his_f = his_prefix_map.get(prefix)
        if his_f is None:
            print(f'  WARNING: no his match for {os.path.basename(dia_f)} '
                  f'({scen}) -- dropped from time mean')
            continue

        with Dataset(dia_f, 'r') as dnc:
            if var_key not in dnc.variables:
                continue
            var3d = clean(dnc[var_key][0]) * mask_plot * scale

            with Dataset(his_f, 'r') as hnc:
                his_time = np.array(hnc['ocean_time'][:])
                his_window_hr = ((his_time[-1] - his_time[0]) / 3600.0
                                  if len(his_time) > 1 else 0.0)
                zeta_mean = np.nanmean(clean(hnc['zeta'][:]), axis=0)
            zr3d = depths.get_zr_zeta(dnc, grdnc, zeta_mean)

        if check_avg_period:
            ap = avg_period_by_file.get(dia_f, np.nan)
            if np.isfinite(ap) and ap > 0 and abs(ap - his_window_hr) / ap > 0.5:
                print(f'  WARNING: {scen} {os.path.basename(dia_f)} -- dia_avg '
                      f'averaging period ~{ap:.0f}h but his-based zeta/rho '
                      f'window is only {his_window_hr:.0f}h; this file\'s '
                      f'contribution to the time mean does not represent its '
                      f'full averaging window')

        if accum_sum is None:
            accum_sum   = [np.zeros((tr['zgrid'].size, tr['n_pts'])) for tr in TRANSECTS]
            accum_count = [np.zeros((tr['zgrid'].size, tr['n_pts'])) for tr in TRANSECTS]

        for ti, tr in enumerate(TRANSECTS):
            field_box = cb.boxavg_section(var3d, zr3d, tr)
            valid = ~np.isnan(field_box)
            accum_sum[ti][valid]   += field_box[valid]
            accum_count[ti][valid] += 1
        n_used += 1

    if accum_sum is None:
        return None, 0
    with np.errstate(invalid='ignore'):
        return [accum_sum[ti] / accum_count[ti] for ti in range(len(TRANSECTS))], n_used


ALL_VARS = RERUN_VARS + FULL_RECORD_VARS

for var_key in ALL_VARS:
    print(f'\n=== {var_key} ===')
    cfg = VAR_CONFIGS[var_key]
    scale = VAR_SCALE.get(var_key, 1.0)
    is_full_record = var_key in FULL_RECORD_VARS

    var_mean = {}
    for scen in SCENARIOS:
        cached = [cache_path(var_key, scen, tr['name']) for tr in TRANSECTS]
        if all(os.path.exists(c) for c in cached):
            print(f'  {scen}: loading cache...')
            var_mean[scen] = [np.load(c)['mean'] for c in cached]
            continue

        if is_full_record:
            files = full_files_by_scen[scen]
        else:
            files = [dia_by_prefix[scen][p] for p in common_prefixes]

        if not files:
            print(f'  WARNING: no dia_avg files for {var_key} | {scen}, skipping')
            continue

        mean_list, n_used = compute_var_mean_native(
            var_key, scale, scen, files, his_by_prefix[scen],
            check_avg_period=is_full_record)
        if mean_list is None:
            print(f'  WARNING: {var_key} not found / no usable files for {scen}, skipping')
            continue
        print(f'  {scen}: averaged over {n_used}/{len(files)} files')
        for ti, arr in enumerate(mean_list):
            np.savez(cached[ti], mean=arr)
        var_mean[scen] = mean_list

    # -------------------------------------------------------------------
    # Plot: panel 1 = base case raw, remaining panels = diff from base case
    # -------------------------------------------------------------------
    diff_scens = [s for s in SCENARIOS if s != BASE_SCEN and s in var_mean]
    if BASE_SCEN not in var_mean or not diff_scens:
        print(f'  WARNING: missing base case or no scenarios to diff for {var_key}, skipping plot')
        continue

    for ti, tr in enumerate(TRANSECTS):
        fname = f'{SAVEPATH}cs_diag_bgcdia_avg_diff_box_3x2_{var_key}_{tr["name"]}.png'
        zgrid = tr['zgrid']

        n_panels = 1 + len(diff_scens)
        ncols = 2
        nrows = math.ceil(n_panels / ncols)
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                                 figsize=(14, 4 * nrows))
        axes_flat = np.atleast_1d(axes).flatten()
        for ax in axes_flat[n_panels:]:
            ax.axis('off')

        def _format_ax(ax, row, col):
            ax.set_ylim([tr['depth_lim'], 0])
            sf = ScalarFormatter(useOffset=False)
            sf.set_scientific(False)
            ax.xaxis.set_major_formatter(sf)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            if col == 0:
                ax.set_ylabel('Depth (m)')
            if row == nrows - 1:
                ax.set_xlabel('Longitude')
            ax.label_outer()

        # panel 1: base case, raw time-mean
        ax_raw = axes_flat[0]
        base_field = var_mean[BASE_SCEN][ti]
        raw_kwargs = dict(vmin=np.nanmin(base_field), vmax=np.nanmax(base_field))
        pc_raw = ax_raw.pcolormesh(tr['lon'], zgrid, base_field,
                                   cmap=cfg['raw_cmap'], shading='nearest', **raw_kwargs)
        ax_raw.set_title(LABELS[BASE_SCEN])
        _format_ax(ax_raw, 0, 0)

        # remaining panels: diff from base case
        diffs = {s: var_mean[s][ti] - var_mean[BASE_SCEN][ti] for s in diff_scens}
        vmax = np.nanpercentile(
            np.abs(np.concatenate([d.ravel() for d in diffs.values()])), 98)

        pc_diff = None
        for i, scen in enumerate(diff_scens, start=1):
            ax = axes_flat[i]
            row, col = divmod(i, ncols)
            pc_diff = ax.pcolormesh(tr['lon'], zgrid, diffs[scen],
                                    cmap=DIFF_CMAP, vmin=-vmax, vmax=vmax, shading='nearest')
            ax.set_title(f'{LABELS[scen]}  −  {LABELS[BASE_SCEN]}')
            _format_ax(ax, row, col)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.3)
        fig.canvas.draw()

        pos_raw = ax_raw.get_position()
        below = axes_flat[ncols] if n_panels > ncols else None
        if below is not None:
            gap_top, gap_bottom = pos_raw.y0, below.get_position().y1
            cb_h = min(0.02, (gap_top - gap_bottom) * 0.4)
            cb_y = gap_bottom + (gap_top - gap_bottom - cb_h) / 2
        else:
            cb_h, cb_y = 0.02, pos_raw.y0 - 0.06
        cax_raw = fig.add_axes([pos_raw.x0, cb_y, pos_raw.width, cb_h])
        cbar_raw = fig.colorbar(pc_raw, cax=cax_raw, orientation='horizontal', label=cfg['raw_label'])
        cbar_raw.ax.xaxis.set_label_position('top')
        cbar_raw.ax.xaxis.set_ticks_position('top')

        pos_tr = axes_flat[1].get_position()
        pos_br = axes_flat[n_panels - 1].get_position()
        cax_diff = fig.add_axes([pos_tr.x1 + 0.015, pos_br.y0, 0.015, pos_tr.y1 - pos_br.y0])
        fig.colorbar(pc_diff, cax=cax_diff, label=cfg['label'])

        plt.savefig(fname, dpi=800, bbox_inches='tight')
        plt.close()
        print(f'  saved -> {fname}')
