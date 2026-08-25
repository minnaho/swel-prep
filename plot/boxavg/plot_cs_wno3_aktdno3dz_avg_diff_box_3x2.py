"""
Box-averaged cross-section differences of time-averaged w'NO3' (resolved
eddy NO3 flux) and -Akt*dNO3/dz (parameterized diffusive NO3 flux), using
zsliced output, laid out as a 3x2 grid -- flux analog of
plot_cs_diag_avg_diff_box_3x2.py / plot_cs_diag_drhodz_diff_box_3x2.py, same
grid/colorbar layout copied from those.

Unlike every other *_diff_box_3x2.py sibling, both quantities here are
PRODUCTS of two fields (w'*NO3' and Akt*dNO3/dz), not a single field or a
linear derivative of one -- so, unlike drho/dz (mean(box(drho/dz)) ==
box(d(mean(rho))/dz), since differentiation/box-avg/time-avg all commute for
a single linear field), the product must be formed at EVERY timestep before
box-averaging or time-averaging: mean(box(w'*NO3')) != box(w')*box(mean(NO3'))
in general. So this script cannot just time-average w and NO3 independently
and multiply at the end -- it has to loop every zsliced timestep, form the
instantaneous flux field, box-average that, and only then accumulate the
time mean (same sum/count accumulation pattern as compute_var_mean, just
with the product computed first).

w'NO3': w'=w-w̄, NO3'=NO3-N̄, where w̄/N̄ are the FULL-RECORD, FULL-DOMAIN time
mean on the zsliced fixed physical-depth grid (same depth levels at every
timestep) -- computed once per scenario and reused for every timestep's
anomaly. Because both the instantaneous field and its mean already sit on
this same fixed depth grid, forming the anomaly needs no vertical
interpolation at all (unlike plot_cs_wno3_aktdno3dz_snap.py, which computes
w'/NO3' against this same zsliced-grid mean but for a RAW native-sigma
instantaneous snapshot, requiring a per-column vertical interpolation of the
mean onto the sigma-level depths -- not needed here since this script stays
entirely on the zsliced product for both instant and mean).

The full-domain time mean is the expensive part. This script tries, in
order: (1) a local domain-mean npz cache from a prior run of this script;
(2) plot_cs_wno3_aktdno3dz_snap.py's precomputed ncra means
(../figs/cs_wno3_aktdno3dz_snap/ncra_means/{w,no3}_mean_<scen>.nc -- see
that script's docstring for the ncra command) -- these already exist on disk
for all 6 scenarios as of this script's creation, so this is the expected
fast path, not a rare fallback; (3) a Python full-record scan over the
zsliced product as a last resort. Note the ncra scenario-name mismatch:
that script's ncra loop key for the amplified-WEC/no-tides case is 'ampwec',
not this script's 'notidesampwec' -- see NCRA_SCEN_ALIAS.

Akt*dNO3/dz: no mean-removal / Reynolds decomposition, same reasoning as
plot_cs_wno3_aktdno3dz_snap.py -- Akt already parameterizes
unresolved/subgrid turbulent transport via a closure scheme, so there's no
resolved advective-eddy analog to isolate. Just the raw instantaneous
product at every timestep. dNO3/dz via np.gradient against the zsliced
grid's actual (non-uniform: 1 m spacing 0 to -50 m, 5 m -50 to -300 m, 30 m
-300 to -1980 m) depth array -- np.gradient handles this fine given the
actual coordinate array, descending order included. Sign: standard
downgradient-diffusion convention (flux = -Akt*dNO3/dz, z positive up,
positive = upward nutrient supply), matching w'NO3''s convention and
plot_wno3_flux_100m.py's / plot_cs_wno3_aktdno3dz_snap.py's AktdNO3dz sign
flip.

Akt lives in its own zsliced subdir (ak/z_mc60_his.*.nc, var 'Akt') rather
than the root his files -- same subdir plot_cs_diag_avg_diff_box_3x2.py's
'Akt'/'Akv' entries read from. Already on the same fixed depth grid as
w/NO3 (the whole point of z-slicing), so no s_w->s_rho averaging is needed
here either (unlike the raw-sigma snap script, which has to average Akt from
s_w to s_rho manually).

Box-averaging uses boxavg_section_fixedz (no vertical interpolation needed,
the zsliced grid is already fixed-depth) with FLUX_MIN_FRAC=0.5 -- same
thin-support-spike protection plot_cs_diag_drhodz_diff_box_3x2.py's
RHO_MIN_FRAC and plot_cs_diag_avg_diff_box_3x2.py's DUDZ_MIN_FRAC use:
right at a transect point's deepest reachable level, box support can
collapse from ~21 offsets to 1-2, and that thin remainder can differ sharply
from the well-supported average one level up. Applies to both wno3 and
aktdno3dz here (not just the gradient-derived one), since both are
box-averaged pointwise product fields, not a box-averaged-then-differentiated
single field like drho/dz.

**Expensive**: a full per-timestep loop over 3 paired zsliced files (w, NO3,
Akt) per scenario, box-averaged across ~21 offsets x 3 transects. Run in
screen, not interactively. Two-tier cache: the domain-mean pass is cached
per scenario (domain_mean_<scen>.npz, or reused from ncra as above); the
flux pass is cached per (scenario, transect) (avg_diff_box_cache_<scen>_
<ts|tn|mid>.npz), shared across reruns of this script only (no single-line
sibling script produces this cache, unlike the other *_diff_box_3x2.py
scripts).

Layout: 3x2 grid of panels -- panel 1 (top-left) is the base case
(notidesnowec) RAW time-mean, on a fixed range matching
plot_cs_wno3_aktdno3dz_snap.py's VMAX_WNO3/VMAX_AKT (its closest available
single-value reference, despite not being a *_box.py single-variable
sibling); the remaining 5 panels are each other scenario's time-mean
DIFFERENCED against the base case, range from the 98th percentile of |diff|.
Two colorbars: one for the raw panel (horizontal, ticks+label above), one
shared across the 5 diff panels.

Output: ./figs/cs_wno3_avg_diff_box_3x2_<ts|tn|mid>.png,
        ./figs/cs_aktdno3dz_avg_diff_box_3x2_<ts|tn|mid>.png
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
import cs_boxavg as cb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'
SCENARIOS_ZSLICE = {
    'tideswec':      f'{ZSLICE_ROOT}/tideswec',
    'tidesnowec':    f'{ZSLICE_ROOT}/tidesnowec',
    'notidesnowec':  f'{ZSLICE_ROOT}/notidesnowec',
    'notideswec':    f'{ZSLICE_ROOT}/notideswec',
    'tidesampwec':   f'{ZSLICE_ROOT}/tidesampwec',
    'notidesampwec': f'{ZSLICE_ROOT}/notidesampwec',
}
BASE_SCEN = 'notidesnowec'

LABELS = {
    'notidesnowec':  'no tides, no WEC',
    'tideswec':      'tides, WEC',
    'tidesnowec':    'tides, no WEC',
    'notideswec':    'no tides, WEC',
    'notidesampwec': 'no tides, 2.5x WEC',
    'tidesampwec':   'tides, 2.5x WEC',
}

# plot_cs_wno3_aktdno3dz_snap.py's ncra precompute loop uses 'ampwec' for
# this scenario, not 'notidesampwec' -- alias for reusing its output files
NCRA_SCEN_ALIAS = {'notidesampwec': 'ampwec'}
NCRA_DIR = '../figs/cs_wno3_aktdno3dz_snap/ncra_means/'

DIFF_CMAP = cmocean.cm.diff

FLUX_CONFIGS = {
    'wno3': dict(
        label=r"$\Delta\,w'NO_3^{-\prime}$ (mmol N m$^{-2}$ s$^{-1}$)",
        raw_label=r"$w'NO_3^{-\prime}$ (mmol N m$^{-2}$ s$^{-1}$)",
        raw_cmap=cmocean.cm.curl, diff_cmap=DIFF_CMAP,
        raw_vmax=0.0065,   # matches ../plot_cs_wno3_aktdno3dz_snap.py's VMAX_WNO3
        out_stem='cs_wno3_avg_diff_box_3x2',
    ),
    'aktdno3dz': dict(
        label=r'$\Delta\,(-A_{kt}\,\partial NO_3^-/\partial z)$ (mmol N m$^{-2}$ s$^{-1}$)',
        raw_label=r'$-A_{kt}\,\partial NO_3^-/\partial z$ (mmol N m$^{-2}$ s$^{-1}$)',
        raw_cmap=cmocean.cm.turbid, diff_cmap=DIFF_CMAP,
        raw_vmax=0.00035,  # matches ../plot_cs_wno3_aktdno3dz_snap.py's VMAX_AKT
        out_stem='cs_aktdno3dz_avg_diff_box_3x2',
    ),
}

# see module docstring -- same thin-box-support spike protection as
# RHO_MIN_FRAC / DUDZ_MIN_FRAC in the sibling _3x2 scripts
FLUX_MIN_FRAC = 0.5

# Diagonal transect geometry (grid index space) -- same as plot_cs_diag_box.py / plot_cs_diag_rho_box.py
ETA0       = 271       # south transect — starting eta index (coast end)
XI0        = 543       # south transect — starting xi index (coast end)
SLOPE      = -0.8      # south transect — deta/dxi
DEPTH_LIM0 = -125      # south transect — y-axis bottom (m)
ETA1       = 832       # north transect — starting eta index (coast end)
XI1        = 646       # north transect — starting xi index (coast end)
SLOPE1     = 0.6       # north transect — deta/dxi
DEPTH_LIM1 = -90       # north transect — y-axis bottom (m)
LENGTH_XI  = 250       # transect length in xi cells (both)
N_PTS      = 300       # interpolation resolution along transect (both)

# Mid transect geometry -- same as plot_cs_mid_o2.py / plot_cs_w_NO3.py
ETA_MID       = 477                 # fixed eta index — inside the bay to offshore
DEPTH_LIM_MID = -300                # y-axis bottom (m)
MID_XLIM      = [-121.96, -121.8]   # longitude window
MID_XTICKS    = np.linspace(MID_XLIM[0], MID_XLIM[1], 5)

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


def clean(arr):
    arr = np.array(arr)
    arr[np.abs(arr) > 1e10] = np.nan   # below-bathymetry zslice fill value
    return arr


# ---------------------------------------------------------------------------
# Build box transect coordinates
# ---------------------------------------------------------------------------
TRANSECTS = [
    cb.build_box_transect(lon, lat, mask_rho, ETA0, XI0, SLOPE,  DEPTH_LIM0,
                          LENGTH_XI, N_PTS, name='ts', title='south transect'),
    cb.build_box_transect(lon, lat, mask_rho, ETA1, XI1, SLOPE1, DEPTH_LIM1,
                          LENGTH_XI, N_PTS, name='tn', title='north transect'),
    cb.build_box_mid_transect(lon, lat, mask_rho, ETA_MID, DEPTH_LIM_MID,
                              xlim=MID_XLIM, xticks=MID_XTICKS, name='mid',
                              title=f'mid transect (eta={ETA_MID})'),
]

# ---------------------------------------------------------------------------
# Per-(scenario, subdir) zsliced file lists -- w (root), NO3 (bgc/), Akt (ak/)
# ---------------------------------------------------------------------------
def zslice_files(scen, subdir, stem, exclude_stamps=()):
    root = SCENARIOS_ZSLICE[scen]
    d = os.path.join(root, subdir) if subdir else root
    files = sorted(glob.glob(os.path.join(d, f'{stem}.*.nc')))
    return [f for f in files if not any(s in f for s in exclude_stamps)]

# tidesampwec's raw source has a trailing 1-timestep file
# (...20190429110056) whose zslice output has no time dimension at all
TIDESAMPWEC_EXCLUDE = ('20190429110056',)

def scenario_files_for(subdir, stem):
    files = {}
    for scen in SCENARIOS_ZSLICE:
        exclude = TIDESAMPWEC_EXCLUDE if scen == 'tidesampwec' else ()
        files[scen] = zslice_files(scen, subdir, stem, exclude_stamps=exclude)
    return files

W_FILES   = scenario_files_for(None,  'z_mc60_his')
NO3_FILES = scenario_files_for('bgc', 'z_mc60_bgc')
AKT_FILES = scenario_files_for('ak',  'z_mc60_his')

for name in SCENARIOS_ZSLICE:
    print(f'  {name}: {len(W_FILES[name])} w, {len(NO3_FILES[name])} no3, '
          f'{len(AKT_FILES[name])} akt zslice files')

with Dataset(W_FILES[BASE_SCEN][0], 'r') as _tmp:
    DEPTH_1D = np.array(_tmp.variables['depth'][:])   # (157,) metres, 0 to -1980, surface-first

# ---------------------------------------------------------------------------
# Full-domain time mean of w and NO3 (fixed zsliced depth grid) -- the
# anomaly reference for w'NO3'. Cached per scenario; reused from
# plot_cs_wno3_aktdno3dz_snap.py's ncra precompute when available (see
# module docstring).
# ---------------------------------------------------------------------------
os.makedirs(SAVEPATH, exist_ok=True)

def domain_mean_cache_path(scen):
    return f'{SAVEPATH}cs_wno3_aktdno3dz_avg_diff_box_domain_mean_{scen}.npz'


def load_domain_mean_ncra(scen):
    nscen = NCRA_SCEN_ALIAS.get(scen, scen)
    w_path   = f'{NCRA_DIR}w_mean_{nscen}.nc'
    no3_path = f'{NCRA_DIR}no3_mean_{nscen}.nc'
    if not (os.path.exists(w_path) and os.path.exists(no3_path)):
        return None
    with Dataset(w_path) as nc:
        depth_z = np.array(nc.variables['depth'][:])
        w_mean  = clean(np.squeeze(np.array(nc.variables['w']))) * mask_plot
    with Dataset(no3_path) as nc:
        no3_mean = clean(np.squeeze(np.array(nc.variables['NO3']))) * mask_plot
    return w_mean, no3_mean, depth_z


def compute_domain_mean_python(scen):
    """Full-domain (n_depth, eta_rho, xi_rho) time mean of w and NO3, read
    per timestep (each is ~525 MB at full domain) -- last-resort fallback
    when no ncra precompute exists for this scenario."""
    w_files, no3_files = W_FILES[scen], NO3_FILES[scen]
    n_pairs = min(len(w_files), len(no3_files))
    print(f'  [{scen}] no ncra mean found -- scanning {n_pairs} zsliced file '
          f'pairs in Python (slow; see module docstring for the ncra '
          f'alternative)', flush=True)

    shape = (DEPTH_1D.size,) + mask_plot.shape
    w_sum,   w_cnt   = np.zeros(shape), np.zeros(shape)
    no3_sum, no3_cnt = np.zeros(shape), np.zeros(shape)

    for fi in range(n_pairs):
        with Dataset(w_files[fi]) as wnc, Dataset(no3_files[fi]) as nnc:
            n_t = wnc.variables['w'].shape[0]
            for t in range(n_t):
                w3d    = clean(wnc.variables['w'][t])   * mask_plot
                no3_3d = clean(nnc.variables['NO3'][t]) * mask_plot
                wv  = ~np.isnan(w3d);    w_sum[wv]     += w3d[wv];      w_cnt[wv]     += 1
                nv  = ~np.isnan(no3_3d); no3_sum[nv]   += no3_3d[nv];   no3_cnt[nv]   += 1
        print(f'  [{scen}] domain mean scan: {fi + 1}/{n_pairs} file pairs done', flush=True)

    with np.errstate(invalid='ignore'):
        return w_sum / w_cnt, no3_sum / no3_cnt, DEPTH_1D


def get_domain_mean(scen):
    cp = domain_mean_cache_path(scen)
    if os.path.exists(cp):
        d = np.load(cp)
        return d['w_mean'], d['no3_mean'], d['depth_z']

    ncra = load_domain_mean_ncra(scen)
    if ncra is not None:
        print(f'  [{scen}] loaded ncra-precomputed domain mean', flush=True)
        w_mean, no3_mean, depth_z = ncra
    else:
        w_mean, no3_mean, depth_z = compute_domain_mean_python(scen)

    np.savez(cp, w_mean=w_mean, no3_mean=no3_mean, depth_z=depth_z)
    print(f'  [{scen}] saved domain mean cache -> {cp}', flush=True)
    return w_mean, no3_mean, depth_z

# ---------------------------------------------------------------------------
# Time+box-averaged w'NO3' and -Akt*dNO3/dz per scenario, per transect (cached)
# ---------------------------------------------------------------------------
def flux_cache_path(scen, tr_name):
    return f'{SAVEPATH}cs_wno3_aktdno3dz_avg_diff_box_cache_{scen}_{tr_name}.npz'


def compute_flux_means(scen, w_mean, no3_mean, depth_z):
    """Time+box-averaged w'NO3' and -Akt*dNO3/dz, one (n_depth, n_pts) array
    per flux per TRANSECTS entry. Forms the instantaneous product fields at
    every zsliced timestep (see module docstring for why this can't be
    deferred to after averaging), box-averages each with
    boxavg_section_fixedz (no vertical interpolation needed -- already a
    fixed depth grid), then accumulates the time mean via the same
    sum/count pattern as plot_cs_diag_avg_diff_box_3x2.py's
    compute_var_mean."""
    w_files, no3_files, akt_files = W_FILES[scen], NO3_FILES[scen], AKT_FILES[scen]
    n_pairs = min(len(w_files), len(no3_files), len(akt_files))
    print(f'  [{scen}] flux scan: {len(w_files)} w, {len(no3_files)} no3, '
          f'{len(akt_files)} akt files, pairing {n_pairs}', flush=True)

    # boxavg_section_fixedz does no vertical interpolation -- output stays on
    # the SAME native depth axis as the input (depth_z, 157 levels), not
    # tr['zgrid'] (that's only for the sigma-level boxavg_section)
    n_depth = depth_z.size
    sums   = {tr['name']: {k: np.zeros((n_depth, tr['n_pts']))
                            for k in ('wno3', 'aktdno3dz')} for tr in TRANSECTS}
    counts = {tr['name']: {k: np.zeros((n_depth, tr['n_pts']))
                            for k in ('wno3', 'aktdno3dz')} for tr in TRANSECTS}

    for fi in range(n_pairs):
        with Dataset(w_files[fi]) as wnc, Dataset(no3_files[fi]) as nnc, \
             Dataset(akt_files[fi]) as anc:
            # per timestep, not the whole file -- each zslice variable is
            # ~6.36 GB per file (12, 157, 1202, 702 float32), ~530 MB/timestep
            n_t = wnc.variables['w'].shape[0]
            for t in range(n_t):
                w3d    = clean(wnc.variables['w'][t])   * mask_plot
                no3_3d = clean(nnc.variables['NO3'][t]) * mask_plot
                akt3d  = clean(anc.variables['Akt'][t]) * mask_plot

                wno3_3d      = (w3d - w_mean) * (no3_3d - no3_mean)
                dno3dz_3d    = np.gradient(no3_3d, depth_z, axis=0)
                aktdno3dz_3d = -akt3d * dno3dz_3d

                for tr in TRANSECTS:
                    name = tr['name']
                    for key, field3d in (('wno3', wno3_3d), ('aktdno3dz', aktdno3dz_3d)):
                        field_box = cb.boxavg_section_fixedz(field3d, tr, min_frac=FLUX_MIN_FRAC)
                        valid = ~np.isnan(field_box)
                        sums[name][key][valid]   += field_box[valid]
                        counts[name][key][valid] += 1
        print(f'  [{scen}] flux scan: {fi + 1}/{n_pairs} file pairs done', flush=True)

    with np.errstate(invalid='ignore'):
        return {name: {key: sums[name][key] / counts[name][key] for key in ('wno3', 'aktdno3dz')}
                for name in sums}


print('\nComputing box+time-averaged w\'NO3\' / -Akt*dNO3/dz per scenario...')
flux_mean = {}   # scen -> {tr_name: {'wno3': arr, 'aktdno3dz': arr}}
for scen in SCENARIOS_ZSLICE:
    cached = [flux_cache_path(scen, tr['name']) for tr in TRANSECTS]
    if all(os.path.exists(c) for c in cached):
        print(f'[{scen}] loading flux cache...', flush=True)
        flux_mean[scen] = {tr['name']: dict(np.load(flux_cache_path(scen, tr['name'])))
                            for tr in TRANSECTS}
        continue

    if not (W_FILES[scen] and NO3_FILES[scen] and AKT_FILES[scen]):
        print(f'  WARNING: missing zslice files for {scen}, skipping', flush=True)
        continue

    print(f'[{scen}] getting domain mean...', flush=True)
    w_mean, no3_mean, depth_z = get_domain_mean(scen)
    print(f'[{scen}] computing box+time-averaged flux...', flush=True)
    result = compute_flux_means(scen, w_mean, no3_mean, depth_z)
    for tr in TRANSECTS:
        np.savez(flux_cache_path(scen, tr['name']), **result[tr['name']])
    flux_mean[scen] = result

# ---------------------------------------------------------------------------
# Plot: panel 1 = base case raw, panels 2-6 = diff from base case, one
# figure per (flux variable, transect), laid out as a 3x2 grid
# ---------------------------------------------------------------------------
for var_key, cfg in FLUX_CONFIGS.items():
    diff_scens = [s for s in SCENARIOS_ZSLICE if s != BASE_SCEN and s in flux_mean]
    if BASE_SCEN not in flux_mean or not diff_scens:
        print(f'  WARNING: missing base case or no scenarios to diff for {var_key}, skipping plot')
        continue

    for ti, tr in enumerate(TRANSECTS):
        fname = f'{SAVEPATH}{cfg["out_stem"]}_{tr["name"]}.png'

        n_panels = 1 + len(diff_scens)
        ncols = 2
        nrows = math.ceil(n_panels / ncols)
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                                 figsize=(14, 4 * nrows))
        axes_flat = np.atleast_1d(axes).flatten()
        for ax in axes_flat[n_panels:]:
            ax.axis('off')

        keep = DEPTH_1D >= tr['depth_lim']

        def _format_ax(ax, row, col):
            if tr.get('xlim') is not None:
                ax.set_xlim(tr['xlim'])
            if tr.get('xticks') is not None:
                ax.set_xticks(tr['xticks'])
            ax.set_ylim([tr['depth_lim'], 0])
            sf = ScalarFormatter(useOffset=False)
            sf.set_scientific(False)
            ax.xaxis.set_major_formatter(sf)
            if tr.get('xticks') is None:
                ax.xaxis.set_major_locator(MaxNLocator(5))
            if col == 0:
                ax.set_ylabel('Depth (m)')
            if row == nrows - 1:
                ax.set_xlabel('Longitude')
            ax.label_outer()

        # panel 1: base case, raw time-mean -- fixed range, see module
        # docstring (matches ../plot_cs_wno3_aktdno3dz_snap.py's VMAX_*)
        ax_raw = axes_flat[0]
        base_field = flux_mean[BASE_SCEN][tr['name']][var_key][keep, :]
        rv = cfg['raw_vmax']
        pc_raw = ax_raw.pcolormesh(tr['lon'], DEPTH_1D[keep], base_field,
                                   cmap=cfg['raw_cmap'], vmin=-rv, vmax=rv,
                                   shading='nearest')
        ax_raw.set_title(LABELS[BASE_SCEN])
        _format_ax(ax_raw, 0, 0)

        # panels 2-6: diff from base case
        diffs = {s: flux_mean[s][tr['name']][var_key] - flux_mean[BASE_SCEN][tr['name']][var_key]
                 for s in diff_scens}
        vmax = np.nanpercentile(
            np.abs(np.concatenate([d[keep, :].ravel() for d in diffs.values()])), 98)

        pc_diff = None
        for i, scen in enumerate(diff_scens, start=1):
            ax = axes_flat[i]
            row, col = divmod(i, ncols)
            pc_diff = ax.pcolormesh(tr['lon'], DEPTH_1D[keep], diffs[scen][keep, :],
                                    cmap=cfg['diff_cmap'], vmin=-vmax, vmax=vmax, shading='nearest')
            ax.set_title(f'{LABELS[scen]}  −  {LABELS[BASE_SCEN]}')
            _format_ax(ax, row, col)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.3)
        fig.canvas.draw()

        # two colorbars: raw (below panel 1 only) + diff (right of the grid) --
        # same centered-in-the-gap placement as the sibling _3x2 scripts
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

        pos_tr  = axes_flat[1].get_position()
        pos_br  = axes_flat[n_panels - 1].get_position()
        cax_diff = fig.add_axes([pos_tr.x1 + 0.015, pos_br.y0, 0.015, pos_tr.y1 - pos_br.y0])
        fig.colorbar(pc_diff, cax=cax_diff, label=cfg['label'])

        plt.savefig(fname, dpi=800, bbox_inches='tight')
        plt.close()
        print(f'  saved -> {fname}')
