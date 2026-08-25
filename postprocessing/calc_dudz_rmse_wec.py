"""
RMSE(depth) profile of the vertical shear of the NATIVE (grid-relative)
u-velocity component (du/dz), testing whether WEC changes the strength of
tidal-bore-driven shear -- shear counterpart of
calc_vort_rmse_wec_shelf.py / _offshore.py. Unlike those two (split into
separate shelf/offshore scripts), shelf and offshore are computed here from
the SAME read in one pass and combined into a single script -- they read
the same u file, only the mask differs, so there's no cost-tier reason to
split them (same rationale as calc_w_std.py's shelf+offshore merge).

Uses native u, not geographic east, and v is not read at all: the grid's
`angle` field is essentially constant over the shelf+offshore mask
(25.0 deg +/- 0.09 deg, verified directly) -- this grid is nearly
rectilinear, just rotated by one fixed angle, so native u is a
well-defined, spatially-consistent axis across the whole domain rather
than a grid artifact that shifts meaning cell-to-cell. It also matches
plot_cs_diag.py's own convention, where -xi is treated as the offshore
direction for its cross-shore transects. Rotating to geographic east
(u_east = u_rho*cos(25deg) - v_rho*sin(25deg)) would only capture ~91% of
the true cross-shore signal while mixing in ~42% of the alongshore (v)
component -- since tidal bores propagate cross-shore, that dilutes the
signal this script is meant to isolate with alongshore variability that
isn't the mechanism of interest. (Earlier version of this script rotated
to east; switched to native u after this reasoning was confirmed.)

Also computes std(depth) (and Delta-std vs the base case) and RMS(depth)
for all 4 scenarios in the same pass -- folded in here rather than separate
calc_dudz_std.py / calc_dudz_rms_wec.py scripts because neither statistic
needs anything dudz_other/dudz_base don't already have in memory each
timestep. RMS in particular is free once std is computed: std's own
intermediate mean_sq = E[dudz^2] (before subtracting mean^2) IS
RMS = sqrt(E[dudz^2]), so no extra accumulator or I/O is needed for it at
all. This mirrors calc_w_rms_pycnocline.py's precedent of computing
multiple related pooled statistics from one expensive read rather than
re-reading the same ~1 TB+ of zslice files once per statistic.

Three comparisons, all against the notidesnowec base case (same BASE_SCEN
convention used throughout plot/postprocessing):
  ampwec_notides -- notidesampwec vs notidesnowec  (WEC alone, tides OFF)
  tides_nowec    -- tidesnowec    vs notidesnowec  (tides alone, no WEC)
  tides_ampwec   -- tidesampwec   vs notidesnowec  (tides + WEC together)

RMSE(depth) is a POOLED spatial+temporal root-mean-square, matching
calc_vort_rmse_wec_shelf.py's definition exactly:

    RMSE(depth) = sqrt( sum_t sum_(eta,xi in mask) (dudz_other - dudz_base)^2
                        / count_valid_samples(depth) )

Std(depth) is the POOLED population standard deviation of the raw field
(not a diff), one running sum + running sum-of-squares per scenario,
divided once at the end -- identical formula/rationale to calc_w_std.py:

    Var(depth) = sum_t sum_(eta,xi in mask) dudz^2 / count
                 - ( sum_t sum_(eta,xi in mask) dudz / count )^2
    std(depth) = sqrt(max(Var(depth), 0))
    dstd(depth) = std(scenario) - std(notidesnowec)

Masks: shelf = eta_rho index > ETA_MIN AND h < H_MAX AND mask_rho==1;
offshore = eta_rho index > ETA_MIN AND h > H_MIN AND mask_rho==1
(grid-index-space convention -- same masks as calc_w_std.py /
calc_vort_rmse_wec_shelf.py).

du/dz derivation: u is read on its native zslice grid (eta_rho x xi_u) at
each of the 157 fixed depth levels, interpolated to the rho grid in the xi
direction only (interior-average + edge-copy, same scheme as
pyfuncs.rho_uv_angle / calc_vort_rmse_wec_shelf.py's uv_to_rho, just the u
half -- v is never read or interpolated), then differentiated once along
the fixed zslice depth axis with np.gradient. No rotation step. No
box-averaging is involved here (unlike
plot_cs_diag_avg_diff_box_3x2.py's transect-based dudz), so there's no
thin-support-spike concern from partial box coverage -- np.gradient simply
returns NaN at a column's deepest valid cell when its one-sided neighbor is
already below the seafloor, which is the expected, non-spurious behavior.

Same "read the base once per timestep, reuse across all 3 comparisons"
optimization as calc_vort_rmse_wec_shelf.py.

Usage:
  python -u calc_dudz_rmse_wec.py [--nfiles N]

Cost note: only u is read now (v dropped along with the rotation step) --
roughly half calc_vort_rmse_wec_shelf.py's ~1 TB+ zslice I/O (which needs
both u and v), same order of magnitude as calc_w_std.py / calc_drhodz_rmse_wec.py.
Use --nfiles for a quick smoke test before committing to the full run.

Output: ./dudz_rmse_wec.npz
  depth -- (157,) m, 0 to -1980, surface-first
  rmse_shelf_<key>, rmse_offshore_<key> -- (157,) each, pooled
    spatiotemporal RMSE(du/dz, native u) profile vs notidesnowec, key in
    ampwec_notides, tides_nowec, tides_ampwec
  n_rmse_shelf_<key>, n_rmse_offshore_<key> -- (157,) each, valid sample
    count backing each RMSE point
  std_shelf_base, std_offshore_base -- (157,) each, pooled std(du/dz)
    for notidesnowec
  std_shelf_<key>, std_offshore_<key> -- (157,) each, pooled std(du/dz)
    for the comparison scenario
  dstd_shelf_<key>, dstd_offshore_<key> -- (157,) each,
    std_..._<key> - std_..._base
  rms_shelf_base, rms_offshore_base -- (157,) each, pooled RMS(du/dz)
    for notidesnowec
  rms_shelf_<key>, rms_offshore_<key> -- (157,) each, pooled RMS(du/dz)
    for the comparison scenario
  n_std_shelf_base, n_std_offshore_base, n_std_shelf_<key>,
    n_std_offshore_<key> -- (157,) each, valid sample count backing each
    std/rms point (RMS and std share the same accumulators, so the same
    count applies to both)
"""

import os
import argparse
import glob
import numpy as np
from netCDF4 import Dataset

ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'
GRD = '../mc60_grd.nc'

ETA_MIN = 186     # keep eta_rho index > this (grid-index space)
H_MAX   = 100.0   # shelf: keep h < this (m)
H_MIN   = 100.0   # offshore: keep h > this (m)

FILL_THRESH = 0.9e33

# tidesampwec's raw source has a trailing 1-timestep file whose zslice
# output has no time dimension at all -- same exclusion used throughout
# postprocessing/ and plot/
TIDESAMPWEC_EXCLUDE = ('20190429110056',)

BASE_SCEN = 'notidesnowec'
COMPARISONS = {
    'ampwec_notides': 'notidesampwec',
    'tides_nowec':     'tidesnowec',
    'tides_ampwec':    'tidesampwec',
}


def zfiles_for(scen):
    files = sorted(glob.glob(os.path.join(ZSLICE_ROOT, scen, 'z_mc60_his.*.nc')))
    if scen == 'tidesampwec':
        files = [f for f in files if not any(x in f for x in TIDESAMPWEC_EXCLUDE)]
    return files


def load_u(nc, t):
    u = np.array(nc.variables['u'][t], dtype=np.float32)   # (depth, eta_rho, xi_u)
    u[np.abs(u) > FILL_THRESH] = np.nan
    return u


def u_to_rho(u):
    """u (depth,eta_rho,xi_u) -> u_rho (depth,eta_rho,xi_rho); same
    interior-average / edge-copy scheme as pyfuncs.rho_uv_angle /
    calc_vort_rmse_wec_shelf.py's uv_to_rho, just the u half."""
    u_temp = 0.5 * (u[:, :, 1:] + u[:, :, :-1])
    u_rho = np.empty((u.shape[0], u.shape[1], u.shape[2] + 1), dtype=np.float32)
    u_rho[:, :, 1:-1] = u_temp
    u_rho[:, :, 0] = u_temp[:, :, 0]
    u_rho[:, :, -1] = u_temp[:, :, -1]
    return u_rho


def load_dudz(nc, t, depth):
    u = load_u(nc, t)
    u_rho = u_to_rho(u)
    return np.gradient(u_rho, depth, axis=0)


def build_masks(eta_rho):
    grdnc = Dataset(GRD, 'r')
    h = np.array(grdnc.variables['h'])
    mask_rho = np.array(grdnc.variables['mask_rho'])
    grdnc.close()
    eta_idx = np.arange(eta_rho)[:, None]
    base = (eta_idx > ETA_MIN) & (mask_rho == 1)
    return {'shelf': base & (h < H_MAX), 'offshore': base & (h > H_MIN)}


def compute_all_profiles(nfiles=None):
    files_base = zfiles_for(BASE_SCEN)
    files_other = {key: zfiles_for(scen) for key, scen in COMPARISONS.items()}
    if nfiles is not None:
        files_base = files_base[:nfiles]
        files_other = {k: v[:nfiles] for k, v in files_other.items()}
    if not files_base:
        raise RuntimeError(f'no zslice files found for {BASE_SCEN}')
    for key, scen in COMPARISONS.items():
        if len(files_other[key]) != len(files_base):
            raise RuntimeError(
                f'{BASE_SCEN} has {len(files_base)} zslice files but {scen} has '
                f'{len(files_other[key])} -- cannot pair by index')

    with Dataset(files_base[0], 'r') as nc0:
        depth = np.array(nc0.variables['depth'][:])   # (157,) 0..-1980, surface-first
        eta_rho = nc0.dimensions['eta_rho'].size
    nz = depth.size
    masks = build_masks(eta_rho)
    print(f'shelf mask: {masks["shelf"].sum()} cells, '
          f'offshore mask: {masks["offshore"].sum()} cells '
          f'(eta_rho > {ETA_MIN})')

    # RMSE (diff-vs-base) accumulators, keyed by COMPARISONS only
    sq_sum_rmse = {key: {r: np.zeros(nz, dtype=np.float64) for r in masks} for key in COMPARISONS}
    count_rmse = {key: {r: np.zeros(nz, dtype=np.float64) for r in masks} for key in COMPARISONS}

    # std (raw-field) accumulators, keyed by scen_keys = base + COMPARISONS
    scen_keys = ['base'] + list(COMPARISONS)
    sums_std = {sk: {r: np.zeros(nz, dtype=np.float64) for r in masks} for sk in scen_keys}
    sq_sums_std = {sk: {r: np.zeros(nz, dtype=np.float64) for r in masks} for sk in scen_keys}
    counts_std = {sk: {r: np.zeros(nz, dtype=np.float64) for r in masks} for sk in scen_keys}

    def accum_std(sk, field):
        for region, mask2d in masks.items():
            f_m = field[:, mask2d]              # (nz, n_masked)
            valid = np.isfinite(f_m)
            f_valid = np.where(valid, f_m, 0.0)
            sums_std[sk][region] += np.sum(f_valid, axis=1)
            sq_sums_std[sk][region] += np.sum(f_valid * f_valid, axis=1)
            counts_std[sk][region] += valid.sum(axis=1)

    n = len(files_base)
    for fi in range(n):
        fb = files_base[fi]
        print(f'[{fi + 1}/{n}] {os.path.basename(fb)}', flush=True)
        ncb = Dataset(fb, 'r')
        nco = {key: Dataset(files_other[key][fi], 'r') for key in COMPARISONS}
        try:
            nt = ncb.dimensions['time'].size
            for t in range(nt):
                dudz_base = load_dudz(ncb, t, depth)
                accum_std('base', dudz_base)
                for key in COMPARISONS:
                    dudz_other = load_dudz(nco[key], t, depth)
                    accum_std(key, dudz_other)
                    diff = dudz_other - dudz_base
                    for region, mask2d in masks.items():
                        diff_m = diff[:, mask2d]          # (nz, n_masked)
                        valid = np.isfinite(diff_m)
                        sq_sum_rmse[key][region] += np.sum(
                            np.where(valid, diff_m * diff_m, 0.0), axis=1)
                        count_rmse[key][region] += valid.sum(axis=1)
                        del diff_m, valid
                    del dudz_other, diff
                del dudz_base
        finally:
            ncb.close()
            for nc in nco.values():
                nc.close()

    result = {'depth': depth}
    with np.errstate(invalid='ignore'):
        for key in COMPARISONS:
            for region in masks:
                cnt = count_rmse[key][region]
                result[f'rmse_{region}_{key}'] = np.sqrt(np.where(
                    cnt > 0, sq_sum_rmse[key][region] / np.maximum(cnt, 1e-30), np.nan))
                result[f'n_rmse_{region}_{key}'] = cnt

        std = {sk: {} for sk in scen_keys}
        for sk in scen_keys:
            for region in masks:
                cnt = counts_std[sk][region]
                mean = np.where(
                    cnt > 0, sums_std[sk][region] / np.maximum(cnt, 1e-30), np.nan)
                mean_sq = np.where(
                    cnt > 0, sq_sums_std[sk][region] / np.maximum(cnt, 1e-30), np.nan)
                var = np.maximum(mean_sq - mean * mean, 0.0)
                std[sk][region] = np.sqrt(var)
                result[f'std_{region}_{sk}'] = std[sk][region]
                result[f'rms_{region}_{sk}'] = np.sqrt(mean_sq)
                result[f'n_std_{region}_{sk}'] = cnt
        for key in COMPARISONS:
            for region in masks:
                result[f'dstd_{region}_{key}'] = std[key][region] - std['base'][region]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nfiles', type=int, default=None,
                        help='process only the first N zslice files per scenario (smoke test)')
    args = parser.parse_args()

    result = compute_all_profiles(nfiles=args.nfiles)

    outfile = 'dudz_rmse_wec.npz'
    np.savez(outfile, **result)
    print(f'saved -> {outfile}')
