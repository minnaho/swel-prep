"""
RMSE(depth) profile of the vertical density gradient (drho/dz), testing
whether WEC changes the strength of tidal-bore-driven stratification --
density-gradient counterpart of calc_dudz_rmse_wec.py, same structure and
same 3-statistic consolidation (RMSE + std/Delta-std + RMS in one pass, one
script, shelf+offshore combined from the same read -- see that script's
docstring for the full rationale, only the derived field differs here).

Three comparisons, all against the notidesnowec base case (same BASE_SCEN
convention used throughout plot/postprocessing):
  ampwec_notides -- notidesampwec vs notidesnowec  (WEC alone, tides OFF)
  tides_nowec    -- tidesnowec    vs notidesnowec  (tides alone, no WEC)
  tides_ampwec   -- tidesampwec   vs notidesnowec  (tides + WEC together)

RMSE(depth) is a POOLED spatial+temporal root-mean-square, matching
calc_dudz_rmse_wec.py's definition exactly:

    RMSE(depth) = sqrt( sum_t sum_(eta,xi in mask) (drhodz_other - drhodz_base)^2
                        / count_valid_samples(depth) )

Std(depth) is the POOLED population standard deviation of the raw field
(not a diff), one running sum + running sum-of-squares per scenario,
divided once at the end -- identical formula/rationale to calc_w_std.py /
calc_dudz_rmse_wec.py:

    Var(depth) = sum_t sum_(eta,xi in mask) drhodz^2 / count
                 - ( sum_t sum_(eta,xi in mask) drhodz / count )^2
    std(depth) = sqrt(max(Var(depth), 0))
    dstd(depth) = std(scenario) - std(notidesnowec)

RMS(depth) = sqrt(mean_sq) is free once std is computed, same as
calc_dudz_rmse_wec.py -- no extra accumulator or I/O.

Masks: shelf = eta_rho index > ETA_MIN AND h < H_MAX AND mask_rho==1;
offshore = eta_rho index > ETA_MIN AND h > H_MIN AND mask_rho==1
(grid-index-space convention -- same masks as calc_w_std.py /
calc_dudz_rmse_wec.py / calc_vort_rmse_wec_shelf.py).

drho/dz derivation: zsliced 'rho' (root z_mc60_his.*.nc, same file as u/v/w)
is a deviation from a reference density (see plot_cs_diag_rho.py's
RHO_OFFSET = RHO_REF - 1000.0) -- that offset is a per-timestep-and-column
CONSTANT, so it cancels exactly under a depth derivative
(d(rho+C)/dz == drho/dz) and is deliberately NOT added back here, unlike
the plot_cs_diag_rho.py/plot_cs_diag_drhodz_diff.py family which needs true
density for other purposes (e.g. isopycnal contours). Differentiated once
along the fixed zslice depth axis with np.gradient, same as
calc_dudz_rmse_wec.py -- no box-averaging/transects, so no thin-support
concern; np.gradient just returns NaN at a column's deepest valid cell when
its one-sided neighbor is already below the seafloor.

Same "read the base once per timestep, reuse across all 3 comparisons"
optimization as calc_dudz_rmse_wec.py.

Usage:
  python -u calc_drhodz_rmse_wec.py [--nfiles N]

Cost note: reads a single field (rho) per scenario, same ~500+ GB order of
magnitude as calc_w_std.py -- about half calc_dudz_rmse_wec.py's cost since
that script needs both u and v. Use --nfiles for a quick smoke test before
committing to the full run.

Output: ./drhodz_rmse_wec.npz
  depth -- (157,) m, 0 to -1980, surface-first
  rmse_shelf_<key>, rmse_offshore_<key> -- (157,) each, pooled
    spatiotemporal RMSE(drho/dz) profile vs notidesnowec, key in
    ampwec_notides, tides_nowec, tides_ampwec
  n_rmse_shelf_<key>, n_rmse_offshore_<key> -- (157,) each, valid sample
    count backing each RMSE point
  std_shelf_base, std_offshore_base -- (157,) each, pooled std(drho/dz) for
    notidesnowec
  std_shelf_<key>, std_offshore_<key> -- (157,) each, pooled std(drho/dz)
    for the comparison scenario
  dstd_shelf_<key>, dstd_offshore_<key> -- (157,) each,
    std_..._<key> - std_..._base
  rms_shelf_base, rms_offshore_base -- (157,) each, pooled RMS(drho/dz) for
    notidesnowec
  rms_shelf_<key>, rms_offshore_<key> -- (157,) each, pooled RMS(drho/dz)
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


def load_rho(nc, t):
    rho = np.array(nc.variables['rho'][t], dtype=np.float32)   # (depth, eta_rho, xi_rho)
    rho[np.abs(rho) > FILL_THRESH] = np.nan
    return rho


def load_drhodz(nc, t, depth):
    rho = load_rho(nc, t)
    return np.gradient(rho, depth, axis=0)


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
                drhodz_base = load_drhodz(ncb, t, depth)
                accum_std('base', drhodz_base)
                for key in COMPARISONS:
                    drhodz_other = load_drhodz(nco[key], t, depth)
                    accum_std(key, drhodz_other)
                    diff = drhodz_other - drhodz_base
                    for region, mask2d in masks.items():
                        diff_m = diff[:, mask2d]          # (nz, n_masked)
                        valid = np.isfinite(diff_m)
                        sq_sum_rmse[key][region] += np.sum(
                            np.where(valid, diff_m * diff_m, 0.0), axis=1)
                        count_rmse[key][region] += valid.sum(axis=1)
                        del diff_m, valid
                    del drhodz_other, diff
                del drhodz_base
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

    outfile = 'drhodz_rmse_wec.npz'
    np.savez(outfile, **result)
    print(f'saved -> {outfile}')
