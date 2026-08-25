"""
RMSE(depth) profile of vertical velocity `w`, testing whether WEC changes
the strength of tidal-bore-driven w offshore. Offshore counterpart of
calc_w_rmse_wec_shelf.py -- same calculation, opposite side of the h=100 m
split (h > H_MIN here instead of h < H_MAX).

Three comparisons, all against the notidesnowec base case (same BASE_SCEN
convention used throughout plot/postprocessing):
  ampwec_notides -- notidesampwec vs notidesnowec  (WEC alone, tides OFF)
  tides_nowec    -- tidesnowec    vs notidesnowec  (tides alone, no WEC)
  tides_ampwec   -- tidesampwec   vs notidesnowec  (tides + WEC together)

Comparing the 3 profiles answers the question directly: if WEC changes
tidal-bore strength (not just adding its own tide-independent w signature),
the tides_ampwec line won't simply look like tides_nowec's line -- it'll
diverge from it by more than ampwec_notides's own (tide-independent)
magnitude would predict.

RMSE(depth) is a POOLED spatial+temporal root-mean-square: at each zslice
depth level, every (time step, masked grid cell) sample across the whole
run is squared, summed, and divided by the total valid-sample count once,
then sqrt'd once at the end -- not RMS-over-time-then-averaged-over-space,
or vice versa:

    RMSE(depth) = sqrt( sum_t sum_(eta,xi in mask) (w_other - w_base)^2
                        / count_valid_samples(depth) )

Mask: offshore region, `eta_rho index > ETA_MIN` AND `h > H_MIN` AND
`mask_rho==1` (grid-index-space convention, same as ETA_MID/SPONGE
elsewhere in plot/ -- not a lat/lon bound). 291,666 cells, h up to ~2352 m
-- unlike the shelf mask (capped at h<100m, so only the shallowest ~60 of
157 depth levels ever have data), this mask reaches every depth level down
to the seafloor's deepest point, so expect a much taller profile.

`w` lives on the zsliced fixed 157-level depth grid (same depth axis every
time step, unlike raw s_rho) -- zslicefull/<scen>/z_mc60_his.*.nc. No
vertical interpolation needed, matching calc_w_rms_pycnocline.py's approach
for the same variable.

All 3 comparisons share the same notidesnowec base -- read it ONCE per
timestep and reused across all 3, rather than once per comparison (which
would reread the same 21 base files 3x over). Cuts total zslice reads from
4x21x12 to (1+3)x21x12, ~33% less I/O on an already ~750 GB job.

Usage:
  python -u calc_w_rmse_wec_offshore.py [--nfiles N]

Cost note: full run is ~21 files x 12 timesteps x 4 reads (1 base + 3
other scenarios) x ~530 MB/read =~ 500+ GB of zslice I/O -- same order of
magnitude as calc_w_rms_pycnocline.py's per-scenario cost (the offshore
mask being larger than the shelf mask doesn't change this -- both masks
subselect from the same full-domain read). Use --nfiles for a quick smoke
test before committing to the full run.

Output: ./w_rmse_wec_offshore.npz
  depth -- (157,) m, 0 to -1980, surface-first
  rmse_ampwec_notides, rmse_tides_nowec, rmse_tides_ampwec -- (157,) each,
    pooled spatiotemporal RMSE(w) profile vs notidesnowec
  n_ampwec_notides, n_tides_nowec, n_tides_ampwec -- (157,) each, valid
    sample count backing each RMSE point
"""

import os
import argparse
import glob
import numpy as np
from netCDF4 import Dataset

ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'
GRD = '../mc60_grd.nc'

ETA_MIN = 186     # keep eta_rho index > this (grid-index space)
H_MIN   = 100.0   # keep h > this (m)

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


def load_w(nc, t):
    arr = np.array(nc.variables['w'][t], dtype=np.float32)   # (depth, eta_rho, xi_rho)
    arr[np.abs(arr) > FILL_THRESH] = np.nan
    return arr


def build_mask(eta_rho):
    grdnc = Dataset(GRD, 'r')
    h = np.array(grdnc.variables['h'])
    mask_rho = np.array(grdnc.variables['mask_rho'])
    grdnc.close()
    eta_idx = np.arange(eta_rho)[:, None]
    return (eta_idx > ETA_MIN) & (h > H_MIN) & (mask_rho == 1)


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
    mask2d = build_mask(eta_rho)
    print(f'offshore mask: {mask2d.sum()} cells (eta_rho > {ETA_MIN}, h > {H_MIN} m)')

    sq_sum = {key: np.zeros(nz, dtype=np.float64) for key in COMPARISONS}
    count = {key: np.zeros(nz, dtype=np.float64) for key in COMPARISONS}

    n = len(files_base)
    for fi in range(n):
        fb = files_base[fi]
        print(f'[{fi + 1}/{n}] {os.path.basename(fb)}', flush=True)
        ncb = Dataset(fb, 'r')
        nco = {key: Dataset(files_other[key][fi], 'r') for key in COMPARISONS}
        try:
            nt = ncb.dimensions['time'].size
            for t in range(nt):
                w_base = load_w(ncb, t)
                for key in COMPARISONS:
                    w_other = load_w(nco[key], t)
                    diff = w_other - w_base
                    diff_m = diff[:, mask2d]              # (nz, n_masked)
                    valid = np.isfinite(diff_m)
                    sq_sum[key] += np.sum(np.where(valid, diff_m * diff_m, 0.0), axis=1)
                    count[key] += valid.sum(axis=1)
                    del w_other, diff, diff_m, valid
                del w_base
        finally:
            ncb.close()
            for nc in nco.values():
                nc.close()

    result = {'depth': depth}
    with np.errstate(invalid='ignore'):
        for key in COMPARISONS:
            result[f'rmse_{key}'] = np.sqrt(np.where(
                count[key] > 0, sq_sum[key] / np.maximum(count[key], 1e-30), np.nan))
            result[f'n_{key}'] = count[key]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nfiles', type=int, default=None,
                        help='process only the first N zslice files per scenario (smoke test)')
    args = parser.parse_args()

    result = compute_all_profiles(nfiles=args.nfiles)

    outfile = 'w_rmse_wec_offshore.npz'
    np.savez(outfile, **result)
    print(f'saved -> {outfile}')
