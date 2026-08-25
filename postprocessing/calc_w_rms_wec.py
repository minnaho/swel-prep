"""
RMS(depth) profile of vertical velocity `w` for each of the 4 scenarios on
its own -- plain magnitude, NOT a difference against the notidesnowec base
case (see calc_w_rmse_wec_shelf.py / _offshore.py for that). RMS-only
counterpart of calc_w_std.py, following the same "rms" vs "rmse" vocabulary
as calc_w_rms_pycnocline.py elsewhere in this directory. Shelf and offshore
are computed from the SAME read in one pass, same as calc_w_std.py.

RMS(depth) is a POOLED spatial+temporal root-mean-square of the raw field
(no mean subtracted, unlike calc_w_std.py's std -- RMS keeps the
mean-square):

    RMS(depth) = sqrt( sum_t sum_(eta,xi in mask) w^2
                        / count_valid_samples(depth) )

Masks: shelf = eta_rho index > ETA_MIN AND h < H_MAX AND mask_rho==1;
offshore = eta_rho index > ETA_MIN AND h > H_MIN AND mask_rho==1 --
identical masks to calc_w_std.py / calc_w_rmse_wec_shelf.py.

All 4 scenarios (base + 3 comparisons) are read in this single invocation --
no per-scenario CLI arg and no launch-all loop, same as calc_w_std.py,
purely so the same 4 npz keys line up 1:1 with calc_w_std.py's/
calc_w_rmse_wec_shelf.py's COMPARISONS keys for easy joint plotting.

Usage:
  python -u calc_w_rms_wec.py [--nfiles N]

Cost note: same ~500+ GB zslice I/O as calc_w_std.py (same 4 scenarios x 1
field). Use --nfiles for a quick smoke test before committing to the full
run.

Output: ./w_rms_wec.npz
  depth -- (157,) m, 0 to -1980, surface-first
  rms_shelf_base, rms_offshore_base -- (157,) each, pooled RMS(w) for
    notidesnowec
  rms_shelf_<key>, rms_offshore_<key> -- (157,) each, pooled RMS(w) for the
    comparison scenario, key in ampwec_notides, tides_nowec, tides_ampwec
  n_shelf_base, n_offshore_base, n_shelf_<key>, n_offshore_<key> -- (157,)
    each, valid sample count backing each RMS point
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


def load_w(nc, t):
    arr = np.array(nc.variables['w'][t], dtype=np.float32)   # (depth, eta_rho, xi_rho)
    arr[np.abs(arr) > FILL_THRESH] = np.nan
    return arr


def build_masks(eta_rho):
    grdnc = Dataset(GRD, 'r')
    h = np.array(grdnc.variables['h'])
    mask_rho = np.array(grdnc.variables['mask_rho'])
    grdnc.close()
    eta_idx = np.arange(eta_rho)[:, None]
    base = (eta_idx > ETA_MIN) & (mask_rho == 1)
    return {'shelf': base & (h < H_MAX), 'offshore': base & (h > H_MIN)}


def compute_all_profiles(nfiles=None):
    scen_files = {'base': zfiles_for(BASE_SCEN)}
    scen_files.update({key: zfiles_for(scen) for key, scen in COMPARISONS.items()})
    if nfiles is not None:
        scen_files = {k: v[:nfiles] for k, v in scen_files.items()}
    if not scen_files['base']:
        raise RuntimeError(f'no zslice files found for {BASE_SCEN}')
    n = len(scen_files['base'])
    for key in COMPARISONS:
        if len(scen_files[key]) != n:
            raise RuntimeError(
                f'{BASE_SCEN} has {n} zslice files but {COMPARISONS[key]} has '
                f'{len(scen_files[key])} -- cannot pair by index')

    with Dataset(scen_files['base'][0], 'r') as nc0:
        depth = np.array(nc0.variables['depth'][:])   # (157,) 0..-1980, surface-first
        eta_rho = nc0.dimensions['eta_rho'].size
    nz = depth.size
    masks = build_masks(eta_rho)
    print(f'shelf mask: {masks["shelf"].sum()} cells, '
          f'offshore mask: {masks["offshore"].sum()} cells')

    scen_keys = ['base'] + list(COMPARISONS)
    sq_sums = {sk: {r: np.zeros(nz, dtype=np.float64) for r in masks} for sk in scen_keys}
    counts = {sk: {r: np.zeros(nz, dtype=np.float64) for r in masks} for sk in scen_keys}

    for fi in range(n):
        fname = os.path.basename(scen_files['base'][fi])
        print(f'[{fi + 1}/{n}] {fname}', flush=True)
        ncs = {sk: Dataset(scen_files[sk][fi], 'r') for sk in scen_keys}
        try:
            nt = ncs['base'].dimensions['time'].size
            for t in range(nt):
                for sk in scen_keys:
                    w = load_w(ncs[sk], t)
                    for region, mask2d in masks.items():
                        w_m = w[:, mask2d]              # (nz, n_masked)
                        valid = np.isfinite(w_m)
                        w_valid = np.where(valid, w_m, 0.0)
                        sq_sums[sk][region] += np.sum(w_valid * w_valid, axis=1)
                        counts[sk][region] += valid.sum(axis=1)
                        del w_m, valid, w_valid
                    del w
        finally:
            for nc in ncs.values():
                nc.close()

    result = {'depth': depth}
    with np.errstate(invalid='ignore'):
        for sk in scen_keys:
            for region in masks:
                cnt = counts[sk][region]
                result[f'rms_{region}_{sk}'] = np.sqrt(np.where(
                    cnt > 0, sq_sums[sk][region] / np.maximum(cnt, 1e-30), np.nan))
                result[f'n_{region}_{sk}'] = cnt
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nfiles', type=int, default=None,
                        help='process only the first N zslice files per scenario (smoke test)')
    args = parser.parse_args()

    result = compute_all_profiles(nfiles=args.nfiles)

    outfile = 'w_rms_wec.npz'
    np.savez(outfile, **result)
    print(f'saved -> {outfile}')
