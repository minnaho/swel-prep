"""
Spatial map of the RMS of sub-pycnocline vertical velocity `w`, and the RMSE
of `w` against the notidesnowec base case, both time- and depth-averaged.

`w` is squared first, then averaged over sub-pycnocline depth AND time
together (dz-weighted, since the zslice depth grid is not uniform), and the
square root is taken exactly once at the end:

    RMS(x,y)  = sqrt( sum_t sum_k dz_k w^2            / sum_t sum_k dz_k )
    RMSE(x,y) = sqrt( sum_t sum_k dz_k (w - w_base)^2 / sum_t sum_k dz_k )

Averaging `w` over depth first (then differencing in time) is NOT used here:
`w` changes sign with depth (internal-wave vertical modes), so a layer mean
largely cancels and would map the RMSE of a small residual rather than the
true sub-pycnocline vertical-velocity difference.

The pycnocline (1025 kg/m^3 isopycnal) is defined per scenario, per time
step, per column, as the shallowest zslice level at or below which stored
`rho` first reaches the threshold -- everything from there down is
"sub-pycnocline" for that column at that instant. The RMSE diff mask uses
the EVALUATED scenario's own pycnocline mask (not the base case's, and not
their intersection).

Two depth floors are accumulated in the same pass: the full water column
below the pycnocline, and a pycnocline-to-300 m band (DEPTH_LIM_MID
convention used elsewhere in plot/). Offshore the full column can run to
~900+ m of nearly quiescent deep water while the pycnocline itself sits at
~25-40 m, so the two floors can look very different -- the 300 m band keeps
the layer thickness comparable between shelf and offshore.

`w` and `rho` are both on the fixed 157-level zslice depth grid in the same
file (zslicefull/<scen>/z_mc60_his.*.nc), so no vertical interpolation is
needed.

Stored `rho` is a deviation from RHO_REF = 1027.4 (NOT full density, NOT
sigma-t) -- confirmed via ncdump: stored values run ~-3 to -1. The 1025
kg/m^3 surface is therefore `rho_stored >= 1025.0 - 1027.4 = -2.4`; compared
directly against the stored value, no offset conversion needed.

Usage:
  python -u calc_w_rms_pycnocline.py <scenario> [--nfiles N]
  scenario in: tideswec, tidesnowec, notidesnowec, notideswec,
               notidesampwec, tidesampwec

Output: ./w_rms_pycnocline_<scenario>.npz
  rms_w_full, rms_w_300m       -- (eta_rho, xi_rho), RMS of w
  weight_full, weight_300m     -- (eta_rho, xi_rho), accumulated sum(dz), a
                                   coverage/thickness diagnostic
  zpyc_mean                    -- (eta_rho, xi_rho), time-mean pycnocline depth (m)
  n_records                    -- int, number of time records processed
  (non-base scenarios only)
  rmse_w_full, rmse_w_300m     -- (eta_rho, xi_rho), RMSE of w vs notidesnowec
"""

import os
import sys
import glob
import argparse
import numpy as np
from netCDF4 import Dataset

ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'
GRD = '../mc60_grd.nc'

SCENARIOS = ['tideswec', 'tidesnowec', 'notidesnowec',
             'notideswec', 'notidesampwec', 'tidesampwec']
BASE_SCEN = 'notidesnowec'

# tidesampwec's raw source has a trailing 1-timestep file whose zslice
# output has no time dimension at all -- same exclusion used throughout
# postprocessing/ and plot/
TIDESAMPWEC_EXCLUDE = ('20190429110056',)

# stored `rho` is a deviation from RHO_REF; compare the pycnocline threshold
# against the stored value directly rather than converting to full density
RHO_REF = 1027.4
PYC_RHO = 1025.0
PYC_THRESH = PYC_RHO - RHO_REF   # = -2.4

FILL_THRESH = 0.9e33

# depth floors accumulated together in one pass: full sub-pycnocline column,
# and a pycnocline-to-300 m band (DEPTH_LIM_MID convention elsewhere in plot/)
FLOORS = {'full': None, '300m': -300.0}


def zfiles_for(scen):
    files = sorted(glob.glob(os.path.join(ZSLICE_ROOT, scen, 'z_mc60_his.*.nc')))
    if scen == 'tidesampwec':
        files = [f for f in files if not any(x in f for x in TIDESAMPWEC_EXCLUDE)]
    return files


def load_var(nc, t, var):
    arr = np.array(nc.variables[var][t], dtype=np.float32)   # (depth, eta_rho, xi_rho)
    arr[np.abs(arr) > FILL_THRESH] = np.nan
    return arr


def pycnocline_mask(rho, depth):
    """rho: (nz, eta, xi) stored values. Returns (below, zpyc):
    below -- (nz, eta, xi) bool, True at and below the shallowest level
             where rho >= PYC_THRESH (all-False where never reached).
    zpyc  -- (eta, xi) depth (m) of that shallowest level, NaN where never
             reached."""
    with np.errstate(invalid='ignore'):
        dense = rho >= PYC_THRESH
    has = dense.any(axis=0)
    kpyc = np.argmax(dense, axis=0)          # shallowest True index; 0 where none
    kk = np.arange(rho.shape[0])[:, None, None]
    below = (kk >= kpyc[None, :, :]) & has[None, :, :]
    zpyc = np.where(has, depth[kpyc], np.nan)
    return below, zpyc


def process_scenario(scen, nfiles=None):
    files = zfiles_for(scen)
    if nfiles is not None:
        files = files[:nfiles]
    if not files:
        print(f'WARNING: no zslice files found for {scen}')
        return
    print(f'{scen}: {len(files)} zslice files')

    is_base = (scen == BASE_SCEN)
    base_files = None
    if not is_base:
        base_files = zfiles_for(BASE_SCEN)
        if nfiles is not None:
            base_files = base_files[:nfiles]
        if len(base_files) != len(files):
            raise RuntimeError(
                f'{scen} has {len(files)} zslice files but {BASE_SCEN} has '
                f'{len(base_files)} -- cannot pair by index')

    with Dataset(files[0], 'r') as nc0:
        depth = np.array(nc0.variables['depth'][:])   # (157,) 0..-1980, surface-first
        eta_rho = nc0.dimensions['eta_rho'].size
        xi_rho = nc0.dimensions['xi_rho'].size
    dz = np.gradient(-depth)   # midpoint cell thickness (m), non-uniform grid
    nz = depth.size

    shape = (eta_rho, xi_rho)
    acc_w2 = {k: np.zeros(shape, dtype=np.float64) for k in FLOORS}
    acc_wt = {k: np.zeros(shape, dtype=np.float64) for k in FLOORS}
    acc_zpyc_sum = np.zeros(shape, dtype=np.float64)
    acc_zpyc_cnt = np.zeros(shape, dtype=np.float64)
    if not is_base:
        acc_d2 = {k: np.zeros(shape, dtype=np.float64) for k in FLOORS}
        acc_dwt = {k: np.zeros(shape, dtype=np.float64) for k in FLOORS}

    n_records = 0
    pairs = zip(files, base_files) if not is_base else zip(files, files)
    for fi, (f, fb) in enumerate(pairs):
        print(f'  [{fi + 1}/{len(files)}] {os.path.basename(f)}', flush=True)
        nc = Dataset(f, 'r')
        ncb = nc if is_base else Dataset(fb, 'r')
        try:
            nt = nc.dimensions['time'].size
            for t in range(nt):
                rho = load_var(nc, t, 'rho')
                w = load_var(nc, t, 'w')
                below, zpyc = pycnocline_mask(rho, depth)
                del rho

                valid_pyc = ~np.isnan(zpyc)
                acc_zpyc_sum[valid_pyc] += zpyc[valid_pyc]
                acc_zpyc_cnt[valid_pyc] += 1

                if not is_base:
                    wb = load_var(ncb, t, 'w')

                for k in range(nz):
                    m = below[k] & np.isfinite(w[k])
                    if not is_base:
                        md = m & np.isfinite(wb[k])
                        d = np.where(md, w[k] - wb[k], 0.0)

                    for floor_key, floor_val in FLOORS.items():
                        if floor_val is not None and depth[k] < floor_val:
                            continue   # depth is surface-first descending
                        acc_w2[floor_key] += dz[k] * np.where(m, w[k] * w[k], 0.0)
                        acc_wt[floor_key] += dz[k] * m
                        if not is_base:
                            acc_d2[floor_key] += dz[k] * d * d
                            acc_dwt[floor_key] += dz[k] * md

                if not is_base:
                    del wb
                del w, below, zpyc
                n_records += 1
        finally:
            nc.close()
            if not is_base:
                ncb.close()

    grdnc = Dataset(GRD, 'r')
    mask_rho = np.array(grdnc.variables['mask_rho'])
    mask_plot = mask_rho.astype(float)
    mask_plot[mask_plot == 0] = np.nan

    result = dict(n_records=n_records)
    with np.errstate(invalid='ignore'):
        result['zpyc_mean'] = np.where(
            acc_zpyc_cnt > 0, acc_zpyc_sum / np.maximum(acc_zpyc_cnt, 1e-30), np.nan
        ) * mask_plot
        for floor_key in FLOORS:
            rms = np.sqrt(np.where(
                acc_wt[floor_key] > 0,
                acc_w2[floor_key] / np.maximum(acc_wt[floor_key], 1e-30),
                np.nan))
            result[f'rms_w_{floor_key}'] = rms * mask_plot
            result[f'weight_{floor_key}'] = acc_wt[floor_key] * mask_plot
            if not is_base:
                rmse = np.sqrt(np.where(
                    acc_dwt[floor_key] > 0,
                    acc_d2[floor_key] / np.maximum(acc_dwt[floor_key], 1e-30),
                    np.nan))
                result[f'rmse_w_{floor_key}'] = rmse * mask_plot

    outfile = f'w_rms_pycnocline_{scen}.npz'
    np.savez(outfile, **result)
    print(f'  saved -> {outfile} ({n_records} records)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario', choices=SCENARIOS)
    parser.add_argument('--nfiles', type=int, default=None,
                        help='process only the first N zslice files (smoke test)')
    args = parser.parse_args()
    process_scenario(args.scenario, nfiles=args.nfiles)
