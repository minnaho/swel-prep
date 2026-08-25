"""
Time-mean bottom boundary-layer depth per scenario, diagnosed from the
Akt > AKT_THRESH criterion (vertical thermal diffusivity threshold marking
actively turbulent boundary-layer water, e.g. the Kz-threshold bottom mixed
layer definition of Perlin et al. 2005). Surface boundary layer
counterpart: calc_bl_depth_sbl.py -- split into two scripts (each with its
own npz output) so the cheap SBL side isn't stuck waiting on this script's
much slower per-timestep depth reconstruction.

Uses raw history Akt on its native s_w grid (101 cell-face levels,
stretched much finer near the seafloor than the zsliced grid's coarse 30 m
bins below -300 m), converted to physical depth per time step with
ROMS_depths.get_zw_zeta (same helper plot_cs_diag_akt.py uses for its
cross-sections). The zsliced grid's 30 m bins below -300 m would make BBL
nearly unresolvable off the shelf, so the native grid is used instead.

For each grid column and time step:
  BBL depth (m) = |z| extent above the seafloor (starting at the deepest
    s_w level, z=-h) over which Akt stays above AKT_THRESH. 0 if the
    bottom level itself is below threshold.

Computed per time step, then arithmetic-averaged over all time steps --
not diagnosed once from a time-averaged Akt profile, since tidal forcing
makes Akt vary by orders of magnitude in time and a time-mean profile
would blur out the (physically real) intermittent turbulent episodes that
individually cross the threshold.

Usage:
  python -u calc_bl_depth_bbl.py <scenario> [--nfiles N]
  scenario in: tideswec, tidesnowec, notidesnowec, notideswec, ampwec,
               tidesampwec   -- 'ampwec' (raw-file naming convention, since
               this script reads raw history files directly).

  # launch all 6 scenarios at once (run from postprocessing/), alongside
  # the SBL jobs -- matches profile_zslice_par_100m.py's convention of
  # backgrounding multiple per-scenario job types together in one loop:
  for scen in tideswec tidesnowec notidesnowec notideswec ampwec tidesampwec; do
    python -u calc_bl_depth_sbl.py $scen > log_bl_sbl_${scen}.txt 2>&1 &
    python -u calc_bl_depth_bbl.py $scen > log_bl_bbl_${scen}.txt 2>&1 &
  done

Output: ./bl_depth_bbl_<scenario>.npz
  bbl_mean -- (eta_rho, xi_rho), time-mean bottom boundary layer depth
              (m), NaN over land
  n_bbl    -- int, number of time records averaged over
"""

import os
import sys
import glob
import argparse
import numpy as np
from netCDF4 import Dataset

sys.path.append('/data/project3/minnaho/global/')
import ROMS_depths as depths

GRD = '../mc60_grd.nc'

AKT_THRESH = 1e-4   # m2 s-1
FILL_THRESH = 0.9e33

SCENARIOS = ['tideswec', 'tidesnowec', 'notidesnowec',
             'notideswec', 'ampwec', 'tidesampwec']

# raw history root per scenario -- same paths as calc_vort_pdf_100m.py /
# zslice_ak.py
RAW_ROOTS = {
    'tideswec':     '/data/project3/minnaho/swel/tides/mc60/wec',
    'tidesnowec':   '/data/project3/minnaho/swel/tides/mc60/nowec/output',
    'notidesnowec': '/data/project3/minnaho/swel/notides/mc60/nowec',
    'notideswec':   '/data/project3/minnaho/swel/notides/mc60/wec/rerun',
    'ampwec':       '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything',
    'tidesampwec':  '/data/project3/minnaho/swel/tides/mc60/ampwec/everything',
}

# tidesampwec's raw source has a trailing 1-timestep file whose zslice
# output has no time dimension at all -- same exclusion used throughout
# postprocessing/ and plot/
TIDESAMPWEC_EXCLUDE = ('20190429110056',)


def raw_files_for(scen):
    """Handle both flat (ampwec/tidesampwec) and his/ subdir layouts."""
    root = RAW_ROOTS[scen]
    sub = os.path.join(root, 'his')
    base = sub if os.path.isdir(sub) else root
    files = sorted(glob.glob(os.path.join(base, 'mc60_his.201904*.nc')))
    if scen == 'tidesampwec':
        files = [f for f in files if not any(x in f for x in TIDESAMPWEC_EXCLUDE)]
    return files


def scan_extent_from_edge(akt, z, thresh):
    """akt: (nz, eta, xi). z: (nz,) or (nz, eta, xi) -- physical depth (m,
    <=0) of each level along the same axis. Index 0 is the edge to scan
    FROM (bottom here). Returns (eta, xi): |z| extent from index 0 over
    which akt stays contiguously above `thresh`. NaN entries in akt count
    as non-turbulent, so a land/below-seafloor column at index 0 naturally
    reports 0 there, turned to NaN below via the has_valid check."""
    z3 = np.broadcast_to(z[:, None, None], akt.shape) if np.ndim(z) == 1 else z
    nz = akt.shape[0]
    with np.errstate(invalid='ignore'):
        not_turbulent = ~(akt > thresh)
    any_stop = not_turbulent.any(axis=0)
    j = np.where(any_stop, np.argmax(not_turbulent, axis=0), nz)
    j_edge = np.clip(j - 1, 0, nz - 1)
    z_edge = np.take_along_axis(z3, j_edge[None, :, :], axis=0)[0]
    extent = np.where(j == 0, 0.0, np.abs(z_edge - z3[0]))
    has_valid = np.isfinite(akt).any(axis=0)
    return np.where(has_valid, extent, np.nan)


def process_scenario(scen, nfiles=None):
    hfiles = raw_files_for(scen)
    if nfiles is not None:
        hfiles = hfiles[:nfiles]
    if not hfiles:
        raise RuntimeError(f'no raw history files found for {scen}')
    print(f'{scen}: {len(hfiles)} raw his files')

    grdnc = Dataset(GRD, 'r')
    mask_rho = np.array(grdnc.variables['mask_rho'])
    mask_plot = mask_rho.astype(float)
    mask_plot[mask_plot == 0] = np.nan

    with Dataset(hfiles[0], 'r') as nc0:
        eta_rho = nc0.dimensions['eta_rho'].size
        xi_rho = nc0.dimensions['xi_rho'].size
    shape = (eta_rho, xi_rho)

    bbl_sum = np.zeros(shape, dtype=np.float64)
    bbl_cnt = np.zeros(shape, dtype=np.float64)
    n_bbl = 0
    for fi, f in enumerate(hfiles):
        print(f'  [bbl {fi + 1}/{len(hfiles)}] {os.path.basename(f)}', flush=True)
        hisnc = Dataset(f, 'r')
        try:
            nt = hisnc.dimensions['time'].size
            for t in range(nt):
                zeta = np.squeeze(np.array(hisnc.variables['zeta'][t, :, :]))
                zw3d = depths.get_zw_zeta(hisnc, grdnc, zeta)   # (s_w, eta, xi)
                akt = np.array(hisnc.variables['Akt'][t], dtype=np.float32)
                akt[np.abs(akt) > FILL_THRESH] = np.nan
                bbl = scan_extent_from_edge(akt, zw3d, AKT_THRESH)
                valid = np.isfinite(bbl)
                bbl_sum[valid] += bbl[valid]
                bbl_cnt[valid] += 1
                n_bbl += 1
                del zeta, zw3d, akt, bbl, valid
        finally:
            hisnc.close()

    with np.errstate(invalid='ignore'):
        bbl_mean = np.where(
            bbl_cnt > 0, bbl_sum / np.maximum(bbl_cnt, 1e-30), np.nan) * mask_plot

    outfile = f'bl_depth_bbl_{scen}.npz'
    np.savez(outfile, bbl_mean=bbl_mean, n_bbl=n_bbl)
    print(f'  saved -> {outfile} (n_bbl={n_bbl})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario', choices=SCENARIOS)
    parser.add_argument('--nfiles', type=int, default=None,
                        help='process only the first N raw history files (smoke test)')
    args = parser.parse_args()
    process_scenario(args.scenario, nfiles=args.nfiles)
