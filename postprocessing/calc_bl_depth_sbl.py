"""
Time-mean surface boundary-layer depth per scenario, diagnosed from the
Akt > AKT_THRESH criterion (vertical thermal diffusivity threshold marking
actively turbulent boundary-layer water, e.g. the Kz-threshold bottom mixed
layer definition of Perlin et al. 2005). Bottom boundary layer counterpart:
calc_bl_depth_bbl.py -- split into two scripts (each with its own npz
output) so the SBL side, which is far cheaper, can finish and be plotted
without waiting on BBL's per-timestep depth reconstruction.

Uses zsliced Akt (zslicefull/<scen>/ak/z_mc60_his.*.nc), fixed 157-level
z-grid, 1 m bins down to -50 m -- fine enough near the surface (where SBL
depths of a few to tens of meters live) and much cheaper to read than the
raw file.

For each grid column and time step:
  SBL depth (m) = |z| extent below the surface (starting at the shallowest
    level) over which Akt stays above AKT_THRESH. 0 if the surface level
    itself is already below threshold.

Computed per time step, then arithmetic-averaged over all time steps --
not diagnosed once from a time-averaged Akt profile, since tidal forcing
makes Akt vary by orders of magnitude in time and a time-mean profile
would blur out the (physically real) intermittent turbulent episodes that
individually cross the threshold.

Usage:
  python -u calc_bl_depth_sbl.py <scenario> [--nfiles N]
  scenario in: tideswec, tidesnowec, notidesnowec, notideswec, ampwec,
               tidesampwec   -- 'ampwec' (raw-file naming convention, kept
               consistent with calc_bl_depth_bbl.py even though this script
               only touches the zsliced side) is looked up under its
               'notidesampwec' directory alias internally.

  # launch all 6 scenarios at once (run from postprocessing/), alongside
  # the BBL jobs -- matches profile_zslice_par_100m.py's convention of
  # backgrounding multiple per-scenario job types together in one loop:
  for scen in tideswec tidesnowec notidesnowec notideswec ampwec tidesampwec; do
    python -u calc_bl_depth_sbl.py $scen > log_bl_sbl_${scen}.txt 2>&1 &
    python -u calc_bl_depth_bbl.py $scen > log_bl_bbl_${scen}.txt 2>&1 &
  done

Output: ./bl_depth_sbl_<scenario>.npz
  sbl_mean -- (eta_rho, xi_rho), time-mean surface boundary layer depth
              (m), NaN over land
  n_sbl    -- int, number of time records averaged over
"""

import os
import glob
import argparse
import numpy as np
from netCDF4 import Dataset

ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'
GRD = '../mc60_grd.nc'

AKT_THRESH = 1e-4   # m2 s-1
FILL_THRESH = 0.9e33

SCENARIOS = ['tideswec', 'tidesnowec', 'notidesnowec',
             'notideswec', 'ampwec', 'tidesampwec']

# zsliced ak/ output uses 'notidesampwec' for the ampwec run -- see README's
# "raw-file scripts use ampwec, zslice-dir scripts use notidesampwec"
ZSLICE_SCEN_DIRS = {'ampwec': 'notidesampwec'}

# tidesampwec's raw source has a trailing 1-timestep file whose zslice
# output has no time dimension at all -- same exclusion used throughout
# postprocessing/ and plot/
TIDESAMPWEC_EXCLUDE = ('20190429110056',)


def zslice_ak_files_for(scen):
    zdir = ZSLICE_SCEN_DIRS.get(scen, scen)
    files = sorted(glob.glob(os.path.join(ZSLICE_ROOT, zdir, 'ak', 'z_mc60_his.*.nc')))
    if scen == 'tidesampwec':
        files = [f for f in files if not any(x in f for x in TIDESAMPWEC_EXCLUDE)]
    return files


def scan_extent_from_edge(akt, z, thresh):
    """akt: (nz, eta, xi). z: (nz,) or (nz, eta, xi) -- physical depth (m,
    <=0) of each level along the same axis. Index 0 is the edge to scan
    FROM (surface here). Returns (eta, xi): |z| extent from index 0 over
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
    zfiles = zslice_ak_files_for(scen)
    if nfiles is not None:
        zfiles = zfiles[:nfiles]
    if not zfiles:
        raise RuntimeError(f'no zsliced ak files found for {scen}')
    print(f'{scen}: {len(zfiles)} zsliced ak files')

    grdnc = Dataset(GRD, 'r')
    mask_rho = np.array(grdnc.variables['mask_rho'])
    mask_plot = mask_rho.astype(float)
    mask_plot[mask_plot == 0] = np.nan

    with Dataset(zfiles[0], 'r') as nc0:
        depth = np.array(nc0.variables['depth'][:])   # (157,) 0..-1980, surface-first
        eta_rho = nc0.dimensions['eta_rho'].size
        xi_rho = nc0.dimensions['xi_rho'].size
    shape = (eta_rho, xi_rho)

    sbl_sum = np.zeros(shape, dtype=np.float64)
    sbl_cnt = np.zeros(shape, dtype=np.float64)
    n_sbl = 0
    for fi, f in enumerate(zfiles):
        print(f'  [sbl {fi + 1}/{len(zfiles)}] {os.path.basename(f)}', flush=True)
        nc = Dataset(f, 'r')
        try:
            nt = nc.dimensions['time'].size
            for t in range(nt):
                akt = np.array(nc.variables['Akt'][t], dtype=np.float32)
                akt[np.abs(akt) > FILL_THRESH] = np.nan
                sbl = scan_extent_from_edge(akt, depth, AKT_THRESH)
                valid = np.isfinite(sbl)
                sbl_sum[valid] += sbl[valid]
                sbl_cnt[valid] += 1
                n_sbl += 1
                del akt, sbl, valid
        finally:
            nc.close()

    with np.errstate(invalid='ignore'):
        sbl_mean = np.where(
            sbl_cnt > 0, sbl_sum / np.maximum(sbl_cnt, 1e-30), np.nan) * mask_plot

    outfile = f'bl_depth_sbl_{scen}.npz'
    np.savez(outfile, sbl_mean=sbl_mean, n_sbl=n_sbl)
    print(f'  saved -> {outfile} (n_sbl={n_sbl})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario', choices=SCENARIOS)
    parser.add_argument('--nfiles', type=int, default=None,
                        help='process only the first N zsliced ak files (smoke test)')
    args = parser.parse_args()
    process_scenario(args.scenario, nfiles=args.nfiles)
