"""
Std(depth) profile of normalized relative vorticity (zeta/f, same convention
as calc_vort_rmse_wec_shelf.py/_offshore.py) for the notidesnowec base case
and its three comparison scenarios, split into shelf (h < 100 m) and
offshore (h > 100 m) masks -- same masks as calc_vort_rmse_wec_shelf.py /
_offshore.py (`eta_rho index > ETA_MIN` AND `mask_rho==1`, in grid-index
space). Vorticity counterpart of calc_w_std.py -- see that script's
docstring for the full Delta-std motivation and the single-pass pooled
variance identity; only the field differs (zeta/f derived from zsliced u/v
instead of the native zslice variable `w`).

Vorticity is derived here exactly as in calc_vort_rmse_wec_shelf.py:
interpolate u/v onto the rho grid, rotate to east/north with the grid angle,
zeta = dv/dx - du/dy via pm/pn, then normalize by f. Reimplemented locally
(not calling pyfuncs) for the same per-timestep-load memory reason given
there.

Std(depth) = sqrt(max(E[(zeta/f)^2] - E[zeta/f]^2, 0)), pooled over every
(timestep, masked grid cell) sample for one scenario in a single pass
(running sum, running sum-of-squares, count) -- see calc_w_std.py's
docstring for the exact formula and the max(...,0) clamp rationale.

Like calc_w_std.py, shelf and offshore are computed from the SAME u/v read
in one pass (no cost-tier reason to split by region here), and all 4
scenarios (base + 3 comparisons) are read in this single invocation -- no
per-scenario CLI arg, no launch-all loop.

Usage:
  python -u calc_vort_std.py [--nfiles N]

Cost note: full run is ~2x calc_w_std.py's I/O since vorticity needs both u
and v where w only reads w (same ~1 TB+ order of magnitude as
calc_vort_rmse_wec_shelf.py). Use --nfiles for a quick smoke test before
committing to the full run.

Output: ./vort_std.npz
  depth -- (157,) m, 0 to -1980, surface-first
  std_shelf_base, std_offshore_base -- (157,) each, pooled std(zeta/f) for
    notidesnowec
  std_shelf_<key>, std_offshore_<key> -- (157,) each, pooled std(zeta/f) for
    the comparison scenario, key in COMPARISONS
  dstd_shelf_<key>, dstd_offshore_<key> -- (157,) each,
    std_..._<key> - std_..._base
  n_shelf_base, n_offshore_base, n_shelf_<key>, n_offshore_<key> -- (157,)
    each, valid sample count backing each std point
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


def load_uv(nc, t):
    u = np.array(nc.variables['u'][t], dtype=np.float32)   # (depth, eta_rho, xi_u)
    v = np.array(nc.variables['v'][t], dtype=np.float32)   # (depth, eta_v, xi_rho)
    u[np.abs(u) > FILL_THRESH] = np.nan
    v[np.abs(v) > FILL_THRESH] = np.nan
    return u, v


def uv_to_rho(u, v):
    """u (depth,eta_rho,xi_u), v (depth,eta_v,xi_rho) -> u_rho, v_rho
    (depth,eta_rho,xi_rho); same interior-average / edge-copy scheme as
    pyfuncs.rho_uv_angle."""
    u_temp = 0.5 * (u[:, :, 1:] + u[:, :, :-1])
    u_rho = np.empty((u.shape[0], u.shape[1], u.shape[2] + 1), dtype=np.float32)
    u_rho[:, :, 1:-1] = u_temp
    u_rho[:, :, 0] = u_temp[:, :, 0]
    u_rho[:, :, -1] = u_temp[:, :, -1]

    v_temp = 0.5 * (v[:, 1:, :] + v[:, :-1, :])
    v_rho = np.empty((v.shape[0], v.shape[1] + 1, v.shape[2]), dtype=np.float32)
    v_rho[:, 1:-1, :] = v_temp
    v_rho[:, 0, :] = v_temp[:, 0, :]
    v_rho[:, -1, :] = v_temp[:, -1, :]
    return u_rho, v_rho


def relative_vorticity(u_east, v_north, pm, pn):
    """dv/dx - du/dy on (depth,eta_rho,xi_rho); same forward/backward
    difference + edge-copy scheme as pyfuncs.vorticity, generalized off the
    (time, s_rho) axes that function assumes onto a bare depth axis."""
    dvdx = np.empty_like(v_north)
    dvdx[:, :, :-1] = (v_north[:, :, 1:] - v_north[:, :, :-1]) * pm[:, :-1]
    dvdx[:, :, -1] = (v_north[:, :, -1] - v_north[:, :, -2]) * pm[:, -1]

    dudy = np.empty_like(u_east)
    dudy[:, :-1, :] = (u_east[:, 1:, :] - u_east[:, :-1, :]) * pn[:-1, :]
    dudy[:, -1, :] = (u_east[:, -1, :] - u_east[:, -2, :]) * pn[-1, :]

    zeta = dvdx - dudy
    zeta[:, :, 0] = zeta[:, :, 1]
    zeta[:, :, -1] = zeta[:, :, -2]
    zeta[:, 0, :] = zeta[:, 1, :]
    zeta[:, -1, :] = zeta[:, -2, :]
    return zeta


def load_vort_norm(nc, t, cosang, sinang, pm, pn, f):
    u, v = load_uv(nc, t)
    u_rho, v_rho = uv_to_rho(u, v)
    u_east = u_rho * cosang - v_rho * sinang
    v_north = u_rho * sinang + v_rho * cosang
    zeta = relative_vorticity(u_east, v_north, pm, pn)
    return zeta / f[np.newaxis]


def build_masks(eta_rho):
    grdnc = Dataset(GRD, 'r')
    h = np.array(grdnc.variables['h'])
    mask_rho = np.array(grdnc.variables['mask_rho'])
    grdnc.close()
    eta_idx = np.arange(eta_rho)[:, None]
    base = (eta_idx > ETA_MIN) & (mask_rho == 1)
    return {'shelf': base & (h < H_MAX), 'offshore': base & (h > H_MIN)}


def load_grid_terms():
    grdnc = Dataset(GRD, 'r')
    angle = np.array(grdnc.variables['angle']).astype(np.float64)
    pm = np.array(grdnc.variables['pm']).astype(np.float64)
    pn = np.array(grdnc.variables['pn']).astype(np.float64)
    f = np.array(grdnc.variables['f']).astype(np.float64)
    grdnc.close()
    return np.cos(angle), np.sin(angle), pm, pn, f


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
    cosang, sinang, pm, pn, f = load_grid_terms()
    print(f'shelf mask: {masks["shelf"].sum()} cells, '
          f'offshore mask: {masks["offshore"].sum()} cells')

    scen_keys = ['base'] + list(COMPARISONS)
    sums = {sk: {r: np.zeros(nz, dtype=np.float64) for r in masks} for sk in scen_keys}
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
                    vort = load_vort_norm(ncs[sk], t, cosang, sinang, pm, pn, f)
                    for region, mask2d in masks.items():
                        v_m = vort[:, mask2d]           # (nz, n_masked)
                        valid = np.isfinite(v_m)
                        v_valid = np.where(valid, v_m, 0.0)
                        sums[sk][region] += np.sum(v_valid, axis=1)
                        sq_sums[sk][region] += np.sum(v_valid * v_valid, axis=1)
                        counts[sk][region] += valid.sum(axis=1)
                        del v_m, valid, v_valid
                    del vort
        finally:
            for nc in ncs.values():
                nc.close()

    result = {'depth': depth}
    std = {sk: {} for sk in scen_keys}
    with np.errstate(invalid='ignore'):
        for sk in scen_keys:
            for region in masks:
                cnt = counts[sk][region]
                mean = np.where(cnt > 0, sums[sk][region] / np.maximum(cnt, 1e-30), np.nan)
                mean_sq = np.where(
                    cnt > 0, sq_sums[sk][region] / np.maximum(cnt, 1e-30), np.nan)
                var = np.maximum(mean_sq - mean * mean, 0.0)
                std[sk][region] = np.sqrt(var)
                result[f'std_{region}_{sk}'] = std[sk][region]
                result[f'n_{region}_{sk}'] = cnt
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

    outfile = 'vort_std.npz'
    np.savez(outfile, **result)
    print(f'saved -> {outfile}')
