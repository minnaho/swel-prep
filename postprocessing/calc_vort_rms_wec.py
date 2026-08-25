"""
RMS(depth) profile of normalized relative vorticity (zeta/f, same convention
as calc_vort_rmse_wec_shelf.py/_offshore.py) for each of the 4 scenarios on
its own -- plain magnitude, NOT a difference against the notidesnowec base
case (see calc_vort_rmse_wec_shelf.py / _offshore.py for that). RMS-only
counterpart of calc_vort_std.py, following the same "rms" vs "rmse"
vocabulary as calc_w_rms_pycnocline.py elsewhere in this directory. Shelf
and offshore are computed from the SAME u/v read in one pass, same as
calc_vort_std.py.

Vorticity is derived here exactly as in calc_vort_rmse_wec_shelf.py /
calc_vort_std.py: interpolate u/v onto the rho grid, rotate to east/north
with the grid angle, zeta = dv/dx - du/dy via pm/pn, then normalize by f.

RMS(depth) is a POOLED spatial+temporal root-mean-square of the raw field
(no mean subtracted, unlike calc_vort_std.py's std -- RMS keeps the
mean-square):

    RMS(depth) = sqrt( sum_t sum_(eta,xi in mask) (zeta/f)^2
                        / count_valid_samples(depth) )

Masks: shelf = eta_rho index > ETA_MIN AND h < H_MAX AND mask_rho==1;
offshore = eta_rho index > ETA_MIN AND h > H_MIN AND mask_rho==1 --
identical masks to calc_vort_std.py / calc_vort_rmse_wec_shelf.py.

All 4 scenarios (base + 3 comparisons) are read in this single invocation --
no per-scenario CLI arg and no launch-all loop, same as calc_vort_std.py,
purely so the same 4 npz keys line up 1:1 with calc_vort_std.py's/
calc_vort_rmse_wec_shelf.py's COMPARISONS keys for easy joint plotting.

Usage:
  python -u calc_vort_rms_wec.py [--nfiles N]

Cost note: same ~1 TB+ zslice I/O as calc_vort_std.py (u and v, all 4
scenarios, both regions computed for free out of the same read). Use
--nfiles for a quick smoke test before committing to the full run.

Output: ./vort_rms_wec.npz
  depth -- (157,) m, 0 to -1980, surface-first
  rms_shelf_base, rms_offshore_base -- (157,) each, pooled RMS(zeta/f) for
    notidesnowec
  rms_shelf_<key>, rms_offshore_<key> -- (157,) each, pooled RMS(zeta/f) for
    the comparison scenario, key in ampwec_notides, tides_nowec,
    tides_ampwec
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
                        sq_sums[sk][region] += np.sum(v_valid * v_valid, axis=1)
                        counts[sk][region] += valid.sum(axis=1)
                        del v_m, valid, v_valid
                    del vort
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

    outfile = 'vort_rms_wec.npz'
    np.savez(outfile, **result)
    print(f'saved -> {outfile}')
