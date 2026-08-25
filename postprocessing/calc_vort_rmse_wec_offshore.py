"""
RMSE(depth) profile of relative vorticity (normalized by the Coriolis
parameter f, matching the vort/f convention used throughout
postprocessing/plot -- calc_vort_pdf*.py, save_vort_surf.py, etc.), testing
whether WEC changes the strength of tidal-bore-driven vorticity offshore.
Offshore counterpart of calc_vort_rmse_wec_shelf.py -- same calculation,
opposite side of the h=100 m split (h > H_MIN here instead of h < H_MAX);
also the vorticity counterpart of calc_w_rmse_wec_offshore.py.

Three comparisons, all against the notidesnowec base case (same BASE_SCEN
convention used throughout plot/postprocessing):
  ampwec_notides -- notidesampwec vs notidesnowec  (WEC alone, tides OFF)
  tides_nowec    -- tidesnowec    vs notidesnowec  (tides alone, no WEC)
  tides_ampwec   -- tidesampwec   vs notidesnowec  (tides + WEC together)

Comparing the 3 profiles answers the question directly: if WEC changes
tidal-bore-driven vorticity (not just adding its own tide-independent
vorticity signature), the tides_ampwec line won't simply look like
tides_nowec's line -- it'll diverge from it by more than ampwec_notides's
own (tide-independent) magnitude would predict.

RMSE(depth) is a POOLED spatial+temporal root-mean-square: at each zslice
depth level, every (time step, masked grid cell) sample across the whole
run is squared, summed, and divided by the total valid-sample count once,
then sqrt'd once at the end -- not RMS-over-time-then-averaged-over-space,
or vice versa:

    RMSE(depth) = sqrt( sum_t sum_(eta,xi in mask) (vort_other - vort_base)^2
                        / count_valid_samples(depth) )

Mask: offshore region, `eta_rho index > ETA_MIN` AND `h > H_MIN` AND
`mask_rho==1` (grid-index-space convention, same as ETA_MID/SPONGE
elsewhere in plot/ -- not a lat/lon bound). Same mask as
calc_w_rmse_wec_offshore.py -- 291,666 cells, h up to ~2352 m, so (unlike
the shelf mask, capped at h<100m) this reaches every depth level down to
the seafloor's deepest point.

Vorticity is not a native zslice variable -- it's derived here from the
zsliced u/v (zslicefull/<scen>/z_mc60_his.*.nc, on their native
eta_rho x xi_u / eta_v x xi_rho grids at each of the 157 fixed depth
levels) the same way pyfuncs.rho_uv_angle + pyfuncs.vorticity derive it for
raw s_rho files: interpolate u/v onto the rho grid, rotate to east/north
with the grid angle, then zeta = dv/dx - du/dy via pm/pn, then normalize by
f. Reimplemented locally (not calling pyfuncs) to keep the per-timestep
load pattern calc_w_rmse_wec_offshore.py uses for `w` -- pyfuncs.rho_uv_angle
reads an entire file's u/v in one shot (`[:]` over all 12 timesteps), which
at zsliced full-depth resolution would double the per-file memory that
load-per-timestep already avoids.

All 3 comparisons share the same notidesnowec base -- read it ONCE per
timestep and reused across all 3, rather than once per comparison (which
would reread the same 21 base files 3x over). Cuts total zslice reads from
4x21x12 to (1+3)x21x12, ~33% less I/O on an already ~1500 GB job (u and v
both read per file, vs w's single field).

Usage:
  python -u calc_vort_rmse_wec_offshore.py [--nfiles N]

Cost note: full run is ~21 files x 12 timesteps x 4 reads (1 base + 3
other scenarios) x 2 fields (u, v) x ~530 MB/read =~ 1 TB+ of zslice I/O --
roughly 2x calc_w_rmse_wec_offshore.py's cost since vorticity needs both u
and v where w_rmse only reads w (the offshore mask being larger than the
shelf mask doesn't change this -- both masks subselect from the same
full-domain read). Use --nfiles for a quick smoke test before committing to
the full run.

Output: ./vort_rmse_wec_offshore.npz
  depth -- (157,) m, 0 to -1980, surface-first
  rmse_ampwec_notides, rmse_tides_nowec, rmse_tides_ampwec -- (157,) each,
    pooled spatiotemporal RMSE(zeta/f) profile vs notidesnowec
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


def build_mask(eta_rho):
    grdnc = Dataset(GRD, 'r')
    h = np.array(grdnc.variables['h'])
    mask_rho = np.array(grdnc.variables['mask_rho'])
    grdnc.close()
    eta_idx = np.arange(eta_rho)[:, None]
    return (eta_idx > ETA_MIN) & (h > H_MIN) & (mask_rho == 1)


def load_grid_terms():
    grdnc = Dataset(GRD, 'r')
    angle = np.array(grdnc.variables['angle']).astype(np.float64)
    pm = np.array(grdnc.variables['pm']).astype(np.float64)
    pn = np.array(grdnc.variables['pn']).astype(np.float64)
    f = np.array(grdnc.variables['f']).astype(np.float64)
    grdnc.close()
    return np.cos(angle), np.sin(angle), pm, pn, f


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
    cosang, sinang, pm, pn, f = load_grid_terms()
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
                vort_base = load_vort_norm(ncb, t, cosang, sinang, pm, pn, f)
                for key in COMPARISONS:
                    vort_other = load_vort_norm(nco[key], t, cosang, sinang, pm, pn, f)
                    diff = vort_other - vort_base
                    diff_m = diff[:, mask2d]              # (nz, n_masked)
                    valid = np.isfinite(diff_m)
                    sq_sum[key] += np.sum(np.where(valid, diff_m * diff_m, 0.0), axis=1)
                    count[key] += valid.sum(axis=1)
                    del vort_other, diff, diff_m, valid
                del vort_base
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

    outfile = 'vort_rmse_wec_offshore.npz'
    np.savez(outfile, **result)
    print(f'saved -> {outfile}')
