"""
Offshore advective flux of rtrace using z-sliced output.

Reads z_mc60_his.*.nc from /data/project1/minnaho/swel/zslicefull/<scenario>/,
which carry rtrace and u on a fixed uniform z-grid (157 levels, 0 to -1980 m).

F(t, z, j) = -u(t, z, j, i_left-1) * rtrace(t, z, j, i_left)

u is taken directly from the u-face at xi_u = i_left - 1 (western face of the
leftmost band cell), so no interpolation to rho-points is needed.  ocean_time
is read from the matching original history file.

dz is computed per (z, j) column from the depth coordinate, with interfaces
clipped to -h(jj, il) so cells straddling the seabed get a partial thickness
and cells fully below the seabed get dz = 0.  h is sampled at the tracer
rho-point (jj, il).

_FillValue entries (1e33) are converted to NaN immediately after reading.
NaN propagates through the flux multiplication so that masked and
below-seabed points remain NaN in the output, distinguishing missing data
from genuine zero flux.

Output (per scenario) saved as offshore_flux_rtrace_zslice_<scenario>.npz:
    offshore_flux : (time, n_z, n_valid)   mmol m^-2 s^-1  (flux per unit area)
    depth         : (n_z,)                 m, negative downward
    dz            : (n_z, n_valid)         m, bathymetry-clipped thickness
    ocean_time    : (time,)                seconds since 1995-01-01
    eta_idx       : (n_valid,)
    xi_idx        : (n_valid,)
    dy_face       : (n_valid,)             m
    h_edge        : (n_valid,)             m, positive bathymetry at band edge
    tracer_name   : str
"""

import glob
import os
import numpy as np
from netCDF4 import Dataset

GRD       = '../mc60_grd.nc'
MASK_FILE = '../plot/coastal_mask.nc'
TRACER    = 'rtrace'
OUT_NAME  = 'offshore_flux_rtrace_zslice'

ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'

scenarios = {
    'notidesnowec': ('notidesnowec',
                     '/data/project3/minnaho/swel/notides/mc60/nowec/his'),
    'ampwec':       ('notidesampwec',
                     '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything'),
    'tidesnowec':   ('tidesnowec',
                     '/data/project3/minnaho/swel/tides/mc60/nowec/output/his'),
    'tidesampwec':  ('tidesampwec',
                     '/data/project3/minnaho/swel/tides/mc60/ampwec/everything'),
}

# tidesampwec's raw source has a trailing 1-timestep file
# (...20190429110056) whose zslice output has no time dimension at all --
# same exclusion as calc_wtrace_flux.py / plot_cs_diag_avg_diff.py
TIDESAMPWEC_EXCLUDE = ('20190429110056',)

# --- coastal band western edge ---
mask = np.array(Dataset(MASK_FILE)['coastal_mask'])
ny, nx = mask.shape
i_left = np.full(ny, -1, dtype=int)
for j in range(ny):
    band = np.where(mask[j, :] == 1)[0]
    if band.size > 0:
        i_left[j] = band.min()

valid   = i_left >= 0
jj      = np.where(valid)[0]
il      = i_left[valid]
n_valid = jj.size
print(f'{n_valid} rows have a coastal-band edge')

grdnc   = Dataset(GRD)
dy_face = 1.0 / np.array(grdnc.variables['pn'])[jj, il]   # (n_valid,)
h_edge  = np.array(grdnc.variables['h'])[jj, il]           # (n_valid,) positive


def _fill_to_nan(arr):
    """Replace _FillValue sentinel (|x| > 1e30) with NaN in a float array."""
    return np.where(np.abs(arr) > 1e30, np.nan, arr)


def compute_dz_2d(depth_vals, h_edge):
    """
    Bathymetry-aware cell thicknesses for a non-uniform z-grid.

    depth_vals : (n_z,)     m, negative downward, cell centers (from file)
    h_edge     : (n_valid,) m, positive bathymetry at the band edge

    Returns dz : (n_z, n_valid) m.
      Cells fully above the seabed: full midpoint-rule thickness.
      Cell straddling the seabed: partial (clipped) thickness.
      Cells fully below the seabed: 0.
    """
    n_z = depth_vals.size
    z_w = np.empty(n_z + 1)
    z_w[0]    = 0.0
    z_w[1:-1] = (depth_vals[:-1] + depth_vals[1:]) / 2.0
    bottom_step = float(depth_vals[-2]) - float(depth_vals[-1])   # positive
    z_w[-1]   = float(depth_vals[-1]) - bottom_step / 2.0

    z_w_clipped = np.maximum(z_w[:, None], -h_edge[None, :])   # (n_z+1, n_valid)

    return z_w_clipped[:-1, :] - z_w_clipped[1:, :]   # (n_z, n_valid)


def compute_flux(zslice_dir, his_dir):
    z_files = sorted(glob.glob(os.path.join(zslice_dir, 'z_mc60_his.*.nc')))
    z_files = [zf for zf in z_files if not any(x in zf for x in TIDESAMPWEC_EXCLUDE)]
    print(f'  {len(z_files)} z-sliced files')

    with Dataset(z_files[0]) as nc:
        depth_vals = np.array(nc.variables['depth'][:])   # (n_z,) negative downward
    dz = compute_dz_2d(depth_vals, h_edge)   # (n_z, n_valid)

    flux_chunks = []
    time_chunks = []

    for zf in z_files:
        hf = os.path.join(his_dir, os.path.basename(zf).replace('z_', ''))

        with Dataset(zf) as nc:
            u_full    = _fill_to_nan(np.array(nc.variables['u'][:]))
            u_edge    = u_full[:, :, jj, il - 1]             # (t, n_z, n_valid)
            del u_full

            trac_full = _fill_to_nan(np.array(nc.variables[TRACER][:]))
            trac_edge = trac_full[:, :, jj, il]              # (t, n_z, n_valid)
            del trac_full

        with Dataset(hf) as his:
            ot = np.array(his.variables['ocean_time'][:])    # (t,)

        flux = (-u_edge * trac_edge)                         # (t, n_z, n_valid) mmol/m2/s
        flux_chunks.append(flux.astype(np.float32))
        time_chunks.append(ot)

    return (np.concatenate(flux_chunks, axis=0),
            depth_vals, dz,
            np.concatenate(time_chunks))


for name, (zscen, his_dir) in scenarios.items():
    print(f'Processing {name} ...')
    zslice_dir = os.path.join(ZSLICE_ROOT, zscen)
    flux, depth_vals, dz, ot = compute_flux(zslice_dir, his_dir)
    out = f'{OUT_NAME}_{name}.npz'
    np.savez(out,
             offshore_flux = flux,
             depth         = depth_vals,
             dz            = dz,
             ocean_time    = ot,
             eta_idx       = jj.astype(np.int32),
             xi_idx        = il.astype(np.int32),
             dy_face       = dy_face,
             h_edge        = h_edge,
             tracer_name   = TRACER)
    print(f'  saved -> {out}  flux {flux.shape}')
