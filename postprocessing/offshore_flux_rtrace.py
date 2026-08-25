"""
Offshore advective flux of rtrace through the offshore (western) edge of the
10 km coastal band defined in plot/coastal_mask.nc.

    F(t, k, j) = -u_rho(t, k, j, i_left[j])
                 * rtrace(t, k, j, i_left[j])
                 * dy(j)
                 * Hz(t, k, j, i_left[j])

u_rho is grid-aligned (xi-direction) via pyfuncs.rho_uv_angle(rotate=False).
Sign: positive u_rho is +xi (onshore); -u_rho is offshore.
rtrace is read directly from the history files.

Output (per scenario) saved as offshore_flux_rtrace_<scenario>.npz with:
    offshore_flux : (time, s_rho, n_valid)   tracer * m^3 / s
    z_r           : (time, s_rho, n_valid)   m, negative downward
    ocean_time    : (time,)                  seconds since 1995-01-01
    eta_idx       : (n_valid,)               eta index of each band-edge column
    xi_idx        : (n_valid,)               xi index of band edge per row
    dy_face       : (n_valid,)               dy at edge cell (m)
    tracer_name   : str
"""

import glob
import os
import sys
import numpy as np
from netCDF4 import Dataset

sys.path.append('/data/project3/minnaho/global/')
import pyfuncs
import ROMS_depths as depths

GRD       = '../mc60_grd.nc'
MASK_FILE = '../plot/coastal_mask.nc'
TRACER    = 'rtrace'
OUT_NAME  = 'offshore_flux_rtrace'

scenarios = {
    'notidesnowec': '/data/project3/minnaho/swel/notides/mc60/nowec/his',
    'ampwec':       '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything',
    'tidesnowec':   '/data/project3/minnaho/swel/tides/mc60/nowec/output/his',
    'tidesampwec':  '/data/project3/minnaho/swel/tides/mc60/ampwec/everything',
}

# --- offshore (west) edge of the coastal band per eta row ---
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
pn      = np.array(grdnc.variables['pn'])
dy_face = 1.0 / pn[jj, il]     # (n_valid,)


def compute_flux(his_dir):
    his_files = sorted(glob.glob(os.path.join(his_dir, 'mc60_his.*.nc')))
    print(f'  {len(his_files)} his files')

    flux_chunks = []
    z_chunks    = []
    time_chunks = []

    for hf in his_files:
        u_rho, _ = pyfuncs.rho_uv_angle(hf, GRD, rotate=False)   # (t, s, eta, xi)
        u_edge   = u_rho[:, :, jj, il]                           # (t, s, n_valid)
        del u_rho

        with Dataset(hf) as his:
            # netCDF4 uses orthogonal (Cartesian) fancy indexing; load full then
            # apply numpy paired fancy indexing.
            trac_full = np.array(his.variables[TRACER][:])
            trac_edge = trac_full[:, :, jj, il]                       # (t, s, n_valid)
            del trac_full
            ot = np.array(his.variables['ocean_time'][:])
            nt = ot.size
            ns = u_edge.shape[1]

            Hz_edge = np.empty((nt, ns, n_valid), dtype=np.float64)
            z_edge  = np.empty((nt, ns, n_valid), dtype=np.float64)
            for t in range(nt):
                zw = depths.get_zw_zeta_tind(his, grdnc, t)
                zr = depths.get_zr_zeta_tind(his, grdnc, t)
                Hz_edge[t] = np.diff(zw, axis=0)[:, jj, il]
                z_edge[t]  = zr[:, jj, il]

        flux = -u_edge * trac_edge * dy_face[None, None, :] * Hz_edge
        flux_chunks.append(flux.astype(np.float32))
        z_chunks.append(z_edge.astype(np.float32))
        time_chunks.append(ot)

    return (np.concatenate(flux_chunks, axis=0),
            np.concatenate(z_chunks,    axis=0),
            np.concatenate(time_chunks))


for name, his_dir in scenarios.items():
    out = f'{OUT_NAME}_{name}.npz'
    if os.path.exists(out):
        print(f'Skipping {name} -- {out} already exists')
        continue
    print(f'Processing {name} ...')
    flux, z_r, ot = compute_flux(his_dir)
    np.savez(out,
             offshore_flux = flux,
             z_r           = z_r,
             ocean_time    = ot,
             eta_idx       = jj.astype(np.int32),
             xi_idx        = il.astype(np.int32),
             dy_face       = dy_face,
             tracer_name   = TRACER)
    print(f'  saved -> {out}  flux {flux.shape}  z_r {z_r.shape}')
