"""
Add s-coordinate global attributes (theta_s, theta_b, hc, Cs_r, Cs_w) to a
mc60_ini file.  These are needed by zslice and ROMS depth routines that read
s-coord parameters from global attributes (as in smode_his files) rather than
from variables.

Cs_w is computed from the UCLA ROMS double-stretch function evaluated at
w-point s-coordinates.  The result is validated against the Cs_r already stored
in the ini file.

Comment/uncomment the INI block to switch between tides and notides.
"""

import sys
import numpy as np
from netCDF4 import Dataset

# ── choose case (comment out the one you don't want) ────────────────────────
#INI = '/data/project3/minnaho/project9copy/swel/tides/mc60/rst/mc60_ini.20190415110120.nc'

INI = '/data/project3/minnaho/project9copy/swel/notides/mc60/rst/mc60_ini.20190415110120.nc'


# ── UCLA ROMS double-stretch (set_scoord.F) ──────────────────────────────────
def stretching(sc, theta_s, theta_b):
    """Shchepetkin/McWilliams stretching function used in UCLA ROMS."""
    sc = np.asarray(sc, dtype=np.float64)
    if theta_s > 0:
        csrf = (1.0 - np.cosh(theta_s * sc)) / (np.cosh(theta_s) - 1.0)
    else:
        csrf = -sc**2
    if theta_b > 0:
        return (np.exp(theta_b * csrf) - 1.0) / (1.0 - np.exp(-theta_b))
    return csrf


# ── read existing variables ───────────────────────────────────────────────────
with Dataset(INI, 'a') as nc:
    theta_s = float(nc.variables['theta_s'][0])
    theta_b = float(nc.variables['theta_b'][0])
    hc      = float(nc.variables['hc'][0])
    sc_r    = np.array(nc.variables['sc_r'][:], dtype=np.float64)
    Cs_r    = np.array(nc.variables['Cs_r'][:], dtype=np.float64)
    N       = len(sc_r)

    print(f'theta_s={theta_s}, theta_b={theta_b}, hc={hc}, N={N}')

    # ── validate stretching formula ──────────────────────────────────────────
    Cs_r_calc = stretching(sc_r, theta_s, theta_b)
    if not np.allclose(Cs_r_calc, Cs_r, atol=1e-5):
        max_err = np.max(np.abs(Cs_r_calc - Cs_r))
        print(f'ERROR: Cs_r mismatch (max |diff| = {max_err:.2e})')
        print('       Stretching formula or parameters do not match stored Cs_r.')
        print('       Aborting — do not write incorrect attributes.')
        sys.exit(1)
    print(f'Cs_r validation passed (max |diff| = {np.max(np.abs(Cs_r_calc - Cs_r)):.2e})')

    # ── compute Cs_w at w-points ─────────────────────────────────────────────
    sc_w = (np.arange(N + 1) - N) / float(N)   # -1 .. 0, length N+1
    Cs_w = stretching(sc_w, theta_s, theta_b)
    print(f'Cs_w range: [{Cs_w.min():.6f}, {Cs_w.max():.6f}]  (N+1={len(Cs_w)} values)')

    # ── write global attributes ───────────────────────────────────────────────
    nc.setncattr('theta_s', theta_s)
    nc.setncattr('theta_b', theta_b)
    nc.setncattr('hc',      hc)
    nc.setncattr('Cs_r',    Cs_r)
    nc.setncattr('Cs_w',    Cs_w)

    print(f'Written global attributes: theta_s, theta_b, hc, Cs_r, Cs_w')
    print(f'Done -> {INI}')
