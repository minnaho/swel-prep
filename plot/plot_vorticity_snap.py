"""
Single 2x2 snapshot of surface (top s_rho level) normalized vorticity
(zeta/f) across 4 scenarios, all at 2019-04-22 07:01 -- matched by hour
only, since each scenario's raw output stamps its records a few seconds
apart.

Layout: top-left = no tides, no WEC; top-right = no tides, 2.5x WEC;
bottom-left = tides, no WEC; bottom-right = tides, 2.5x WEC.

The target hour falls within mc60_his.2019042123*.nc for all 4 scenarios
(the file whose 12 hourly records start at 2019-04-21 23:01 and run
through 2019-04-22 11:01).

Cache:  ./figs/snapshots/vorticity/vort_snap_cache_<key>_20190422_07.npz
"""

import sys
import os
import glob
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
from netCDF4 import Dataset
import pyfuncs as pf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import cmocean

grd = '/data/project3/minnaho/project9copy/swel/mc60_grd.nc'

grdnc  = Dataset(grd, 'r')
lat_nc = np.array(grdnc.variables['lat_rho'])
lon_nc = np.array(grdnc.variables['lon_rho']) - 360
f_nc   = np.array(grdnc.variables['f'])
maskc  = np.array(grdnc.variables['mask_rho'])

# Drop the 15-point sponge layer on the northern, western, and southern
# boundaries (eta=0 is south, eta=-1 is north, xi=0 is west -- confirmed via
# the grid's lat/lon corners). Eastern boundary is left untouched.
SPONGE = 15
lat_nc = lat_nc[SPONGE:-SPONGE, SPONGE:]
lon_nc = lon_nc[SPONGE:-SPONGE, SPONGE:]
maskc  = maskc[SPONGE:-SPONGE, SPONGE:]

TARGET_GLOB = 'mc60_his.2019042123*.nc'
TARGET_HOUR = 7

# (key, label, directory) in the requested panel order: top-left, top-right,
# bottom-left, bottom-right
SCENARIOS = [
    ('notidesnowec', 'no tides, no WEC',   '/data/project3/minnaho/swel/notides/mc60/nowec/his'),
    ('ampwec',       'no tides, 2.5x WEC', '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything'),
    ('tidesnowec',   'tides, no WEC',      '/data/project3/minnaho/swel/tides/mc60/nowec/output/his'),
    ('tidesampwec',  'tides, 2.5x WEC',    '/data/project3/minnaho/swel/tides/mc60/ampwec/everything'),
]

SAVEFIG_DIR = './figs/snapshots/vorticity'
os.makedirs(SAVEFIG_DIR, exist_ok=True)


def cache_path(key):
    return f'{SAVEFIG_DIR}/vort_snap_cache_{key}_20190422_07.npz'

figh   = 12
figw   = 14
axfont = 16
c_map  = cmocean.cm.balance
v_min  = -5
v_max  =  5


def surface_vort_at_hour(directory, hour):
    files = sorted(glob.glob(os.path.join(directory, TARGET_GLOB)))
    assert len(files) == 1, f'expected 1 file matching {TARGET_GLOB} in {directory}, found {len(files)}'
    f = files[0]

    oceantime = np.array(Dataset(f, 'r').variables['ocean_time'])
    oceandt   = pf.numdate(oceantime, 'second since 1995-01-01')
    matches   = [i for i, dt in enumerate(oceandt) if dt.hour == hour]
    assert len(matches) == 1, f'expected 1 record with hour=={hour} in {f}, found {len(matches)}'
    t_idx = matches[0]

    urho, vrho = pf.rho_uv_angle(f, grd, rotate=True)
    vort = pf.vorticity(grd, urho, vrho) / f_nc
    return vort[t_idx, -1, :, :], oceandt[t_idx]


print('Computing surface vorticity for 4 scenarios...')
panel_data = []
for key, label, directory in SCENARIOS:
    cpath = cache_path(key)
    if os.path.exists(cpath):
        print(f'  {label}: loading cache...')
        with np.load(cpath) as npz:
            surf_vort = npz['vort']
            dt = str(npz['dt'])
    else:
        surf_vort, dt = surface_vort_at_hour(directory, TARGET_HOUR)
        np.savez(cpath, vort=surf_vort, dt=str(dt))
    print(f'  {label}: matched {dt}')
    panel_data.append((label, surf_vort[SPONGE:-SPONGE, SPONGE:]))

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[figw, figh])

pc = None
for ax, (label, surf_vort) in zip(axes.flat, panel_data):
    pc = ax.pcolormesh(lon_nc, lat_nc, surf_vort, cmap=c_map, vmin=v_min, vmax=v_max)
    ax.contour(lon_nc, lat_nc, maskc, colors='k', linewidths=1)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.tick_params(axis='both', which='major', labelsize=axfont)
    ax.set_title(label, fontsize=axfont)

axes[0, 0].set_xlim(right=-121.78)
axes[0, 0].set_ylim(top=37.07)

fig.tight_layout()

pos_tr = axes[0, 1].get_position().get_points().flatten()
pos_br = axes[1, 1].get_position().get_points().flatten()
cb_ax = fig.add_axes([pos_tr[2] + .02, pos_br[1], .015, pos_tr[3] - pos_br[1]])
cb = fig.colorbar(pc, cax=cb_ax, orientation='vertical')
cb.set_label(r'$\zeta/f$', fontsize=axfont)
cb.ax.tick_params(axis='both', which='major', labelsize=axfont)

out_path = f'{SAVEFIG_DIR}/vort_snap_4scen_20190422_07.png'
plt.savefig(out_path, bbox_inches='tight', dpi=600)
plt.close()
print(f'saved -> {out_path}')
