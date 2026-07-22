import sys
import os
import glob
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
from netCDF4 import Dataset, num2date
import pyfuncs as pf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cmocean

grd = '/data/project3/minnaho/project9copy/swel/mc60_grd.nc'

grdnc  = Dataset(grd, 'r')
lat_nc = np.array(grdnc.variables['lat_rho'])
lon_nc = np.array(grdnc.variables['lon_rho']) - 360
f_nc   = np.array(grdnc.variables['f'])
maskc  = np.array(grdnc.variables['mask_rho'])

#WEC_DIR   = '/data/project3/minnaho/swel/tides/mc60/wec/his'
WEC_DIR   = '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything'
NOWEC_DIR = '/data/project3/minnaho/swel/notides/mc60/nowec/output/his'

wec_files   = sorted(glob.glob(os.path.join(WEC_DIR,   'mc60_his.*.nc')))
nowec_files = sorted(glob.glob(os.path.join(NOWEC_DIR, 'mc60_his.*.nc')))

SAVEFIG_DIR = './figs/snapshots/vorticity'
os.makedirs(SAVEFIG_DIR, exist_ok=True)

figh   = 9
figw   = 15
axfont = 16
c_map  = cmocean.cm.balance
v_min  = -5
v_max  =  5

for wf, nwf in zip(wec_files, nowec_files):
    print(f'processing {os.path.basename(wf)}')

    # cheap pre-check: read times, build expected PNG paths, skip if all done
    oceantime = np.array(Dataset(wf, 'r').variables['ocean_time'])
    oceandt   = pf.numdate(oceantime, 'second since 1995-01-01')
    png_paths = [f'{SAVEFIG_DIR}/vort5_notides_'
                 f'{dt.year}{dt.month:02d}{dt.day:02d}{dt.hour:02d}{dt.minute:02d}.png'
                 for dt in oceandt]
    todo = [i for i, p in enumerate(png_paths) if not os.path.exists(p)]
    if not todo:
        print('  all time steps done, skipping')
        continue

    oceantime_nw = np.array(Dataset(nwf, 'r').variables['ocean_time'])

    print('  calculating urho, vrho')
    wec_urho,   wec_vrho   = pf.rho_uv_angle(wf,  grd, rotate=True)
    nowec_urho, nowec_vrho = pf.rho_uv_angle(nwf, grd, rotate=True)

    print('  calculating vorticity')
    wec_normvort   = pf.vorticity(grd, wec_urho,   wec_vrho)   / f_nc
    nowec_normvort = pf.vorticity(grd, nowec_urho, nowec_vrho) / f_nc

    # save NetCDF
    stem = os.path.basename(wf)
    with Dataset(f'notideampwec_rossby_{stem}', 'w', format='NETCDF4') as nc:
        nc.createDimension('time',    wec_normvort.shape[0])
        nc.createDimension('s_rho',   wec_normvort.shape[1])
        nc.createDimension('eta_rho', wec_normvort.shape[2])
        nc.createDimension('xi_rho',  wec_normvort.shape[3])
        tv = nc.createVariable('ocean_time', 'f8', ('time',))
        rv = nc.createVariable('Rossby',     'f8', ('time', 's_rho', 'eta_rho', 'xi_rho'))
        tv[:] = oceantime;  tv.units = 'seconds since 1995-01-01'
        rv[:] = wec_normvort

    #stem_nw = os.path.basename(nwf)
    #with Dataset(f'nowec_rossby_{stem_nw}', 'w', format='NETCDF4') as nc:
    #    nc.createDimension('time',    nowec_normvort.shape[0])
    #    nc.createDimension('s_rho',   nowec_normvort.shape[1])
    #    nc.createDimension('eta_rho', nowec_normvort.shape[2])
    #    nc.createDimension('xi_rho',  nowec_normvort.shape[3])
    #    tv = nc.createVariable('ocean_time', 'f8', ('time',))
    #    rv = nc.createVariable('Rossby',     'f8', ('time', 's_rho', 'eta_rho', 'xi_rho'))
    #    tv[:] = oceantime_nw;  tv.units = 'seconds since 1995-01-01'
    #    rv[:] = nowec_normvort

    for t_i in range(len(oceandt)):
        if os.path.exists(png_paths[t_i]):
            print(f'  {t_i} of {len(oceandt)} — already exists, skipping')
            continue
        print(f'  {t_i} of {len(oceandt)}')
        dt = oceandt[t_i]

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[figw, figh])
        p_plot0 = ax.flat[0].pcolormesh(lon_nc, lat_nc, wec_normvort[t_i, -1, :, :],
                                        cmap=c_map, vmin=v_min, vmax=v_max)
        ax.flat[1].pcolormesh(lon_nc, lat_nc, nowec_normvort[t_i, -1, :, :],
                              cmap=c_map, vmin=v_min, vmax=v_max)
        for a in ax.flat:
            a.contour(lon_nc, lat_nc, maskc, colors='k', linewidths=1)
            a.tick_params(axis='both', which='major', labelsize=axfont)
        ax.flat[0].set_title('no tides, 2.5x WEC',       fontsize=axfont)
        ax.flat[1].set_title('no tides, no WEC', fontsize=axfont)
        fig.suptitle(f'{dt.year}-{dt.month:02d}-{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}',
                     fontsize=axfont)
        fig.tight_layout()

        p1 = ax.flat[1].get_position().get_points().flatten()
        cb_ax = fig.add_axes([p1[2] + .015, p1[1], .01, p1[3] - p1[1]])
        cb = fig.colorbar(p_plot0, cax=cb_ax, orientation='vertical')
        cb.set_label(r'$\zeta/f$', fontsize=axfont)
        cb.ax.tick_params(axis='both', which='major', labelsize=axfont)

        plt.savefig(png_paths[t_i], bbox_inches='tight', dpi=600)
        plt.close()
