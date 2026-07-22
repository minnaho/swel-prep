import sys
sys.path.append('/data/project3/minnaho/global/')
from netCDF4 import Dataset,num2date
import numpy as np
import matplotlib.pyplot as plt
import cftime
import ROMS_depths as depths

grd0 = Dataset('../tides/mc60/mc60_grd.nc','r')
grd1 = Dataset('../smode200_grd.nc','r')
grd2 = Dataset('/data/project3/minnaho/project9copy/sftracer/frc/sfbay_grd.nc','r')

hisavg = Dataset('/data/project3/minnaho/swel/tides/mc60/wec/his/mc60_wec_his_7dayavg.nc','r')

zr = np.squeeze(depths.get_zr_loop(hisavg,grd0))

wavg = np.squeeze(hisavg.variables['w'][:])

mask0 = np.array(grd0.variables['mask_rho'])
pip0 = np.array(grd0.variables['pipe_flux'])

pip0j,pip0i = np.where(pip0>0)
pip0num = pip0[pip0j,pip0i]
h0 = grd0.variables['h'][:]
print(str(h0[pip0j,pip0i]))

mask1 = np.array(grd1.variables['mask_rho'])
pip1 = np.array(grd1.variables['pipe_flux'])

pip1j,pip1i = np.where(pip1>0)
pip1num = pip1[pip1j,pip1i]

mask2 = np.array(grd2.variables['mask_rho'])
pip2 = np.array(grd2.variables['pipe_flux'])

pip2j,pip2i = np.where(pip2>0)
pip2num = pip2[pip2j,pip2i]

plt.ion()
plt.imshow(mask0,origin='lower')
plt.scatter(pip0i,pip0j,c=pip0num,cmap='Paired')
plt.colorbar()

pip0nc = Dataset('../tides/mc60/rivpip/mc60_pip_trace.nc','r')
pip0vol = np.array(pip0nc.variables['pipe_volume'])
pip0dt = np.array(pip0nc.variables['pipe_time'])
pip0t = np.array(pip0nc.variables['pipe_tracer'])
pip0date = num2date(pip0dt,'days since 1995-01-01')

pip1nc = Dataset('../smode200_pip.nc','r')
pip1vol = np.array(pip1nc.variables['pipe_volume'])
pip1dt = np.array(pip1nc.variables['pipe_time'])
pip1t = np.array(pip1nc.variables['pipe_tracer'])
pip1date = num2date(pip1dt,'days since 1995-01-01')

#target_date = cftime.DatetimeGregorian(2019,4,15)
#dtidx = np.where(target_date==rivdate)[0][0]
#rivvol[dtidx-10:dtidx+10,3]
