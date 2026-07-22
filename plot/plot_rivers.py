import sys
sys.path.append('/data/project3/minnaho/global/')
from netCDF4 import Dataset,num2date
import numpy as np
import matplotlib.pyplot as plt
import pyfuncs as pf
import cftime
from datetime import datetime

grd = Dataset('../tides/mc60/mc60_grd.nc','r')

mask = np.array(grd.variables['mask_rho'])
riv = np.array(grd.variables['river_flux'])

rivj,rivi = np.where(riv>0)
rivnum = riv[rivj,rivi]

plt.ion()
plt.imshow(mask,origin='lower')
plt.scatter(rivi,rivj,c=rivnum,cmap='Paired')
plt.colorbar()

rivnc = Dataset('../tides/mc60/rivpip/mc60_riv_trace.nc','r')
rivvol = np.array(rivnc.variables['river_volume'])
rivdt = np.array(rivnc.variables['river_time'])
rivt = np.array(rivnc.variables['river_tracer'])
rivdate = pf.numdate(rivdt,'days since 1995-01-01')

target_date = datetime(2019,4,15)
dtidx = np.where(target_date==rivdate)[0][0]
rivvol[dtidx-10:dtidx+10,3]
