# ramp up WEC variables exponentially from 0-1 for first few days

import shutil
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
#plt.ion()

reg = './tides/mc60/frc/mc60'
name = 'mc60'

shutil.copy2(reg+'_wec_smooth.20190415.nc',reg+'_wec_smooth_ramp.20190415.nc')

grdnc = Dataset(name+'_grd.nc','r')
hnc = np.array(grdnc.variables['h'])
masknc = np.array(grdnc.variables['mask_rho'])

nc = Dataset(reg+'_wec_smooth_ramp.20190415.nc','r+')

awave = np.array(nc.variables['Awave'])
pwave = np.array(nc.variables['Pwave'])
eb = np.array(nc.variables['eb'])
ed = np.array(nc.variables['ed'])
#lmw = np.array(nc.variables['lmw'])
qb = np.array(nc.variables['qb'])
sup = np.array(nc.variables['sup'])
uorb = np.array(nc.variables['uorb'])
ust0 = np.array(nc.variables['ust0'])
ust2d = np.array(nc.variables['ust2d'])
vorb = np.array(nc.variables['vorb'])
vst0 = np.array(nc.variables['vst0'])
vst2d = np.array(nc.variables['vst2d'])


# apply scaling to first 3 days (half hour time steps)
dttime = 144
t_norm = np.linspace(0,1,dttime)
alpha = 3.0  # Controls steepness of exponential scale
# shape: (time,)
scale = (np.exp(alpha*t_norm)-1)/(np.exp(alpha)-1)  

#plt.plot(scale)

scale2d = np.ones((dttime,hnc.shape[0],hnc.shape[1]))*np.nan
for s_i in range(len(scale)):
    scale2d[s_i,:,:] = np.ones((hnc.shape[0],hnc.shape[1]))*scale[s_i]

awave_ramp = awave[:dttime,:,:]*scale2d
pwave[pwave<2.0] = 2.0
eb_ramp = eb[:dttime,:,:]*scale2d
ed_ramp = ed[:dttime,:,:]*scale2d
#lmw_ramp = lmw[:dttime,:,:]*scale2d
qb_ramp = qb[:dttime,:,:]*scale2d
sup_ramp = sup[:dttime,:,:]*scale2d
uorb_ramp = uorb[:dttime,:,:]*scale2d
ust0_ramp = ust0[:dttime,:,:]*scale2d
ust2d_ramp = ust2d[:dttime,:,:]*scale2d
vorb_ramp = vorb[:dttime,:,:]*scale2d
vst0_ramp = vst0[:dttime,:,:]*scale2d
vst2d_ramp = vst2d[:dttime,:,:]*scale2d



awavenc = nc.variables['Awave']
pwavenc = nc.variables['Pwave']
ebnc = nc.variables['eb']
ednc = nc.variables['ed']
lmwnc = nc.variables['lmw']
qbnc = nc.variables['qb']
supnc = nc.variables['sup']
uorbnc = nc.variables['uorb']
ust0nc = nc.variables['ust0']
ust2dnc = nc.variables['ust2d']
vorbnc = nc.variables['vorb']
vst0nc = nc.variables['vst0']
vst2dnc = nc.variables['vst2d']

awavenc[:dttime,:,:] = awave_ramp
pwavenc[:,:,:] = pwave
ebnc[:dttime,:,:] = eb_ramp
ednc[:dttime,:,:] = ed_ramp
#lmwnc[:dttime,:,:] = lmw_ramp
qbnc[:dttime,:,:] = qb_ramp
supnc[:dttime,:,:] = sup_ramp
uorbnc[:dttime,:,:] = uorb_ramp
ust0nc[:dttime,:,:] = ust0_ramp
ust2dnc[:dttime,:,:] = ust2d_ramp
vorbnc[:dttime,:,:] = vorb_ramp
vst0nc[:dttime,:,:] = vst0_ramp
vst2dnc[:dttime,:,:] = vst2d_ramp

nc.close()
