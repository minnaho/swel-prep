# create WEC file where nearshore (30 m or shallower) 
# WEC variables are set to 0

import numpy as np
from netCDF4 import Dataset
#import matplotlib.pyplot as plt

# copied WEC file into new _0 file
nc = Dataset('/data/project9/minnaho/swel/tides/mc60/frc/mc60_wec_0.20190415.nc','a')

grdnc = Dataset('/data/project9/minnaho/swel/tides/mc60/mc60_grd.nc','r')
hnc = np.array(grdnc.variables['h'])
masknc = np.array(grdnc.variables['mask_rho'])

# set mask shallower than 30 m to 0
hj,hi = np.where(hnc<30)
masknc[hj,hi] = 0
#plt.imshow(masknc,origin='lower')

awave = nc.variables['Awave']
pwave = nc.variables['Pwave']
eb    = nc.variables['eb']
ed    = nc.variables['ed']
lmw   = nc.variables['lmw']
qb    = nc.variables['qb']
sup   = nc.variables['sup']
uorb  = nc.variables['uorb']
ust0  = nc.variables['ust0']
ust2d = nc.variables['ust2d']
vorb  = nc.variables['vorb']
vst0  = nc.variables['vst0']
vst2d = nc.variables['vst2d']

awave[:] = awave[:]*masknc
pwave[:] = pwave[:]*masknc
eb[:] = eb[:]*masknc
ed[:] = ed[:]*masknc
lmw[:] = lmw[:]*masknc
qb[:] = qb[:]*masknc
sup[:] = sup[:]*masknc
uorb[:] = uorb[:]*masknc
ust0[:] = ust0[:]*masknc
ust2d[:] = ust2d[:]*masknc
vorb[:] = vorb[:]*masknc
vst0[:] = vst0[:]*masknc
vst2d[:] = vst2d[:]*masknc

nc.close()
