# apply tapering near mask
# H < 20 or 30 m 
# linearly decrease Awave and Stakes velocity components

import shutil
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
#plt.ion()

reg = './tides/mc60/frc/mc60'
name = 'mc60'

innc = reg+'_wec_smooth_ramp_trim.20190415.nc'
outnc = reg+'_wec_smooth_ramp_trim_taper.20190415.nc'

shutil.copy2(innc,outnc)

H = 20 # conservative bathymetric contour to start tapering
       # could try 20 m or 30 m

grdnc = Dataset(name+'_grd.nc','r')
hnc = np.array(grdnc.variables['h'])
masknc = np.array(grdnc.variables['mask_rho'])

nc = Dataset(outnc,'r+')

awave = np.array(nc.variables['Awave'])
ust0 = np.array(nc.variables['ust0'])
ust2d = np.array(nc.variables['ust2d'])
vst0 = np.array(nc.variables['vst0'])
vst2d = np.array(nc.variables['vst2d'])

# taper Awave and ust towards coast
hj,hi = np.where(hnc<H) 
awave_tape = np.copy(awave)
awave_tape[:,hj,hi] = (hnc[hj,hi]/H)*awave[:,hj,hi]

ust0_tape = np.copy(ust0)
ust0_tape[:,hj,hi] = ((hnc[hj,hi]/H)**2)*ust0[:,hj,hi]

ust2d_tape = np.copy(ust2d)
ust2d_tape[:,hj,hi] = ((hnc[hj,hi]/H)**2)*ust2d[:,hj,hi]

vst0_tape = np.copy(vst0)
vst0_tape[:,hj,hi] = ((hnc[hj,hi]/H)**2)*vst0[:,hj,hi]

vst2d_tape = np.copy(vst2d)
vst2d_tape[:,hj,hi] = ((hnc[hj,hi]/H)**2)*vst2d[:,hj,hi]

# plot to see taper
plt.imshow(awave[100,:,:],cmap='rainbow')
plt.colorbar()
plt.contour(hnc,[H])
plt.contour(masknc,colors='k')
plt.imshow(awave[100,:,:]-awave_tape[100,:,:],cmap='rainbow')

plt.imshow(ust0[100,:,:],cmap='rainbow')
plt.colorbar()
plt.contour(hnc,[H])
plt.contour(masknc,colors='k')
plt.imshow(ust0[100,:,:]-ust0_tape[100,:,:],cmap='rainbow')

awavenc = nc.variables['Awave']
ust0nc = nc.variables['ust0']
ust2dnc = nc.variables['ust2d']
vst0nc = nc.variables['vst0']
vst2dnc = nc.variables['vst2d']

awavenc[:,:,:] = awave_tape[:,:,:]
ust0nc[:,:,:] = ust0_tape[:,:,:]
ust2dnc[:,:,:] = ust2d_tape[:,:,:]
vst0nc[:,:,:] = vst0_tape[:,:,:]
vst2dnc[:,:,:] = vst2d_tape[:,:,:]

nc.close()
