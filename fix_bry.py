# bry file mc60_bry.20190415110220.nc
# has NaN in the last time step for all variables
# --> take average between previous time and next time

import numpy as np
from netCDF4 import Dataset,num2date,date2num
from datetime import datetime

ncfile1 = '/data/project9/minnaho/swel/tides/mc60/bry/mc60_bry_new.20190415110220.nc'
ncfile2 = '/data/project9/minnaho/swel/tides/mc60/bry/mc60_bry.20190419110200.nc'

nc1 = Dataset(ncfile1,'r+')
nc2 = Dataset(ncfile2,'r')

dict1 = list(nc1.variables.keys())

for d_i in range(len(dict1)):
    print(str(d_i)+' of '+str(len(dict1)))
    if dict1[d_i] == 'bry_time':
        continue
    var1 = nc1.variables[dict1[d_i]]
    temp1 = np.array(nc1.variables[dict1[d_i]])
    temp2 = np.array(nc2.variables[dict1[d_i]])
    if len(temp1.shape) == 2:
        if np.isnan(temp1[-1,0]) == True:
            temp1[-1,:] = np.nanmean([temp1[-2,:],temp2[0,:]],axis=0)
            var1[-1,:] = temp1[-1,:]
    elif len(temp1.shape) == 3:
        if np.isnan(temp1[-1,0,0]) == True:
            temp1[-1,:,:] = np.nanmean([temp1[-2,:,:],temp2[0,:,:]],axis=0)
            var1[-1,:,:] = temp1[-1,:,:]

nc1.close()

