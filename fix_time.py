# frc files were made using time since 2000-01-01
# while bry and rst are from 1995-01-01
# and since paracas isn't working, manually fix dates of frc
#
# make sure to only run once for each file! 
# will mess up time more if run more than once

import numpy as np
from netCDF4 import Dataset,num2date,date2num
from datetime import datetime

ncfile = '/data/project9/minnaho/swel/tides/sfshelf60/frc/sfshelf60_frc.201905.nc'

nc = Dataset(ncfile,'r')

time = np.array(nc.variables['time'])
rad_time = np.array(nc.variables['rad_time'])

od = 'days since 2000-01-01'
nd = 'days since 1995-01-01'

old_time = num2date(time,od)
old_rad_time = num2date(rad_time,od)

temp_time = num2date(time,nd)
temp_rad_time = num2date(rad_time,nd)

delta = datetime(old_time[0].year,old_time[0].month,old_time[0].day,old_time[0].hour,old_time[0].minute,old_time[0].second,old_time[0].microsecond)-datetime(temp_time[0].year,temp_time[0].month,temp_time[0].day,temp_time[0].hour,temp_time[0].minute,temp_time[0].second,temp_time[0].microsecond)

dt = delta.days

delta_rad_time = datetime(old_rad_time[0].year,old_rad_time[0].month,old_rad_time[0].day,old_rad_time[0].hour,old_rad_time[0].minute,old_rad_time[0].second,old_rad_time[0].microsecond)-datetime(temp_rad_time[0].year,temp_rad_time[0].month,temp_rad_time[0].day,temp_rad_time[0].hour,temp_rad_time[0].minute,temp_rad_time[0].second,temp_rad_time[0].microsecond)

dt_rad_time = delta_rad_time.days

fixed_time = time+dt
fixed_rad_time = rad_time+dt_rad_time

nc.close()

newnc = Dataset(ncfile,'a')
time_var = newnc.variables['time']
rad_time_var = newnc.variables['rad_time']

time_var.long_name = 'time since 1995-01-01'
rad_time_var.long_name='time since 1995-01-01'

time_var[:] = fixed_time
rad_time_var[:] = fixed_rad_time


newnc.close()

# verify it changed
nc = Dataset(ncfile,'r')

time = np.array(nc.variables['time'])
rad_time = np.array(nc.variables['rad_time'])

new_time = num2date(time,nd)
new_rad_time = num2date(rad_time,nd)


nc.close()

