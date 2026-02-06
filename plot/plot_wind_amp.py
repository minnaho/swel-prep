# plot wind and wave amplitude 
import os
import sys
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
from netCDF4 import Dataset,num2date
import matplotlib.pyplot as plt
import pyfuncs as pf
import ROMS_depths as rd

sim = 'mc60'
month = '201904'

frc = '/data/project9/minnaho/swel/tides/'+sim+'/frc/'+sim+'_frc.'+month+'.nc'
wec = '/data/project9/minnaho/swel/tides/'+sim+'/frc/'+sim+'_wec_taper.'+month+'15.nc'
his = '/data/project9/minnaho/swel/tides/'+sim+'/output/mc60_his.20190415110120.nc'
grd = '/data/project9/minnaho/swel/tides/'+sim+'/mc60_grd_fill.nc'

frcnc = Dataset(frc,'r')
wecnc = Dataset(wec,'r')
hisnc = Dataset(his,'r')
grdnc = Dataset(grd,'r')

frctime = np.array(frcnc.variables['time'])
wectime = np.array(wecnc.variables['wwv_time'])
histime = np.array(hisnc.variables['ocean_time'])

frcdt = pf.numdate(frctime,'days since 1995-01-01')
wecdt = pf.numdate(wectime,'days since 1995-01-01')
hisdt = pf.numdate(histime,'seconds since 1995-01-01')

wecvar = list(wecnc.variables.keys())



uwnd = np.array(frcnc.variables['uwnd'])
vwnd = np.array(frcnc.variables['vwnd'])

wspd = np.sqrt((uwnd**2)+(vwnd**2))

avgwspd = np.nanmean(wspd,axis=(1,2))

awave = np.array(wecnc.variables['Awave'])
avgawave = np.nanmean(awave,axis=(1,2))

zr = rd.get_zr_loop(hisnc,grdnc)

fig,ax = plt.subplots(1,1)
ax.plot(frcdt[335:],avgwspd[335:]/np.nanmax(avgwspd[335:]),label='wind speed')
ax.plot(wecdt,avgawave/np.nanmax(avgawave),linestyle='--',label='wave amplitude')
ax.legend()
ax.set_ylabel('normalized magnitude')
