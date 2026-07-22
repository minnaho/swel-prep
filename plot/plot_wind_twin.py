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
wec = '/data/project3/minnaho/project9copy/swel/tides/'+sim+'/frc/mc60_wec_smooth_ramp_trim_sup.20190415.nc'
rampscale = '/data/project3/minnaho/project9copy/swel/tides/'+sim+'/frc/mc60_wec_rampscale.20190415.nc'
#grd = '/data/project3/minnaho/project9copy/swel/tides/'+sim+'/mc60_grd.nc'

frcnc = Dataset(frc,'r')
wecnc = Dataset(wec,'r')
rampnc = Dataset(rampscale,'r')
# hisnc = Dataset(his,'r') # 'his' is not defined in this snippet, left as is if you need it
#grdnc = Dataset(grd,'r')

frctime = np.array(frcnc.variables['time'])
wectime = np.array(wecnc.variables['wwv_time'])
ramptime = np.array(rampnc.variables['wwv_time'])

frcdt = pf.numdate(frctime,'days since 1995-01-01')
wecdt = pf.numdate(wectime,'days since 1995-01-01')
rampdt = pf.numdate(ramptime,'days since 1995-01-01')

wecvar = list(wecnc.variables.keys())

uwnd = np.array(frcnc.variables['uwnd'])
vwnd = np.array(frcnc.variables['vwnd'])

wspd = np.sqrt((uwnd**2)+(vwnd**2))

avgwspd = np.nanmean(wspd,axis=(1,2))

awave = np.array(wecnc.variables['Awave'])
avgawave = np.nanmean(awave,axis=(1,2))

ramp_awave = np.array(rampnc.variables['Awave'])
avg_ramp_awave = np.nanmean(ramp_awave,axis=(1,2))

# --- Plotting Section ---
axfont = 14
fig, ax1 = plt.subplots(1, 1, figsize=[12, 5])

# Primary Y-Axis (Wind Speed)
color1 = 'orange'
line1 = ax1.plot(frcdt[335:], avgwspd[335:], color=color1, label='wind speed')
ax1.set_ylabel('Wind Speed (m s$^{-1}$)', color=color1, fontsize=axfont)
# Added labelsize=axfont for y-ticks and applied it to the x-axis ticks as well
ax1.tick_params(axis='y', labelcolor=color1, labelsize=axfont)
ax1.tick_params(axis='x', labelsize=axfont)

# Secondary Y-Axis (Wave Amplitude)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color2 = 'navy'
line2 = ax2.plot(wecdt, avgawave, linestyle='--', color=color2, label='wave amplitude')
color3 = 'navy'
line3 = ax2.plot(rampdt, avg_ramp_awave, linestyle=':', color=color3, label='wave amplitude 2.5x scaled')
ax2.set_ylabel('Wave Amplitude (m)', color=color2, fontsize=axfont)
# Added labelsize=axfont for secondary y-ticks
ax2.tick_params(axis='y', labelcolor=color2, labelsize=axfont)

# Combine the legends from both axes
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
# Added fontsize=axfont to the legend
ax1.legend(lines, labels, loc='upper right', fontsize=axfont)

# Formats the time axis (optional, but keeps it clean)
import matplotlib.dates as mdates
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

fig.tight_layout() # Ensures the right y-label isn't cut off by the figure edge
plt.savefig('./figs/wind_wave_amp.png', bbox_inches='tight')
print('saved ./figs/wind_wave_amp.png')
