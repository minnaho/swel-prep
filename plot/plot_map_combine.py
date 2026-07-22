import sys
import os
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import cmocean
import cartopy.crs as ccrs
from scipy.ndimage import map_coordinates
import pyfuncs as pf

# --- Grid ---
sfnc = Dataset('/data/project3/minnaho/project9copy/swel/mc60_grd.nc', 'r')
sf_h    = np.array(sfnc.variables['h'])
sf_lon  = np.array(sfnc.variables['lon_rho']) - 360
sf_lat  = np.array(sfnc.variables['lat_rho'])
sf_mask = np.array(sfnc.variables['mask_rho'])
sf_riv  = np.array(sfnc.variables['river_flux'])
sf_pip  = np.array(sfnc.variables['pipe_flux'])

cmasknc      = Dataset('/data/project3/minnaho/project9copy/swel/plot/coastal_mask.nc', 'r')
coastal_mask = np.array(cmasknc.variables['coastal_mask'])

rivy, rivx = np.where(sf_riv > 0)
rivlon = sf_lon[rivy[1::3], rivx[1::3]]
rivlat = sf_lat[rivy[1::3], rivx[1::3]]

pipy, pipx = np.where(sf_pip > 0)
piplon = sf_lon[pipy, pipx]
piplat = sf_lat[pipy, pipx]

# --- Wave forcing (WEC file) ---
wec_file  = '/data/project3/minnaho/project9copy/swel/tides/mc60/frc/mc60_wec_smooth_ramp_trim_sup.20190415.nc'
ramp_file = '/data/project3/minnaho/project9copy/swel/tides/mc60/frc/mc60_wec_rampscale.20190415.nc'
frc_file  = '/data/project9/minnaho/swel/tides/mc60/frc/mc60_frc.201904.nc'

wecnc  = Dataset(wec_file,  'r')
rampnc = Dataset(ramp_file, 'r')
frcnc  = Dataset(frc_file,  'r')

awave      = np.array(wecnc.variables['Awave'])          # (time, eta, xi)
ramp_awave = np.array(rampnc.variables['Awave'])

# time-mean wave amplitude, masked to ocean cells
awave_mean = np.nanmean(awave, axis=0)
awave_mean[sf_mask == 0] = np.nan

# time-series averages
avgawave      = np.nanmean(awave,      axis=(1, 2))
avg_ramp_awave = np.nanmean(ramp_awave, axis=(1, 2))

uwnd = np.array(frcnc.variables['uwnd'])
vwnd = np.array(frcnc.variables['vwnd'])
avgwspd = np.nanmean(np.sqrt(uwnd**2 + vwnd**2), axis=(1, 2))

wectime  = np.array(wecnc.variables['wwv_time'])
ramptime = np.array(rampnc.variables['wwv_time'])
frctime  = np.array(frcnc.variables['time'])

wecdt  = pf.numdate(wectime,  'days since 1995-01-01')
rampdt = pf.numdate(ramptime, 'days since 1995-01-01')
frcdt  = pf.numdate(frctime,  'days since 1995-01-01')

# --- Layout ---
axfont = 14
fig = plt.figure(figsize=[22, 13])
gs  = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.25, wspace=0.12)

ax1   = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())  # bathymetry
ax2   = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())  # wave amplitude
ax_ts = fig.add_subplot(gs[1, :])                                  # time series

XLIM = [sf_lon.min(), sf_lon.max()]
YLIM = [sf_lat.min(), sf_lat.max()]
ax1.set_extent([XLIM[0], XLIM[1], YLIM[0], YLIM[1]], crs=ccrs.PlateCarree())
ax2.set_extent([XLIM[0], -121.77,  YLIM[0], YLIM[1]], crs=ccrs.PlateCarree())

# --- Left panel: bathymetry ---
p_bathy = ax1.pcolormesh(sf_lon, sf_lat, -sf_h, transform=ccrs.PlateCarree(),
                         cmap=cmocean.cm.deep_r, vmin=-2000, vmax=0)
ax1.contour(sf_lon, sf_lat, sf_mask, colors='k', transform=ccrs.PlateCarree(), linewidths=1)
ax1.scatter(rivlon, rivlat, marker='o', c='lightblue', edgecolors='navy', s=60,
            transform=ccrs.PlateCarree(), label='Rivers', zorder=2)
ax1.scatter(piplon, piplat, marker='^', c='purple', edgecolors='None', s=80,
            transform=ccrs.PlateCarree(), label='Outfalls', zorder=3)

_LENGTH_XI = 250
_N_PTS     = 300
for _eta0, _xi0, _slope, _color, _label in [
    (271, 543, -0.8, 'red',  'Transect S'),
    (832, 646,  0.6, 'blue', 'Transect N'),
]:
    _xi  = np.linspace(_xi0,  _xi0  - _LENGTH_XI, _N_PTS)
    _eta = np.linspace(_eta0, _eta0 + _slope * (-_LENGTH_XI), _N_PTS)
    _crd = np.array([_eta, _xi])
    _tlon = map_coordinates(sf_lon, _crd, order=1, mode='nearest')
    _tlat = map_coordinates(sf_lat, _crd, order=1, mode='nearest')
    ax1.plot(_tlon, _tlat, '-', color=_color, linewidth=2,
             transform=ccrs.PlateCarree(), label=_label)
    ax1.plot(_tlon[0], _tlat[0], 'o', color=_color, markersize=7,
             transform=ccrs.PlateCarree())

ax1.legend(loc='lower right', fontsize=axfont - 2)

contours = ax1.contour(sf_lon, sf_lat, sf_h, [50, 100, 500, 1000, 1500, 2000],
                       colors='black', linewidths=0.5)
manual_pos = [
    (-122.15521671603335, 36.951516679194675),
    (-122.18968283307998, 36.89677462037397),
    (-122.14653235180069, 36.79974502146389),
    (-122.19880748200471, 36.702231681647184),
    (-122.16556182910833, 36.6486139904713),
    (-122.15735317111466, 36.632662098707534),
]
ax1.clabel(contours, inline=False, fontsize=8, fmt='%d', manual=manual_pos)

gl1 = ax1.gridlines(draw_labels=True, linestyle='--')
gl1.xlabels_top   = False
gl1.ylabels_right = False
gl1.ylabel_style  = {'size': axfont - 1}
gl1.xlabel_style  = {'size': axfont - 1}

# --- Right panel: time-mean wave amplitude ---
vmax_wave = np.nanpercentile(awave_mean[sf_mask > 0], 99)
p_wave = ax2.pcolormesh(sf_lon, sf_lat, awave_mean, transform=ccrs.PlateCarree(),
                        cmap=cmocean.cm.haline, vmin=0, vmax=vmax_wave)
ax2.contour(sf_lon, sf_lat, coastal_mask, levels=[0.5], colors='white',
            transform=ccrs.PlateCarree(), linewidths=1.5, linestyles='--')
ax2.contour(sf_lon, sf_lat, sf_mask, colors='k', transform=ccrs.PlateCarree(), linewidths=1)

gl2 = ax2.gridlines(draw_labels=True, linestyle='--')
gl2.xlabels_top   = False
gl2.ylabels_left  = False
gl2.ylabels_right = False
gl2.ylabel_style  = {'size': axfont - 1}
gl2.xlabel_style  = {'size': axfont - 1}

# --- Bottom: wind speed + wave amplitude time series ---
color_wind = 'orange'
line1 = ax_ts.plot(frcdt[335:], avgwspd[335:], color=color_wind, label='wind speed')
ax_ts.set_ylabel('Wind Speed (m s$^{-1}$)', color=color_wind, fontsize=axfont)
ax_ts.tick_params(axis='y', labelcolor=color_wind, labelsize=axfont)
ax_ts.tick_params(axis='x', labelsize=axfont)

ax_ts2 = ax_ts.twinx()
color_wave = 'navy'
line2 = ax_ts2.plot(wecdt,  avgawave,       linestyle='--', color=color_wave, label='wave amplitude')
line3 = ax_ts2.plot(rampdt, avg_ramp_awave, linestyle=':',  color=color_wave, label='wave amplitude 2.5x scaled')
ax_ts2.set_ylabel('Wave Amplitude (m)', color=color_wave, fontsize=axfont)
ax_ts2.tick_params(axis='y', labelcolor=color_wave, labelsize=axfont)

lines  = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax_ts.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.01),
             ncol=len(lines), frameon=False, fontsize=axfont - 2)
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

# --- Colorbars + time series alignment (after all content is drawn) ---
fig.canvas.draw()

pos1   = ax1.get_position()
pos2   = ax2.get_position()
pos_ts = ax_ts.get_position()

cb_ax1 = fig.add_axes([pos1.x1 + 0.008, pos1.y0, 0.010, pos1.y1 - pos1.y0])
cb1    = fig.colorbar(p_bathy, cax=cb_ax1, orientation='vertical')
cb1.set_ticks(np.array([0, -100, -500, -1000, -1500, -2000]))
cb1.set_label('Depth (m)', fontsize=axfont)
cb1.ax.tick_params(labelsize=axfont - 2)

cb_ax2 = fig.add_axes([pos2.x1 + 0.008, pos2.y0, 0.010, pos2.y1 - pos2.y0])
cb2    = fig.colorbar(p_wave, cax=cb_ax2, orientation='vertical')
cb2.set_label('Wave amplitude (m)', fontsize=axfont)
cb2.ax.tick_params(labelsize=axfont - 2)

# Align time series width to ax1.x0 → ax2.x1; apply to both axes (twinx has its own position)
new_pos = [pos1.x0, pos_ts.y0, pos2.x1 - pos1.x0, pos_ts.height]
ax_ts.set_position(new_pos)
ax_ts2.set_position(new_pos)

fig.savefig('./figs/grid_combined.png', bbox_inches='tight', dpi=600)
print('saved ./figs/grid_combined.png')
