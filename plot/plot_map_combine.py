import sys
import os
sys.path.append('/data/project3/minnaho/global/')
sys.path.append('/data/project3/minnaho/project9copy/swel/plot/boxavg/')
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
import pyfuncs as pf
import cs_boxavg as cb

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

ramp_awave_mean = np.nanmean(ramp_awave, axis=0)
ramp_awave_mean[sf_mask == 0] = np.nan

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
fig = plt.figure(figsize=[16, 13])
gs  = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.16, wspace=0.06)

def add_axis_labels(ax, gl, xlabel=None, ylabel=None, pad=0.03):
    """'Longitude'/'Latitude' labels for a cartopy GeoAxes. ax.set_xlabel/
    set_ylabel render off-figure here (cartopy disables the normal x/y axis,
    so matplotlib's default label offset doesn't account for the gridliner's
    own tick labels) -- placed as free text instead, just past however far
    out the gridliner's own tick labels (gl.xlabel_artists/ylabel_artists)
    actually extend, measured after a draw rather than guessed."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if xlabel is not None:
        ymins = [t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted()).y0
                 for t in gl.xlabel_artists]
        y = (min(ymins) if ymins else 0.0) - pad
        ax.text(0.5, y, xlabel, transform=ax.transAxes, ha='center', va='top', fontsize=axfont)
    if ylabel is not None:
        xmins = [t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted()).x0
                 for t in gl.ylabel_artists]
        x = (min(xmins) if xmins else 0.0) - pad
        ax.text(x, 0.5, ylabel, transform=ax.transAxes, ha='right', va='center',
                rotation='vertical', fontsize=axfont)

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
for _eta0, _xi0, _slope, _depth_lim, _color, _label in [
    (271, 543, -0.8, -125, 'red',  'Transect S'),
    (832, 646,  0.6, -90,  'blue', 'Transect N'),
]:
    _tr = cb.build_box_transect(sf_lon, sf_lat, sf_mask, _eta0, _xi0, _slope,
                                _depth_lim, _LENGTH_XI, _N_PTS)
    ax1.plot(_tr['lon'], _tr['lat'], '-', color=_color, linewidth=2,
             transform=ccrs.PlateCarree(), label=_label)
    ax1.plot(_tr['lon'][0], _tr['lat'][0], 'o', color=_color, markersize=7,
             transform=ccrs.PlateCarree())

    # box-average footprint (see cs_boxavg.py) -- the band the *_box.py
    # scripts actually average over, not just the centre line above
    (_elo_lon, _elo_lat), (_ehi_lon, _ehi_lat) = cb.box_edges(_tr)
    ax1.plot(_elo_lon, _elo_lat, '--', color=_color, linewidth=1,
             transform=ccrs.PlateCarree())
    ax1.plot(_ehi_lon, _ehi_lat, '--', color=_color, linewidth=1,
             transform=ccrs.PlateCarree())

# Mid transect -- fixed-eta bay-to-offshore slice, same geometry as
# plot_cs_mid_o2.py / plot_cs_diag_avg_diff.py's 'mid' transect
_ETA_MID       = 477
_DEPTH_LIM_MID = -300
_tr_mid = cb.build_box_mid_transect(sf_lon, sf_lat, sf_mask, _ETA_MID, _DEPTH_LIM_MID)
_mid_lat_row = sf_lat[_ETA_MID, :]
ax1.plot(_tr_mid['lon'], _mid_lat_row, '-', color='green', linewidth=2,
         transform=ccrs.PlateCarree(), label='Transect Mid')
ax1.plot(_tr_mid['lon'][0], _mid_lat_row[0], 'o', color='green', markersize=7,
         transform=ccrs.PlateCarree())

(_elo_lon, _elo_lat), (_ehi_lon, _ehi_lat) = cb.box_edges(_tr_mid)
ax1.plot(_elo_lon, _elo_lat, '--', color='green', linewidth=1,
         transform=ccrs.PlateCarree())
ax1.plot(_ehi_lon, _ehi_lat, '--', color='green', linewidth=1,
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
add_axis_labels(ax1, gl1, xlabel='Longitude', ylabel='Latitude')

# --- Right panel: time-mean wave amplitude ---
vmax_wave = np.nanpercentile(ramp_awave_mean[sf_mask > 0], 99)
p_wave = ax2.pcolormesh(sf_lon, sf_lat, ramp_awave_mean, transform=ccrs.PlateCarree(),
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
add_axis_labels(ax2, gl2, xlabel='Longitude', ylabel=None)

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
ax_ts.set_xlabel('Date', fontsize=axfont)

lines  = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax_ts.legend(lines, labels, loc='upper right', ncol=len(lines),
             frameon=True, framealpha=0.8, fontsize=axfont - 3)
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

# --- Colorbars + time series alignment (after all content is drawn) ---
fig.canvas.draw()

pos1   = ax1.get_position()
pos2   = ax2.get_position()
pos_ts = ax_ts.get_position()

# Push the maps down to the bottom of their gridspec row instead of leaving
# them vertically centered. Cartopy's aspect lock height-shrinks ax1 to fit
# the cell's width and then centers it in the leftover vertical room by
# default -- that unused margin below ax1 is exactly the visible gap to the
# time-series panel. Bottom-align to the row's actual cell so nothing sits
# below the maps but hspace itself.
cell0 = ax1.get_subplotspec().get_position(fig)
ax1.set_position([pos1.x0, cell0.y0, pos1.width, pos1.height])
fig.canvas.draw()
pos1 = ax1.get_position()

# Match ax2's box height to ax1's. Cartopy's aspect-locked GeoAxes fit each
# panel to preserve its true physical aspect ratio within the same nominal
# gridspec cell -- ax2 covers a narrower longitude span than ax1 at the same
# latitude span, so it fills the cell's full height while ax1 (proportionally
# wider) gets height-shrunk to fit the cell's width instead. Left as-is, ax2
# renders visibly taller than ax1 with mismatched top/bottom edges. Keep
# ax2's own aspect ratio (its width comes along for the ride) and re-center
# it vertically in the row so both panels share the same y-span.
aspect2 = pos2.width / pos2.height
new_h2  = pos1.height
new_w2  = new_h2 * aspect2
ax2.set_position([pos2.x0, pos1.y0, new_w2, new_h2])
fig.canvas.draw()
pos2 = ax2.get_position()

cb_ax1 = fig.add_axes([pos1.x1 + 0.006, pos1.y0, 0.010, pos1.y1 - pos1.y0])
cb1    = fig.colorbar(p_bathy, cax=cb_ax1, orientation='vertical')
cb1.set_ticks(np.array([0, -100, -500, -1000, -1500, -2000]))
cb1.set_label('Depth (m)', fontsize=axfont)
cb1.ax.tick_params(labelsize=axfont - 2)

# Move the right map directly next to the left map's colorbar. Cartopy's
# aspect-locked GeoAxes ignore GridSpec's wspace almost entirely -- each
# panel is shrunk/centered within its cell to preserve the map's lat/lon
# aspect ratio, so wspace alone barely changes the actual on-screen gap.
# An explicit reposition after layout is the only way to close it.
fig.canvas.draw()
cb1_pos = cb_ax1.get_position()
# clears the colorbar's own tick-label text, which extends ~0.041
# fig-fraction beyond cb_ax1's right edge (measured directly, longest tick
# label is '-2000') -- 0.004 left almost no margin, 0.05 still felt tight.
MAP_GAP = 0.06
pos2 = [cb1_pos.x1 + MAP_GAP, pos2.y0, pos2.width, pos2.height]
ax2.set_position(pos2)
fig.canvas.draw()
pos2 = ax2.get_position()

cb_ax2 = fig.add_axes([pos2.x1 + 0.006, pos2.y0, 0.010, pos2.y1 - pos2.y0])
cb2    = fig.colorbar(p_wave, cax=cb_ax2, orientation='vertical')
cb2.set_label('Wave amplitude (m)', fontsize=axfont)
cb2.ax.tick_params(labelsize=axfont - 2)

# Align time series width to ax1.x0 → ax2.x1; apply to both axes (twinx has its own position)
new_pos = [pos1.x0, pos_ts.y0, pos2.x1 - pos1.x0, pos_ts.height]
ax_ts.set_position(new_pos)
ax_ts2.set_position(new_pos)

fig.savefig('./figs/grid_combined.png', bbox_inches='tight', dpi=600)
print('saved ./figs/grid_combined.png')
