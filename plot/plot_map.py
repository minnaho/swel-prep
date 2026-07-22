import sys
import os
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
from netCDF4 import Dataset,num2date
import glob as glob
import ROMS_depths as depths
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import cmocean
import datetime as datetime
import calendar
import cartopy.crs as ccrs
import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import h5py
from scipy.ndimage import map_coordinates
plt.ion()

sfnc = Dataset('/data/project3/minnaho/project9copy/swel/mc60_grd.nc','r')
#coastal_mask = np.array(Dataset('coastal_mask.nc','r').variables['coastal_mask'])


sf_h = np.array(sfnc.variables['h'])
sf_lon = np.array(sfnc.variables['lon_rho'])-360
sf_lat = np.array(sfnc.variables['lat_rho'])
sf_mask = np.array(sfnc.variables['mask_rho'])
sf_riv = np.array(sfnc.variables['river_flux'])
sf_pip = np.array(sfnc.variables['pipe_flux'])

# rivers in dx = 60 m sfbay
rivy,rivx = np.where(sf_riv>0)
rivlon = sf_lon[rivy[1::3],rivx[1::3]]
rivlat = sf_lat[rivy[1::3],rivx[1::3]]

pipy,pipx = np.where(sf_pip>0)
piplon = sf_lon[pipy,pipx]
piplat = sf_lat[pipy,pipx]

# plot
axfont = 16
figw = 12
figh = 8

fig,ax = plt.subplots(1,1,figsize=[figw,figh],subplot_kw=dict(projection=ccrs.PlateCarree()))

#p_plot1 = ax.pcolormesh(sf_lon,sf_lat,-1*sf_h,transform=ccrs.PlateCarree(),cmap=cmocean.cm.topo,norm=mcolors.DivergingNorm(vmin=-2000,vmax=0,vcenter=-75))
p_plot1 = ax.pcolormesh(sf_lon,sf_lat,-1*sf_h,transform=ccrs.PlateCarree(),cmap=cmocean.cm.deep_r,vmin=-2000,vmax=0)
#ax.contour(sf_lon,sf_lat,coastal_mask,colors='orange',transform=ccrs.PlateCarree(),linewidth=1)
ax.contour(sf_lon,sf_lat,sf_mask,colors='k',transform=ccrs.PlateCarree(),linewidth=1)

#ax.scatter(rivlon,rivlat,marker='o',s=300,transform=ccrs.PlateCarree())
ax.scatter(rivlon,rivlat,marker='o',c='lightblue',edgecolors='navy',s=60,transform=ccrs.PlateCarree(),label='Rivers',zorder=2)
ax.scatter(piplon,piplat,marker='^',c='purple',edgecolors='None',s=80,transform=ccrs.PlateCarree(),label='Outfalls',zorder=3)

# Transect lines (geometry mirrors plot_cs_diag.py)
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
    ax.plot(_tlon, _tlat, '-', color=_color, linewidth=2,
            transform=ccrs.PlateCarree(), label=_label)
    ax.plot(_tlon[0], _tlat[0], 'o', color=_color, markersize=7,
            transform=ccrs.PlateCarree())

ax.legend(loc='lower right',fontsize=axfont)

#ax.contour(sf_lon,sf_lat,sf_h,[100],colors='gray',linewidths=3)
contours = ax.contour(sf_lon,sf_lat,sf_h,[50,100,500,1000,1500,2000],colors='black',linewidths=.5)
manual_pos = [(-122.15521671603335, 36.951516679194675), (-122.18968283307998, 36.89677462037397), (-122.14653235180069, 36.79974502146389), (-122.19880748200471, 36.702231681647184), (-122.16556182910833, 36.6486139904713), (-122.15735317111466, 36.632662098707534)]
clabels = ax.clabel(contours,inline=False,fontsize=8,fmt='%d',manual=manual_pos)


# After you press Enter:
positions = [t.get_position() for t in clabels]

print(positions)

gl1 = ax.gridlines(draw_labels=True,linestyle='--')
gl1.xlabels_top = False
gl1.ylabels_right = False
gl1.ylabel_style = {'size':axfont-1}
gl1.xlabel_style = {'size':axfont-1}

fig.tight_layout()

p1 = ax.get_position().get_points().flatten()
cb_ax1 = fig.add_axes([p1[2]+.015,p1[1],.01,p1[3]-p1[1]])
cb1 = fig.colorbar(p_plot1,cax=cb_ax1,orientation='vertical')
cbticks = np.array([0,-100,-500,-1000,-1500,-2000])
cb1.set_ticks(cbticks)
cb1.set_label('Depth (m)',fontsize=axfont)
cb1.ax.tick_params(axis='both',which='major',labelsize=axfont-2)

fig.savefig('./figs/grid.png',bbox_inches='tight',dpi=600)
