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
smodenc = Dataset('/data/project3/minnaho/project9copy/swel/smode200_grd.nc','r')
#nepacnc = Dataset('/data/project9/minnaho/swel/nepac4km_grd.nc','r')
uswcnc = Dataset('/data/project3/minnaho/project9copy/swel/nepac2km_grd.nc','r')
westcnc = Dataset('/data/project3/minnaho/project9copy/swel/westc600_grd.nc','r')
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

smode_h = np.array(smodenc.variables['h'])
smode_lon = np.array(smodenc.variables['lon_rho'])-360
smode_lat = np.array(smodenc.variables['lat_rho'])
smode_mask = np.array(smodenc.variables['mask_rho'])

#nepac_h = np.array(nepacnc.variables['h'])
#nepac_lon = np.array(nepacnc.variables['lon_rho'])
#nepac_lat = np.array(nepacnc.variables['lat_rho'])
#nepac_mask = np.array(nepacnc.variables['mask_rho'])

uswc_h = np.array(uswcnc.variables['h'])
uswc_lon = np.array(uswcnc.variables['lon_rho'])-360
uswc_lat = np.array(uswcnc.variables['lat_rho'])
uswc_mask = np.array(uswcnc.variables['mask_rho'])

westc_h = np.array(westcnc.variables['h'])
westc_lon = np.array(westcnc.variables['lon_rho'])-360
westc_lat = np.array(westcnc.variables['lat_rho'])
westc_mask = np.array(westcnc.variables['mask_rho'])

# plot
axfont = 16
figw = 23
figh = 8

fig,ax = plt.subplots(1,2,figsize=[figw,figh],subplot_kw=dict(projection=ccrs.PlateCarree()))
#p_plot0 = ax.flat[0].pcolormesh(uswc_lon,uswc_lat,-1*uswc_h/1000,transform=ccrs.PlateCarree(),cmap=cmocean.cm.deep_r,vmin=-7,vmax=0)
#ax.flat[0].contour(uswc_lon,uswc_lat,uswc_mask,colors='k',transform=ccrs.PlateCarree())
p_plot0 = ax.flat[0].pcolormesh(westc_lon,westc_lat,-1*westc_h/1000,transform=ccrs.PlateCarree(),cmap=cmocean.cm.deep_r,vmin=-6,vmax=0)
ax.flat[0].contour(westc_lon,westc_lat,westc_mask,colors='k',transform=ccrs.PlateCarree())

#ax.flat[0].plot(uswc_lon[:,0],uswc_lat[:,0],color='blue',linewidth=3,transform=ccrs.PlateCarree(),label='dx = 2 km')
#ax.flat[0].plot(uswc_lon[:,-1],uswc_lat[:,-1],color='blue',linewidth=3,transform=ccrs.PlateCarree())
#ax.flat[0].plot(uswc_lon[0,:],uswc_lat[0,:],color='blue',linewidth=3,transform=ccrs.PlateCarree())
#ax.flat[0].plot(uswc_lon[-1,:],uswc_lat[-1,:],color='blue',linewidth=3,transform=ccrs.PlateCarree())

ax.flat[0].plot(westc_lon[:,0],westc_lat[:,0],color='gray',linewidth=3,transform=ccrs.PlateCarree(),label='dx = 600 m')
ax.flat[0].plot(westc_lon[:,-1],westc_lat[:,-1],color='gray',linewidth=3,transform=ccrs.PlateCarree())
ax.flat[0].plot(westc_lon[0,:],westc_lat[0,:],color='gray',linewidth=3,transform=ccrs.PlateCarree())
ax.flat[0].plot(westc_lon[-1,:],westc_lat[-1,:],color='gray',linewidth=3,transform=ccrs.PlateCarree())

ax.flat[0].plot(smode_lon[:,0],smode_lat[:,0],color='purple',linewidth=3,transform=ccrs.PlateCarree(),label='dx = 200 m')
ax.flat[0].plot(smode_lon[:,-1],smode_lat[:,-1],color='purple',linewidth=3,transform=ccrs.PlateCarree())
ax.flat[0].plot(smode_lon[0,:],smode_lat[0,:],color='purple',linewidth=3,transform=ccrs.PlateCarree())
ax.flat[0].plot(smode_lon[-1,:],smode_lat[-1,:],color='purple',linewidth=3,transform=ccrs.PlateCarree())

ax.flat[0].plot(sf_lon[:,0],sf_lat[:,0],color='orange',linewidth=3,transform=ccrs.PlateCarree(),label='dx = 60 m')
ax.flat[0].plot(sf_lon[:,-1],sf_lat[:,-1],color='orange',linewidth=3,transform=ccrs.PlateCarree())
ax.flat[0].plot(sf_lon[0,:],sf_lat[0,:],color='orange',linewidth=3,transform=ccrs.PlateCarree())
ax.flat[0].plot(sf_lon[-1,:],sf_lat[-1,:],color='orange',linewidth=3,transform=ccrs.PlateCarree())

#p_plot1 = ax.flat[1].pcolormesh(sf_lon,sf_lat,-1*sf_h,transform=ccrs.PlateCarree(),cmap=cmocean.cm.topo,norm=mcolors.DivergingNorm(vmin=-2000,vmax=0,vcenter=-75))
p_plot1 = ax.flat[1].pcolormesh(sf_lon,sf_lat,-1*sf_h,transform=ccrs.PlateCarree(),cmap=cmocean.cm.deep_r,vmin=-2000,vmax=0)
#ax.flat[1].contour(sf_lon,sf_lat,coastal_mask,colors='orange',transform=ccrs.PlateCarree(),linewidth=1)
ax.flat[1].contour(sf_lon,sf_lat,sf_mask,colors='k',transform=ccrs.PlateCarree(),linewidth=1)

#ax.flat[1].plot(sf_lon[:,0],sf_lat[:,0],color='orange',linewidth=3,transform=ccrs.PlateCarree())
#ax.flat[1].plot(sf_lon[:,-1],sf_lat[:,-1],color='orange',linewidth=3,transform=ccrs.PlateCarree())
#ax.flat[1].plot(sf_lon[0,:],sf_lat[0,:],color='orange',linewidth=3,transform=ccrs.PlateCarree())
#ax.flat[1].plot(sf_lon[-1,:],sf_lat[-1,:],color='orange',linewidth=3,transform=ccrs.PlateCarree())

#ax.flat[1].scatter(rivlon,rivlat,marker='o',s=300,transform=ccrs.PlateCarree())
ax.flat[1].scatter(rivlon,rivlat,marker='o',c='lightblue',edgecolors='navy',s=60,transform=ccrs.PlateCarree(),label='Rivers',zorder=2)
ax.flat[1].scatter(piplon,piplat,marker='^',c='purple',edgecolors='None',s=80,transform=ccrs.PlateCarree(),label='Outfalls',zorder=3)

ax.flat[0].legend(loc='lower left',fontsize=axfont-2)

# Transect lines (geometry mirrors plot_cs_diag.py)
_LENGTH_XI = 250
_N_PTS     = 300
for _eta0, _xi0, _slope, _color, _label in [
    (271, 543, -0.8, 'red',  'Transect 0'),
    (832, 646,  0.6, 'blue', 'Transect 1'),
]:
    _xi  = np.linspace(_xi0,  _xi0  - _LENGTH_XI, _N_PTS)
    _eta = np.linspace(_eta0, _eta0 + _slope * (-_LENGTH_XI), _N_PTS)
    _crd = np.array([_eta, _xi])
    _tlon = map_coordinates(sf_lon, _crd, order=1, mode='nearest')
    _tlat = map_coordinates(sf_lat, _crd, order=1, mode='nearest')
    ax.flat[1].plot(_tlon, _tlat, '-', color=_color, linewidth=2,
                    transform=ccrs.PlateCarree(), label=_label)
    ax.flat[1].plot(_tlon[0], _tlat[0], 'o', color=_color, markersize=7,
                    transform=ccrs.PlateCarree())

ax.flat[1].legend(loc='lower right',fontsize=axfont-4)

#ax.flat[1].contour(sf_lon,sf_lat,sf_h,[100],colors='gray',linewidths=3)
contours = ax.flat[1].contour(sf_lon,sf_lat,sf_h,[50,100,500,1000,1500,2000],colors='black',linewidths=.5)
manual_pos = [(-122.15521671603335, 36.951516679194675), (-122.18968283307998, 36.89677462037397), (-122.14653235180069, 36.79974502146389), (-122.19880748200471, 36.702231681647184), (-122.16556182910833, 36.6486139904713), (-122.15735317111466, 36.632662098707534)]
clabels = ax.flat[1].clabel(contours,inline=False,fontsize=8,fmt='%d',manual=manual_pos)


# After you press Enter:
positions = [t.get_position() for t in clabels]

print(positions)

gl1 = ax.flat[1].gridlines(draw_labels=True,linewidth=0)
gl1.xlabels_top = False
gl1.ylabels_right = False
gl1.ylabel_style = {'size':axfont-4}
gl1.xlabel_style = {'size':axfont-4}

fig.tight_layout()

p0 = ax.flat[0].get_position().get_points().flatten()
cb_ax0 = fig.add_axes([p0[2]+.015,p0[1],.01,p0[3]-p0[1]])
cb0 = fig.colorbar(p_plot0,cax=cb_ax0,orientation='vertical')
cb0.set_label('Depth (km)',fontsize=axfont)
cb0.ax.tick_params(axis='both',which='major',labelsize=axfont-2)

p1 = ax.flat[1].get_position().get_points().flatten()
cb_ax1 = fig.add_axes([p1[2]+.015,p1[1],.01,p1[3]-p1[1]])
cb1 = fig.colorbar(p_plot1,cax=cb_ax1,orientation='vertical')
cbticks = np.array([0,-100,-500,-1000,-1500,-2000])
cb1.set_ticks(cbticks)
cb1.set_label('Depth (m)',fontsize=axfont)
cb1.ax.tick_params(axis='both',which='major',labelsize=axfont-2)

fig.savefig('./figs/grid.png',bbox_inches='tight')

