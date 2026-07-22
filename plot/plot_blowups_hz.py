import sys
import os
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
from netCDF4 import Dataset,num2date
import glob as glob
import ROMS_depths as depths
import pyfuncs as pf
import matplotlib
matplotlib.use('Agg')
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
#plt.ion()

sfnc = Dataset('sfshelf60_grd.nc','r')
mcnc = Dataset('mc60_grd.nc','r')
smodenc = Dataset('smode200_grd.nc','r')

sf_h = np.array(sfnc.variables['h'])
sf_lon = np.array(sfnc.variables['lon_rho'])
sf_lat = np.array(sfnc.variables['lat_rho'])

mc_h = np.array(mcnc.variables['h'])
mc_lon = np.array(mcnc.variables['lon_rho'])
mc_lat = np.array(mcnc.variables['lat_rho'])
mc_mask = np.array(mcnc.variables['mask_rho'])
mc_r = np.array(mcnc.variables['river_flux'])
mc_rj, mc_ri = np.where(mc_r>0)

smode_h = np.array(smodenc.variables['h'])
smode_lon = np.array(smodenc.variables['lon_rho'])
smode_lat = np.array(smodenc.variables['lat_rho'])

coast_10m = cpf.NaturalEarthFeature('physical','coastline','10m')

# river list
riv_lat = [
36.28136294992721,
36.53638148010817,
36.75166011624930,
36.80678097419682,
36.84562369424969,
36.96866405277619,
36.97120764105607,
36.96345181932923,
37.26663740861312,
37.26663740861312,
37.32098341191214,
37.47543615109713,
37.46514537006239,
37.46514537006239,
37.465145370062395,
37.5629189882314, 
37.66939995851737,
38.14558098205704,
38.01274174736791,
38.05548931641736,
38.27166996723769,
38.17360674083887,
38.10116543333019,
37.94258975968380,
37.85944662236578,
38.09100090822576,
38.09100090822576,
38.45005315183953,
38.45005315183953,
38.76842712687647,
38.76842712687647,
38.95521108034522]

riv_lon = [
-121.85962289437829,
-121.92809386446135,
-121.80447014532206,
-121.78875045035736,
-121.80744090234150,
-121.90685050703406,
-121.95189373160380,
-122.01315634322728,
-122.41200564424240,
-122.41200564424240,
-122.40342988581470,
-122.44935282864355,
-122.02153266586895,
-122.02153266586895,
-122.02153266586895,
-122.13065973280075,
-122.16449183593014,
-121.68866743047246,
-121.65933890812958,
-121.68739564164085,
-122.28674917542557,
-122.41562863842931,
-122.48639319974718,
-122.50680819177369,
-122.57808080729819,
-122.83180204115443,
-122.83180204115443,
-123.13031249059986,
-123.13031249059986,
-123.53467539654342,
-123.53467539654342,
-123.73347439169544]


#fig,ax = plt.subplots(1,1,subplot_kw=dict(projection=ccrs.PlateCarree()))
#p_plot = ax.pcolormesh(sf_lon,sf_lat,sf_h,transform=ccrs.PlateCarree(),cmap=cmocean.cm.deep,vmin=0,vmax=200)
#ax.add_feature(coast_10m,facecolor='None',edgecolor='k')
#gl = ax.gridlines(draw_labels=True,linestyle='--')
#gl.xlabels_top = False
#gl.ylabels_right = False
#cb = fig.colorbar(p_plot)
#cb.set_label('depth (m)',fontsize=16)

blow_nowec_i = [226,228,641,561,690]
blow_nowec_j = [54,38,499,390,563]

blow_wec_i = [670,669]
blow_wec_j = [489,490]

blow_nowec_i_hard = [281]
blow_nowec_j_hard = [116]

taper10i = [670]
taper10j = [489]


mc_lon_small = mc_lon[::2, ::2]
mc_lat_small = mc_lat[::2, ::2]
mc_h_small   = mc_h[::2, ::2]

fig,ax = plt.subplots(1,1,subplot_kw=dict(projection=ccrs.PlateCarree()))
p_plot = ax.pcolormesh(mc_lon,mc_lat,mc_h,transform=ccrs.PlateCarree(),cmap=cmocean.cm.deep,vmin=0,vmax=200,rasterized=True)
#ax.add_feature(coast_10m,facecolor='None',edgecolor='k')
#ax.contour(mc_lon,mc_lat,mc_h,[2])
ax.contour(mc_lon,mc_lat,mc_mask,[0], rasterized=True)
ax.contour(mc_lon,mc_lat,mc_h,[30], rasterized=True)
ax.contour(mc_lon,mc_lat,mc_h,[10], rasterized=True)
ax.scatter(mc_lon[blow_nowec_j,blow_nowec_i],mc_lat[blow_nowec_j,blow_nowec_i],label='no WEC',color='pink')
ax.scatter(mc_lon[blow_wec_j,blow_wec_i],mc_lat[blow_wec_j,blow_wec_i],label='WEC',color='orange')
ax.scatter(mc_lon[blow_nowec_j_hard,blow_nowec_i_hard],mc_lat[blow_nowec_j_hard,blow_nowec_i_hard],label='no WEC hard',color='maroon')
ax.scatter(mc_lon[mc_rj,mc_ri],mc_lat[mc_rj,mc_ri],label='rivers',color='blue')
ax.scatter(mc_lon[taper10j,taper10i],mc_lat[taper10j,taper10i],label='taper10',color='brown')
ax.set_extent([-121.85,-121.77,36.79,36.87])
gl = ax.gridlines(draw_labels=True,linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
cb = fig.colorbar(p_plot)
cb.set_label('depth (m)',fontsize=16)
ax.legend()
#fig.savefig('testhz.png',bbox_inches='tight')
fig.savefig('/tmp/testhz.png',dpi=600)

#fig,ax = plt.subplots(1,1,subplot_kw=dict(projection=ccrs.PlateCarree()))
#p_plot = ax.pcolormesh(mc_lon,mc_lat,mc_h,transform=ccrs.PlateCarree(),cmap='rainbow',vmin=0,vmax=100)
##ax.add_feature(coast_10m,facecolor='None',edgecolor='k')
#ax.contour(mc_lon,mc_lat,mc_h,[2])
#ax.contour(mc_lon,mc_lat,mc_h,[30])
#ax.scatter(mc_lon[blow_wec_j,blow_wec_i],mc_lat[blow_wec_j,blow_wec_i],label='WEC',color='orange')
#gl = ax.gridlines(draw_labels=True,linestyle='--')
#gl.xlabels_top = False
#gl.ylabels_right = False
#cb = fig.colorbar(p_plot)
#cb.set_label('depth (m)',fontsize=16)
#
#
#fig,ax = plt.subplots(1,1,subplot_kw=dict(projection=ccrs.PlateCarree()))
#p_plot = ax.pcolormesh(smode_lon,smode_lat,smode_h,transform=ccrs.PlateCarree(),cmap=cmocean.cm.deep,vmin=0,vmax=1500)
#ax.add_feature(coast_10m,facecolor='None',edgecolor='k')
#p_scat = ax.scatter(riv_lon,riv_lat)
#gl = ax.gridlines(draw_labels=True,linestyle='--')
#gl.xlabels_top = False
#gl.ylabels_right = False
#cb = fig.colorbar(p_plot)
#cb.set_label('depth (m)',fontsize=16)

