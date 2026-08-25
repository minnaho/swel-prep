# cross section of tracer
import os
import sys
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
from netCDF4 import Dataset,num2date
import glob as glob
import ROMS_depths as depths
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cmocean

#plt.ion()

grd = 'mc60_grd.nc'

# tides wec
sce = 'tideswec'
hisfolder1 = '/data/project3/minnaho/swel/tides/mc60/wec/his/mc60_his.*.nc'
bgcfolder1 = '/data/project3/minnaho/swel/tides/mc60/wec/bgc/mc60_bgc.*.nc'

# tides no wec
sce = 'tidesnowec'
hisfolder2 = '/data/project3/minnaho/swel/tides/mc60/nowec/output/his/mc60_his.*.nc'
bgcfolder2 = '/data/project3/minnaho/swel/tides/mc60/nowec/output/bgc/mc60_bgc.*.nc'

# no tides no wec
sce = 'notidesnowec'
hisfolder3 = '/data/project3/minnaho/swel/notides/mc60/nowec/his/mc60_his.*.nc'
bgcfolder3 = '/data/project3/minnaho/swel/notides/mc60/nowec/bgc/mc60_bgc.*.nc'

hisfiles1 = list(sorted(glob.glob(hisfolder1)))
hisfiles2 = list(sorted(glob.glob(hisfolder2)))
hisfiles3 = list(sorted(glob.glob(hisfolder3)))
bgcfiles = list(sorted(glob.glob(bgcfolder1)))

savepath = './figs/snapshots/'

grdnc = Dataset(grd,'r')
lat_nc = np.array(grdnc.variables['lat_rho'])
lon_nc = np.array(grdnc.variables['lon_rho'])-360
eta_slice = 477 # slice that captures inside the bay to outside

# do this so land is masked in white
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc==0] = np.nan
# mask to contour
maskc = np.array(grdnc.variables['mask_rho'])

lon_slice = lon_nc[eta_slice,:] # put in degrees east

figw = 18
figh = 7.5

c_map = cmocean.cm.matter
c_map.set_bad(color='w')

v_min = -7
v_max = -1

axfont = 16

for hf in range(len(hisfiles1)):
    print(str(hisfiles1[hf]))
    hisnc1 = Dataset(hisfiles1[hf],'r')
    hisnc2 = Dataset(hisfiles2[hf],'r')
    hisnc3 = Dataset(hisfiles3[hf],'r')
    his_time = np.array(hisnc1.variables['ocean_time'])
    for t_i in range(his_time.shape[0]):
        dt0 = num2date(his_time,'seconds since 1995-01-01')[t_i]

        tracer1mask = np.squeeze(hisnc1.variables['ptrace'])[t_i,-1]*masknc # get surface 
        tracer2mask = np.squeeze(hisnc2.variables['ptrace'])[t_i,-1]*masknc # get surface 
        tracer3mask = np.squeeze(hisnc3.variables['ptrace'])[t_i,-1]*masknc # get surface 
        tracer1mask[tracer1mask<0] = 0
        tracer2mask[tracer2mask<0] = 0
        tracer3mask[tracer3mask<0] = 0
        tracer1mask = np.log10(tracer1mask) # log makes 0 inf
        tracer2mask = np.log10(tracer2mask) # log makes 0 inf
        tracer3mask = np.log10(tracer3mask) # log makes 0 inf

        tracer1mask[np.isinf(tracer1mask)]=-38 # put inf back to small value
        tracer2mask[np.isinf(tracer2mask)]=-38 # put inf back to small value
        tracer3mask[np.isinf(tracer3mask)]=-38 # put inf back to small value

        fig,ax = plt.subplots(1,3,sharex=True,figsize=[figw,figh])
        p_plot1 = ax.flat[0].pcolormesh(lon_nc,lat_nc,tracer1mask,cmap=c_map,vmin=v_min,vmax=v_max)
        p_plot2 = ax.flat[1].pcolormesh(lon_nc,lat_nc,tracer2mask,cmap=c_map,vmin=v_min,vmax=v_max)
        p_plot3 = ax.flat[2].pcolormesh(lon_nc,lat_nc,tracer3mask,cmap=c_map,vmin=v_min,vmax=v_max)
        ax.flat[0].contour(lon_nc,lat_nc,maskc,colors='k',linewidths=0.5)
        ax.flat[1].contour(lon_nc,lat_nc,maskc,colors='k',linewidths=0.5)
        ax.flat[2].contour(lon_nc,lat_nc,maskc,colors='k',linewidths=0.5)

        ax.flat[0].set_title('WEC, tides',fontsize=axfont)
        ax.flat[1].set_title(str(dt0.year)+'-'+'%02d'%dt0.month+'-'+'%02d'%dt0.day+' '+'%02d'%dt0.hour+':'+'%02d'%dt0.minute+'\nno WEC, tides',fontsize=axfont)
        ax.flat[2].set_title('no WEC, no tides',fontsize=axfont)

        ax.flat[0].set_ylim([36.47,37.05]) 
        ax.flat[0].set_xlim([-122.4,-121.75])
        ax.flat[1].set_ylim([36.47,37.05]) 
        ax.flat[1].set_xlim([-122.4,-121.75])
        ax.flat[2].set_ylim([36.47,37.05]) 
        ax.flat[2].set_xlim([-122.4,-121.75])

        ax.flat[0].tick_params(axis='both',which='major',labelsize=axfont-2)
        ax.flat[1].tick_params(axis='both',which='major',labelsize=axfont-2)
        ax.flat[2].tick_params(axis='both',which='major',labelsize=axfont-2)
        ax.flat[1].set_yticklabels([])
        ax.flat[2].set_yticklabels([])
        ax.flat[0].set_xlabel('Longitude',fontsize=axfont)
        ax.flat[1].set_xlabel('Longitude',fontsize=axfont)
        ax.flat[2].set_xlabel('Longitude',fontsize=axfont)
        ax.flat[0].set_ylabel('Latitude',fontsize=axfont)

        # put this before the colorbar to prevent cb axis issues!
        fig.tight_layout()

        p2 = ax.flat[2].get_position().get_points().flatten()
        cb_ax1 = fig.add_axes([p2[2]+.015,p2[1],.01,p2[3]-p2[1]])
        cb1 = fig.colorbar(p_plot1,cax=cb_ax1,orientation='vertical')
        cb1.set_label(r'log$_{10}$(ptrace)',fontsize=axfont)
        cb1.ax.tick_params(axis='both',which='major',labelsize=axfont)

        plt.savefig(savepath+'surf_ptrace-'+str(dt0.year)+'-'+'%02d'%dt0.month+'-'+'%02d'%dt0.day+'-'+'%02d'%dt0.hour+'.png',bbox_inches='tight')
        plt.close()
