# snapshots of surface and cross section of w + tracer
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
#sce = 'tideswec'
#hisfolder1 = '/data/project3/minnaho/swel/tides/mc60/wec/his/mc60_his.*.nc'
#bgcfolder1 = '/data/project3/minnaho/swel/tides/mc60/wec/bgc/mc60_bgc.*.nc'

# tides no wec
#sce = 'tidesnowec'
#hisfolder1 = '/data/project3/minnaho/swel/tides/mc60/nowec/output/his/mc60_his.*.nc'
#bgcfolder1 = '/data/project3/minnaho/swel/tides/mc60/nowec/output/bgc/mc60_bgc.*.nc'

# no tides no wec
sce = 'notidesnowec'
hisfolder1 = '/data/project3/minnaho/swel/notides/mc60/nowec/output/his/mc60_his.*.nc'
bgcfolder1 = '/data/project3/minnaho/swel/notides/mc60/nowec/output/bgc/mc60_bgc.*.nc'

hisfiles = list(sorted(glob.glob(hisfolder1)))
bgcfiles = list(sorted(glob.glob(bgcfolder1)))

savepath = './figs/snapshots/'

grdnc = Dataset(grd,'r')
lat_nc = np.array(grdnc.variables['lat_rho'])
lon_nc = np.array(grdnc.variables['lon_rho'])-360
eta_slice = 477 # slice that captures inside the bay to outside

# do this so land is masked in white
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc==0] = np.nan

lon_slice = lon_nc[eta_slice,:] # put in degrees east

figw = 18
figh = 15

c_map1 = cmocean.cm.balance
c_map1.set_bad(color='w')
c_map0 = cmocean.cm.matter
c_map0.set_bad(color='w')

axfont = 20

vmin1 = -0.01
vmax1 = 0.01

vmin2 = 0
vmax2 = 40

for hf in range(len(hisfiles)):
    print(hisfiles[hf])
    hisnc = Dataset(hisfiles[hf],'r')
    bgcnc = Dataset(bgcfiles[hf],'r')
    his_time = np.array(hisnc.variables['ocean_time'])

    for t_i in range(his_time.shape[0]):
        zeta = np.squeeze(hisnc.variables['zeta'][t_i,:,:])
        zr0 = depths.get_zr_zeta(hisnc,grdnc,zeta)[:,eta_slice,:]

        tracer1nc = np.squeeze(hisnc.variables['w'][t_i])*masknc
        #tracer1mask = tracer1nc[-1,:,:] # get surface
        tracer1mask = np.nanmean(tracer1nc,axis=0) # vertical mean 
        tracer1 = tracer1nc[:,eta_slice,:]
    
        tracer2nc = np.squeeze(bgcnc.variables['NO3'][t_i])*masknc
        tracer2mask = tracer2nc[-1,:,:] # get surface
        #tracer2mask = np.nanmean(tracer2nc,axis=0) # vertical mean 
        tracer2 = tracer2nc[:,eta_slice,:]

        #tracer0mask[tracer0mask<0] = 0
        #tracer0mask = np.log10(tracer0mask) # log makes 0 inf
        #tracer0mask[np.isinf(tracer0mask)]=-38 # put inf back to small value

        #tracer2[tracer2<0] = 0
        #tracer2 = np.log10(tracer2)
        #tracer2[np.isinf(tracer2)]=-38 # put inf back to small value

        dt0 = num2date(his_time,'seconds since 1995-01-01')[t_i]


        fig,ax = plt.subplots(2,2,sharex=True,figsize=[figw,figh])
        
        p_plot1 = ax.flat[0].pcolormesh(lon_nc,lat_nc,tracer1mask,cmap=c_map1,vmin=vmin1,vmax=vmax1)
        p_plot0 = ax.flat[1].pcolormesh(lon_nc,lat_nc,tracer2mask,cmap=c_map0,vmin=vmin2,vmax=vmax2)
        
        #ax.flat[0].set_title(str(dt0.year)+'-'+'%02d'%dt0.month+'-'+'%02d'%dt0.day+' '+'%02d'%dt0.hour+':'+'%02d'%dt0.minute,fontsize=axfont)
        
        ax.flat[0].plot(lon_slice,lat_nc[eta_slice],color='k')
        ax.flat[1].plot(lon_slice,lat_nc[eta_slice],color='k')
        
        ax.flat[0].set_ylim([36.47,37])
        ax.flat[0].set_xlim([-122.2,-121.8])
        ax.flat[1].set_ylim([36.47,37])
        ax.flat[1].set_xlim([-122.2,-121.8])
        
        p_plot2  = ax.flat[2].pcolormesh(lon_slice,zr0,tracer1,cmap=c_map1,vmin=vmin1,vmax=vmax1)
        p_plot21 = ax.flat[3].pcolormesh(lon_slice,zr0,tracer2,cmap=c_map0,vmin=vmin2,vmax=vmax2)
        ax.flat[2].set_ylim([-1800,0])
        ax.flat[2].set_xlim([-122.2,-121.8])
        ax.flat[3].set_ylim([-1800,0])
        ax.flat[3].set_xlim([-122.2,-121.8])
        
        ax.flat[0].tick_params(axis='both',which='major',labelsize=axfont)
        ax.flat[1].tick_params(axis='both',which='major',labelsize=axfont)
        ax.flat[2].tick_params(axis='both',which='major',labelsize=axfont)
        ax.flat[3].tick_params(axis='both',which='major',labelsize=axfont)
        
        ax.flat[1].set_yticklabels([])
        ax.flat[3].set_yticklabels([])
        
        ax.flat[0].set_ylabel('Latitude',fontsize=axfont)
        ax.flat[2].set_ylabel('Depth (m)',fontsize=axfont)
        ax.flat[2].set_xlabel('Longitude',fontsize=axfont)
        ax.flat[3].set_xlabel('Longitude',fontsize=axfont)
        
        # put this before the colorbar to prevent cb axis issues!
        #fig.tight_layout()
        plt.subplots_adjust(hspace=0.03)
        
        ax1pos = ax.flat[1].get_position()
        ax3pos = ax.flat[3].get_position()
        ax1right = ax1pos.x0+.017
        ax3right = ax3pos.x0+.017
        
        newax1 = [ax1right,ax1pos.y0,ax1pos.width,ax1pos.height]
        ax.flat[1].set_position(newax1)
        newax3 = [ax3right,ax3pos.y0,ax3pos.width,ax3pos.height]
        ax.flat[3].set_position(newax3)
        
        
        p0 = ax.flat[0].get_position().get_points().flatten()
        p1 = ax.flat[2].get_position().get_points().flatten()
        p2 = ax.flat[1].get_position().get_points().flatten()
        p3 = ax.flat[3].get_position().get_points().flatten()
        cb_ax0 = fig.add_axes([p1[2]+.005,p1[1],.01,p0[3]-p1[1]])
        cb_ax1 = fig.add_axes([p3[2]+.005,p3[1],.01,p2[3]-p3[1]])
        cb0 = fig.colorbar(p_plot1,cax=cb_ax0,orientation='vertical')
        cb1 = fig.colorbar(p_plot0,cax=cb_ax1,orientation='vertical')
        cb0.set_label(r'w [m s$^{-1}$]',fontsize=axfont)
        cb1.set_label(r'NO$_{3}$ [mmol m$^{-3}$]',fontsize=axfont)
        cb0.ax.tick_params(axis='both',which='major',labelsize=axfont)
        cb1.ax.tick_params(axis='both',which='major',labelsize=axfont)
        
        fig.text(0.52,0.895,str(dt0.year)+'-'+'%02d'%dt0.month+'-'+'%02d'%dt0.day+' '+'%02d'%dt0.hour+':'+'%02d'%dt0.minute+' UTC',ha='center',va='center',fontsize=axfont)
        

        plt.savefig('./figs/snapshots/map_cs_'+sce+'-'+str(dt0.year)+'-'+'%02d'%dt0.month+'-'+'%02d'%dt0.day+'-'+'%02d'%dt0.hour+'.png',bbox_inches='tight')
        plt.close()
