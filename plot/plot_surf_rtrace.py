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

# no tides, no WEC
sce = 'notidesnowec'
hisfolder1 = '/data/project3/minnaho/swel/notides/mc60/nowec/his/mc60_his.*.nc'

# no tides, 2.5x WEC
sce = 'ampwec'
hisfolder4 = '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything/mc60_his.*.nc'

# tides, no WEC
sce = 'tidesnowec'
hisfolder2 = '/data/project3/minnaho/swel/tides/mc60/nowec/output/his/mc60_his.*.nc'

# tides, 2.5x WEC
sce = 'tidesampwec'
hisfolder3 = '/data/project3/minnaho/swel/tides/mc60/ampwec/everything/mc60_his.*.nc'

hisfiles1 = list(sorted(glob.glob(hisfolder1)))
hisfiles2 = list(sorted(glob.glob(hisfolder2)))
hisfiles3 = list(sorted(glob.glob(hisfolder3)))
hisfiles4 = list(sorted(glob.glob(hisfolder4)))

savepath = './figs/snapshots/rtrace/'

grdnc = Dataset(grd,'r')
lat_nc = np.array(grdnc.variables['lat_rho'])
lon_nc = np.array(grdnc.variables['lon_rho'])-360
h_nc = np.array(grdnc.variables['h'])

# do this so land is masked in white
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc==0] = np.nan
# mask to contour
maskc = np.array(grdnc.variables['mask_rho'])

# Adjusted figure size for a 2x2 layout
figw = 12
figh = 15

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
    hisnc4 = Dataset(hisfiles4[hf],'r')
    his_time = np.array(hisnc1.variables['ocean_time'])
    
    for t_i in range(his_time.shape[0]):
        dt0 = num2date(his_time,'seconds since 1995-01-01')[t_i]
        time_str = f"{dt0.year}-{dt0.month:02d}-{dt0.day:02d} {dt0.hour:02d}:{dt0.minute:02d}"

        out_file = savepath+'surf_rtrace-'+str(dt0.year)+'-'+'%02d'%dt0.month+'-'+'%02d'%dt0.day+'-'+'%02d'%dt0.hour+'.png'
        if os.path.exists(out_file):
            print(f'  skipping {out_file}, already exists')
            continue

        tracer1mask = np.squeeze(hisnc1.variables['rtrace'])[t_i,-1]*masknc # get surface
        tracer2mask = np.squeeze(hisnc2.variables['rtrace'])[t_i,-1]*masknc # get surface 
        tracer3mask = np.squeeze(hisnc3.variables['rtrace'])[t_i,-1]*masknc # get surface 
        tracer4mask = np.squeeze(hisnc4.variables['rtrace'])[t_i,-1]*masknc # get surface 
        
        tracer1mask[tracer1mask<0] = 0
        tracer2mask[tracer2mask<0] = 0
        tracer3mask[tracer3mask<0] = 0
        tracer4mask[tracer4mask<0] = 0
        
        tracer1mask = np.log10(tracer1mask) # log makes 0 inf
        tracer2mask = np.log10(tracer2mask) # log makes 0 inf
        tracer3mask = np.log10(tracer3mask) # log makes 0 inf
        tracer4mask = np.log10(tracer4mask) # log makes 0 inf

        tracer1mask[np.isinf(tracer1mask)]=-38 # put inf back to small value
        tracer2mask[np.isinf(tracer2mask)]=-38 # put inf back to small value
        tracer3mask[np.isinf(tracer3mask)]=-38 # put inf back to small value
        tracer4mask[np.isinf(tracer4mask)]=-38 # put inf back to small value

        fig,ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=[figw,figh])
        
        # Add a super title that spans the entire figure
        fig.suptitle(time_str, fontsize=axfont+2, fontweight='bold')
        
        p_plot1 = ax.flat[0].pcolormesh(lon_nc,lat_nc,tracer1mask,cmap=c_map,vmin=v_min,vmax=v_max) # TL
        p_plot4 = ax.flat[1].pcolormesh(lon_nc,lat_nc,tracer4mask,cmap=c_map,vmin=v_min,vmax=v_max) # TR
        p_plot2 = ax.flat[2].pcolormesh(lon_nc,lat_nc,tracer2mask,cmap=c_map,vmin=v_min,vmax=v_max) # BL
        p_plot3 = ax.flat[3].pcolormesh(lon_nc,lat_nc,tracer3mask,cmap=c_map,vmin=v_min,vmax=v_max) # BR
        
        ax.flat[0].contour(lon_nc,lat_nc,maskc,colors='k',linewidths=1)
        ax.flat[1].contour(lon_nc,lat_nc,maskc,colors='k',linewidths=1)
        ax.flat[2].contour(lon_nc,lat_nc,maskc,colors='k',linewidths=1)
        ax.flat[3].contour(lon_nc,lat_nc,maskc,colors='k',linewidths=1)

        # Removed time_str from the first subplot title
        ax.flat[0].set_title('no tides, no WEC',fontsize=axfont)
        ax.flat[1].set_title('no tides, 2.5x WEC',fontsize=axfont)
        ax.flat[2].set_title('tides, no WEC',fontsize=axfont)
        ax.flat[3].set_title('tides, 2.5x WEC',fontsize=axfont)

        for i in range(4):
            ax.flat[i].set_ylim([36.47,37.05]) 
            ax.flat[i].set_xlim([-122.4,-121.75])
            ax.flat[i].tick_params(axis='both',which='major',labelsize=axfont-2)

        # X labels on bottom plots
        ax.flat[2].set_xlabel('Longitude',fontsize=axfont)
        ax.flat[3].set_xlabel('Longitude',fontsize=axfont)
        
        # Y labels on left plots
        ax.flat[0].set_ylabel('Latitude',fontsize=axfont)
        ax.flat[2].set_ylabel('Latitude',fontsize=axfont)

        # Adjust tight_layout so it leaves 5% space at the top for the suptitle
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Custom colorbar positioning spanning the right side
        p_top = ax.flat[1].get_position().get_points().flatten() # Top right plot
        p_bot = ax.flat[3].get_position().get_points().flatten() # Bottom right plot
        cb_ax1 = fig.add_axes([p_top[2]+.02, p_bot[1], .015, p_top[3]-p_bot[1]])
        
        cb1 = fig.colorbar(p_plot1,cax=cb_ax1,orientation='vertical')
        cb1.set_label(r'log$_{10}$(rtrace)',fontsize=axfont)
        cb1.ax.tick_params(axis='both',which='major',labelsize=axfont)

        plt.savefig(out_file,bbox_inches='tight')
        plt.close()
