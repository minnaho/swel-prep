import os
import sys
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
from netCDF4 import Dataset, num2date
import glob as glob
import ROMS_depths as depths
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cmocean

#plt.ion()

grd = 'mc60_grd.nc'

# 1. tides wec
hisfolder1 = '/data/project3/minnaho/swel/tides/mc60/wec/his/mc60_his.*.nc'
bgcfolder1 = '/data/project3/minnaho/swel/tides/mc60/wec/bgc/mc60_bgc.*.nc'

# 2. tides no wec
hisfolder2 = '/data/project3/minnaho/swel/tides/mc60/nowec/output/his/mc60_his.*.nc'
bgcfolder2 = '/data/project3/minnaho/swel/tides/mc60/nowec/output/bgc/mc60_bgc.*.nc'

# 3. no tides no wec
hisfolder3 = '/data/project3/minnaho/swel/notides/mc60/nowec/his/mc60_his.*.nc'
bgcfolder3 = '/data/project3/minnaho/swel/notides/mc60/nowec/bgc/mc60_bgc.*.nc'

# 4. no tides wec
hisfolder4 = '/data/project3/minnaho/swel/notides/mc60/wec/rerun/his/mc60_his.*.nc'
bgcfolder4 = '/data/project3/minnaho/swel/notides/mc60/wec/rerun/bgc/mc60_bgc.*.nc'

hisfiles1 = list(sorted(glob.glob(hisfolder1)))
bgcfiles1 = list(sorted(glob.glob(bgcfolder1)))

hisfiles2 = list(sorted(glob.glob(hisfolder2)))
bgcfiles2 = list(sorted(glob.glob(bgcfolder2)))

hisfiles3 = list(sorted(glob.glob(hisfolder3)))
bgcfiles3 = list(sorted(glob.glob(bgcfolder3)))

hisfiles4 = list(sorted(glob.glob(hisfolder4)))
bgcfiles4 = list(sorted(glob.glob(bgcfolder4)))

savepath = './figs/snapshots/cs_w_no3/'

grdnc = Dataset(grd,'r')
lat_nc = np.array(grdnc.variables['lat_rho'])
lon_nc = np.array(grdnc.variables['lon_rho'])-360
eta_slice = 477 # slice that captures inside the bay to outside

# do this so land is masked in white
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc==0] = np.nan

lon_slice = lon_nc[eta_slice,:] # put in degrees east

# Taller figure for 4 rows
figw = 14
figh = 18

c_map1 = cmocean.cm.balance
c_map1.set_bad(color='w')
c_map0 = cmocean.cm.matter
c_map0.set_bad(color='w')

axfont = 16

# limits for w
vmin1 = -0.02
vmax1 = 0.02

# limits for NO3
vmin2 = 0
vmax2 = 25

for hf in range(len(hisfiles1)):
    print(hisfiles1[hf])
    
    hisnc1 = Dataset(hisfiles1[hf],'r')
    bgcnc1 = Dataset(bgcfiles1[hf],'r')
    
    hisnc2 = Dataset(hisfiles2[hf],'r')
    bgcnc2 = Dataset(bgcfiles2[hf],'r')
    
    hisnc3 = Dataset(hisfiles3[hf],'r')
    bgcnc3 = Dataset(bgcfiles3[hf],'r')
    
    hisnc4 = Dataset(hisfiles4[hf],'r')
    bgcnc4 = Dataset(bgcfiles4[hf],'r')
    
    his_time = np.array(hisnc1.variables['ocean_time'])

    for t_i in range(his_time.shape[0]):
        dt0 = num2date(his_time,'seconds since 1995-01-01')[t_i]
        time_str = f"{dt0.year}-{dt0.month:02d}-{dt0.day:02d} {dt0.hour:02d}:{dt0.minute:02d} UTC"

        # --- Data Extraction ---
        # Scen 1: tides, WEC
        z_1 = np.squeeze(hisnc1.variables['zeta'][t_i,:,:])
        zr1 = depths.get_zr_zeta(hisnc1,grdnc,z_1)[:,eta_slice,:]
        w1 = (np.squeeze(hisnc1.variables['w'][t_i])*masknc)[:,eta_slice,:]
        no1 = (np.squeeze(bgcnc1.variables['NO3'][t_i])*masknc)[:,eta_slice,:]

        # Scen 2: tides, no WEC
        z_2 = np.squeeze(hisnc2.variables['zeta'][t_i,:,:])
        zr2 = depths.get_zr_zeta(hisnc2,grdnc,z_2)[:,eta_slice,:]
        w2 = (np.squeeze(hisnc2.variables['w'][t_i])*masknc)[:,eta_slice,:]
        no2 = (np.squeeze(bgcnc2.variables['NO3'][t_i])*masknc)[:,eta_slice,:]

        # Scen 3: no tides, no WEC
        z_3 = np.squeeze(hisnc3.variables['zeta'][t_i,:,:])
        zr3 = depths.get_zr_zeta(hisnc3,grdnc,z_3)[:,eta_slice,:]
        w3 = (np.squeeze(hisnc3.variables['w'][t_i])*masknc)[:,eta_slice,:]
        no3 = (np.squeeze(bgcnc3.variables['NO3'][t_i])*masknc)[:,eta_slice,:]

        # Scen 4: no tides, WEC
        z_4 = np.squeeze(hisnc4.variables['zeta'][t_i,:,:])
        zr4 = depths.get_zr_zeta(hisnc4,grdnc,z_4)[:,eta_slice,:]
        w4 = (np.squeeze(hisnc4.variables['w'][t_i])*masknc)[:,eta_slice,:]
        no4 = (np.squeeze(bgcnc4.variables['NO3'][t_i])*masknc)[:,eta_slice,:]

        # --- Plotting Setup ---
        fig, ax = plt.subplots(4, 2, sharex=True, sharey=True, figsize=[figw, figh])
        fig.suptitle(time_str, fontsize=axfont, fontweight='bold')
        
        # Row 0: tides, WEC (Scenario 1)
        pw_1 = ax[0,0].pcolormesh(lon_slice, zr1, w1, cmap=c_map1, vmin=vmin1, vmax=vmax1)
        pn_1 = ax[0,1].pcolormesh(lon_slice, zr1, no1, cmap=c_map0, vmin=vmin2, vmax=vmax2)
        ax[0,0].set_ylabel('tides, WEC\nDepth (m)', fontsize=axfont)
        
        # Row 1: no tides, WEC (Scenario 4)
        pw_4 = ax[1,0].pcolormesh(lon_slice, zr4, w4, cmap=c_map1, vmin=vmin1, vmax=vmax1)
        pn_4 = ax[1,1].pcolormesh(lon_slice, zr4, no4, cmap=c_map0, vmin=vmin2, vmax=vmax2)
        ax[1,0].set_ylabel('no tides, WEC\nDepth (m)', fontsize=axfont)
        
        # Row 2: tides, no WEC (Scenario 2)
        pw_2 = ax[2,0].pcolormesh(lon_slice, zr2, w2, cmap=c_map1, vmin=vmin1, vmax=vmax1)
        pn_2 = ax[2,1].pcolormesh(lon_slice, zr2, no2, cmap=c_map0, vmin=vmin2, vmax=vmax2)
        ax[2,0].set_ylabel('tides, no WEC\nDepth (m)', fontsize=axfont)

        # Row 3: no tides, no WEC (Scenario 3)
        pw_3 = ax[3,0].pcolormesh(lon_slice, zr3, w3, cmap=c_map1, vmin=vmin1, vmax=vmax1)
        pn_3 = ax[3,1].pcolormesh(lon_slice, zr3, no3, cmap=c_map0, vmin=vmin2, vmax=vmax2)
        ax[3,0].set_ylabel('no tides, no WEC\nDepth (m)', fontsize=axfont)

        # Column Titles
        ax[0,0].set_title('w [m s$^{-1}$]', fontsize=axfont+2)
        ax[0,1].set_title('NO$_3$ [mmol m$^{-3}$]', fontsize=axfont+2)

        # Axis Limits and formatting
        # For the plot x-axis
        x_ticks = np.linspace(-121.96, -121.8, 5) # Start, End, and how many ticks total
        for i in range(4):
            for j in range(2):
                ax[i,j].set_ylim([-300, 0])
                ax[i,j].set_xticks(x_ticks)
                ax[i,j].set_xlim([-121.96, -121.8])
                ax[i,j].tick_params(axis='both', which='major', labelsize=axfont-2)
                #ax[i,j].xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        # X-axis labels
        ax[3,0].set_xlabel('Longitude', fontsize=axfont)
        ax[3,1].set_xlabel('Longitude', fontsize=axfont)

        # Tight layout (leaves room at bottom for horizontal colorbars)
        fig.tight_layout(rect=[0, 0.08, 1, 0.96])

        # --- Colorbars ---
        # Position colorbars directly beneath their respective columns
        pos_w = ax[3,0].get_position().get_points().flatten() # Left column
        pos_n = ax[3,1].get_position().get_points().flatten() # Right column
        
        # [left, bottom, width, height]
        cb_ax_w = fig.add_axes([pos_w[0], pos_w[1]-0.06, pos_w[2]-pos_w[0], 0.015])
        cb_ax_n = fig.add_axes([pos_n[0], pos_n[1]-0.06, pos_n[2]-pos_n[0], 0.015])
        
        cb_w = fig.colorbar(pw_1, cax=cb_ax_w, orientation='horizontal')
        cb_w.set_label(r'w [m s$^{-1}$]', fontsize=axfont)
        w_ticks = np.linspace(vmin1, vmax1, 5) # [-.2, -.1, 0, .1, .2]
        cb_w.set_ticks(w_ticks)
        cb_w.ax.tick_params(axis='both', which='major', labelsize=axfont-2)

        cb_n = fig.colorbar(pn_1, cax=cb_ax_n, orientation='horizontal')
        cb_n.set_label(r'NO$_3$ [mmol m$^{-3}$]', fontsize=axfont)
        cb_n.ax.tick_params(axis='both', which='major', labelsize=axfont-2)
        #cb_n.ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        plt.savefig(savepath + 'cs_w_no3-' + str(dt0.year) + '-' + '%02d'%dt0.month + '-' + '%02d'%dt0.day + '-' + '%02d'%dt0.hour + '.png', bbox_inches='tight')
        plt.close()
