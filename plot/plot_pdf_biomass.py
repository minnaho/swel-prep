import sys
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

# biomass netCDFs
wecnc = '/data/project3/minnaho/swel/tides/mc60/wec/extract_biomass/mc60_wec_7dayavg.nc'
nowecnc = '/data/project3/minnaho/swel/tides/mc60/nowec/output/extract_biomass/mc60_nowec_7dayavg.nc'
notidesnc = '/data/project3/minnaho/swel/notides/mc60/nowec/extract_biomass/mc60_notides_7dayavg.nc'

wec_ds = Dataset(wecnc, 'r')
nowec_ds = Dataset(nowecnc, 'r')
notides_ds = Dataset(notidesnc, 'r')

print('summing phyto components')
#wec_phyto = wec_ds['SPC'][-1,:,:,:] + wec_ds['DIATC'][-1,:,:,:] + wec_ds['DIAZC'][-1,:,:,:]
#nowec_phyto = nowec_ds['SPC'][-1,:,:,:] + nowec_ds['DIATC'][-1,:,:,:] + nowec_ds['DIAZC'][-1,:,:,:]
#notides_phyto = notides_ds['SPC'][-1,:,:,:] + notides_ds['DIATC'][-1,:,:,:] + notides_ds['DIAZC'][-1,:,:,:]
wec_phyto = wec_ds['SPC'][:] + wec_ds['DIATC'][:] + wec_ds['DIAZC'][:]
nowec_phyto = nowec_ds['SPC'][:] + nowec_ds['DIATC'][:] + nowec_ds['DIAZC'][:]
notides_phyto = notides_ds['SPC'][:] + notides_ds['DIATC'][:] + notides_ds['DIAZC'][:]

wec_phyto[wec_phyto<0] = 0
nowec_phyto[nowec_phyto<0] = 0
notides_phyto[notides_phyto<0] = 0

#wec_zoo = wec_ds.variables['ZOOC'][:]
#nowec_zoo = nowec_ds.variables['ZOOC'][:]
#notides_zoo = notides_ds.variables['ZOOC'][:]

grd = '/data/project3/minnaho/project9copy/swel/mc60_grd.nc'

grdnc = Dataset(grd,'r')
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc==0] = np.nan

# multiply by mask to make all land values nan
wec_phyto = wec_phyto*masknc
nowec_phyto = nowec_phyto*masknc
notides_phyto = notides_phyto*masknc

coastal_mask = np.array(Dataset('coastal_mask.nc','r').variables['coastal_mask'])
coastal_mask[coastal_mask==0] = np.nan

c_wec_phyto = wec_phyto*coastal_mask
c_nowec_phyto = nowec_phyto*coastal_mask
c_notides_phyto = notides_phyto*coastal_mask

print('bin stats')
# stats
nbins = 500
n_wec,bins_wec,patch_wec = plt.hist(wec_phyto.flatten(),nbins)
n_nowec,bins_nowec,patch_nowec = plt.hist(nowec_phyto.flatten(),nbins)
n_notides,bins_notides,patch_notides = plt.hist(notides_phyto.flatten(),nbins)

c_n_wec,c_bins_wec,c_patch_wec = plt.hist(c_wec_phyto.flatten(),nbins)
c_n_nowec,c_bins_nowec,c_patch_nowec = plt.hist(c_nowec_phyto.flatten(),nbins)
c_n_notides,c_bins_notides,c_patch_notides = plt.hist(c_notides_phyto.flatten(),nbins)

# non nan values
flt_wec = np.where(~np.isnan(wec_phyto.flatten()))[0].shape[0]
flt_nowec = np.where(~np.isnan(nowec_phyto.flatten()))[0].shape[0]
flt_notides = np.where(~np.isnan(notides_phyto.flatten()))[0].shape[0]

c_flt_wec = np.where(~np.isnan(c_wec_phyto.flatten()))[0].shape[0]
c_flt_nowec = np.where(~np.isnan(c_nowec_phyto.flatten()))[0].shape[0]
c_flt_notides = np.where(~np.isnan(c_notides_phyto.flatten()))[0].shape[0]

# make same size as bins
n_wec = np.append(n_wec,0)
n_nowec = np.append(n_nowec,0)
n_notides = np.append(n_notides,0)

c_n_wec = np.append(c_n_wec,0)
c_n_nowec = np.append(c_n_nowec,0)
c_n_notides = np.append(c_n_notides,0)

print('plotting')

# total domain
figw = 12
figh = 8
axisfont = 16

fig,ax = plt.subplots(1,1,figsize=[figw,figh])
ax.plot(bins_wec,n_wec/flt_wec,color='lightblue',linewidth=1.5,label='WEC')
ax.plot(bins_nowec,n_nowec/flt_nowec,color='navy',linestyle=':',linewidth=1.5,label='no WEC')
ax.plot(bins_notides,n_notides/flt_notides,color='orange',linestyle='--',linewidth=1.5,label='no WEC, no tides')

ax.set_yscale('log')
ax.set_ylim([1E-6,1])
ax.set_xlim([-1,150])

ax.legend(fontsize=axisfont)
ax.set_xlabel('phytoplankton biomass '+r'mmol m$^{-3}$',fontsize=axisfont)
ax.set_ylabel('PDF',fontsize=axisfont)
ax.tick_params(axis='both',which='major',labelsize=axisfont)
plt.savefig('./figs/pdf_biomass.png',bbox_inches='tight')

# coastal 10 km
figw = 12
figh = 8
axisfont = 16

fig,ax = plt.subplots(1,1,figsize=[figw,figh])
ax.plot(c_bins_wec,c_n_wec/c_flt_wec,color='lightblue',linewidth=1.5,label='WEC')
ax.plot(c_bins_nowec,c_n_nowec/c_flt_nowec,color='navy',linestyle=':',linewidth=1.5,label='no WEC')
ax.plot(c_bins_notides,c_n_notides/c_flt_notides,color='orange',linestyle='--',linewidth=1.5,label='no WEC, no tides')

ax.set_yscale('log')
ax.set_ylim([1E-6,1])
ax.set_xlim([-1,150])

ax.legend(fontsize=axisfont)
ax.set_xlabel('phytoplankton biomass '+r'mmol m$^{-3}$',fontsize=axisfont)
ax.set_ylabel('PDF',fontsize=axisfont)
ax.tick_params(axis='both',which='major',labelsize=axisfont)
plt.savefig('./figs/pdf_biomass_coastal.png',bbox_inches='tight')
