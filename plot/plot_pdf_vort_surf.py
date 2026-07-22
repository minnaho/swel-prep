import sys
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# load npz 
# wecvort.files to see variables
wecvort = np.load('../postprocessing/surfvort_tideswec.npz')
nowecvort = np.load('../postprocessing/surfvort_tidesnowec.npz')
notidesnowecvort = np.load('../postprocessing/surfvort_notidesnowec.npz')

wecvort_mask = wecvort['vort_mask']
wecvort_cmask = wecvort['vort_cmask']

nowecvort_mask = nowecvort['vort_mask']
nowecvort_cmask = nowecvort['vort_cmask']

notidesnowecvort_mask = notidesnowecvort['vort_mask']
notidesnowecvort_cmask = notidesnowecvort['vort_cmask']

print('bin stats')
# stats
nbins = 500
n_wec,bins_wec,patch_wec = plt.hist(wecvort_mask.flatten(),nbins)
n_nowec,bins_nowec,patch_nowec = plt.hist(nowecvort_mask.flatten(),nbins)
n_notides,bins_notides,patch_notides = plt.hist(notidesnowecvort_mask.flatten(),nbins)

c_n_wec,c_bins_wec,c_patch_wec = plt.hist(wecvort_cmask.flatten(),nbins)
c_n_nowec,c_bins_nowec,c_patch_nowec = plt.hist(nowecvort_cmask.flatten(),nbins)
c_n_notides,c_bins_notides,c_patch_notides = plt.hist(notidesnowecvort_cmask.flatten(),nbins)

# non nan values
flt_wec = np.where(~np.isnan(wecvort_mask.flatten()))[0].shape[0]
flt_nowec = np.where(~np.isnan(nowecvort_mask.flatten()))[0].shape[0]
flt_notides = np.where(~np.isnan(notidesnowecvort_mask.flatten()))[0].shape[0]

c_flt_wec = np.where(~np.isnan(wecvort_cmask.flatten()))[0].shape[0]
c_flt_nowec = np.where(~np.isnan(nowecvort_cmask.flatten()))[0].shape[0]
c_flt_notides = np.where(~np.isnan(notidesnowecvort_cmask.flatten()))[0].shape[0]

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
ax.set_xlim([-20,20])

ax.legend(fontsize=axisfont)
ax.set_xlabel(r'$\zeta/f$',fontsize=axisfont)
ax.set_ylabel('PDF',fontsize=axisfont)
ax.tick_params(axis='both',which='major',labelsize=axisfont)
plt.savefig('./figs/pdf_vort_surf.png',bbox_inches='tight')

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
ax.set_xlim([-20,20])

ax.legend(fontsize=axisfont)
ax.set_xlabel(r'$\zeta/f$',fontsize=axisfont)
ax.set_ylabel('PDF',fontsize=axisfont)
ax.tick_params(axis='both',which='major',labelsize=axisfont)
plt.savefig('./figs/pdf_vort_surf_coastal.png',bbox_inches='tight')
