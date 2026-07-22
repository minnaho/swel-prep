import sys
import glob
import numpy as np
from netCDF4 import Dataset

sys.path.append('/data/project3/minnaho/global/')
import pyfuncs as pf

# ==========================================
# 1. Configuration & File Paths
# ==========================================
grd = '/data/project3/minnaho/project9copy/swel/mc60_grd.nc'

# Use glob to grab all relevant PV history files
tideswec_files = sorted(glob.glob('/data/project3/minnaho/swel/tides/mc60/wec/his/pv/mc60_his.2019*_pv.nc'))
tidesnowec_files = sorted(glob.glob('/data/project3/minnaho/swel/tides/mc60/nowec/output/his/pv/mc60_his.2019*_pv.nc'))
notidesnowec_files = sorted(glob.glob('/data/project3/minnaho/swel/notides/mc60/nowec/output/his/pv/mc60_his.2019*_pv.nc'))

# Output paths for the individual arrays
out_tideswec_path = './pdf8_pv_tideswec.npz'
out_tidesnowec_path = './pdf8_pv_tidesnowec.npz'
out_notidesnowec_path = './pdf8_pv_notidesnowec.npz'

# ==========================================
# 2. Load Grid & Mask
# ==========================================
print('Loading grid data...')
grdnc = Dataset(grd, 'r')
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc == 0] = np.nan
grdnc.close()

# ==========================================
# 3. Initialize PDF (Histogram) Variables
# ==========================================
# IMPORTANT: PV values are usually very small (e.g., 10^-8). 
# Adjust these min/max limits to fit your specific PV range.
bin_edges = np.linspace(-1e-8, 1e-8, 501) 

# ==========================================
# 4. Iteratively Process WEC Files
# ==========================================
print(f'Processing {len(tideswec_files)} WEC files...')
counts_wec = np.zeros(len(bin_edges) - 1)
total_flt_wec = 0

for f_idx, file in enumerate(tideswec_files):
    print(f'  -> WEC {f_idx+1}/{len(tideswec_files)}: {file}')
    
    # Read PV directly from the new files
    nc = Dataset(file, 'r')
    pv = np.array(nc.variables['pv'])
    nc.close()
    
    # Apply 2D mask to 3D/4D PV array (NumPy handles the broadcasting)
    masked_pv = pv * masknc 
    
    flat_pv = masked_pv.flatten()
    valid_pv = flat_pv[~np.isnan(flat_pv)]
    
    hist_counts, _ = np.histogram(valid_pv, bins=bin_edges)
    counts_wec += hist_counts
    total_flt_wec += valid_pv.shape[0]

# Normalize and Save WEC array
counts_wec_plot = np.append(counts_wec, 0)
pdf_wec = counts_wec_plot / total_flt_wec
np.savez(out_tideswec_path, bin_edges=bin_edges, pdf_wec=pdf_wec)
print(f'Saved WEC data to {out_tideswec_path}')

# ==========================================
# 5. Iteratively Process no-WEC Files
# ==========================================
#print(f'Processing {len(tidesnowec_files)} tides-no-WEC files...')
#counts_nowec = np.zeros(len(bin_edges) - 1)
#total_flt_nowec = 0
#
#for f_idx, file in enumerate(tidesnowec_files):
#    print(f'  -> tides-no-WEC {f_idx+1}/{len(tidesnowec_files)}: {file}')
#    
#    nc = Dataset(file, 'r')
#    pv = np.array(nc.variables['pv'])
#    nc.close()
#    
#    masked_pv = pv * masknc 
#    
#    flat_pv = masked_pv.flatten()
#    valid_pv = flat_pv[~np.isnan(flat_pv)]
#    
#    hist_counts, _ = np.histogram(valid_pv, bins=bin_edges)
#    counts_nowec += hist_counts
#    total_flt_nowec += valid_pv.shape[0]
#
## Normalize and Save no-WEC array
#counts_nowec_plot = np.append(counts_nowec, 0)
#pdf_nowec = counts_nowec_plot / total_flt_nowec
#np.savez(out_tidesnowec_path, bin_edges=bin_edges, pdf_nowec=pdf_nowec)
#print(f'Saved tides-no-WEC data to {out_tidesnowec_path}')
#
## ==========================================
## 6. Iteratively Process no tides no-WEC Files
## ==========================================
#print(f'Processing {len(notidesnowec_files)} no-tides-no-WEC files...')
#counts_notidesnowec = np.zeros(len(bin_edges) - 1)
#total_flt_notidesnowec = 0
#
#for f_idx, file in enumerate(notidesnowec_files):
#    print(f'  -> no-tides-no-WEC {f_idx+1}/{len(notidesnowec_files)}: {file}')
#    
#    nc = Dataset(file, 'r')
#    pv = np.array(nc.variables['pv'])
#    nc.close()
#    
#    masked_pv = pv * masknc 
#    
#    flat_pv = masked_pv.flatten()
#    valid_pv = flat_pv[~np.isnan(flat_pv)]
#    
#    hist_counts, _ = np.histogram(valid_pv, bins=bin_edges)
#    counts_notidesnowec += hist_counts
#    total_flt_notidesnowec += valid_pv.shape[0]
#
## Normalize and Save no-WEC array
#counts_notidesnowec_plot = np.append(counts_notidesnowec, 0)
#pdf_notidesnowec = counts_notidesnowec_plot / total_flt_notidesnowec
#np.savez(out_notidesnowec_path, bin_edges=bin_edges, pdf_notidesnowec=pdf_notidesnowec)
#print(f'Saved no-tides-no-WEC data to {out_notidesnowec_path}')

print('All calculations complete.')
