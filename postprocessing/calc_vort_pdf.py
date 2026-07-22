import os
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

# Full simulated period: 2019-04-18 through 2019-04-28 (the entire raw
# archive for each scenario -- no day-range restriction needed)
SCENARIOS = {
    'tideswec':     '/data/project3/minnaho/swel/tides/mc60/wec',
    'tidesnowec':   '/data/project3/minnaho/swel/tides/mc60/nowec/output',
    'notidesnowec': '/data/project3/minnaho/swel/notides/mc60/nowec/output',
    'notideswec':   '/data/project3/minnaho/swel/notides/mc60/wec/rerun',
    'ampwec':       '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything',
    'tidesampwec':  '/data/project3/minnaho/swel/tides/mc60/ampwec/everything',
}

# array key inside each output npz -- kept backward compatible for the 3
# scenarios plot_pdf_vort.py already reads (pdf_wec/pdf_nowec/pdf_notidesnowec)
ARRAY_KEY = {
    'tideswec':     'pdf_wec',
    'tidesnowec':   'pdf_nowec',
    'notidesnowec': 'pdf_notidesnowec',
    'notideswec':   'pdf_notideswec',
    'ampwec':       'pdf_ampwec',
    'tidesampwec':  'pdf_tidesampwec',
}

def src_glob(root):
    """Handle both flat (ampwec/tidesampwec) and his/ subdir layouts."""
    sub = os.path.join(root, 'his')
    base = sub if os.path.isdir(sub) else root
    return sorted(glob.glob(os.path.join(base, 'mc60_his.201904*.nc')))

# tidesampwec's raw archive has a trailing 1-timestep file
# (mc60_his.20190429110056.nc); no exclusion needed here since this script
# derives array shapes per-file (via pf.rho_uv_angle) rather than assuming a
# fixed tdim across all files
SCEN_FILES = {name: src_glob(root) for name, root in SCENARIOS.items()}

# ==========================================
# 2. Load Grid & Mask
# ==========================================
print('Loading grid data...')
grdnc = Dataset(grd, 'r')
f_nc = np.array(grdnc.variables['f'])
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc == 0] = np.nan

# ==========================================
# 3. Initialize PDF (Histogram) bins
# ==========================================
bin_edges = np.linspace(-20, 20, 501)

# ==========================================
# 4. Process each scenario
# ==========================================
def process_scenario(name, files):
    print(f'Processing {len(files)} {name} files...')
    counts = np.zeros(len(bin_edges) - 1)
    total_flt = 0

    for f_idx, file in enumerate(files):
        print(f'  -> {name} {f_idx+1}/{len(files)}: {file}')
        urho, vrho = pf.rho_uv_angle(file, grd, rotate=True)
        vort = pf.vorticity(grd, urho, vrho)
        norm_vort = (vort / f_nc) * masknc

        flat_vort = norm_vort.flatten()
        valid_vort = flat_vort[~np.isnan(flat_vort)]

        hist_counts, _ = np.histogram(valid_vort, bins=bin_edges)
        counts += hist_counts
        total_flt += valid_vort.shape[0]

    counts_plot = np.append(counts, 0)
    pdf = counts_plot / total_flt

    out_path = f'./pdf_vorticity_{name}.npz'
    np.savez(out_path, bin_edges=bin_edges, **{ARRAY_KEY[name]: pdf})
    print(f'Saved {name} data to {out_path}')

for name, files in SCEN_FILES.items():
    if not files:
        print(f'WARNING: no files for {name}, skipping')
        continue
    process_scenario(name, files)

print('All calculations complete.')
