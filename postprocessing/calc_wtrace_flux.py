import os
import glob
import numpy as np
from netCDF4 import Dataset

FILL_THRESH = 0.9e33

grd = '../mc60_grd.nc'
grdnc = Dataset(grd,'r')
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc==0] = np.nan

# Paths now point to separate directories where filenames are identical
scenarios = {
    'tides_wec':     ('/data/project3/minnaho/swel/tides/mc60/wec/his/zslice_10m',
                      '/data/project3/minnaho/swel/tides/mc60/wec/his/zslice_10m_trace'),
    'tides_nowec':   ('/data/project3/minnaho/swel/tides/mc60/nowec/output/his/zslice_10m',
                      '/data/project3/minnaho/swel/tides/mc60/nowec/output/his/zslice_10m_trace'),
    'notides_nowec': ('/data/project3/minnaho/swel/notides/mc60/nowec/output/his/zslice_10m',
                      '/data/project3/minnaho/swel/notides/mc60/nowec/output/his/zslice_10m_trace'),
    'notides_wec':   ('/data/project3/minnaho/swel/notides/mc60/wec/rerun/his/zslice_10m',
                      '/data/project3/minnaho/swel/notides/mc60/wec/rerun/his/zslice_10m_trace'),
}

def get_matched_pairs(his_dir, trace_dir):
    # Filenames are now identical in both folders
    his_files = sorted(glob.glob(os.path.join(his_dir, 'z_mc60_his.*.nc')))
    trace_files = sorted(glob.glob(os.path.join(trace_dir, 'z_mc60_his.*.nc')))
    
    # We match them by their basename
    his_map = {os.path.basename(f): f for f in his_files}
    trace_map = {os.path.basename(f): f for f in trace_files}
                 
    common = sorted(set(his_map) & set(trace_map))
    # Return path pairs and the timestamp string (extracted from filename for metadata)
    return [(his_map[f], trace_map[f], f.replace('z_mc60_his.', '').replace('.nc', '')) for f in common]

def load_masked(path, var):
    data = np.array(Dataset(path)[var][:], dtype=np.float32)
    data[data > FILL_THRESH] = np.nan
    data = data*masknc
    return data

def compute_env(his_dir, trace_dir, tracer_name):
    pairs = get_matched_pairs(his_dir, trace_dir)
    print(f'  {len(pairs)} file pairs found')

    w_list, trc_list, file_meta = [], [], []
    for hf, tf, stamp in pairs:
        w   = load_masked(hf, 'w')
        trc = load_masked(tf, tracer_name)
        file_meta.append((stamp, w.shape[0]))
        w_list.append(w)
        trc_list.append(trc)

    w_all   = np.concatenate(w_list,   axis=0)  
    trc_all = np.concatenate(trc_list, axis=0)
    del w_list, trc_list

    # Time mean at each grid point
    mean_w   = np.nanmean(w_all,   axis=0)  
    mean_trc = np.nanmean(trc_all, axis=0)  

    flux = (w_all - mean_w) * (trc_all - mean_trc)  
    del w_all, trc_all

    # --- ENVELOPE & STATISTICS ---
    time_series_max  = np.nanmax(flux, axis=(1, 2))
    time_series_min  = np.nanmin(flux, axis=(1, 2))
    time_series_mean = np.nanmean(flux, axis=(1, 2))

    # Get ocean_time from original (non-sliced) his files
    orig_his_dir = os.path.dirname(his_dir)
    ocean_times  = []
    for stamp, _ in file_meta:
        orig = os.path.join(orig_his_dir, f'mc60_his.{stamp}.nc')
        ocean_times.extend(np.array(Dataset(orig)['ocean_time'][:]).tolist())

    # PDF calculation
    all_flux = flux[~np.isnan(flux)]
    plo, phi = np.nanpercentile(all_flux, [0.5, 99.5])
    counts, edges = np.histogram(all_flux, bins=200, range=(plo, phi), density=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    return dict(
        ts_max      = time_series_max,
        ts_min      = time_series_min,
        ts_mean     = time_series_mean, # This is the same as the old "time_series"
        ocean_times = np.array(ocean_times),
        bin_centers = bin_centers,
        pdf         = counts,
        mean_flux   = float(np.nanmean(all_flux)),
    )

# --- Main Execution Loop ---
for tracer in ['ptrace', 'rtrace']:
    print(f'\nProcessing {tracer}...')
    for name, (his_dir, trace_dir) in scenarios.items():
        print(f'  Scenario: {name}')
        
        resultenv = compute_env(his_dir, trace_dir, tracer)
        
        outfileenv = f'w{tracer}_env_{name}.npz'
        np.savez(outfileenv, **resultenv)
        
        print(f'    Mean {tracer} flux = {resultenv["mean_flux"]:.4e}')
        print(f'    Saved -> {outfileenv}')
