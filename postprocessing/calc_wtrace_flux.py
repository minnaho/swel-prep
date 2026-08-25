import os
import glob
import numpy as np
from netCDF4 import Dataset

FILL_THRESH = 0.9e33

grd = '../mc60_grd.nc'
grdnc = Dataset(grd,'r')
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc==0] = np.nan

# 10 m level from the general zslicefull product -- replaces the legacy
# per-scenario zslice_10m/zslice_10m_trace dirs, which were only ever built
# for notidesnowec/tidesnowec and never existed for ampwec/tidesampwec.
# w and both tracers live together in the same zslicefull root his file, so
# no more matching two separate directories by basename.
ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'
# (depth index, output filename suffix) -- confirmed depth[10] == -10.0 m
# and depth[20] == -20.0 m exactly, same zslicefull depth grid used by
# calc_wno3_flux_10m.py/_20m.py
DEPTHS = [(10, ''), (20, '_20m')]

# key: (zslice subdirectory alias, raw his dir for ocean_time lookup) --
# same tuple shape/values as offshore_flux_ptrace_zslice.py's scenarios dict
scenarios = {
    'notidesnowec': ('notidesnowec',  '/data/project3/minnaho/swel/notides/mc60/nowec/his'),
    'ampwec':       ('notidesampwec', '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything'),
    'tidesnowec':   ('tidesnowec',    '/data/project3/minnaho/swel/tides/mc60/nowec/output/his'),
    'tidesampwec':  ('tidesampwec',   '/data/project3/minnaho/swel/tides/mc60/ampwec/everything'),
}

# tidesampwec's raw source has a trailing 1-timestep file
# (...20190429110056) whose zslice output has no time dimension at all
TIDESAMPWEC_EXCLUDE = ('20190429110056',)


def get_zslice_files(zslice_alias):
    files = sorted(glob.glob(os.path.join(ZSLICE_ROOT, zslice_alias, 'z_mc60_his.*.nc')))
    if zslice_alias == 'tidesampwec':
        files = [f for f in files if not any(s in f for s in TIDESAMPWEC_EXCLUDE)]
    return files

def load_masked(path, var, depth_idx):
    data = np.array(Dataset(path)[var][:, depth_idx, :, :], dtype=np.float32)
    data[data > FILL_THRESH] = np.nan
    data = data*masknc
    return data

def compute_env(zslice_alias, raw_his_dir, tracer_name, depth_idx):
    zfiles = get_zslice_files(zslice_alias)
    print(f'  {len(zfiles)} zslice files found')

    w_list, trc_list, file_meta = [], [], []
    for zf in zfiles:
        stamp = os.path.basename(zf).replace('z_mc60_his.', '').replace('.nc', '')
        w   = load_masked(zf, 'w', depth_idx)
        trc = load_masked(zf, tracer_name, depth_idx)
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

    # Get ocean_time from the raw (non-sliced) his files -- zslicefull files
    # have no ocean_time variable at all
    ocean_times  = []
    for stamp, _ in file_meta:
        orig = os.path.join(raw_his_dir, f'mc60_his.{stamp}.nc')
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
    for depth_idx, suffix in DEPTHS:
        print(f' Depth index {depth_idx} (suffix "{suffix}")...')
        for name, (zslice_alias, raw_his_dir) in scenarios.items():
            outfileenv = f'w{tracer}_env{suffix}_{name}.npz'
            if os.path.exists(outfileenv):
                print(f'  Skipping {name} -- {outfileenv} already exists')
                continue
            print(f'  Scenario: {name}')

            resultenv = compute_env(zslice_alias, raw_his_dir, tracer, depth_idx)

            np.savez(outfileenv, **resultenv)

            print(f'    Mean {tracer} flux = {resultenv["mean_flux"]:.4e}')
            print(f'    Saved -> {outfileenv}')
