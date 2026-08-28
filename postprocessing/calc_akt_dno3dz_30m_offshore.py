"""
h>100m-restricted script computing the turbulent diffusive vertical NO3
flux, Akt * dNO3/dz, at 30 m depth -- diffusive-flux counterpart of
calc_wno3_flux_30m_offshore.py's resolved eddy flux w'NO3'. Same 4-scenario
set, same offshore mask (mask[h <= 100] = np.nan applied to the base
mask_rho land mask), same ~20-M2-cycle record trim (START_TRIM=3, see
calc_wno3_flux_20m_100m.py's docstring for the exact cycle-count reasoning
-- kept here to match calc_akt_dno3dz_30m_100m.py's own convention, even
though the wno3 offshore siblings don't trim; START_TRIM is about
leading-record data availability, not the spatial shelf/offshore
restriction, so it's kept consistent within the Akt*dNO3/dz family instead),
and the same output npz schema (flux file: time_series/ocean_times/
bin_centers/pdf/mean_flux; env file: ts_min/ts_max/ts_mean/ocean_times/
bin_centers/pdf/mean_flux) so it's a drop-in alternate npz source for a
plot_wno3_flux_100m.py-style config entry.

Sources: Akt from the zsliced zslicefull/<scenario>/ak/z_mc60_his.*.nc,
NO3 from zslicefull/<scenario>/bgc/z_mc60_bgc.*.nc -- the same bgc source
calc_wno3_flux_30m_offshore.py already uses. Both share the exact same fixed
157-level zslice depth grid (confirmed directly) -- **NOT uniformly spaced**:
1m spacing 0 to -50m (indices 0-50), 5m spacing -50 to -300m (indices
50-100), 30m spacing -300 to -1980m (indices 100-156). 30m falls well inside
the 1m-spaced tier, so the centered difference at the two levels bracketing
it (index 29/31 around index 30) is still exact (spans exactly ±1m, no
interpolation needed) -- but this only holds because TARGET_DEPTH is in the
top tier; a deeper target would need to check which tier it lands in rather
than assume ±1 index means ±1m. No w'/mean-removal decomposition is applied
since Akt*dNO3/dz is already the whole instantaneous diffusive flux at each
grid point/time, not an eddy correlation that needs anomalies isolated first.

Sign convention: this computes the literal Akt*dNO3/dz product as asked, not
the standard oceanographic downgradient-diffusion sign flip
(flux = -Akt*dNO3/dz, z positive up); flip the sign at consumption time if
the standard convention is wanted.

Output: akt_dno3dz_flux_30m_offshore_<scenario>.npz, akt_dno3dz_env_30m_offshore_<scenario>.npz
"""

import os
import glob
import numpy as np
from netCDF4 import Dataset

FILL_THRESH  = 0.9e33
TARGET_DEPTH = -30   # m — select the level bracketing this from the zslicefull depth array
START_TRIM   = 3     # drop this many leading hourly samples -- trims the
                      # record to ~20 M2 tidal cycles, see module docstring

ZSLICE_ROOT = '/data/project1/minnaho/swel/zslicefull'

grdnc  = Dataset('../mc60_grd.nc', 'r')
masknc = np.array(grdnc.variables['mask_rho'])
masknc[masknc == 0] = np.nan
h_nc = np.array(grdnc.variables['h'])
masknc[h_nc <= 100] = np.nan   # restrict to h>100m (offshore) only

# same 4 scenarios as calc_wno3_flux_30m_offshore.py
scenarios = {
    'notidesnowec': (f'{ZSLICE_ROOT}/notidesnowec',
                     '/data/project3/minnaho/swel/notides/mc60/nowec/his'),
    'tidesnowec':   (f'{ZSLICE_ROOT}/tidesnowec',
                     '/data/project3/minnaho/swel/tides/mc60/nowec/output/his'),
    'ampwec':       (f'{ZSLICE_ROOT}/notidesampwec',
                     '/data/project3/minnaho/swel/notides/mc60/wec/ampwec/everything'),
    'tidesampwec':  (f'{ZSLICE_ROOT}/tidesampwec',
                     '/data/project3/minnaho/swel/tides/mc60/ampwec/everything'),
}

# tidesampwec's raw source has a trailing 1-timestep file
# (...20190429110056) whose zslice output has no time dimension at all --
# present in both ak/ and bgc/ subdirs, confirmed
TIDESAMPWEC_EXCLUDE = ('20190429110056',)


def get_matched_pairs(zslice_dir):
    """Match ak/z_mc60_his and bgc/z_mc60_bgc files by timestamp."""
    ak_files  = sorted(glob.glob(os.path.join(zslice_dir, 'ak', 'z_mc60_his.*.nc')))
    bgc_files = [f for f in sorted(glob.glob(os.path.join(zslice_dir, 'bgc', 'z_mc60_bgc.*.nc')))
                 if 'dia_avg' not in f]
    if os.path.basename(zslice_dir.rstrip('/')) == 'tidesampwec':
        ak_files  = [f for f in ak_files  if not any(x in f for x in TIDESAMPWEC_EXCLUDE)]
        bgc_files = [f for f in bgc_files if not any(x in f for x in TIDESAMPWEC_EXCLUDE)]
    ak_map  = {os.path.basename(f).replace('z_mc60_his.', '').replace('.nc', ''): f
               for f in ak_files}
    bgc_map = {os.path.basename(f).replace('z_mc60_bgc.', '').replace('.nc', ''): f
               for f in bgc_files}
    common = sorted(set(ak_map) & set(bgc_map))
    return [(ak_map[s], bgc_map[s], s) for s in common]


def get_bracket_indices(nc):
    """Return (idx_minus, idx0, idx_plus, dz) bracketing TARGET_DEPTH.
    Exact (no interpolation needed) only because TARGET_DEPTH=-30 falls in
    the zslice depth grid's 1m-spaced top tier (0 to -50m); the grid is NOT
    uniformly spaced overall (5m spacing -50 to -300m, 30m beyond) -- a
    different TARGET_DEPTH must be checked against the actual grid, not
    assumed ±1 index == ±1m."""
    depth = np.array(nc.variables['depth'][:])
    idx0  = int(np.argmin(np.abs(depth - TARGET_DEPTH)))
    idx_minus, idx_plus = idx0 - 1, idx0 + 1
    dz = depth[idx_plus] - depth[idx_minus]   # negative -- z decreases downward
    return idx_minus, idx0, idx_plus, dz


def load_level(path, var, idx):
    """Load var at a specific depth index from a zslicefull file → (time, eta, xi)."""
    with Dataset(path) as nc:
        data = np.array(nc.variables[var][:, idx, :, :], dtype=np.float32)
    data[data > FILL_THRESH] = np.nan
    data = data * masknc[None, :, :]
    return data


def read_ocean_times(file_meta, orig_his_dir):
    ocean_times = []
    for stamp, _ in file_meta:
        orig = os.path.join(orig_his_dir, f'mc60_his.{stamp}.nc')
        ocean_times.extend(np.array(Dataset(orig)['ocean_time'][:]).tolist())
    return np.array(ocean_times)


def compute_diff_flux(zslice_dir):
    """Akt*dNO3/dz at every grid point/time, concatenated + trimmed."""
    pairs = get_matched_pairs(zslice_dir)
    print(f'  {len(pairs)} file pairs found')

    with Dataset(pairs[0][1]) as nc0:   # bgc file shares the ak file's depth grid
        idx_minus, idx0, idx_plus, dz = get_bracket_indices(nc0)

    flux_list, file_meta = [], []
    for akf, bf, stamp in pairs:
        akt    = load_level(akf, 'Akt', idx0)
        no3_lo = load_level(bf,  'NO3', idx_minus)
        no3_hi = load_level(bf,  'NO3', idx_plus)
        dno3dz = (no3_hi - no3_lo) / dz
        flux_list.append(akt * dno3dz)
        file_meta.append((stamp, akt.shape[0]))

    flux_all = np.concatenate(flux_list, axis=0)  # (total_nt, eta_rho, xi_rho)
    del flux_list
    return flux_all, file_meta


def compute_flux(zslice_dir, orig_his_dir):
    flux_all, file_meta = compute_diff_flux(zslice_dir)
    ocean_times = read_ocean_times(file_meta, orig_his_dir)

    # trim to ~20 M2 tidal cycles, see module docstring
    flux_all    = flux_all[START_TRIM:]
    ocean_times = ocean_times[START_TRIM:]

    time_series = np.nanmean(flux_all, axis=(1, 2))

    all_flux = flux_all[~np.isnan(flux_all)]
    plo, phi = np.nanpercentile(all_flux, [0.5, 99.5])
    counts, edges = np.histogram(all_flux, bins=200, range=(plo, phi), density=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    return dict(
        time_series = time_series,
        ocean_times = ocean_times,
        bin_centers = bin_centers,
        pdf         = counts,
        mean_flux   = float(np.nanmean(all_flux)),
    )


def compute_env(zslice_dir, orig_his_dir):
    flux_all, file_meta = compute_diff_flux(zslice_dir)
    ocean_times = read_ocean_times(file_meta, orig_his_dir)

    # trim to ~20 M2 tidal cycles, see module docstring
    flux_all    = flux_all[START_TRIM:]
    ocean_times = ocean_times[START_TRIM:]

    time_series_max  = np.nanmax(flux_all, axis=(1, 2))
    time_series_min  = np.nanmin(flux_all, axis=(1, 2))
    time_series_mean = np.nanmean(flux_all, axis=(1, 2))

    all_flux = flux_all[~np.isnan(flux_all)]
    plo, phi = np.nanpercentile(all_flux, [0.5, 99.5])
    counts, edges = np.histogram(all_flux, bins=200, range=(plo, phi), density=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    return dict(
        ts_max      = time_series_max,
        ts_min      = time_series_min,
        ts_mean     = time_series_mean,
        ocean_times = ocean_times,
        bin_centers = bin_centers,
        pdf         = counts,
        mean_flux   = float(np.nanmean(all_flux)),
    )


for name, (zslice_dir, orig_his_dir) in scenarios.items():
    print(f'Processing {name}...')
    result    = compute_flux(zslice_dir, orig_his_dir)
    resultenv = compute_env(zslice_dir, orig_his_dir)
    outfile    = f'akt_dno3dz_flux_30m_offshore_{name}.npz'
    outfileenv = f'akt_dno3dz_env_30m_offshore_{name}.npz'
    np.savez(outfile,    **result)
    np.savez(outfileenv, **resultenv)
    print(f'  mean flux = {result["mean_flux"]:.4e} mmol N m-2 s-1')
    print(f'  saved -> {outfile}')
