# offshore flux plot — status and known issues

## Current state (end of session 2025-05-06)

All three issues from the previous session have been resolved and a full
zslice-based pipeline has been built alongside the legacy scripts.

---

## Zslice pipeline (new, preferred)

### Step 1 — `postprocessing/zslice_uniform.py`

Zslices `his/` and `bgc/` files onto a non-uniform z-grid for each scenario.
Run one instance per scenario (4 in parallel):

```
SCENARIO = 'tideswec'   # edit at top of script
python zslice_uniform.py
```

**Z grid (157 levels, positive integers passed as negatives to CLI):**

| Range | Step | Count |
|-------|------|-------|
| 0 → 50 m | 1 m | 51 |
| 55 → 300 m | 5 m | 50 |
| 330 → 1980 m | 30 m | 56 |

Output: `/data/project1/minnaho/swel/zslicefull/<scenario>/z_mc60_{his,bgc}.<ts>.nc`

Each file has a `depth` dimension (157 levels, negative downward) and a `time`
dimension but **no `ocean_time` variable** (read from original his files in
postprocessing).

### Step 2 — postprocessing scripts

```bash
cd postprocessing/
python offshore_flux_ptrace_zslice.py   # → offshore_flux_ptrace_zslice_<scenario>.npz
python offshore_flux_rtrace_zslice.py   # → offshore_flux_rtrace_zslice_<scenario>.npz
python offshore_flux_zslice.py          # → offshore_flux_zslice_<scenario>.npz  (NO3)
```

**NPZ format (new, differs from legacy):**

| Key | Shape | Description |
|-----|-------|-------------|
| `offshore_flux` | (time, n_z, n_valid) | mmol s⁻¹ |
| `depth` | (n_z,) | m, negative downward |
| `dz` | (n_z,) | m, bin thickness |
| `ocean_time` | (time,) | s since 1995-01-01 |
| `eta_idx`, `xi_idx` | (n_valid,) | band-edge grid indices |
| `dy_face` | (n_valid,) | m |

`u` is taken from the u-face at `xi_u = i_left - 1` directly (no
interpolation to rho-points). `dz` is computed from adjacent depth-level
midpoints (midpoint rule).

### Step 3 — plot scripts

```bash
cd plot/
python plot_offshore_flux_profile_zslice.py
python plot_offshore_flux_hov_zslice.py
python plot_offshore_flux_hov_time_zslice.py
```

No depth-binning in the plot scripts — data is already on the z-grid.
`shading='nearest'` used throughout (depth values are cell centers).

Cache files: `./figs/offshore_flux_{profile,hov,hov_time}_zslice_cache_<TRACER>.npz`

---

## Legacy scripts (still present, partially fixed)

The original `offshore_flux*.py` + `plot_offshore_flux*.py` scripts remain
and have received two fixes this session:

### Fix 1 — date fix in `plot_offshore_flux_hov_time.py`

Replaced `netCDF4.num2date` with `pyfuncs.numdate`. The old path returned
`cftime` objects that `mdates.date2num` converted to JDN-like integers,
causing two symptoms:
- X-axis labels read 6732-05-25 instead of 2019
- 252 hourly steps collapsed to ~11 visible columns

**Cache must be deleted before re-running:**
```bash
rm ./figs/offshore_flux_hov_time_cache_*.npz
```

### Fix 2 — pcolormesh cell-center fix in both hov scripts

`depth_axis` changed from bin lower edges (`shared_bins[:-1][::-1]`) to bin
centers (`(shared_bins[:-1] + BIN_SZ/2)[::-1]`) in both
`plot_offshore_flux_hov.py` and `plot_offshore_flux_hov_time.py`.
Eliminates a `BIN_SZ/2 = 1 m` upward shift under `shading='auto'` on
matplotlib ≥ 3.5.

**Both hov caches must be deleted before re-running:**
```bash
rm ./figs/offshore_flux_hov_cache_*.npz
rm ./figs/offshore_flux_hov_time_cache_*.npz
```

### Remaining known issues in legacy scripts

- **Extensive flux bias**: `rsum/rcnt` per depth bin weights every s_rho cell
  equally regardless of `Hz`. Thin near-surface cells are
  over-represented. Not fixed in legacy path — moot once zslice pipeline
  is used instead.
- **White patches at depth**: `BIN_SZ = 2 m` is finer than ROMS s_rho
  spacing below ~200 m. Addressed in zslice pipeline; legacy scripts
  unchanged.

---

## File map

| File | Role |
|------|------|
| `postprocessing/zslice_uniform.py` | zslice his vars (ptrace,rtrace,w,rho,u,v); BGC removed |
| `postprocessing/zslice_bgc.py` | zslice all 12 BGC vars per scenario → `bgc/` subdir |
| `postprocessing/offshore_flux_ptrace_zslice.py` | ptrace flux from zsliced his |
| `postprocessing/offshore_flux_rtrace_zslice.py` | rtrace flux from zsliced his |
| `postprocessing/offshore_flux_zslice.py` | BGC flux — 7 tracers per scenario |
| `postprocessing/profile_zslice.py` | mean depth profiles for all vars (full domain) |
| `postprocessing/profile_zslice_coastal.py` | mean depth profiles restricted to 10 km coastal mask |
| `plot/plot_profile_zslice.py` | depth profile plots (full + coastal side-by-side, all vars) |
| `plot/plot_offshore_flux_profile_zslice.py` | profile plot |
| `plot/plot_offshore_flux_hov_zslice.py` | lat×depth hov |
| `plot/plot_offshore_flux_hov_time_zslice.py` | time×depth hov |
| `plot/plot_offshore_flux_profile.py` | legacy, unchanged |
| `plot/plot_offshore_flux_hov.py` | legacy, cell-center fix applied |
| `plot/plot_offshore_flux_hov_time.py` | legacy, date fix + cell-center fix applied |

---

## Session 2026-05-08 — changes and pending work

### Changes made

#### `postprocessing/zslice_uniform.py`
Removed the BGC entry from `SOURCES` (was `('bgc', 'mc60_bgc.*.nc', 'NO3')`).
BGC zslicing is now fully handled by `zslice_bgc.py`, which writes all 12 vars
to `<scenario>/bgc/`. The his entry (ptrace, rtrace, w, rho, u, v) is unchanged.

#### `postprocessing/offshore_flux_zslice.py` — expanded to all BGC tracers
Was single-tracer (NO3 only). Now computes flux for 7 tracers per scenario:

| Type | Tracers |
|------|---------|
| Single | NO3, NH4, O2, DIC, DOC |
| Summed | TOTC = SPC+DIATC+DIAZC, TOTCHL = SPCHL+DIATCHL+DIAZCHL |

Output: `offshore_flux_zslice_<scenario>_<tracer>.npz` (7 NPZs × 4 scenarios = 28 total).
Phytoplankton individuals (SPC, DIATC, DIAZC, SPCHL, etc.) are summed before the flux
multiply, not saved separately.

#### `offshore_flux_*trace_zslice.py` + `offshore_flux_zslice.py` — dz and NaN fixes
- `_fill_to_nan(arr)`: converts `|x| > 1e30` → NaN **before** multiplying so
  float32 overflow (1e33² = inf) is avoided and masked cells stay NaN in output.
- `compute_dz_2d(depth_vals, h_edge)` replaces the old scalar dz: builds
  interface positions from depth midpoints (top fixed at 0, bottom extrapolated
  half-step), then clips all interfaces to `−h_edge` via `np.maximum`. Returns
  `(n_z, n_valid)` dz where below-seabed cells have `dz = 0`. Below-seabed
  cells multiplied by `dz=0` remain NaN (NaN × 0 = NaN per IEEE 754), which
  preserves the distinction between "no water" and "genuine zero flux".
- NPZ schema updated: `dz` is now `(n_z, n_valid)` instead of `(n_z,)`;
  `h_edge` array added.

#### `zslice.F` — binary rebuilt by user
String buffers widened from their original length to `len=256`. This fixed a
silent truncation bug where the last variable in `--vars=v1,v2,...` was dropped
whenever the full var-list string exceeded the old buffer size. ZOOC was always
last in `BGC_VARS` and was therefore always silently missing from output.
No Python-side workaround needed — `zslice_bgc.py` already listed all 12 vars
correctly. **All four scenarios need to be re-run with the new binary** so that
existing zsliced BGC files actually contain ZOOC (if desired — see note below).

**Note:** User decided to skip ZOOC for now ("I'll live without zooc"). ZOOC is
not included in `offshore_flux_zslice.py` `SINGLE_TRACERS`. To add it later:
1. Re-run `zslice_bgc.py` for all four scenarios with the rebuilt zslice binary.
2. Add `'ZOOC'` to `SINGLE_TRACERS` in `offshore_flux_zslice.py` and
   `BGC_VARS` in `profile_zslice.py`.

Cosmetic issues introduced by the buffer widening (not blocking):
- Uninitialised grid_file buffer prints garbage when no input file is given.
- VertCoordType prints with 256 trailing spaces instead of being `trim()`-ed.

---

---

## Session 2026-05-11 — changes

### `postprocessing/profile_zslice.py` — written and debugged

Computes time- and horizontally-averaged depth profiles for all zsliced variables
across all four scenarios. Applies ocean `mask_rho` from the grid file so land
cells are excluded (NaN) before accumulation.

**Variables:**
- HIS (157 z-levels): ptrace, rtrace, w, rho, u, v
- BGC (157 z-levels): NO3, NH4, SPC, DIATC, DIAZC, SPCHL, DIATCHL, DIAZCHL, O2, DIC, DOC
- DIA (101 z-levels, 0–200 m every 2 m): TOT_PROD

**Output:** `postprocessing/zslice_profiles.npz` (full merge) + per-scenario
`postprocessing/zslice_profiles_<scen>.npz` saved progressively after each scenario.

| Key | Shape | Description |
|-----|-------|-------------|
| `depth` | (157,) | m, negative downward (his/bgc vars) |
| `depth_dia` | (101,) | m, negative downward (dia vars) |
| `<var>_<scenario>` | (n_z,) | mean profile, NaN where no wet cell |

**BGC glob fix:** BGC z-sliced files live in the scenario root
(`{ZSLICE_ROOT}/{scen}/z_mc60_bgc.*.nc`), not a `bgc/` subdirectory.
All four scenarios confirmed to use the root-level path.

**Mask:** `mask_rho` loaded from `../plot/mc60_grd.nc`, land cells set to NaN
(`mask_rho[mask_rho==0] = np.nan`). Staggered variants `mask_u` and `mask_v`
derived as `mask_rho[:, :-1]` and `mask_rho[:-1, :]`. Each variable array is
multiplied by its appropriate mask before `nansum`.

---

### `postprocessing/profile_zslice_coastal.py` — written

Same as `profile_zslice.py` but restricted to cells within the 10 km coastal mask
(`../plot/coastal_mask.nc`). Mask loaded as float with land set to NaN; staggered
variants for u/v. Per-scenario saves: `zslice_profiles_coastal_<scen>.npz`;
merged: `zslice_profiles_coastal.npz`.

---

### `postprocessing/zslice_npp.py` — written

Zslices `TOT_PROD` from `mc60_bgc_dia_avg.*.nc` files onto a 101-level z-grid
(0–200 m every 2 m). Run one instance per scenario (edit `SCENARIO` at top).

Output: `/data/project1/minnaho/swel/zslicefull/<scenario>/dia/z_mc60_bgc_dia_avg.<ts>.nc`

Status: all four scenarios complete.

---

### Offshore flux normalization change — flux per unit area

**Problem:** Below ~300 m the non-uniform z-grid has coarser bins (30 m steps),
causing apparent magnitude jumps when flux is multiplied by `dz`.

**Fix:** Switched all three postprocessing flux scripts to flux per unit area
(`-u × C`, units mmol m⁻² s⁻¹) — removes both `dy` and `dz` from the multiply.
The `dz` and `dy_face` arrays are still saved in NPZ for optional downstream use.

Updated `TRACER_UNITS` in all three plot scripts to `mmol m$^{-2}$ s$^{-1}`.

**Old versions preserved** as `*_old.py` (multiply by `dy × dz`, units mmol s⁻¹):
- `offshore_flux_ptrace_zslice_old.py`
- `offshore_flux_rtrace_zslice_old.py`
- `offshore_flux_zslice_old.py`

**Fix-up plot scripts** (`*_fix.py`) divide old NPZ data by `dz × dy_face` at
load time to convert legacy outputs to per-unit-area units. Named with `_fix_`
cache prefix to avoid colliding with new-format caches.

---

### `plot/plot_npp_depth_integrated.py` — written and debugged

Depth-integrates `TOT_PROD` from zsliced dia files and plots a 2×2 panel figure:
- Top-left: `notidesnowec` absolute (mmol m⁻² d⁻¹, algae colormap)
- Other three panels: difference from `notidesnowec` (balance colormap)

Depth integration: `nansum(arr, axis=0) × DZ × 86400` (DZ = 2 m, ×86400 s→d).
Land overlay and coastline drawn from `mask_rho` (gray pcolormesh + black contour
at 0.5) rather than `cartopy.feature`.

Output: `./figs/npp_depth_integrated.png`
Cache: `./figs/npp_depth_integrated_cache.npz`

**Bug fixes (session 2026-05-11, late):**

*Root cause of `pcolormesh` `ValueError: not enough values to unpack`*: the
zsliced dia files have **no time dimension** — `TOT_PROD` is `(depth, eta_rho,
xi_rho)` 3D, not `(t, z, eta, xi)` 4D. The original `load_npp` did
`nansum(arr, axis=1)` which collapsed `eta` instead of `depth`, then
`nansum(di, axis=0)` collapsed `depth`, yielding a `(xi,)` 1D array.

Fix: `load_npp` now accepts a list of `{'zf', 'ot', 'date'}` rows (one per
file / time step) and does `nansum(arr, axis=0) * DZ * 86400` to
depth-integrate correctly, then accumulates across files for the time mean.

*Cross-scenario time alignment*: `ocean_time` is not stored in the zsliced dia
files — it must be read from the matching original `mc60_bgc_dia_avg.<TS>.nc`
(same `<TS>` substring, different dir). Timestamps differ across scenarios by
up to ~12 hours (23:01 vs 11:01) because model restarts shift the averaging
window. Matching is therefore done by **calendar date** (not by a seconds
tolerance), using `_ot_to_date(ot_sec)` → `datetime.date`.

New helpers added:
- `build_index(scen)` — globs zsliced dia files, reads `ocean_time` from
  original, returns list of `{'zf', 'ot', 'date'}` rows.
- `intersect_times(index_by_scen)` — builds per-scenario `{date: row}` dicts,
  takes set intersection of dates, prints matched/dropped counts with UTC times.

`ORIG_DIA_ROOTS` dict maps each scenario to its original dia directory (same
paths as `zslice_npp.py` `SCENARIO_ROOTS`).

Cache now validated on load: if any array is not 2D, the script recomputes
rather than passing 1D data silently to `pcolormesh`.

---

### Plot script updates

- `plot_offshore_flux_hov_time_zslice.py`: x-axis date format changed to `'%m-%d'`
  (no year).
- All three `plot_offshore_flux_*zslice.py` scripts updated for new NPZ naming
  convention (`offshore_flux_zslice_<scenario>_<tracer>.npz`) via `npz_path()`
  helper.

---

## Session 2026-05-12 — changes

### `postprocessing/profile_zslice.py` + `profile_zslice_coastal.py` — improved I/O (glob, not concatenated)

Both scripts previously globbed per-timestep zslice files and reopened each
file once per variable (6 HIS + 11 BGC + 1 DIA ≈ 18 reopens per file).

**New behaviour:** scripts still glob the per-timestep files but now open each
file only once and read all variables in a single pass inside the `with Dataset`
block. This reduces file-open overhead by ~6–11× per file.

Note: a concatenated-file approach was attempted but abandoned — concatenating
all zslice files would produce files too large to read into memory.
The `ncecat` command is still relevant for DIA avg files (which lack a time
dimension and require `ncecat` rather than `ncrcat`), documented here for reference:
```bash
ncecat z_mc60_bgc_dia_avg.*.nc concat_z_mc60_bgc_dia_avg.nc  # adds new record dim
```

**Output NPZ format is unchanged** — same keys, same shapes. Plot scripts that
read `zslice_profiles*.npz` need no changes.

### New plot scripts written this session

| Script | Description |
|--------|-------------|
| `plot/plot_npp_depth_integrated.py` | 2×2 map of depth-integrated NPP; fully tuned (colorbars, extent, fonts, layout) |
| `plot/plot_buoy_grad.py` | Time-mean horizontal buoyancy gradient \|∇b\| at 0/10/20/50 m; one PNG per depth |
| `plot/plot_buoy_grad_snapshot.py` | Same as above but for a single snapshot (set `TIMESTAMP` and `TIDX` at top) |

### `plot/plot_buoy_grad*.py` — layout and plotting changes

- All four panels show absolute \|∇b\| fields (one per scenario); no difference
  plots. Colormap `cmocean.amp`, vmin=0, vmax=99th percentile across all four
  scenarios.
- Single shared colorbar placed to the right of the right column via
  `fig.add_axes` + `fig.canvas.draw()` + `get_position()` — no space stolen
  from data panels, so all four plots are equal size.
- `plot_npp_depth_integrated.py` received the same `add_axes` colorbar treatment
  for the same reason.
- `fig.suptitle(..., y=0.92)` in `plot_buoy_grad_snapshot.py` — cartopy's fixed
  aspect ratio means panels don't fill the gridspec cell vertically; the default
  `y≈0.98` left a large blank band between the title and the top row of panels.
  `y=0.92` pins the title just above where the panels actually render.

### `plot/plot_npp_depth_integrated.py` — layout tuning (this session)

- Colorbars moved to `fig.add_axes` / `fig.canvas.draw()` + `get_position()`
  approach so panels are not shrunk.
- `set_extent` applied with `XLIM`/`YLIM` constants; lat/lon tick labels shown
  on outer edges only via `tick_params(labelbottom=, labelleft=)`.
- `plt.rcParams({'font.size': 14})` set globally; all explicit fontsize/labelsize
  overrides updated to match.
- `gridspec_kw=dict(hspace=..., wspace=...)` used for inter-panel spacing
  (no `constrained_layout` — conflicts with cartopy aspect ratio enforcement).

### Output quality — `dpi=600` across all plot scripts

All scripts that previously used `dpi=150` were updated to `dpi=600` for
publication-quality output. Affected scripts include all `plot_*.py` files
in `plot/`.

---

## Session 2026-05-19 — changes and findings

### `plot/plot_profile_zslice.py` — written

New plot script for the output of `postprocessing/profile_zslice.py` and
`postprocessing/profile_zslice_coastal.py`. Reads both
`postprocessing/zslice_profiles.npz` (full domain) and
`postprocessing/zslice_profiles_coastal.npz` (10 km coastal mask) and
produces one PNG per variable (18 total) with two side-by-side panels:

- **Left panel:** full-domain time/space mean profile
- **Right panel:** coastal-mask mean profile
- Four scenarios overlaid as colored lines (C0–C3) with legend on left panel only

Per-variable y-axis depth limits:

| Vars | Depth range |
|------|-------------|
| `ptrace`, `rtrace`, `w`, `u`, `v`, `SPC`, `DIATC`, `DIAZC`, `SPCHL`, `DIATCHL`, `DIAZCHL` | −150 → 0 m |
| `TOT_PROD` | −200 → 0 m |
| `NO3`, `NH4`, `O2`, `DIC`, `DOC` | −500 → 0 m |
| `rho` | full depth |

Output: `./figs/profile_zslice_<var>.png`
Registered in `run_plots.py` under the `pdf` category.

---

### `notideswec` BGC data — complete model blow-up (BLOCKING)

**The `notideswec` BGC simulation has diverged catastrophically and its output
is unusable.** All water-column BGC variables contain values of ±10⁵–10⁸
(physically impossible for concentrations). Only sediment diagnostics
(`Sed_POC`, `Sed_CaCO3`, `Sed_Si`) are unaffected.

Confirmed at two levels:
1. **Original ROMS output** (`/data/project3/minnaho/swel/notides/mc60/wec/output/bgc/mc60_bgc.20190418230115.nc`): values reach SiO3 = ±9×10⁷, O2 = ±6×10⁸, DIATC = ±2×10⁸.
2. **Zsliced files** (`/data/project1/minnaho/swel/zslicefull/notideswec/z_mc60_bgc.*.nc`): same corruption propagated through.

The HIS variables for `notideswec` (ptrace, rtrace, w, rho, u, v) are clean —
the blow-up is specific to the BGC solver.

The `_fill_to_nan` guard (`|x| > 1e30`) in `profile_zslice.py` does not catch
these values because the corrupt data falls below the fill-value threshold
(fill = ~1e33) but far exceeds physical ranges. As a result,
`zslice_profiles*.npz` contains garbage for all BGC keys ending in `_notideswec`.

**Corrupt variable summary (first file, non-fill range):**

| Variable | min | max |
|----------|-----|-----|
| O2 | −6.2×10⁸ | +8.9×10⁸ |
| DIC | −3.1×10⁸ | +2.2×10⁸ |
| DIATC | −1.6×10⁸ | +2.3×10⁸ |
| SiO3 | −9.1×10⁷ | +1.6×10⁸ |
| SPC | −3.8×10⁷ | +6.9×10⁷ |
| NH4 | −3.3×10⁷ | +2.3×10⁷ |
| NO3 | −5.5×10⁶ | +3.9×10⁶ |
| Fe, DOFE, DIATFE, DIAZFE | ≤ ±10³ | (small but still negative) |

**Required fix:** re-run the `notideswec` BGC model from a stable restart
(e.g., restart from the `notidesnowec` run or from the `spinup/` directory),
then re-run `zslice_bgc.py` and `profile_zslice.py` / `profile_zslice_coastal.py`
for `notideswec`.

### File map update

| File | Role |
|------|------|
| `plot/plot_profile_zslice.py` | NEW — depth profiles (full + coastal) for all zsliced vars |

---

## Session 2026-05-20 — new scripts

### `plot/plot_cs_zslice.py` — diagonal cross-section (z-sliced)

Plots a time-mean depth cross-section along a diagonal transect defined in
grid index space, using z-sliced BGC or HIS files. The transect goes in the
**−xi direction** (offshore) from a coast-side starting point.

**Transect definition (set at top of script):**

| Parameter | Meaning |
|-----------|---------|
| `ETA0`, `XI0` | starting grid index near the coast |
| `SLOPE` | `deta/dxi` — diagonal angle; `0` = purely zonal, `+` = eta decreases going offshore, `−` = eta increases |
| `LENGTH_XI` | transect length in xi cells (does not need to reach domain edge) |
| `N_PTS` | number of interpolation points along transect |

Interpolation uses `scipy.ndimage.map_coordinates` (bilinear, order=1) on
each depth level. Land cells and NaN source cells are masked before
interpolation and restored after. X-axis is along-transect distance in km
computed from `lat_rho`/`lon_rho`.

Output: `./figs/cs_zslice_<SCENARIO>_<VAR>.png`

### `plot/plot_cs_diag.py` — diagonal cross-section (native s_rho, snapshot)

Same transect geometry as `plot_cs_zslice.py` but uses native ROMS HIS/BGC
files and `ROMS_depths.get_zr_zeta` for the depth axis — same pattern as
`plot_cs_surf.py`. Saves **one PNG per time step**.

Key differences from `plot_cs_zslice.py`:

- Depth axis is a **2D array** `(s_rho, N_PTS)` because z varies across
  the transect with bathymetry and free surface. `pcolormesh` receives both
  `dist_2d` and `zr_plot` as 2D arrays so the cross-section naturally follows
  the seafloor.
- Two-panel layout: surface map (top) with transect line overlaid, depth
  cross-section (bottom).
- `VAR_SRC = 'his'` or `'bgc'` selects which file set to read.

Output: `./figs/snapshots/cs_diag_<SCENARIO>-YYYY-MM-DD-HH.png`

**Shared transect geometry between both scripts** — set the same `ETA0`,
`XI0`, `SLOPE`, `LENGTH_XI` in both files to compare z-sliced (time-mean)
and snapshot cross-sections on the same transect.

---

## Session 2026-05-29 — `plot_cs_diag.py` major refactor

### Reynolds decomposition — averaging period discussion

The `calc_wno3_flux.py` and `calc_wtrace_flux.py` scripts compute `w'C'` by
subtracting the **full simulation mean** (2019-04-18 23:01 → 2019-04-29 11:01,
252 hours) at each grid point. This is the standard Reynolds decomposition where
"prime" = deviation from the time-mean residual circulation.

**Tidal residual in the mean:** 252 h / 12.42 h (M2) ≈ 20.29 cycles — not an
integer, so a small tidal signal (~1% of M2 amplitude) leaks into `mean_w` and
`mean_no3`. The time-averaged identity `⟨w'C'⟩ = ⟨wC⟩ − ⟨w⟩⟨C⟩` is exact
regardless, so the time-mean eddy flux is unaffected; only the instantaneous
time series carries a small systematic offset. Considered acceptable for the
comparison purpose.

**Shorter averaging periods (1 day, 3 days):** Would define "mean" as a
running/block average, leaving only sub-period fluctuations in `w'`. For the
tides runs this would largely remove the tidal pumping contribution from `w'C'`
(tidal period ~12.4 h ≈ 2 cycles per day). The full-simulation mean is the
correct choice for capturing tidal pumping as an eddy flux mechanism.

---

### `plot/plot_cs_diag.py` — refactored

#### Distance axis removed
X-axis changed from along-transect distance in km (computed via
`lat_rho`/`lon_rho` spherical geometry) to **longitude** directly, following the
pattern of `plot_cs_surf.py` (`lon_slice`). The `dlat`/`dlon`/`dist_km`
calculation and `dist_2d` tile are gone.

#### Transect preview
Interactive preview (shown before plotting) draws the transect line(s) on the
domain `mask_rho` map. User presses **Enter** in the figure window to proceed or
**Esc** to cancel. Preview is skipped automatically when the `Agg` backend is
active (screen session / no display).

#### Multi-scenario stacked layout
Replaced single-scenario two-panel figure (surface map + cross-section) with
**one cross-section row per scenario**, stacked vertically in a single figure.
One shared colorbar on the right. No surface map panel.

**Scenarios (`SCENARIOS` dict):**

| Key | Path |
|-----|------|
| `tideswec` | `/data/project3/minnaho/swel/tides/mc60/wec` |
| `tidesnowec` | `/data/project3/minnaho/swel/tides/mc60/nowec/output` |
| `notidesnowec` | `/data/project3/minnaho/swel/notides/mc60/nowec` |
| `notideswec` | `/data/project3/minnaho/swel/notides/mc60/wec/rerun` |
| `ampwec` | `/data/project3/minnaho/swel/notides/mc60/wec/ampwec/notrace` |

`notideswec` path updated to `.../wec/rerun` (previously `.../wec/base`, previously `.../wec/output`).
`notidesnowec` path updated to `.../nowec` (previously `.../nowec/output`).
`ampwec` is a new scenario with wave amplitude amplified 2.5×.

`src_glob(root, kind)` helper transparently handles the flat file layout of
`ampwec/notrace` (no `his/` subdir) vs. the standard `<root>/his/` layout.
File alignment across scenarios is by sorted-glob index (all five runs have
21 `his` files in the same chronological order).

#### Dual transect support
Two transects can be defined simultaneously:

| Parameter | Transect 0 | Transect 1 |
|-----------|-----------|-----------|
| `ETA0`/`ETA1` | 271 | 832 |
| `XI0`/`XI1` | 543 | 646 |
| `SLOPE`/`SLOPE1` | −0.8 | 0.3 |

`build_transect(eta0, xi0, slope)` builds the coords, lon, lat, and mask arrays
for each transect. `TRANSECTS` is a list of two dicts. The preview shows **both**
lines (red = t0, blue = t1). `interp_transect` and `interp_section` now take
explicit `coords` and `mask_t` arguments (no longer closing over globals).

Per time step the main loop reads each scenario's `zr3d`/`var3d` **once**, then
iterates over both transects to produce **two figures**:

```
./figs/snapshots/cs_diag_allscen-t0-YYYY-MM-DD-HH.png
./figs/snapshots/cs_diag_allscen-t1-YYYY-MM-DD-HH.png
```

---

## Session 2026-05-30 — `plot_cs_diag.py` cosmetics + density copy

### `plot/plot_cs_diag.py` — changes

#### Per-transect depth limits
Replaced the single `DEPTH_LIM` scalar with per-transect `DEPTH_LIM0` / `DEPTH_LIM1`
in the config block. `build_transect(eta0, xi0, slope, depth_lim)` now takes a
`depth_lim` arg and stores it in the transect dict as `tr['depth_lim']`. The plot
loop uses `tr['depth_lim']` for both the `keep` (`depth_mean >= tr['depth_lim']`)
filter and `ax.set_ylim`. Current values: t0 = −125 m, t1 = −90 m.

#### Figure cosmetics
- `plt.rcParams.update({'font.size': 14})` at top — all text (titles, axis labels,
  ticks, colorbar numbers + label) is 14 pt.
- `suptitle` reduced to just the timestamp (`YYYY-MM-DD HH:MM UTC`), `y=0.92` to
  pull it close to the top panel (default left a large white gap).
- Longitude x-axis: `ScalarFormatter(useOffset=False)` + `set_scientific(False)`
  removes the `-1.22e2` offset notation so real longitudes show; `MaxNLocator(5)`
  caps to ~5 ticks. Applied per-axis.

#### Colorbar — thin, full-height via `add_axes`
`fig.colorbar(pc, ax=axes.tolist(), ...)` with `fraction`/`aspect` made the bar
thin but too SHORT (aspect forces length = aspect×width). Replaced with an
explicit `add_axes` colorbar that spans the full panel stack:
```python
fig.canvas.draw()
pos_top = axes[0].get_position()
pos_bot = axes[-1].get_position()
cax = fig.add_axes([pos_top.x1 + 0.015, pos_bot.y0,
                    0.012, pos_top.y1 - pos_bot.y0])
fig.colorbar(pc, cax=cax, label=VAR_LABEL)
```
Width `0.012` (fig fraction) = thin; height = bottom-panel `y0` → top-panel `y1`
= full span. Does not steal space from panels. Same pattern as the other plot
scripts' `add_axes` colorbars.

#### Scenario labels
Added `LABELS` dict mapping scenario keys to readable panel titles; `ax.set_title`
uses `LABELS[name]` instead of the raw key:
| key | label |
|-----|-------|
| `notidesnowec` | no tides, no WEC |
| `tideswec` | tides, WEC |
| `tidesnowec` | tides, no WEC |
| `notideswec` | no tides, WEC |
| `ampwec` | no tides, 2.5x WEC |

### `plot/plot_cs_diag_rho.py` — NEW (density copy)

Copy of `plot_cs_diag.py` configured for density (`rho`, from his files):
- `VAR='rho'`, `VAR_SRC='his'`, `VAR_CMAP=cmocean.cm.dense`, `VMIN=23`, `VMAX=27`.
- Output prefix `cs_diag_rho_allscen-tN-...png` (distinct, won't overwrite the
  `w` figures).
- **Density reference offset:** stored `rho` is a deviation from `RHO_REF = 1027.4`.
  The script adds `RHO_OFFSET = RHO_REF - 1000 = 27.4` to the field so it plots
  density anomaly relative to 1000 kg m⁻³. Offset applied *before* the land-mask
  multiply (NaN + 27.4 = NaN, so land stays masked). Label: `ρ − 1000 (kg m⁻³)`.

All other geometry/cosmetics identical to `plot_cs_diag.py`.

---

## Session 2026-05-30 (continued) — `plot_map.py` transects + `plot_cs_diag_no3.py`

### `plot/plot_map.py` — transects added, reduced to single panel

#### Transect lines on mc60 map
`from scipy.ndimage import map_coordinates` added. A loop over both transects
(same `ETA0/XI0/SLOPE`, `ETA1/XI1/SLOPE1`, `LENGTH_XI = 250`, `N_PTS = 300` as
`plot_cs_diag.py`) interpolates onto `sf_lon`/`sf_lat` and plots each line on
the mc60 bathymetry panel: red = Transect 0, blue = Transect 1, filled circle
at the coast-side start point. Labels appear in the panel legend.

#### Reduced to single panel
The left panel (westc600 nested-grid overview) was removed. Only the mc60
bathymetry panel remains. Removed dataset loads for `smodenc`, `uswcnc`,
`westcnc` and all their data arrays. Changed to `plt.subplots(1,1)`,
`figw = 12`. All `ax.flat[1]` references replaced with `ax`. Left colorbar
removed.

#### Other settings
- `dpi=600` in `fig.savefig`.
- Gridlines kept as-is from the original script (`draw_labels=True`,
  `xlabels_top=False`, `ylabels_right=False`).

### `plot/plot_map_old.py` — gridline labels removed
The two gridline blocks (`gl0`, `gl1`) and all `set_xlabel`/`set_ylabel` calls
were removed. The two-panel layout and transect lines are unchanged.

### `plot/plot_cs_diag_no3.py` — NEW (NO3 copy)

Copy of `plot_cs_diag_rho.py` configured for NO3 (from bgc files):
- `VAR='NO3'`, `VAR_SRC='bgc'`, `VAR_CMAP=cmocean.cm.matter`, `VMIN=0`, `VMAX=25`.
- `RHO_OFFSET = 0.0` (no offset needed).
- `VAR_LABEL = r'NO$_3$ (mmol m$^{-3}$)'`.
- Output prefix `cs_diag_no3_allscen-tN-...png`.

All other geometry/cosmetics identical to `plot_cs_diag_rho.py`.

---

## Session 2026-06-02 — KE spectra expansion + plot_map_combine.py

### `plot/plot_map_combine.py` — new combined figure

New script combining the mc60 bathymetry map, a time-mean wave amplitude map, and
the wind/wave time series into one figure.

Layout: `GridSpec(2, 2, height_ratios=[3, 1])`
- Top-left (`ax1`): mc60 bathymetry (copy of `plot_map.py` content)
- Top-right (`ax2`): time-mean `Awave` from `mc60_wec_smooth_ramp_trim_sup.20190415.nc`,
  colormap `cmocean.cm.haline`, vmax = 99th percentile of ocean cells
- Bottom (full width): twin-axis wind speed / wave amplitude time series
  (replicates `plot_wind_twin.py`; orange wind left axis, navy wave amplitude right axis)

Both maps use identical `set_extent` bounds and `PlateCarree` projection so they
render at the same physical size. Colorbars placed via `fig.canvas.draw()` +
`fig.add_axes` to avoid shrinking panels. The time series axes are explicitly
resized to `[pos1.x0, …, pos2.x1 - pos1.x0, …]` after `canvas.draw()` so both
the primary and twin axes span exactly `ax1.x0 → ax2.x1`.

Legend: `bbox_to_anchor=(0.5, 1.01)`, `ncol=3`, `frameon=False` — one row above
the time series axes.

Output: `./figs/grid_combined.png`

---

### KE spectra — `notideswec` and `ampwec` added; new depth + wavenumber scripts

New scenarios added to all KE scripts:
| Key | Path | Notes |
|-----|------|-------|
| `notideswec` | `/data/project3/minnaho/swel/notides/mc60/wec/base/his/` | standard layout |
| `ampwec` | `/data/project3/minnaho/swel/notides/mc60/wec/ampwec/notrace/` | flat layout (no `his/` subdir) |

ROMS grid parameters confirmed from his file global attributes:
- `s_rho = 100`, `theta_s = 6.0`, `theta_b = 6.0`, `hc = 250.0`
- `u(time, s_rho, eta_rho, xi_u)`, `v(time, s_rho, eta_v, xi_rho)` — staggered in xi/eta

#### `plot/calc_ke_surf.py` — updated

Added `notideswec_files` and `ampwec_files` glob lists. Both scenarios processed
through the same `calculate_dataset_spectra` function. NPZ output extended with
`psd_notideswec_masknc`, `psd_notideswec_coastal`, `psd_ampwec_masknc`,
`psd_ampwec_coastal`. Existing keys unchanged.

#### `plot/calc_ke_depth.py` — NEW

Depth-averaged frequency spectrum using ROMS his files. Processes one `s_rho`
level at a time to keep memory manageable (~500 MB peak per level):

1. For each of 100 s_rho levels:
   - Load `u(time, s_lvl, eta_rho, xi_u)` and `v(time, s_lvl, eta_v, xi_rho)`
     across all files for the scenario
   - Interpolate to rho-points (average adjacent faces; boundaries padded)
   - `rfft` along time axis; accumulate masked PSD
2. Arithmetic mean over all 100 levels → depth-averaged spectrum
3. All 5 scenarios output to `ke_spectra_depth.npz`

Uses `scipy.fft.rfft` (one-sided) instead of complex FFT — no rotary
decomposition needed. Frequency array: `rfftfreq(n_time, d=dt_hours)`.

#### `plot/calc_ke_horiz_wavenumber.py` — NEW

Horizontal wavenumber spectrum following Hypolite et al. (2021). For each time step:
1. Load surface `u` and `v`, interpolate to rho-points
2. Subtract time-mean (computed from first scenario pass)
3. 2D `rfft2` on the horizontal field; form `E_2d = |û|² + |v̂|²`
4. Azimuthally average into 1D `E(k_h)` using `k_h = sqrt(kx²+ky²)` with
   logarithmically spaced bins
5. Accumulate time-average

Grid treated as approximately uniform: `dx = dy = 60 m` (true resolution of mc60).
Wavenumber range: 0 to `k_Nyquist = 1/(2×60 m)`. Output plotted vs wavelength
`λ = 1/k` in km for readability. All 5 scenarios → `ke_horiz_wavenumber.npz`.

**Approximation note:** the mc60 grid is curvilinear; treating it as Cartesian
introduces small spectral leakage, acceptable for slope comparison.

#### `plot/calc_ke_vert_wavenumber.py` — NEW

Vertical wavenumber spectrum from zslice his files (4 scenarios; no ampwec
zslice data yet). Uses the near-surface **uniform depth section only**:
zslice levels 0–50 (depth 0 to −50 m, dz = 1 m, 51 levels).

For each horizontal point and each time step, `rfft` of `u'(z)` and `v'(z)`
along depth (mean subtracted). Spatially averaged over ocean mask. Time-averaged.
Output: `ke_vert_wavenumber.npz`.

Vertical wavenumber range: 0 to `k_z,Nyquist = 0.5 m⁻¹` (λ_z from 2 m to 50 m).

#### Updated plot scripts

| Script | Change |
|--------|--------|
| `plot_energy_cascade.py` | Added `notideswec` (teal) and `ampwec` (magenta) lines |
| `plot_energy_cascade_coastal.py` | Same |
| `plot_ke.py` | Added `notideswec` and `ampwec` to all three figures |
| `plot_ke_depth.py` | NEW — mirrors `plot_energy_cascade.py` but reads `ke_spectra_depth.npz` |
| `plot_ke_wavenumber.py` | NEW — two-panel: horizontal k_h spectrum (left) + vertical k_z spectrum (right) |

### notidesampwec — tracer index fix confirmed (2026-06-03)

The `notidesampwec` (notrace) scenario had incorrect tracer indices in its BGC/dia
output. **Re-run with corrected tracers completed** — output is now valid.

### notideswec BGC — re-run in progress (2026-06-03)

`notideswec` BGC was blown up (see session 2026-05-19 blocking issue). A corrected
BGC re-run is currently in progress. Once complete, re-run `zslice_bgc.py` and
`profile_zslice.py` / `profile_zslice_coastal.py` for `notideswec`.

### Planned scenario — `tidesampwec`

A new scenario with tides + 2.5× wave amplitude (`tidesampwec`) is planned. All
KE, flux, and profile scripts will need this scenario added once output is available.

---

#### File map update

| File | Role |
|------|------|
| `plot/calc_ke_surf.py` | Frequency spectrum — surface; now 5 scenarios |
| `plot/calc_ke_depth.py` | Frequency spectrum — depth-averaged; 5 scenarios |
| `plot/calc_ke_horiz_wavenumber.py` | Horizontal wavenumber spectrum; 5 scenarios |
| `plot/calc_ke_vert_wavenumber.py` | Vertical wavenumber spectrum (0–50 m); 4 scenarios (no ampwec zslice) |
| `plot/plot_energy_cascade.py` | Plots `ke_spectra_comparison.npz` — full domain; 5 scenarios |
| `plot/plot_energy_cascade_coastal.py` | Same — coastal mask; 5 scenarios |
| `plot/plot_ke.py` | Rotary spectrum figures; 5 scenarios |
| `plot/plot_ke_depth.py` | NEW — depth-averaged frequency cascade |
| `plot/plot_ke_wavenumber.py` | NEW — horizontal + vertical wavenumber spectra |

---

## Session 2026-06-04

### Offshore flux `_norm` plot scripts — 100 m artifact fix

The `_fix` plot scripts showed an artifact at ~100 m: ~30 % of coastal-band
columns have `h_edge` ≈ 90–105 m (a shelf feature) and drop out of the spatial
average all at once, making the mean jump discontinuously.

**Fix:** restrict the spatial average to columns where `h_edge ≥ max_depth`
(the bottom of `DEPTH_YLIM` for the chosen tracer) so the sample is constant
at every depth level.  For NO3 (max_depth = 500 m) this retains 38 % of
columns (the deep shelf/canyon cells); for ptrace/rtrace (max_depth = 50 m)
all 1202 columns qualify.

Three new scripts written (read old-format NPZ directly, cache with `_norm_`
prefix):

| Script | Fix applied |
|--------|-------------|
| `plot/plot_offshore_flux_profile_zslice_norm.py` | h_edge filter; `nansum(F_tm)/n_valid` → `nanmean` over filtered columns |
| `plot/plot_offshore_flux_hov_zslice_norm.py` | vmin/vmax restricted to `DEPTH_YLIM` depth range only |
| `plot/plot_offshore_flux_hov_time_zslice_norm.py` | h_edge filter + vmin/vmax restriction |

### `postprocessing/zslice_ak.py` — NEW

Zslices `Akt` (thermal diffusivity) and `Akv` (viscosity) from his files onto
the standard 157-level z grid.  Both variables are on the `s_w` grid (101
w-level faces, not 100 s_rho centres) — verify zslice output depth coordinate
before downstream use.

- 5 scenarios; `ampwec` reads from `ampwec/everything` (flat layout)
- `ampwec` output → `zslicefull/notidesampwec/ak/`
- `notideswec` corrected to `wec/base/his`

### `plot/plot_profile_ak_npp.py` — NEW

Standalone profile script that reads directly from `zslicefull/<scenario>/dia/`
(TOT_PROD) and `zslicefull/<scenario>/ak/` (Akt, Akv) without requiring a
pre-computed NPZ.  Full-domain and coastal (10 km) panels per variable.
Scenarios that have no files for a given variable are silently skipped.
`VARS_AK` currently set to `[]` until `zslice_ak.py` finishes running for all
scenarios.

### `profile_zslice.py` + `profile_zslice_coastal.py` — ampwec added

- `SCENARIOS` list extended to include `'ampwec'`
- `SCEN_DIRS = {'ampwec': 'notidesampwec'}` maps label → zslice directory
- `if depth is not None` guard added to prevent overwriting depth arrays with
  `None` when his/bgc files are absent for a scenario
- Only `TOT_PROD_ampwec` will be written to the merged NPZ until his/bgc
  zslice files are generated for ampwec

### `plot/plot_profile_zslice.py` — ampwec added

Added `'ampwec'` to `SCENARIOS`, `LABELS` (`'no tides, 2.5× WEC'`), and
`COLORS` (`'magenta'`).  Panels for variables with no ampwec data are silently
skipped via the existing `if key not in full: continue` guard.

### `plot/plot_wno3_flux.py` — restructured

- Added envelope panel (shaded min/max, skip first 12 steps for spin-up)
  and peak-normalized PDF (`pdf / np.max(pdf)`) → 3-panel layout
- Further restructured to produce **3 separate figures** (one per tracer)
  in a single run: `wno3_flux.png`, `wptrace_flux.png`, `wrtrace_flux.png`
- Per-tracer config dict (`TRACERS`) specifies NPZ paths, ts key, scale
  factors, units, and output filename
- ptrace/rtrace use `ts_mean` from `w{tracer}_env_{name}.npz`
  (no separate flux NPZ exists for trace); scaled by `1e5` (ts) / `1e7` (PDF)
- Fixed broken ylabel LaTeX (`\langle` missing opening `$`, `_{x_y}` →
  `_{x,y}`, missing `(`)

### `postprocessing/calc_wno3_flux_20m.py` — zslicefull + new scenarios

Rewrote to use `zslicefull` paths (multi-level files) instead of the
non-existent `zslice_20m` single-depth directories:

- `TARGET_DEPTH = -20` — `get_depth_idx` finds the closest level in the
  file's depth array (index 20 in the 0–50 m every-1-m section, exact match)
- `load_masked` slices `[:, idx, :, :]` → `(time, eta, xi)`
- `scenarios` dict: `(zslicefull_dir, orig_his_dir)` tuples; `orig_his_dir`
  passed explicitly for `ocean_time` lookup (zslicefull files have no
  `ocean_time` variable)
- Added `notides_wec` (→ `zslicefull/notideswec`, orig `wec/base/his`) and
  `ampwec` (→ `zslicefull/notidesampwec`, orig `ampwec/everything`)
- Output: `wno3_flux_20m_{name}.npz` / `wno3_env_20m_{name}.npz`

### `ampwec/notrace` → `ampwec/everything` — global rename

All 11 scripts updated in one pass:

`plot_cs_diag.py`, `plot_cs_diag_no3.py`, `plot_cs_diag_rho.py`,
`plot_npp_depth_integrated.py`, `calc_ke_surf.py`, `calc_ke_depth.py`,
`calc_ke_horiz_wavenumber.py`, `zslice_npp.py`, `zslice_npp_ampwec.py`,
`zslice_ak.py`, `calc_wno3_flux_20m.py`

### `postprocessing/zslice_bgc.py` — ampwec/everything + notideswec path fix

- `notidesampwec` entry already pointed to `ampwec/everything`; added flat
  layout handling: `src_dir = scen_root if SCENARIO == 'notidesampwec' else
  os.path.join(scen_root, 'bgc')` (bgc files are flat in `everything/`)
- `notideswec` path corrected: `wec/output` → `wec/base`

### `postprocessing/zslice_uniform.py` — notidesampwec added

- Added `'notidesampwec'` to `SCENARIO_ROOTS` (→ `ampwec/everything`)
- Fixed stale `notideswec` path: `wec/output` → `wec/base`
- Added flat layout handling: `src_dir = scen_root if SCENARIO ==
  'notidesampwec' else os.path.join(scen_root, src_sub)` (his files are
  flat in `everything/`, no `his/` subdir)
- **Currently running** for `notidesampwec` (launched in screen, 2026-06-04)

### h_edge weighting fix scope — profile/NPP scripts unaffected

The `h_edge >= max_depth` filter and `nansum/n_valid` normalization applied
to the `_norm` offshore flux plot scripts do **not** affect:

- `plot_npp_depth_integrated.py` — integrates over the full 2D domain at
  each grid point; no coastal-edge 1D sampling
- `plot_profile_zslice.py` / `plot_profile_ak_npp.py` — use
  `nansum / wet_cell_count` over the 2D domain; wet-cell count decreases
  smoothly with depth (no discrete shelf-edge drop-off like the flux band)

### Pending / next steps

- `zslice_uniform.py` for `notidesampwec`: **running** (screen, 2026-06-04)
- Run `zslice_ak.py` for remaining 4 scenarios (tideswec, tidesnowec,
  notidesnowec, notideswec) once ready
- Run `zslice_bgc.py` for `notidesampwec` to generate BGC zslice files
- Re-run `profile_zslice.py` + `profile_zslice_coastal.py` after
  notidesampwec his/bgc zslice files exist to add all ampwec variables
- notideswec BGC re-run: when complete, re-run `zslice_bgc.py` and
  `profile_zslice.py` / `profile_zslice_coastal.py` for `notideswec`
- Update offshore flux scripts (postprocessing + plot) to add `notideswec`
  and `ampwec` (paths and scenario keys)

#### File map update

| File | Role |
|------|------|
| `plot/plot_offshore_flux_profile_zslice_norm.py` | NEW — profile with h_edge filter |
| `plot/plot_offshore_flux_hov_zslice_norm.py` | NEW — hov with depth-range vmin/vmax |
| `plot/plot_offshore_flux_hov_time_zslice_norm.py` | NEW — hov_time with h_edge filter |
| `postprocessing/zslice_ak.py` | NEW — zslice Akt/Akv from his files; 5 scenarios |
| `plot/plot_profile_ak_npp.py` | NEW — direct-read profile for TOT_PROD + Akt/Akv |
| `postprocessing/calc_wno3_flux_20m.py` | Rewrote to use zslicefull; 5 scenarios |
| `postprocessing/zslice_bgc.py` | Flat layout for notidesampwec; notideswec path fix |
| `postprocessing/profile_zslice.py` | Added ampwec; SCEN_DIRS mapping; depth None guard |
| `postprocessing/profile_zslice_coastal.py` | Same |
| `plot/plot_profile_zslice.py` | Added ampwec (magenta) |
| `plot/plot_wno3_flux.py` | 3-panel layout; 3 separate figures per tracer |
