# swel-prep

## Forcing / prep files

Prepared forcing files using a combination of:
- `./prep_scripts/` — see readme there
- `/data/project1/minnaho/roms-tools/bgc/make_s2r_bgcSRF.m`

Initial file created in `bryini_bgcnest/` in this directory.

Created WEC from SMODE 200 m WEC (`/data/project1/minnaho/roms-tools/bry_ini/*wec*`),
then used `ramp.py` to ramp up the WEC over time.

Created river file from the SMODE 200 m river file using nco commands
(`mc60_river_commands.txt`).

Interpolated river positions and fractionated in `rivers/make_rivers_grid.py`.

Created pipe file from Marco's `.mat` file:
- `/data/project1/minnaho/roms-tools/rivers/ROMS_CC_new.mat`
- `/data/project1/minnaho/roms-tools/rivers/potw_grd.py`
- `/data/project1/minnaho/roms-tools/rivers/potw_pip.py`

Created grid bathymetry from `monterey_13_mhw_2012.nc` using
`/data/project9/minnaho/ucla-tools/bathy/topo.m`.

## Scenarios

Nearly everything in `postprocessing/` and `plot/` compares the same 6 run
configurations (tides on/off × WEC amplitude):

| Scenario key | Tides | WEC | Notes |
|---|---|---|---|
| `tideswec` | yes | 1x | |
| `tidesnowec` | yes | none | |
| `notidesnowec` | no | none | baseline for most diff/% change comparisons |
| `notideswec` | no | 1x | |
| `ampwec` / `notidesampwec` | no | 2.5x | **same run**, two names — raw-file scripts use `ampwec`, zslice-dir scripts use `notidesampwec` |
| `tidesampwec` | yes | 2.5x | added later than the other 5; some scripts still lack it |

## Postprocessing (`postprocessing/`)

Aggregates raw ROMS history/bgc output into the smaller derived products the `plot/`
scripts read. Grouped by purpose (not every file is listed):

- **zslice pipeline** (`zslice_uniform.py`, `zslice_ak.py`, `zslice_bgc.py`,
  `zslice_npp.py`, `zslice_init.py`, `zslice_dia_avg_rerun.py`, `zslice_w_no3.py`,
  `zslice_trace.py`) — regrids raw sigma-coordinate (`s_rho`) output onto a fixed
  157-level depth grid, since averaging/differentiating is only straightforward on a
  grid that doesn't move with `zeta` each timestep. Output lands under
  `/data/project1/minnaho/swel/zslicefull/<scenario>/`.
- **domain-averaged depth profiles** (`profile_zslice_par.py` and its `_100m`
  h≤100m-restricted, `_bgcdia`, and `_scen` variants) — fully parallel, one job per
  scenario × filetype, merged into `zslice_profiles*.npz`; full-domain, coastal, and
  bathymetry depth-zone masks.
- **PDFs** (`calc_vort_pdf.py` / `_100m`, `calc_pv_pdf*.py`) — normalized
  vorticity (ζ/f) and potential-vorticity distributions from raw history files.
- **fluxes** (`offshore_flux*.py` family, `calc_wno3_flux_10m.py`/`_20m.py`,
  `calc_wtrace_flux.py`) — cross-shelf tracer/water flux calculations.
- **misc**: `save_vort_surf.py`/`save_vort_bot.py` (surface/bottom vorticity dumps),
  `add_scoord_attrs.py` (patches s-coordinate global attrs onto ini files so `zslice`
  can read them), `ncks_extract.py`.

## Plotting (`plot/`)

All plotting/derived-figure scripts, driven by `run_plots.py` (see `plot/CLAUDE.md`
for the full registry and usage — `python run_plots.py --list`,
`--categories {spectral,pdf,map,snapshot,util}`, `--dry-run`, etc.). Categories:

| Category | What it covers |
|---|---|
| `spectral` | Heavy computation reading history files, writing derived NetCDF/NPZ (e.g. `calc_ke_surf.py`, `calc_ke_10m.py`/`_50m.py`, `save_vort.py`) |
| `pdf` | Fast plots reading pre-computed NPZ, writing static PNGs (`plot_ke*.py`, `plot_energy_cascade*.py`, `plot_pdf_*.py`, `plot_profile_*.py`) |
| `map` | Static domain/bathymetry maps (`plot_map*.py`, `plot_depth.py`) |
| `snapshot` | Time-loop scripts writing one PNG per timestep — can take hours (`plot_cs_diag*.py`, `plot_surf_*.py`, `plot_vorticity.py`) |
| `util` | Mask generation / exploratory (`make_mask.py`, `plot_pipes.py`, `plot_rivers.py`) |

Major analysis families:
- **cross-sections** — `plot_cs_diag*.py` (per-variable, per-timestep transect
  snapshots), `plot_cs_diag_avg_diff.py` / `plot_cs_diag_drhodz_diff.py` (time-averaged,
  differenced against `notidesnowec`, 3 transects: south/north diagonal + a fixed-eta
  "mid" section)
- **front transects** — `plot_cs_front_transect.py`: per-scenario, user-placed
  diagonal transects (own geometry per scenario, since the front's location/shape
  differs by run) at one fixed instant. Per scenario: a locator map (surface temp +
  all transect lines) plus one cached figure per variable (temp, correctly-rotated
  ζ/f, NO3, w, Akt, Akv, total phyto C) stacking all of that scenario's transects.
  Headless by default; `--preview` pops up the old interactive placement window.
- **KE spectra** — `calc_ke_surf.py`/`_10m.py`/`_50m.py` (complex-velocity FFT power
  spectra by depth) + `plot_energy_cascade*.py`; `calc_ke_horiz_wavenumber.py`/
  `_vert_wavenumber.py` + `plot_ke_wavenumber.py` (Hypolite et al. 2021 convention)
- **Hovmöller diagrams** — `plot_hov_transect_zslice.py` (zsliced) and `_raw.py`/
  `_raw_bgcdia.py` (native s_rho) siblings
- **NPP / productivity** — `plot_npp_depth_integrated.py`, `plot_profile_ak_npp.py`
- **vorticity** — `plot_vorticity.py` (time series), `plot_vorticity_snap.py`
  (single fixed-instant 2×2 scenario comparison), `plot_pdf_vort*.py`

## Conventions worth knowing

- `plot/scenario_style.py` is the single source of truth for per-scenario
  color/linestyle/label/linewidth — import it rather than redefining styling per script.
- zsliced output (fixed depth grid, same z-levels every timestep) vs. raw `s_rho`
  output (grid moves with `zeta`) — averaging and differentiation only commute on the
  former; scripts that need `d/dz` of a time-mean field read zsliced input for exactly
  this reason.
- `ampwec` (raw-file scripts) and `notidesampwec` (zslice-dir scripts) are the same
  simulation under two different key names — see the Scenarios table above.
- Generated output (`*.nc`, `*.npz`, `*.png`, `*.gif`, `__pycache__/`, per-job
  `log_*.txt`) is gitignored — only source scripts and small config/docs are tracked.
