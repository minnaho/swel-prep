# SWEL Plot Directory

Plotting and post-processing scripts for the SWEL (Surf-Wave-Estuary-Lagoon) mc60 ROMS simulation.

## Quick start

```bash
cd /data/project3/minnaho/project9copy/swel/plot

# List every registered script and its category:
python run_plots.py --list

# Preview what would run (no execution):
python run_plots.py --dry-run

# Run all scripts (4 parallel workers):
python run_plots.py

# Run only the fast PDF/spectral plots:
python run_plots.py --categories pdf

# Run specific scripts:
python run_plots.py --scripts plot_ke plot_energy_cascade

# Skip the heavy spectral computations:
python run_plots.py --exclude calc_ke_surf calc_ke_bot save_vort

# 8 workers, 1-hour timeout per script, print all output:
python run_plots.py --workers 8 --timeout 3600 --verbose
```

## Script categories

| Category   | Scripts | Description |
|------------|---------|-------------|
| `spectral` | `calc_ke_surf`, `calc_ke_bot`, `save_vort` | Heavy computation; read history files, write derived NetCDF/NPZ |
| `pdf`      | `plot_ke`, `plot_energy_cascade*`, `plot_pdf_*` | Fast; read pre-computed NPZ files, write static PNGs |
| `map`      | `plot_depth`, `plot_blowups*`, `plot_map` | Static domain/bathymetry maps using cartopy |
| `snapshot` | `plot_cs_*`, `plot_surf_*`, `plot_vorticity`, `plot_wind_amp` | Time-loop over history files; write one PNG per time step |
| `util`     | `make_mask`, `plot_pipes`, `plot_rivers` | Mask generation or incomplete exploratory scripts |

### Execution order for a full run

```
spectral → pdf → snapshot/map
```

`calc_ke_surf` must finish before `plot_ke` / `plot_energy_cascade` because those read `ke_spectra_comparison.npz` that `calc_ke_surf` writes.

## Simulation scenarios

All multi-scenario scripts reference four ROMS runs in this order:

| Label          | Tides | WEC | Base path |
|----------------|-------|-----|-----------|
| `tideswec`     | yes   | yes | `/data/project3/minnaho/swel/tides/mc60/wec/` |
| `tidesnowec`   | yes   | no  | `/data/project3/minnaho/swel/tides/mc60/nowec/output/` |
| `notidesnowec` | no    | no  | `/data/project3/minnaho/swel/notides/mc60/nowec/` |
| `notideswec`   | no    | yes | `/data/project3/minnaho/swel/notides/mc60/wec/rerun/` |

History files follow the glob pattern: `<scenario_dir>/his/mc60_his.*.nc`

## Key local files

| File | Purpose |
|------|---------|
| `mc60_grd.nc` | mc60 ROMS grid (lat/lon, mask_rho, h, f) |
| `sfshelf60_grd.nc` | Outer shelf grid (used by `plot_map.py`) |
| `coastal_mask.nc` | 10 km coastal boolean mask (written by `make_mask.py`) |
| `ke_spectra_comparison.npz` | Pre-computed KE spectra (written by `calc_ke_surf.py`) |
| `figs/` | All output figures land here |

## External dependencies

| Library | Used by |
|---------|---------|
| `numpy`, `scipy` | All computation scripts |
| `netCDF4` | All scripts that read/write NetCDF |
| `matplotlib`, `cmocean` | All plotting scripts |
| `cartopy` | Map scripts (`plot_map`, `plot_blowups*`, `plot_depth`) |
| `pyfuncs` | Spectral scripts (from `/data/project3/minnaho/global/`) |
| `ROMS_depths` | Snapshot scripts needing depth levels |

`pyfuncs` and `ROMS_depths` are in `/data/project3/minnaho/global/` — each script that needs them appends that path via `sys.path.append(...)`.

## Output structure

```
figs/
├── *.png                    # static map and PDF figures
├── snapshots/
│   ├── ptrace/              # surf_ptrace-YYYY-MM-DD-HH.png
│   ├── rtrace/
│   └── ...
└── ...
```

## run_plots.py reference

```
usage: run_plots.py [-h]
                    [--scripts SCRIPT [SCRIPT ...]]
                    [--categories CAT [CAT ...]]
                    [--exclude SCRIPT [SCRIPT ...]]
                    [--workers N] [--timeout SEC] [--python PATH]
                    [--dry-run] [--list] [--verbose]
```

| Flag | Default | Notes |
|------|---------|-------|
| `--scripts` | all | Explicit whitelist of script stems |
| `--categories` | all | Filter by category |
| `--exclude` | none | Scripts to skip after the above filters |
| `--workers` | 4 | Parallel subprocesses |
| `--timeout` | none | Seconds before a script is killed |
| `--python` | `sys.executable` | Use a different interpreter |
| `--dry-run` | off | Preview selection without running |
| `--list` | off | Print registry and exit |
| `--verbose` | off | Show stdout/stderr even for successes |

`MPLBACKEND=Agg` is injected into every child process so scripts run without a display.

## Adding a new script

1. Write `my_new_plot.py` in this directory (keep CWD-relative paths like `./figs/...`).
2. Add an entry to the `SCRIPTS` dict in `run_plots.py`:
   ```python
   "my_new_plot": ("pdf", "One-line description of what it plots"),
   ```
3. Verify it appears: `python run_plots.py --list`

## Notes

- All scripts assume they are run from this directory (`swel/plot/`). `run_plots.py` enforces this via `cwd=PLOT_DIR`.
- Snapshot scripts (`plot_surf_*`, `plot_cs_*`) loop over every history file and can take hours end-to-end. Use `--timeout` if you need a wall-clock guard.
- `edit_mask.py` opens an interactive matplotlib GUI and cannot be run via `run_plots.py`; launch it directly.
- `plot_cs_front_transect.py` is headless by default (runs fine via `run_plots.py`) and produces, per scenario: a locator map (surface temp + all transect lines) and one cached cross-section figure per variable (temp/vort/NO3/w/Akt/Akv/phytoC) with all of that scenario's transects stacked as rows. Pass `--preview` to instead pop up the old interactive transect-placement window (Enter/Esc to proceed/cancel) before plotting; that mode needs a working `DISPLAY`, same as `edit_mask.py`.

## Standalone analysis scripts (not registered in run_plots.py)

A large batch of one-off/derived scripts have accumulated in `plot/` and `../postprocessing/`, launched directly rather than through `run_plots.py`:

- **WEC/tide RMSE & std diagnostics vs. base case** (~20 scripts, plot+postprocessing): `calc_{w,vort,dudz,drhodz}_rmse_wec_{shelf,offshore}.py` compute RMSE(depth)/RMS(depth)/std(depth) profiles of w, zeta/f, du/dz, and drho/dz against the `notidesnowec` baseline across three comparisons (WEC alone, tides alone, tides+WEC together), testing whether WEC amplifies tidal-bore-driven variability rather than just adding its own signature. `plot_{w,vort}_rmse_wec_{shelf,offshore}[_std].py` render them, with `_std` variants overlaying Δstd to separate "more variable" from "phase-shifted." `plot_rmse_std_grid.py` combines eight of these into one 2×4 labeled figure.
- **Boundary-layer depth** (2 calc + 1 plot): `calc_bl_depth_{sbl,bbl}.py` diagnose surface/bottom boundary-layer depth via an Akt-threshold criterion (SBL from zsliced Akt, BBL from raw native-grid Akt since the zsliced grid is too coarse below -300m); `plot_bl_depth.py` maps both as 3×2 cartopy grids.
- **NO3 flux at multiple depths/domains** (~10 calc + ~6 plot): `calc_wno3_flux_{10m,20m,30m}_100m.py`/`_offshore.py`/`_100m_daily.py` compute the resolved eddy flux w'NO3' restricted to shelf, offshore, or daily-decomposed windows; `calc_akt_dno3dz_{20m,30m}_100m.py` compute the parameterized diffusive counterpart. Corresponding `plot_wno3_flux_100m.py`/`_20m_offshore.py`/`_20m_100m_daily.py` render them in the standard 4-panel (time series/envelope/PDF/box) layout.
- **Shelf-vs-offshore & coastal depth profiles**: `profile_zslice_bgcdia_100m_offshore.py`/`profile_zslice_par_offshore.py` (postprocessing) and `plot_profile_zslice_{100m,100m_coastal,shelf_offshore,shelf_offshore_4}.py`/`plot_profile_zslice_bgcdia_shelf_offshore.py`/`plot_profile_drhodz_coastal.py` (plot) — time/horizontally-averaged depth profiles split by shelf/offshore/coastal domain, for zsliced variables generally and bgcdia limitation/uptake diagnostics specifically.
- **Single-instant snapshot cross-sections**: `plot_cs_{pv,vorticity,wno3_aktdno3dz}_snap.py` plot PV, normalized vorticity, and w'NO3'/Akt·dNO3/dz at one matched timestep across scenarios.
- **`boxavg/` subsystem** (18 scripts + `cs_boxavg.py` helper): perpendicular-box-averaged variants of the `plot_cs_*.py` family — averages over a ~1.2 km box straddling each transect line instead of sampling one grid line, on a fixed depth grid via `boxavg_section`/`boxavg_section_fixedz`. Includes `plot_cs_wno3_aktdno3dz_avg_diff_box_3x2.py`, a box+time-averaged 3x2 scenario-diff figure for w'NO3'/-Akt·dNO3/dz. The bgcdia family was split by cost tier: `plot_cs_diag_bgcdia_box_rerun4.py` (fast, 4-timestep rerun window) and `_box_alltime.py` (adds PAR/TOT_PROD from each scenario's full record) replace the old single `plot_cs_diag_bgcdia_box.py`; `plot_cs_diag_bgcdia_avg_diff_box_3x2.py` adds a time-mean, box-averaged, 4-scenario diff figure for the same diagnostics.
- **Misc**: restricted-window KE spectra (`calc_ke_surf_20190421_20190423.py`, `plot_energy_cascade_20190421_20190423.py`); raw-sigma surface snapshot maps (`plot_surf_no3_phytoc_zslice.py`, `plot_surf_rtrace_ptrace_zslice.py`); smode200 parent-vs-child w'NO3' comparisons (`plot_smode_wno3_{10m,20m}.py`); 4-scenario Hovmöller variant; launcher scripts (`run_all_screen.sh`, `run_cs_replots.sh`, `run_ncra_means.sh`, `run_par_only.sh`).
