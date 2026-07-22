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
| `notidesnowec` | no    | no  | `/data/project3/minnaho/swel/notides/mc60/nowec/output/` |
| `notideswec`   | no    | yes | `/data/project3/minnaho/swel/notides/mc60/wec/output/` |

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
