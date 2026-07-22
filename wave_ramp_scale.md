# Wave Forcing Pipeline — ramp_scale.py

## Overview

`ramp_scale.py` produces the final ROMS-WEC wave forcing file from the spatially smoothed
WaveWatch3 output in a single pass. It consolidates what was previously done by several
separate steps:

| Old step | Tool | Now handled by |
|---|---|---|
| Exponential 3-day spin-up ramp | `ramp.py` | half-cosine ramp in script |
| Drop first 23 time steps | `ncks -d wwv_time,23,,1` | `trim_start` parameter |
| Fix wrong units on 8 variables | `ncatted` | `out_units` dict |
| Flip sign of set-down | `ncap2 -s sup=-1.0*sup` | sign flip before scaling |
| Coastal taper near land mask | `taper.py` | taper pass in script |
| 2.5× amplitude amplification | — | `amp_factor` parameter |

---

## Usage

Edit the three path variables at the top of the script, then run from the `swel/` directory:

```
python ramp_scale.py
```

**Paths**

| Variable | Description |
|---|---|
| `in_path` | Smoothed source file (`mc60_wec_smooth.20190415.nc`), opened read-only |
| `out_path` | Output file, created fresh each run |
| `grd_path` | ROMS grid file (`mc60_grd.nc`), used for bathymetry and land mask |

**Tunable parameters**

| Parameter | Default | Meaning |
|---|---|---|
| `trim_start` | 23 | Drop the first N time steps from the source (770 → 747) |
| `ramp_hours` | 72.0 | Duration of half-cosine spin-up (hours) |
| `amp_factor` | 2.5 | Final wave-amplitude multiplier at full ramp |
| `taper_H` | 20.0 | Depth contour (m) below which coastal taper is applied |

---

## Operations applied to each variable

For every scaled variable, operations are applied in this order:

1. **Trim** — read `src[trim_start:]`, so the output starts at the model's actual
   initial time rather than 11.5 h before it.
2. **Sign flip** — `sup` only: multiply by −1 so set-down is negative, matching
   ROMS-WEC sign convention.
3. **Ramp + amplitude scaling** — multiply by `(amp_factor · R(t))^power`, where
   `R(t)` is the half-cosine ramp from 0 at t = 0 to 1 at t = ramp_hours.
   At full ramp: power-1 variables reach ×2.5, power-2 variables reach ×6.25.
4. **Coastal taper** — for ocean grid cells with h < taper_H, multiply by
   `(h / taper_H)^power`. This smoothly attenuates wave fields toward zero at the
   land mask to prevent numerical instabilities at the boundary.
5. **Land mask preserved** — fill/masked values are restored after all scaling.

---

## Variable reference

### Scaled variables (`var_scales`)

| Variable | Amp. power | Notes |
|---|---|---|
| `Awave` | 1 | Wave amplitude; linear in A |
| `uorb`, `vorb` | 1 | Bottom orbital velocities; u_orb = A·σ/sinh(kh) |
| `sup` | 2 | Set-down; sign-flipped first; scales as A² (Longuet-Higgins) |
| `ust0`, `vst0` | 2 | Surface Stokes drift; ∝ A² |
| `ust2d`, `vst2d` | 2 | Depth-averaged Stokes drift; ∝ A² |
| `eb` | 2 | Breaking dissipation; approx. A²; true scaling ~ A²·f_br |
| `ed` | 2 | Bed friction dissipation; approx. A²; canonical scaling is A³ via u_orb³ |

### Coastal taper (`taper_vars`)

All variables in `var_scales` are tapered at the same amplitude power. Tapering all
variables at matching power keeps the wave physics internally consistent: if Awave → 0
near the coast, then dissipation and drift should also → 0.

### Pass-through variables (no scaling)

| Variable | Treatment |
|---|---|
| `Dwave` | Copied unchanged; direction is independent of amplitude |
| `Pwave` | Copied with lower clip at 2.0 s to prevent degenerate dispersion |
| `lmw` | Copied unchanged; wavelength is independent of amplitude |
| `qb` | Copied unchanged; not read by ROMS-WEC |

---

## Corrected units

The source file has incorrect units on several variables (all labelled "meter").
The output file writes the correct values:

| Variable | Correct units |
|---|---|
| `Pwave` | `seconds` |
| `uorb`, `vorb`, `ust0`, `vst0`, `ust2d`, `vst2d` | `m/s` |
| `eb`, `ed` | `m3/s3` |
| `qb` | ` ` (dimensionless) |
| `Awave`, `sup`, `lmw` | `meter` |
| `Dwave` | `degree` |

---

## Verification checklist

After running the script, confirm the following:

```python
import numpy as np
from netCDF4 import Dataset

src = Dataset("mc60_wec_smooth.20190415.nc")
dst = Dataset("mc60_wec_rampscale.20190415.nc")
```

1. **Time dimension**: `len(dst.dimensions['wwv_time'])` == 747
2. **Pwave lower bound**: `np.nanmin(dst.variables['Pwave'][:])` >= 2.0
3. **Units**: `ncdump -h mc60_wec_rampscale.20190415.nc` — check Pwave, uorb, eb
4. **Amplitude at full ramp** (deep-water grid point, t > 72 h):
   - `Awave_out / Awave_in[23:]` ≈ 2.5
   - `ust0_out / ust0_in[23:]` ≈ 6.25
5. **Ramp start**: all scaled vars near zero at t = 0 (first output step)
6. **sup sign**: `np.nanmean(dst.variables['sup'][:])` should be negative
7. **Coastal taper**: at a shallow grid cell (h < 20 m), check
   `Awave_out[:, j, i] / (2.5 * Rt)` ≈ `h[j,i] / 20` at full ramp
8. **Short ROMS run**: no blow-ups in KPP surface layer or surf zone
