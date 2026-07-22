import numpy as np
from netCDF4 import Dataset

# ── Configuration ──────────────────────────────────────────────────────────────
in_path    = "./tides/mc60/frc/mc60_wec_smooth.20190415.nc"      # source (read-only)
out_path   = "./tides/mc60/frc/mc60_wec_rampscale.20190415.nc"   # created fresh
grd_path   = "./mc60_grd.nc"

trim_start = 23       # drop first N time steps (770 → 747)
ramp_hours = 72.0     # half-cosine spin-up duration (hours)
amp_factor = 2.5      # final amplitude multiplier at full ramp
taper_H    = 20.0     # taper coastal cells shallower than this depth (m)

# Ramp/scale power relative to wave amplitude A.
# Variables not listed are passed through unchanged (Dwave, Pwave, lmw, qb).
var_scales = {
    "Awave": 1,
    "uorb":  1, "vorb":  1,
    "sup":   2,           # sign-flipped before scaling (set-down is negative)
    "ust0":  2, "vst0":  2,
    "ust2d": 2, "vst2d": 2,
    "eb":    2,           # approximate: true scaling ~ A^2 * f_br
    "ed":    2,           # approximate: canonical scaling is A^3 via u_orb^3
}

# Coastal taper: multiply by (h/taper_H)^power for h < taper_H.
# All vars in var_scales are tapered at the same power for internal consistency.
taper_vars = {
    "Awave": 1,
    "uorb":  1, "vorb":  1,
    "sup":   2,
    "ust0":  2, "vst0":  2,
    "ust2d": 2, "vst2d": 2,
    "eb":    2, "ed":    2,
}

# Corrected units for each output variable
out_units = {
    "Awave":    "meter",
    "Dwave":    "degree",
    "Pwave":    "seconds",
    "uorb":     "m/s",  "vorb":  "m/s",
    "ust0":     "m/s",  "vst0":  "m/s",
    "ust2d":    "m/s",  "vst2d": "m/s",
    "sup":      "meter",
    "eb":       "m3/s3",
    "ed":       "m3/s3",
    "qb":       " ",
    "lmw":      "meter",
    "wwv_time": "days",
}


def half_cosine_ramp(t_sec, ramp_hours):
    Tr = ramp_hours * 3600.0
    x  = np.clip(t_sec / Tr, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))   # 0 → 1


# ── Bathymetry for coastal taper ───────────────────────────────────────────────
with Dataset(grd_path, 'r') as grd:
    h    = np.array(grd.variables['h'])
    mask = np.array(grd.variables['mask_rho'])

tj, ti     = np.where((h < taper_H) & (mask == 1))
taper_frac = h[tj, ti] / taper_H              # (N_shallow,), values in (0, 1)


# ── Build output ───────────────────────────────────────────────────────────────
with Dataset(in_path, 'r') as src, \
     Dataset(out_path, 'w', format='NETCDF4_CLASSIC') as dst:

    print(f"[read]  {in_path}")
    print(f"[write] {out_path}\n")

    # Trimmed time and ramp
    t_days = np.array(src.variables['wwv_time'][trim_start:])   # (747,)
    t_sec  = (t_days - t_days[0]) * 86400.0
    Rt     = half_cosine_ramp(t_sec, ramp_hours)
    Rt3d   = Rt[:, None, None]                                   # (747, 1, 1)

    # Dimensions
    neta = len(src.dimensions['eta_rho'])
    nxi  = len(src.dimensions['xi_rho'])
    dst.createDimension('wwv_time', None)
    dst.createDimension('eta_rho',  neta)
    dst.createDimension('xi_rho',   nxi)

    # Global attributes
    dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

    # wwv_time
    tv           = dst.createVariable('wwv_time', 'f4', ('wwv_time',))
    tv.long_name = src.variables['wwv_time'].long_name
    tv.units     = out_units['wwv_time']
    tv[:]        = t_days.astype('f4')

    # 3-D variables
    for name in (v for v in src.variables if v != 'wwv_time'):
        sv  = src.variables[name]
        raw = sv[trim_start:, :, :]          # masked array from netCDF4

        # work in float64 with NaN standing in for land/fill cells
        arr     = raw.filled(np.nan).astype(np.float64)
        nanmask = np.isnan(arr)

        if name in var_scales:
            power = var_scales[name]

            if name == 'sup':
                arr = -arr                   # sign fix: set-down is negative

            arr *= (amp_factor * Rt3d) ** power

            if name in taper_vars:
                arr[:, tj, ti] *= taper_frac ** taper_vars[name]

        elif name == 'Pwave':
            arr = np.maximum(arr, 2.0)       # prevent degenerate dispersion

        arr[nanmask] = np.nan
        out_data = np.ma.array(arr, mask=nanmask)

        fv = getattr(sv, '_FillValue', None)
        kw = {'fill_value': fv} if fv is not None else {}
        dv           = dst.createVariable(name, 'f4',
                                          ('wwv_time', 'eta_rho', 'xi_rho'), **kw)
        dv.long_name = sv.long_name
        dv.units     = out_units.get(name, getattr(sv, 'units', ''))
        dv[:]        = out_data.astype('f4')

        tag = ('×(%.4g·Rt)^%d' % (amp_factor, var_scales[name])
               if name in var_scales else 'passthrough')
        print(f"[done] {name:6s}  {tag}")

    print(f"\nWrote {len(t_days)} time steps to {out_path}")
