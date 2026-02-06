import numpy as np
from netCDF4 import Dataset

wave_path = "./tides/mc60/frc/mc60_wec_smooth_ramp_trim.20190415.nc"
varlist   = ["Awave","ust2d","vst2d","ust0","vst0","sup"]
ramp_hours = 72.0  # try 3648 if needed

def half_cosine_ramp(t_sec, ramp_hours):
    Tr = ramp_hours * 3600.0
    x = np.clip(t_sec / Tr, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))  # 0 -> 1 smoothly

with Dataset(wave_path, "r+") as nc:
    # time: days since 1995-01-01
    t_days = nc.variables["wwv_time"][:]
    t_sec  = (t_days - t_days[0]) * 86400.0
    Rt     = half_cosine_ramp(t_sec, ramp_hours)       # (Lt,)
    Rt3d   = Rt[:, None, None]                         # broadcast to (Lt,1,1)

    for name in varlist:
        if name not in nc.variables:
            print(f"[skip] {name} not found")
            continue

        v = nc.variables[name]
        data = v[:]
        if data.ndim != 3:
            print(f"[skip] {name}: expected (t,y,x), got {data.shape}")
            continue

        # respect fill values / masks
        fill = getattr(v, "_FillValue", None)
        if np.ma.isMaskedArray(data):
            arr = data.filled(np.nan).astype(np.float64, copy=False)
            nanmask = np.isnan(arr)
            arr *= Rt3d
            arr[nanmask] = np.nan
            v[:] = np.ma.array(arr, mask=np.isnan(arr))
        else:
            arr = data.astype(np.float64, copy=False)
            if fill is not None:
                nanmask = (arr == fill)
                arr = arr.copy()
                arr[~nanmask] *= Rt3d[~np.isnan(Rt3d)] if Rt3d.shape != arr.shape else Rt3d
                arr[nanmask] = fill
            else:
                # plain numeric with possible NaNs
                nanmask = np.isnan(arr)
                arr *= Rt3d
                arr[nanmask] = np.nan
            v[:] = arr

    print(f"[done] Applied {ramp_hours:.0f} h time-only ramp using wwv_time (days since 1995-01-01).")


