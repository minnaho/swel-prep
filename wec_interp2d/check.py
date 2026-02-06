import numpy as np
from netCDF4 import Dataset

def check_netcdf_missing(fname):
    with Dataset(fname, 'r') as ds:
        print(f"\nChecking file: {fname}\n")

        for name, var in ds.variables.items():

            # Skip non-numeric variables (e.g. time as char)
            if not np.issubdtype(var.dtype, np.number):
                continue

            data = var[:]  # returns masked array if fill values exist

            n_total = data.size

            # Masked (fill/missing values)
            n_masked = np.ma.count_masked(data)

            # NaNs (only relevant for floats)
            if np.issubdtype(data.dtype, np.floating):
                n_nan = np.isnan(data.data).sum()
            else:
                n_nan = 0

            print(f"{name:15s} | "
                  f"masked: {n_masked:8d} | "
                  f"NaNs: {n_nan:8d} | "
                  f"total: {n_total:10d}")

if __name__ == "__main__":
    check_netcdf_missing("../tides/mc60/frc/mc60_wec_smooth_taper.20190415.nc")


