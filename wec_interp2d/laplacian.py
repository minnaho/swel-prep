import numpy as np
from netCDF4 import Dataset

# -----------------------------
# Files
# -----------------------------
input_file = "../tides/mc60/frc/mc60_wec_smooth_taper.20190415.nc"
grid_file  = "../tides/mc60/mc60_grd_fill.nc"
output_file = "./mc60_wec_smooth_taper_laplacian.nc"

# -----------------------------
# Open files
# -----------------------------
src = Dataset(input_file, "r")
grd = Dataset(grid_file, "r")
dst = Dataset(output_file, "w", format="NETCDF4")

# -----------------------------
# Read grid variables
# -----------------------------
mask = grd.variables["mask_rho"][:].astype(np.int8)   # (eta, xi)
pm   = grd.variables["pm"][:]                         # 1/dx
pn   = grd.variables["pn"][:]                         # 1/dy

# -----------------------------
# Metric-aware Laplacian
# -----------------------------
def laplacian_rho(field, mask, pm, pn):
    """
    field: (eta, xi)
    mask, pm, pn: (eta, xi)
    """

    # Enforce land = 0
    f = field * mask

    # Pad with edge values (zero-gradient BCs)
    fpad  = np.pad(f, 1, mode="edge")
    mpad  = np.pad(mask, 1, mode="edge")
    pmpad = np.pad(pm, 1, mode="edge")
    pnpad = np.pad(pn, 1, mode="edge")

    # Metric terms at half points
    pm2_ip = 0.5 * (pmpad[1:-1, 1:-1]**2 + pmpad[1:-1, 2:]**2)
    pm2_im = 0.5 * (pmpad[1:-1, 1:-1]**2 + pmpad[1:-1, :-2]**2)

    pn2_jp = 0.5 * (pnpad[1:-1, 1:-1]**2 + pnpad[2:, 1:-1]**2)
    pn2_jm = 0.5 * (pnpad[1:-1, 1:-1]**2 + pnpad[:-2, 1:-1]**2)

    # Second derivatives
    d2dx2 = (
        pm2_ip * (fpad[1:-1, 2:] - fpad[1:-1, 1:-1]) -
        pm2_im * (fpad[1:-1, 1:-1] - fpad[1:-1, :-2])
    )

    d2dy2 = (
        pn2_jp * (fpad[2:, 1:-1] - fpad[1:-1, 1:-1]) -
        pn2_jm * (fpad[1:-1, 1:-1] - fpad[:-2, 1:-1])
    )

    lap = d2dx2 + d2dy2

    # Require all 5 points to be ocean
    ocean5 = (
        mpad[1:-1, 1:-1] &
        mpad[1:-1, 2:] &
        mpad[1:-1, :-2] &
        mpad[2:, 1:-1] &
        mpad[:-2, 1:-1]
    )

    lap[~ocean5] = 0.0

    return lap

# -----------------------------
# Copy dimensions
# -----------------------------
for name, dim in src.dimensions.items():
    dst.createDimension(name, len(dim) if not dim.isunlimited() else None)

# -----------------------------
# Copy time variable
# -----------------------------
t_in = src.variables["wwv_time"]
t_out = dst.createVariable("wwv_time", t_in.datatype, t_in.dimensions)
t_out[:] = t_in[:]
t_out.setncatts({a: t_in.getncattr(a) for a in t_in.ncattrs()})

# -----------------------------
# Loop over variables
# -----------------------------
for varname, var in src.variables.items():

    if var.dimensions == ("wwv_time", "eta_rho", "xi_rho"):

        print(f"Processing {varname}")

        out = dst.createVariable(
            f"{varname}_laplacian",
            "f4",
            var.dimensions,
            zlib=True,
            complevel=4
        )

        out.long_name = f"Laplacian of {getattr(var, 'long_name', varname)}"
        out.units = f"{getattr(var, 'units', '')} / meter^2"

        nt = var.shape[0]

        for t in range(nt):
            field = var[t, :, :].astype(np.float64)
            out[t, :, :] = laplacian_rho(field, mask, pm, pn)

# -----------------------------
# Global attributes
# -----------------------------
dst.Title = "ROMS metric-aware Laplacian (rho grid)"
dst.Source = input_file
dst.Grid = grid_file

# -----------------------------
# Close files
# -----------------------------
src.close()
grd.close()
dst.close()

print("Finished writing:", output_file)

