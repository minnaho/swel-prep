import netCDF4 as nc
import numpy as np

infile  = "mc60_riv.nc"
outfile = "mc60_riv_trace.nc"

# Open input file
src = nc.Dataset(infile, "r")

# Create output file
dst = nc.Dataset(outfile, "w", format="NETCDF4")

# -----------------------
# Copy dimensions
# -----------------------
for name, dim in src.dimensions.items():
    if name == "ntracers":
        dst.createDimension("ntracers", 33)
    else:
        dst.createDimension(name, len(dim))

# -----------------------
# Copy variables
# -----------------------
for name, var in src.variables.items():
    if name != "river_tracer":
        out = dst.createVariable(name, var.dtype, var.dimensions)
        out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
        out[:] = var[:]

# -----------------------
# Create new river_tracer
# -----------------------
rt_src = src.variables["river_tracer"]

rt_dst = dst.createVariable(
    "river_tracer",
    rt_src.dtype,
    ("river_time", "ntracers", "nriver"),
    zlib=True
)

rt_dst.setncatts({k: rt_src.getncattr(k) for k in rt_src.ncattrs()})

# Initialize everything to zero
rt_dst[:] = 0.0

# -----------------------
# Preserve tracers 0 and 1
# -----------------------
rt_dst[:, 0, :] = rt_src[:, 0, :]
rt_dst[:, 1, :] = rt_src[:, 1, :]

# -----------------------
# Shift tracers 230 → 432
# -----------------------
rt_dst[:, 4:33, :] = rt_src[:, 2:31, :]

# -----------------------
# Set new tracer meanings
# -----------------------
# tracer index 2 = river tracer
rt_dst[:, 2, :] = 1.0

# tracer index 3 = pipe tracer
rt_dst[:, 3, :] = 0.0

# -----------------------
# Copy global attributes
# -----------------------
dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

# Close files
src.close()
dst.close()

print("Wrote:", outfile)

