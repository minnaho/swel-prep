# see ./tides/mc60/readme_tracer
# add ptrace and rtrace to rst file

import netCDF4 as nc
import numpy as np
import glob
import os
import shutil

# Input pattern for restart files
file_pattern = "/data/project3/minnaho/swel/tides/mc60/ampwec/rst/mc60_rst.20190418230120.???.nc"

# Loop over all matching files
for infile in glob.glob(file_pattern):
    print(f"Processing {infile}...")

    # Construct output filename
    dirname, basename = os.path.split(infile)
    outfile = os.path.join(dirname, basename.replace("mc60_rst", "mc60_rst_trace"))
    print(f"  Writing to {outfile}...")

    # Copy the original file to the new one first (preserves everything)
    shutil.copy(infile, outfile)

    # Open new file in append mode
    with nc.Dataset(outfile, "a") as ds:

        # Define new variables if they don't exist
        for var_name in ["ptrace", "rtrace"]:
            if var_name not in ds.variables:
                dims = ("time", "s_rho", "eta_rho", "xi_rho")
                new_var = ds.createVariable(var_name, "f8", dims, zlib=True)

                # Copy attributes from temp if available
                if "temp" in ds.variables:
                    temp_var = ds.variables["temp"]
                    new_var.setncatts({k: temp_var.getncattr(k) for k in temp_var.ncattrs()})

                # Override specific attributes
                if var_name == "ptrace":
                    new_var.long_name = "pipe tracer"
                elif var_name == "rtrace":
                    new_var.long_name = "river tracer"

                new_var.units = "nondimensional"

                # Initialize to zeros
                new_var[:] = 0.0
                print(f"  Created {var_name} with zeros.")

    print(f"Done: {outfile}\n")

