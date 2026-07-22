import os
import glob
import subprocess

depth = 10
grid_path = '/data/project3/minnaho/project9copy/swel/plot/mc60_grd.nc'
b = -1 * depth

# Simplified list since we are doing both variables at once
folders = [
    '/data/project3/minnaho/swel/tides/mc60/wec/his',
    '/data/project3/minnaho/swel/tides/mc60/nowec/output/his',
    '/data/project3/minnaho/swel/notides/mc60/nowec/output/his',
    '/data/project3/minnaho/swel/notides/mc60/wec/rerun/his',
]

for folder in folders:
    out_dir = os.path.join(folder, 'zslice_10m_trace')
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(folder, 'mc60_*.*.nc')))
    for f in files:
        basename = os.path.basename(f)
        
        # Updated output filename format
        
        print('zslice: ' + f)
        # Updated command to call both variables at once
        subprocess.call(
            'zslice ' + str(b) + ' --vars=rtrace,ptrace ' + grid_path
            + ' ' + os.path.relpath(f, out_dir),
            shell=True,
            cwd=out_dir,
        )
