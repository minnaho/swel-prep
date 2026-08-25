import os
import glob
import subprocess
import sys  # Imported sys to allow us to exit the script

depth = 10
grid_path = '/data/project3/minnaho/project9copy/swel/plot/mc60_grd.nc'
b = -1 * depth

folders = [
    ('/data/project3/minnaho/swel/tides/mc60/wec/his',            'w'),
    ('/data/project3/minnaho/swel/tides/mc60/wec/bgc',            'NO3'),
    ('/data/project3/minnaho/swel/tides/mc60/nowec/output/his',  'w'),
    ('/data/project3/minnaho/swel/tides/mc60/nowec/output/bgc',  'NO3'),
    ('/data/project3/minnaho/swel/notides/mc60/nowec/his','w'),
    ('/data/project3/minnaho/swel/notides/mc60/nowec/bgc','NO3'),
    ('/data/project3/minnaho/swel/notides/mc60/wec/rerun/his',  'w'),
    ('/data/project3/minnaho/swel/notides/mc60/wec/rerun/bgc',  'NO3'),
]

for folder, var in folders:
    out_dir = os.path.join(folder, 'zslice_10m')
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(folder, 'mc60_*.*.nc')))
    for f in files:
        basename = os.path.basename(f)
        out_file = os.path.join(out_dir, 'z_' + basename)
        
        # Commented out so it forces an overwrite of the bad file
        # if os.path.exists(out_file):
        #     print('skip: ' + out_file)
        #     continue
            
        print('zslice: ' + f)
        subprocess.call(
            'zslice ' + str(b) + ' --vars=' + var + ' ' + grid_path
            + ' ' + os.path.relpath(f, out_dir),
            shell=True,
            cwd=out_dir,
        )
        
        print('Finished remaking the first file. Exiting script.')
        sys.exit(0)  # This stops the entire script right here
