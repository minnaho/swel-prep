import subprocess
import glob
import os

inpath = '/data/project3/minnaho/swel/notides/mc60/output/bgc/'
outpath = '/data/project3/minnaho/swel/notides/mc60/output/extract_biomass/'

files = sorted(glob.glob(inpath + 'mc60_bgc.*.nc'))

extract_vars = 'SPC,DIATC,DIAZC,ZOOC'

for f_i in files:
    fname = os.path.basename(f_i)
    print(fname)
    
    # Extract the wildcard part
    suffix = fname.replace('mc60_bgc.', '') 
    
    out_file = os.path.join(outpath, 'mc60_biomass.' + suffix)
    
    subprocess.run([
        'ncks',
        '-v', extract_vars,
        f_i,
        out_file
    ], check=True)

