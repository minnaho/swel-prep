import os
import sys
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import scenario_style as ss

# ../postprocessing/calc_vort_pdf.py PDFs
print('plotting')

SCEN_KEYS = ['tideswec', 'tidesnowec', 'notidesnowec', 'notideswec', 'ampwec', 'tidesampwec']

# array key inside each npz -- matches ARRAY_KEY in postprocessing/calc_vort_pdf.py
ARRAY_KEY = {
    'tideswec':     'pdf_wec',
    'tidesnowec':   'pdf_nowec',
    'notidesnowec': 'pdf_notidesnowec',
    'notideswec':   'pdf_notideswec',
    'ampwec':       'pdf_ampwec',
    'tidesampwec':  'pdf_tidesampwec',
}

# total domain
figw = 12
figh = 8
axisfont = 16

fig, ax = plt.subplots(1, 1, figsize=[figw, figh])

for scen in SCEN_KEYS:
    npz_path = f'../postprocessing/pdf_vorticity_{scen}.npz'
    if not os.path.exists(npz_path):
        print(f'WARNING: {npz_path} not found, skipping {scen}')
        continue
    pdf = np.load(npz_path)
    bin_edges = pdf['bin_edges']
    num = pdf[ARRAY_KEY[scen]]
    ax.plot(bin_edges[1:-1], num[1:-1], **ss.line_kwargs(scen, base_lw=1.5))

ax.set_yscale('log')
ax.set_ylim([1E-7, 1])
ax.set_xlim([-21, 21])

ax.legend(fontsize=axisfont)
ax.set_xlabel(r'$\zeta/f$', fontsize=axisfont)
ax.set_ylabel('PDF', fontsize=axisfont)
ax.tick_params(axis='both', which='major', labelsize=axisfont)
plt.savefig('./figs/pdf_vort.png', bbox_inches='tight')
print('Saved ./figs/pdf_vort.png')
