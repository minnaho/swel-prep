"""
Depth-averaged frequency KE spectrum — mirrors plot_energy_cascade.py but
reads ke_spectra_depth.npz (one-sided rfft frequencies, no rotary decomposition).
"""
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scenario_style as ss

print('Loading depth spectra...')
data  = np.load('./ke_spectra_depth.npz')
freqs = data['freqs']   # positive only (rfft)
# exclude DC (freq=0) for log-log plot
pos_idx   = freqs > 0
freqs_pos = freqs[pos_idx]

grd  = Dataset('mc60_grd.nc', 'r')
f_nc = np.nanmean(grd.variables['f'])

SCEN_KEYS = ['tideswec', 'tidesnowec', 'notidesnowec', 'notideswec', 'ampwec']

fit_mask = (freqs_pos > 1/10) & (freqs_pos < 1/2)

def log_slope(f, psd, mask):
    return np.polyfit(np.log10(f[mask]), np.log10(psd[mask]), 1)

axisfont = 16
fig, ax = plt.subplots(1, 1, figsize=[12, 8])

for scen in SCEN_KEYS:
    key = f'psd_{scen}_masknc'
    if key not in data:
        continue
    ke = data[key][pos_idx]
    ax.loglog(freqs_pos, ke, **ss.line_kwargs(scen, base_lw=2))

for scen, color in [('tideswec', 'red'), ('notidesnowec', 'green')]:
    key = f'psd_{scen}_masknc'
    if key not in data:
        continue
    ke   = data[key][pos_idx]
    poly = log_slope(freqs_pos, ke, fit_mask)
    fit  = (10 ** poly[1]) * (freqs_pos ** poly[0])
    ax.loglog(freqs_pos[fit_mask], fit[fit_mask], color=color, linewidth=2,
              linestyle='-.', label=f'{scen} slope: {poly[0]:.2f}')

# Frequency markers
f_inertial = f_nc * 3600 / (2 * np.pi)
trans = ax.get_xaxis_transform()
for fval, label, color, ha in [
    (f_inertial, '$f$',    'red',    'left'),
    (0.0418,     'K$_1$', 'gray',   'left'),
    (0.0387,     'O$_1$', 'gray',   'right'),
    (0.0805,     'M$_2$', 'purple', 'left'),
    (0.161,      'M$_4$', 'coral',  'left'),
]:
    ax.axvline(x=fval, color=color, linestyle='--', alpha=0.7)
    offset = 1.05 if ha == 'left' else 0.98
    ax.text(fval * offset, 0.85, label, transform=trans,
            color=color, fontsize=axisfont - 2, ha=ha, va='center')

ax.set_xlim([3e-3, 1])
psd_vals = [data[f'psd_{s}_masknc'][pos_idx] for s in SCEN_KEYS if f'psd_{s}_masknc' in data]
ax.set_ylim([1e-7, np.nanmax(psd_vals) * 5])

ax.legend(fontsize=axisfont - 2, loc='lower left')
ax.set_xlabel('Frequency [cycles per hour]', fontsize=axisfont)
ax.set_ylabel('Depth-averaged KE Density [(m s$^{-1}$)$^2$ cph$^{-1}$]', fontsize=axisfont)
ax.tick_params(axis='both', which='major', labelsize=axisfont)
plt.grid(True, which='both', ls='--', alpha=0.5)

plt.savefig('./figs/energy_cascade_depth.png', bbox_inches='tight', dpi=600)
print('Saved ./figs/energy_cascade_depth.png')
