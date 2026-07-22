import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scenario_style as ss

print('Loading spectra data...')
data = np.load('./ke_spectra_comparison.npz')

freqs     = data['freqs']
pos_idx   = freqs > 0
freqs_pos = freqs[pos_idx]

SCEN_KEYS = ['tideswec', 'tidesnowec', 'notidesnowec', 'notideswec', 'ampwec']

def fold_total_ke(psd_full):
    total = np.zeros_like(freqs_pos)
    for i, f in enumerate(freqs_pos):
        p = np.argmin(np.abs(freqs - f))
        n = np.argmin(np.abs(freqs + f))
        total[i] = psd_full[p] + psd_full[n]
    return total

fit_mask = (freqs_pos > 0.15) & (freqs_pos < 0.45)

def log_slope(f, psd, mask):
    return np.polyfit(np.log10(f[mask]), np.log10(psd[mask]), 1)

axisfont = 16
fig, ax = plt.subplots(1, 1, figsize=[12, 8])

for scen in SCEN_KEYS:
    key = f'psd_{scen}_coastal'
    if key not in data:
        continue
    ke = fold_total_ke(data[key])
    ax.loglog(freqs_pos, ke, **ss.line_kwargs(scen, base_lw=2))

for scen, color in [('tideswec', 'red'), ('notidesnowec', 'green')]:
    key = f'psd_{scen}_coastal'
    if key not in data:
        continue
    ke   = fold_total_ke(data[key])
    poly = log_slope(freqs_pos, ke, fit_mask)
    fit  = (10 ** poly[1]) * (freqs_pos ** poly[0])
    ax.loglog(freqs_pos[fit_mask], fit[fit_mask], color=color, linewidth=2,
              linestyle='-.', label=f'{scen} slope: {poly[0]:.2f}')

ax.set_xlim([3e-3, 1])
psd_vals = [fold_total_ke(data[f'psd_{s}_coastal']) for s in SCEN_KEYS if f'psd_{s}_coastal' in data]
ax.set_ylim([1e-6, np.nanmax(psd_vals) * 5])

ax.legend(fontsize=axisfont - 2, loc='lower left')
ax.set_xlabel('Frequency [cycles per hour]', fontsize=axisfont)
ax.set_ylabel('Coastal KE Density [(m s$^{-1}$)$^2$ cph$^{-1}$]', fontsize=axisfont)
ax.tick_params(axis='both', which='major', labelsize=axisfont)
ax.set_title('Coastal Kinetic Energy Cascade', fontsize=axisfont + 2)
plt.grid(True, which='both', ls='--', alpha=0.5)

plt.savefig('./figs/energy_cascade_coastal.png', bbox_inches='tight', dpi=600)
print('Saved ./figs/energy_cascade_coastal.png')
