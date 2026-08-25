"""
Energy cascade plot for the 2019-04-21 00:00 -- 2019-04-23 00:00 window,
direct structural copy of plot_energy_cascade.py (same folding/slope-fit
method, same tidal/inertial frequency markers, same axis limits) -- only
the npz source and output filenames differ, since
calc_ke_surf_20190421_20190423.py writes a separate npz rather than
overwriting calc_ke_surf.py's full-record output.

xlim/ylim are kept numerically identical to plot_energy_cascade.py's on
purpose (not re-fit to this window's data), even though the shorter 48-hour
record can't resolve frequencies below ~0.021 cph (vs. the full record's
~3e-3 cph) -- the resulting blank margin on the low-frequency side of the
plot is real, expected, and informative: it's the direct visual signature
of the window being too short to resolve those periods, not a bug.
"""

import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scenario_style as ss

print('Loading spectra data...')
data = np.load('./ke_spectra_comparison_20190421_20190423.npz')

grd    = Dataset('mc60_grd.nc', 'r')
f_nc   = np.nanmean(grd.variables['f'])

freqs     = data['freqs']
pos_idx   = freqs > 0
freqs_pos = freqs[pos_idx]

SCEN_KEYS = ['tideswec', 'tidesnowec', 'notidesnowec', 'notideswec', 'ampwec', 'tidesampwec']

WINDOW_LABEL = '2019-04-21 00:00 -- 04-23 00:00'

def fold_total_ke(psd_full):
    total = np.zeros_like(freqs_pos)
    for i, f in enumerate(freqs_pos):
        p = np.argmin(np.abs(freqs - f))
        n = np.argmin(np.abs(freqs + f))
        total[i] = psd_full[p] + psd_full[n]
    return total

# ==========================================
# Fit slope
# ==========================================
fit_mask = (freqs_pos > 1/10) & (freqs_pos < 1/2)

def log_slope(f, psd, mask):
    poly = np.polyfit(np.log10(f[mask]), np.log10(psd[mask]), 1)
    return poly

# ==========================================
# Plotting
# ==========================================
axisfont = 16

# frequency markers, shared by every plot
f_inertial    = f_nc * 3600 / (2 * np.pi)
f_o1, f_k1    = 0.0387, 0.0418
f_m2, f_m4, f_m6 = 0.0805, 0.161, 0.2415
FREQ_MARKERS = [
    (f_inertial, '$f$',     'red',    'left'),
    (f_k1,       'K$_1$',   'gray',   'left'),
    (f_o1,       'O$_1$',   'gray',   'right'),
    (f_m2,       'M$_2$',   'purple', 'left'),
    (f_m4,       'M$_4$',   'gold',  'left'),
    (f_m6,       'M$_6$',   'gold',  'left'),
]

# 'notidesampwec' is the zslice-dir name for this script's raw-scenario key 'ampwec'
FIT_SCENARIOS = [('tideswec', 'darkviolet'), ('notidesnowec', 'olive')]


def make_energy_cascade_plot(scen_keys, out_path, title=None):
    fig, ax = plt.subplots(1, 1, figsize=[12, 8])

    plotted = []
    for scen in scen_keys:
        key = f'psd_{scen}_masknc'
        if key not in data:
            continue
        ke = fold_total_ke(data[key])
        ax.loglog(freqs_pos, ke, **ss.line_kwargs(scen, base_lw=2))
        plotted.append(scen)

    for scen, color in FIT_SCENARIOS:
        key = f'psd_{scen}_masknc'
        if scen not in scen_keys or key not in data:
            continue
        ke   = fold_total_ke(data[key])
        poly = log_slope(freqs_pos, ke, fit_mask)
        fit  = (10 ** poly[1]) * (freqs_pos ** poly[0])
        ax.loglog(freqs_pos[fit_mask], fit[fit_mask], color=color, linewidth=2,
                  linestyle='-.', label=f'slope: {poly[0]:.2f}')

    trans = ax.get_xaxis_transform()
    for fval, label, color, ha in FREQ_MARKERS:
        ax.axvline(x=fval, color=color, linestyle='--', alpha=0.7)
        offset = 1.03 if ha == 'left' else 0.98
        ax.text(fval * offset, 0.89, label, transform=trans,
                color=color, fontsize=axisfont - 2, ha=ha, va='center')

    ax.set_xlim([3e-3, 1])
    psd_vals = [fold_total_ke(data[f'psd_{s}_masknc']) for s in plotted]
    if psd_vals:
        #ax.set_ylim([9e-7, np.nanmax(psd_vals) * 3])
        ax.set_ylim([3e-7, 1E-2])

    if title:
        ax.set_title(title, fontsize=axisfont)
    ax.legend(fontsize=axisfont - 2, loc='lower left')
    ax.set_xlabel('Frequency [cycles per hour]', fontsize=axisfont)
    ax.set_ylabel('Surface KE Density [(m s$^{-1}$)$^2$ cph$^{-1}$]', fontsize=axisfont)
    ax.tick_params(axis='both', which='major', labelsize=axisfont)
    ax.grid(True, which='both', ls='--', alpha=0.5)

    fig.savefig(out_path, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f'Saved {out_path}')


make_energy_cascade_plot(SCEN_KEYS, './figs/energy_cascade_20190421_20190423.png',
                          title=WINDOW_LABEL)

make_energy_cascade_plot(
    ['tidesnowec', 'notidesnowec', 'ampwec'],
    './figs/energy_cascade_notidesnowec_group_20190421_20190423.png',
    title=f'tides no WEC / no tides no WEC / no tides 2.5× WEC, {WINDOW_LABEL}',
)

make_energy_cascade_plot(
    ['tideswec', 'notideswec', 'tidesampwec'],
    './figs/energy_cascade_wec_group_20190421_20190423.png',
    title=f'tides WEC / no tides WEC / tides 2.5× WEC, {WINDOW_LABEL}',
)
