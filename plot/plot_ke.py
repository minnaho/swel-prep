import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scenario_style as ss

print('Loading spectra data...')
data  = np.load('./ke_spectra_comparison.npz')
freqs = data['freqs']

pos_idx  = freqs > 0
neg_idx  = freqs < 0
freqs_pos = freqs[pos_idx]
freqs_neg = np.abs(freqs[neg_idx])

SCEN_KEYS = ['tideswec', 'tidesnowec', 'notidesnowec', 'notideswec', 'ampwec']

axisfont = 16

# ==========================================
# Figure 1: Zoomed rotary spectrum
# ==========================================
fig1, ax1 = plt.subplots(1, 1, figsize=[10, 6])
for scen in SCEN_KEYS:
    key = f'psd_{scen}_masknc'
    if key not in data:
        continue
    ax1.plot(freqs, data[key], **ss.line_kwargs(scen, base_lw=2))

ax1.set_yscale('log')
ax1.set_xlim([-0.15, 0.15])
ax1.legend(fontsize=axisfont - 2)
ax1.set_xlabel('Frequency [cph]', fontsize=axisfont)
ax1.set_ylabel('KE Density [m$^2$ s$^{-2}$ cph$^{-1}$]', fontsize=axisfont)
ax1.tick_params(axis='both', which='major', labelsize=axisfont)
ax1.set_title('Rotary Spectrum (Zoomed)', fontsize=axisfont + 2)
plt.savefig('./figs/ke_spectra_zoomed.png', bbox_inches='tight', dpi=600)
plt.close(fig1)

# ==========================================
# Figure 2: Folded CCW vs CW
# ==========================================
fig2, (ax_ccw, ax_cw) = plt.subplots(1, 2, figsize=[14, 6], sharey=True)
for scen in SCEN_KEYS:
    key = f'psd_{scen}_masknc'
    if key not in data:
        continue
    psd = data[key]
    ax_ccw.plot(freqs_pos, psd[pos_idx], **ss.line_kwargs(scen, base_lw=2))
    ax_cw.plot( freqs_neg, psd[neg_idx], **ss.line_kwargs(scen, base_lw=2))

for ax, title, xlabel in [
    (ax_ccw, 'Cyclonic (CCW)',    'Frequency [cph]'),
    (ax_cw,  'Anticyclonic (CW)', 'Absolute Frequency [|cph|]'),
]:
    ax.set_yscale('log')
    ax.set_xlim([0, 0.15])
    ax.set_xlabel(xlabel, fontsize=axisfont)
    ax.tick_params(axis='both', which='major', labelsize=axisfont)
    ax.set_title(title, fontsize=axisfont)

ax_ccw.set_ylabel('KE Density [m$^2$ s$^{-2}$ cph$^{-1}$]', fontsize=axisfont)
ax_cw.legend(fontsize=axisfont - 2, loc='best')

plt.tight_layout()
plt.savefig('./figs/ke_spectra_folded.png', bbox_inches='tight', dpi=600)
plt.close(fig2)

print('Saved ./figs/ke_spectra_zoomed.png and ke_spectra_folded.png')
