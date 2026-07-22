"""
Two-panel wavenumber KE spectrum following Hypolite et al. (2021):
  Left  — horizontal wavenumber spectrum E(k_h) from ke_horiz_wavenumber.npz
  Right — vertical wavenumber spectrum E(k_z) from ke_vert_wavenumber.npz

X-axes shown as wavelength (λ = 1/k) for readability.
Reference slopes (-5/3 and -3) are drawn on the horizontal panel.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scenario_style as ss

horiz = np.load('./ke_horiz_wavenumber.npz')
vert  = np.load('./ke_vert_wavenumber.npz')

k_h = horiz['k_h']   # cycles m^-1
k_z = vert['k_z']    # cycles m^-1

# Convert to wavelength in metres (skip DC = 0)
lam_h = 1.0 / k_h[k_h > 0]
lam_z = 1.0 / k_z[k_z > 0]

HORIZ_KEYS = ['tideswec', 'tidesnowec', 'notidesnowec', 'notideswec', 'ampwec', 'tidesampwec']
# vert npz uses the zslice-dir key 'notidesampwec' for the run that horiz
# npz (raw-file naming) calls 'ampwec' -- same simulation, different key
VERT_KEYS = ['tideswec', 'tidesnowec', 'notidesnowec', 'notideswec', 'notidesampwec', 'tidesampwec']

axisfont = 15
fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=[16, 7])

# ---- Left: horizontal wavenumber ----
for scen in HORIZ_KEYS:
    key = f'E_{scen}_masknc'
    if key not in horiz:
        continue
    E = horiz[key][k_h > 0]
    ax_h.loglog(lam_h, E, **ss.line_kwargs(scen, base_lw=2))

# Reference slopes: E(k) ~ k^n → E(λ) ~ λ^{-n} (since λ = 1/k)
# Plot vs wavelength so increasing x = larger scale
# -5/3 slope: shallow cascade (submesoscale)
# -3 slope: geostrophic turbulence
lam_ref = np.logspace(np.log10(lam_h.min()), np.log10(lam_h.max()), 100)
k_ref   = 1.0 / lam_ref

# anchor reference lines to centre of plot
E_anchor = np.nanmedian([horiz[f'E_{s}_masknc'][k_h > 0].mean()
                          for s in HORIZ_KEYS if f'E_{s}_masknc' in horiz])
k_anchor = k_h[k_h > 0][len(k_h[k_h > 0]) // 2]
for slope, ls_ref, label_ref in [(-5/3, ':', r'$k^{-5/3}$'), (-3, '--', r'$k^{-3}$')]:
    E_ref = E_anchor * (k_ref / k_anchor) ** slope
    ax_h.loglog(lam_ref, E_ref, 'k', linestyle=ls_ref, linewidth=1.5,
                alpha=0.6, label=label_ref)

ax_h.invert_xaxis()   # larger wavelength on left
ax_h.set_xlabel('Horizontal wavelength λ$_h$ (m)', fontsize=axisfont)
ax_h.set_ylabel('E(k$_h$) [(m s$^{-1}$)$^2$ (cycles m$^{-1}$)$^{-1}$]', fontsize=axisfont)
ax_h.set_title('Horizontal wavenumber spectrum', fontsize=axisfont)
ax_h.legend(fontsize=axisfont - 2, loc='upper right')
ax_h.tick_params(axis='both', which='major', labelsize=axisfont - 1)
ax_h.grid(True, which='both', ls='--', alpha=0.4)

# ---- Right: vertical wavenumber ----
for scen in VERT_KEYS:
    key = f'E_{scen}_masknc'
    if key not in vert:
        continue
    E = vert[key][k_z > 0]
    ax_v.loglog(lam_z, E, **ss.line_kwargs(scen, base_lw=2))

ax_v.invert_xaxis()   # larger vertical scale on left
ax_v.set_xlabel('Vertical wavelength λ$_z$ (m)', fontsize=axisfont)
ax_v.set_ylabel('E(k$_z$) [(m s$^{-1}$)$^2$ (cycles m$^{-1}$)$^{-1}$]', fontsize=axisfont)
ax_v.set_title('Vertical wavenumber spectrum (0–50 m)', fontsize=axisfont)
ax_v.legend(fontsize=axisfont - 2, loc='upper right')
ax_v.tick_params(axis='both', which='major', labelsize=axisfont - 1)
ax_v.grid(True, which='both', ls='--', alpha=0.4)

plt.tight_layout()
plt.savefig('./figs/ke_wavenumber.png', bbox_inches='tight', dpi=600)
print('Saved ./figs/ke_wavenumber.png')
