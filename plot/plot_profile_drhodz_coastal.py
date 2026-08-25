"""
d(rho)/dz depth profile, coastal (10 km) band only -- coastal sibling of
plot_profile_drhodz_100m.py, itself the profile-average counterpart to
plot_cs_diag_drhodz_diff.py's cross-section version.

Reads the already time- and horizontally-averaged 'rho' profile from
../postprocessing/zslice_profiles_coastal.npz -- the plain 10 km coastal
mask (not combined with any h<=100m restriction; see
plot_profile_zslice_100m_coastal.py, whose middle panel was fixed to read
this same file rather than the doubly-restricted zslice_profiles_100m_coastal.npz).
Differentiation and averaging are both linear operators that commute
exactly on the zslice output's fixed depth grid, so d(mean(rho))/dz ==
mean(drho/dz) -- same reasoning as plot_cs_diag_drhodz_diff.py, just for a
single domain-mean profile instead of a lon-depth cross-section. A constant
offset (rho is stored as a deviation from a reference density) drops
out of the derivative, so no RHO_OFFSET is needed here.

Unlike the h<=100m version, the coastal band is not depth-restricted, so
valid rho data extends much deeper (down to ~-1440 m for notidesnowec) --
the plot is capped at DEPTH_LIM = -150 m (deeper than
plot_profile_drhodz_100m.py's fixed -100 m, since the coastal band itself
reaches somewhat deeper than the h<=100m mask, but still well short of the
full ~-1440 m of valid data, which is mostly featureless below the
pycnocline).

Left panel:  drho/dz vs depth, one line per scenario.
Right panel: each scenario's drho/dz minus the notidesnowec baseline.

Output: ./figs/profile_drhodz_coastal.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scenario_style as ss

NPZ = '../postprocessing/zslice_profiles_coastal.npz'

SCENARIOS = ['tideswec', 'tidesnowec', 'notidesnowec', 'notideswec', 'ampwec', 'tidesampwec']
BASELINE  = 'notidesnowec'
LABELS  = ss.LABELS
COLORS  = ss.COLORS
LSTYLES = ss.LSTYLES

DEPTH_LIM = -150
DEPTH_YLIM = [DEPTH_LIM, 0]
DRHODZ_LABEL = r'$\partial\rho/\partial z$ (kg m$^{-4}$)'

plt.rcParams.update({'font.size': 12})

data  = dict(np.load(NPZ, allow_pickle=False))
depth = data['depth']

drhodz = {}
for scen in SCENARIOS:
    key = f'rho_{scen}'
    if key not in data:
        continue
    drhodz[scen] = np.gradient(data[key], depth)

fig, (axL, axD) = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

base = drhodz.get(BASELINE)
for scen in SCENARIOS:
    if scen not in drhodz:
        continue
    lw = ss.lw(scen, base_lw=1.5)
    axL.plot(drhodz[scen], depth, color=COLORS[scen], linestyle=LSTYLES[scen],
             label=LABELS[scen], linewidth=lw)
    if scen != BASELINE and base is not None:
        axD.plot(drhodz[scen] - base, depth, color=COLORS[scen], linestyle=LSTYLES[scen],
                 label=f'{LABELS[scen]} − baseline', linewidth=lw)

axL.set_ylim(DEPTH_YLIM)
for ax in (axL, axD):
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)
    ax.set_xlabel(DRHODZ_LABEL)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3g'))

axL.set_ylabel('depth (m)')
axL.set_title(r'$\partial\rho/\partial z$ (coastal, 10 km)')
axD.set_title(f'difference from {LABELS[BASELINE]}')
axL.legend(loc='best', fontsize=9)
axD.legend(loc='best', fontsize=9)

plt.tight_layout()
os.makedirs('./figs', exist_ok=True)
out = './figs/profile_drhodz_coastal.png'
plt.savefig(out, bbox_inches='tight', dpi=800)
plt.close(fig)
print(f'saved -> {out}')
