"""
Shelf (h<=100m) vs. offshore (h>100m) depth profiles for the DIAT/SP
nutrient-limitation and uptake diagnostics -- bgcdia analog of
plot_profile_zslice_shelf_offshore_4.py, same 4-panel layout (h<=100m mean /
diff, h>100m mean / diff), reading zslice_profiles_bgcdia_100m.npz and
zslice_profiles_bgcdia_offshore.npz (written by
postprocessing/profile_zslice_bgcdia_100m_offshore.py) from
../postprocessing/.

Restricted to the 3 scenarios the bgc_dia_avg rerun covers (tidesampwec,
tidesnowec, notidesnowec) -- same set as plot_profile_zslice_bgcdia.py, not
the 4-scenario set plot_profile_zslice_shelf_offshore_4.py uses (bgcdia has
no ampwec data at all).

No coastal-mask panel here, and no depth-zone (0-50/50-200/200+) breakdown --
same reasoning as plot_profile_zslice_shelf_offshore_4.py: the new
postprocessing script doesn't compute h0to50/h50to200/h200p bins any more
(dropped, would silently re-bin within an already depth-restricted domain),
so there's nothing to plot there.

For each variable, writes one PNG with four side-by-side panels:
  1 -- h<=100m domain mean
  2 -- difference from the notidesnowec baseline (h<=100m)
  3 -- h>100m domain mean
  4 -- difference from the notidesnowec baseline (h>100m)

Output: ./figs/bgcprofiles/profile_zslice_bgcdia_shelf_offshore_<var>.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scenario_style as ss

NPZ_SHELF    = '../postprocessing/zslice_profiles_bgcdia_100m.npz'
NPZ_OFFSHORE = '../postprocessing/zslice_profiles_bgcdia_offshore.npz'

DIAT_LIM_VARS    = ['DIAT_N_LIM', 'DIAT_FE_LIM', 'DIAT_PO4_LIM',
                     'DIAT_SIO3_LIM', 'DIAT_LIGHT_LIM', 'DIAT_P_LIM']
SP_LIM_VARS      = ['SP_N_LIM', 'SP_FE_LIM', 'SP_PO4_LIM',
                     'SP_LIGHT_LIM', 'SP_P_LIM']
DIAT_UPTAKE_VARS = ['DIAT_NO3_UPTAKE', 'DIAT_NH4_UPTAKE', 'DIAT_NO2_UPTAKE',
                     'DIAT_SI_UPTAKE']
SP_UPTAKE_VARS   = ['SP_NO3_UPTAKE', 'SP_NH4_UPTAKE', 'SP_NO2_UPTAKE']
PAR_VARS         = ['PAR']

LIM_VARS    = DIAT_LIM_VARS + SP_LIM_VARS
UPTAKE_VARS = DIAT_UPTAKE_VARS + SP_UPTAKE_VARS
VARS        = LIM_VARS + UPTAKE_VARS + PAR_VARS

SCENARIOS = ['tidesampwec', 'tidesnowec', 'notidesnowec']
BASELINE  = 'notidesnowec'
LABELS  = ss.LABELS
COLORS  = ss.COLORS
LSTYLES = ss.LSTYLES

VAR_LONG_NAME = {
    'DIAT_N_LIM':        'Diatom N limitation',
    'DIAT_FE_LIM':       'Diatom Fe limitation',
    'DIAT_PO4_LIM':      'Diatom PO4 limitation',
    'DIAT_SIO3_LIM':     'Diatom SiO3 limitation',
    'DIAT_LIGHT_LIM':    'Diatom light limitation',
    'DIAT_P_LIM':        'Diatom P limitation',
    'SP_N_LIM':          'Small phyto N limitation',
    'SP_FE_LIM':         'Small phyto Fe limitation',
    'SP_PO4_LIM':        'Small phyto PO4 limitation',
    'SP_LIGHT_LIM':      'Small phyto light limitation',
    'SP_P_LIM':          'Small phyto P limitation',
    'DIAT_NO3_UPTAKE':   'Diatom NO3 uptake',
    'DIAT_NH4_UPTAKE':   'Diatom NH4 uptake',
    'DIAT_NO2_UPTAKE':   'Diatom NO2 uptake',
    'DIAT_SI_UPTAKE':    'Diatom Si uptake',
    'SP_NO3_UPTAKE':     'Small phyto NO3 uptake',
    'SP_NH4_UPTAKE':     'Small phyto NH4 uptake',
    'SP_NO2_UPTAKE':     'Small phyto NO2 uptake',
    'PAR':               'PAR',
}

VAR_UNITS = {v: '' for v in LIM_VARS}
VAR_UNITS.update({v: r'mmol m$^{-2}$ s$^{-1}$' for v in UPTAKE_VARS})
VAR_UNITS.update({v: r'W m$^{-2}$' for v in PAR_VARS})

# LIM factors are bounded [0, 1] by definition — fix the x-axis for
# interpretability across panels. UPTAKE vars are left data-driven.
XLIM = {v: [0, 1] for v in LIM_VARS}

# shelf never has water below -100m; offshore's zsliced grid only spans to
# -200m in the first place (the bgc_dia_avg rerun's z-grid), so there's no
# need for a separate deeper offshore range the way the his/bgc/dia/ak
# shelf_offshore_4 script needs one
YLIM_SHELF    = [-100, 0]
YLIM_OFFSHORE = [-200, 0]

plt.rcParams.update({'font.size': 12})

shelf    = dict(np.load(NPZ_SHELF,    allow_pickle=False))
offshore = dict(np.load(NPZ_OFFSHORE, allow_pickle=False))

SAVEPATH = './figs/bgcprofiles/'
os.makedirs(SAVEPATH, exist_ok=True)

for var in VARS:
    depth_shelf    = shelf['depth']
    depth_offshore = offshore['depth']

    var_label = VAR_LONG_NAME.get(var, var)
    units = VAR_UNITS.get(var, '')
    xlabel = f'{var_label} [{units}]' if units else var_label

    fig, (axSL, axSD, axOL, axOD) = plt.subplots(1, 4, figsize=(20, 8))

    base_key = f'{var}_{BASELINE}'
    for data, depth, axL, axD in (
            (shelf, depth_shelf, axSL, axSD),
            (offshore, depth_offshore, axOL, axOD)):
        for scen in SCENARIOS:
            key = f'{var}_{scen}'
            if key not in data:
                continue
            prof = data[key]
            lw = ss.lw(scen, base_lw=1.5)
            axL.plot(prof, depth, color=COLORS[scen], linestyle=LSTYLES[scen],
                      label=LABELS[scen], linewidth=lw)
            if scen != BASELINE and base_key in data:
                diff = data[key] - data[base_key]
                axD.plot(diff, depth, color=COLORS[scen], linestyle=LSTYLES[scen],
                          label=f'{LABELS[scen]} − baseline', linewidth=lw)

    axSL.set_ylim(YLIM_SHELF)
    axSD.set_ylim(YLIM_SHELF)
    axOL.set_ylim(YLIM_OFFSHORE)
    axOD.set_ylim(YLIM_OFFSHORE)

    for ax in (axSL, axSD, axOL, axOD):
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3g'))
    if var in XLIM:
        axSL.set_xlim(XLIM[var])
        axOL.set_xlim(XLIM[var])

    for ax in (axSD, axOL, axOD):
        ax.tick_params(axis='y', labelleft=False)

    axSL.set_ylabel('depth (m)')
    axOL.set_ylabel('depth (m)')
    axSL.set_title('h<=100m domain')
    axSD.set_title(f'diff from {LABELS[BASELINE]} (h<=100m)')
    axOL.set_title('h>100m domain')
    axOD.set_title(f'diff from {LABELS[BASELINE]} (h>100m)')
    axSL.legend(loc='best', fontsize=12)

    plt.tight_layout()
    out = f'{SAVEPATH}profile_zslice_bgcdia_shelf_offshore_{var}.png'
    plt.savefig(out, bbox_inches='tight', dpi=800)
    plt.close(fig)
    print(f'saved -> {out}')
