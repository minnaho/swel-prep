"""
4-scenario version of plot_profile_zslice_shelf_offshore.py -- same 4-panel
layout (h<=100m mean / diff, h>100m mean / diff), same npz sources, same
per-variable y/x-limit handling, only SCENARIOS restricted to
notidesnowec / tidesnowec / ampwec (the "notidesampwec" -- no tides,
2.5x WEC -- scenario's dict key) / tidesampwec.

Reads zslice_profiles_100m.npz (h<=100m domain) and zslice_profiles_offshore.npz
(h>100m domain) from ../postprocessing/. No coastal-mask panel here -- just
the two domain means and their diffs from baseline.

For each variable, writes one PNG with four side-by-side panels:
  1 -- h<=100m domain mean
  2 -- difference from the notidesnowec baseline (h<=100m)
  3 -- h>100m domain mean
  4 -- difference from the notidesnowec baseline (h>100m)
Four scenarios are overlaid as colored lines on each mean panel.

Output: ./figs/profiles/profile_zslice_shelf_offshore_4_<var>.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scenario_style as ss

NPZ_SHELF    = '../postprocessing/zslice_profiles_100m.npz'
NPZ_OFFSHORE = '../postprocessing/zslice_profiles_offshore.npz'

VARS_HIS = ['ptrace', 'rtrace', 'w', 'rho', 'u', 'v']
VARS_BGC = ['NO3', 'NH4', 'SPC', 'DIATC', 'DIAZC', 'TOTC',
            'SPCHL', 'DIATCHL', 'DIAZCHL', 'O2', 'DIC', 'DOC']
VARS_DIA = ['TOT_PROD']
VARS_AK  = ['Akt', 'Akv']
TOTC_VARS = ['DIATC', 'DIAZC', 'SPC']

SCENARIOS = ['notidesnowec', 'tidesnowec', 'ampwec', 'tidesampwec']
BASELINE  = 'notidesnowec'
LABELS  = ss.LABELS
COLORS  = ss.COLORS
LSTYLES = ss.LSTYLES

# shelf y-axis ranges -- verbatim from plot_profile_zslice_100m.py, all
# clamped to -100 m since there's no valid shelf data below that
_s40  = [-40, 0]
_s80  = [-80, 0]
_s100 = [-100, 0]
DEPTH_YLIM_SHELF = {
    'ptrace':   _s80,
    'rtrace':   _s40,
    'w':        _s100,
    'u':        _s100,
    'v':        _s100,
    'rho':      _s100,
    'NO3':      _s100,
    'NH4':      _s80,
    'O2':       _s100,
    'DIC':      _s100,
    'DOC':      _s100,
    'SPC':      _s100,
    'DIATC':    _s100,
    'DIAZC':    _s100,
    'SPCHL':    _s100,
    'DIATCHL':  _s100,
    'DIAZCHL':  _s100,
    'TOTC':     _s100,
    'TOT_PROD': _s40,
    'Akt':      _s100,
    'Akv':      _s100,
}

# offshore y-axis ranges -- deeper, since offshore data is valid to -1980m
_o40  = [-40, 0]
_o80  = [-80, 0]
_o150 = [-150, 0]
_o500 = [-500, 0]
DEPTH_YLIM_OFFSHORE = {
    'ptrace':   _o80,
    'rtrace':   _o40,
    'w':        _o500,
    'u':        _o500,
    'v':        _o500,
    'rho':      _o500,
    'NO3':      _o150,
    'NH4':      _o80,
    'O2':       _o500,
    'DIC':      _o500,
    'DOC':      _o500,
    'SPC':      _o150,
    'DIATC':    _o150,
    'DIAZC':    _o150,
    'SPCHL':    _o150,
    'DIATCHL':  _o150,
    'DIAZCHL':  _o150,
    'TOTC':     _o150,
    'TOT_PROD': _o40,
    'Akt':      _o500,
    'Akv':      _o500,
}

XLIM_SHELF = {
    'NO3': [0, 20],
}
XLIM_OFFSHORE = {}

VAR_UNITS = {
    'ptrace':   r'mmol m$^{-3}$',
    'rtrace':   r'mmol m$^{-3}$',
    'w':        r'm s$^{-1}$',
    'u':        r'm s$^{-1}$',
    'v':        r'm s$^{-1}$',
    'rho':      r'kg m$^{-3}$',
    'NO3':      r'mmol N m$^{-3}$',
    'NH4':      r'mmol N m$^{-3}$',
    'O2':       r'mmol O$_2$ m$^{-3}$',
    'DIC':      r'mmol C m$^{-3}$',
    'DOC':      r'mmol C m$^{-3}$',
    'SPC':      r'mmol C m$^{-3}$',
    'DIATC':    r'mmol C m$^{-3}$',
    'DIAZC':    r'mmol C m$^{-3}$',
    'SPCHL':    r'mg Chl m$^{-3}$',
    'DIATCHL':  r'mg Chl m$^{-3}$',
    'DIAZCHL':  r'mg Chl m$^{-3}$',
    'TOTC':     r'mmol C m$^{-3}$',
    'TOT_PROD': r'mmol C m$^{-3}$ d$^{-1}$',
    'Akt':      r'm$^2$ s$^{-1}$',
    'Akv':      r'm$^2$ s$^{-1}$',
}

VAR_LABELS = {
    'TOTC':     'Total phyto C',
    'TOT_PROD': 'NPP',
    'Akt':      r'$K_t$',
    'Akv':      r'$K_v$',
}

plt.rcParams.update({'font.size': 12})

shelf    = dict(np.load(NPZ_SHELF, allow_pickle=False))
offshore = dict(np.load(NPZ_OFFSHORE, allow_pickle=False))

# pre-compute TOTC = DIATC + DIAZC + SPC, in both npz dicts
for data in (shelf, offshore):
    for scen in SCENARIOS:
        src_keys = [f'{v}_{scen}' for v in TOTC_VARS]
        if all(k in data for k in src_keys):
            data[f'TOTC_{scen}'] = sum(data[k] for k in src_keys)

SAVEPATH = './figs/profiles/'
os.makedirs(SAVEPATH, exist_ok=True)

for var in VARS_HIS + VARS_BGC + VARS_DIA + VARS_AK:
    depth_key = 'depth_dia' if var in VARS_DIA else 'depth'
    depth_shelf    = shelf[depth_key]
    depth_offshore = offshore[depth_key]

    ylim_shelf    = DEPTH_YLIM_SHELF.get(var, _s100)
    ylim_offshore = DEPTH_YLIM_OFFSHORE.get(var, _o500)
    units = VAR_UNITS.get(var, '')
    var_label = VAR_LABELS.get(var, var)
    xlabel = f'{var_label} [{units}]' if units else var_label
    scale = 86400.0 if var == 'TOT_PROD' else 1.0

    fig, (axSL, axSD, axOL, axOD) = plt.subplots(1, 4, figsize=(20, 8))

    base_key = f'{var}_{BASELINE}'
    for data, depth, axL, axD in (
            (shelf, depth_shelf, axSL, axSD),
            (offshore, depth_offshore, axOL, axOD)):
        for scen in SCENARIOS:
            key = f'{var}_{scen}'
            if key not in data:
                continue
            prof = data[key] * scale
            lw = ss.lw(scen, base_lw=1.5)
            axL.plot(prof, depth, color=COLORS[scen], linestyle=LSTYLES[scen],
                      label=LABELS[scen], linewidth=lw)
            if scen != BASELINE and base_key in data:
                diff = (data[key] - data[base_key]) * scale
                axD.plot(diff, depth, color=COLORS[scen], linestyle=LSTYLES[scen],
                          label=f'{LABELS[scen]} − baseline', linewidth=lw)

    axSL.set_ylim(ylim_shelf)
    axSD.set_ylim(ylim_shelf)
    axOL.set_ylim(ylim_offshore)
    axOD.set_ylim(ylim_offshore)

    for ax in (axSL, axSD, axOL, axOD):
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3g'))
    if var in XLIM_SHELF:
        axSL.set_xlim(XLIM_SHELF[var])
    if var in XLIM_OFFSHORE:
        axOL.set_xlim(XLIM_OFFSHORE[var])

    for ax in (axSD, axOL, axOD):
        ax.tick_params(axis='y', labelleft=False)

    axSL.set_ylabel('depth (m)')
    axOL.set_ylabel('depth (m)')
    axSL.set_title('h<=100m domain')
    axSD.set_title(f'diff from {LABELS[BASELINE]} (h<=100m)')
    axOL.set_title('h>100m domain')
    axOD.set_title(f'diff from {LABELS[BASELINE]} (h>100m)')
    axSL.legend(loc='best', fontsize=12)
    #axSD.legend(loc='best', fontsize=12)

    plt.tight_layout()
    out = f'{SAVEPATH}profile_zslice_shelf_offshore_4_{var}.png'
    plt.savefig(out, bbox_inches='tight', dpi=800)
    plt.close(fig)
    print(f'saved -> {out}')
