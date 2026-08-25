"""
Time- and horizontally-averaged depth profiles for every zsliced variable,
restricted to the h<=100m domain, with a coastal (10 km) mask panel added
(3-panel analog of plot_profile_zslice.py for profile_zslice_par_100m.py's
output).
Reads zslice_profiles_100m.npz from ../postprocessing/ for the left/right
panels, and zslice_profiles_coastal.npz (the same plain 10 km coastal mask
plot_profile_zslice.py uses, NOT h<=100m-restricted) for the middle panel --
the h<=100m + coastal combined mask lives in zslice_profiles_100m_coastal.npz
but is deliberately not used here, so the middle panel reads as "coastal"
on its own rather than a doubly-restricted domain.

No depth-zone (h0to50/h50to200/h200p) breakdown -- profile_zslice_par_100m.py
deliberately drops those bins since they'd be misleading once already restricted
to h<=100m.

For each variable, writes one PNG with three side-by-side panels:
  left   -- h<=100m domain mean
  middle -- coastal (10 km) domain mean
  right  -- difference from the notidesnowec baseline (h<=100m domain)
Six scenarios are overlaid as colored lines on each panel.

Output: ./figs/profiles/profile_zslice_100m_coastal_<var>.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scenario_style as ss

NPZ_FULL    = '../postprocessing/zslice_profiles_100m.npz'
NPZ_COASTAL = '../postprocessing/zslice_profiles_coastal.npz'

VARS_HIS = ['ptrace', 'rtrace', 'w', 'rho', 'u', 'v']
VARS_BGC = ['NO3', 'NH4', 'SPC', 'DIATC', 'DIAZC', 'TOTC',
            'SPCHL', 'DIATCHL', 'DIAZCHL', 'O2', 'DIC', 'DOC']
VARS_DIA = ['TOT_PROD']
TOTC_VARS = ['DIATC', 'DIAZC', 'SPC']

SCENARIOS = ['tideswec', 'tidesnowec', 'notidesnowec', 'notideswec', 'ampwec', 'tidesampwec']
BASELINE  = 'notidesnowec'
LABELS  = ss.LABELS
COLORS  = ss.COLORS
LSTYLES = ss.LSTYLES

# y-axis depth range per variable, clamped to -100 m -- plot_profile_zslice.py's
# ylims go deeper (down to -500 m) but the h<=100m restriction means there's no
# valid data below -100 m here, so anything deeper would just be empty axis.
_s40  = [-40, 0]
_s80  = [-80, 0]
_s100 = [-100, 0]
DEPTH_YLIM = {
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
}

XLIM = {
    'NO3': [0, 20],
}

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
}

VAR_LABELS = {
    'TOTC':     'Total phyto C',
    'TOT_PROD': 'NPP',
}

plt.rcParams.update({'font.size': 12})

full    = dict(np.load(NPZ_FULL, allow_pickle=False))
coastal = dict(np.load(NPZ_COASTAL, allow_pickle=False))

# pre-compute TOTC = DIATC + DIAZC + SPC in both domains
for scen in SCENARIOS:
    src_keys = [f'{v}_{scen}' for v in TOTC_VARS]
    if all(k in full for k in src_keys):
        full[f'TOTC_{scen}'] = sum(full[k] for k in src_keys)
    if all(k in coastal for k in src_keys):
        coastal[f'TOTC_{scen}'] = sum(coastal[k] for k in src_keys)

SAVEPATH = './figs/profiles/'
os.makedirs(SAVEPATH, exist_ok=True)

for var in VARS_HIS + VARS_BGC + VARS_DIA:
    depth_key = 'depth_dia' if var in VARS_DIA else 'depth'
    depth = full[depth_key]

    ylim = DEPTH_YLIM.get(var, _s100)
    units = VAR_UNITS.get(var, '')
    var_label = VAR_LABELS.get(var, var)
    xlabel = f'{var_label} [{units}]' if units else var_label
    scale = 86400.0 if var == 'TOT_PROD' else 1.0

    fig, (axL, axC, axD) = plt.subplots(1, 3, figsize=(15, 8), sharey=True)

    base_key = f'{var}_{BASELINE}'
    for scen in SCENARIOS:
        key = f'{var}_{scen}'
        lw = ss.lw(scen, base_lw=1.5)

        if key in full:
            prof_full = full[key] * scale
            axL.plot(prof_full, depth, color=COLORS[scen], linestyle=LSTYLES[scen], label=LABELS[scen], linewidth=lw)
            if scen != BASELINE and base_key in full:
                diff = (full[key] - full[base_key]) * scale
                axD.plot(diff, depth, color=COLORS[scen], linestyle=LSTYLES[scen],
                         label=f'{LABELS[scen]} − baseline', linewidth=lw)

        if key in coastal:
            prof_coastal = coastal[key] * scale
            axC.plot(prof_coastal, depth, color=COLORS[scen], linestyle=LSTYLES[scen], label=LABELS[scen], linewidth=lw)

    axL.set_ylim(ylim)
    for ax in (axL, axC, axD):
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3g'))
    if var in XLIM:
        axL.set_xlim(XLIM[var])

    axL.set_ylabel('depth (m)')
    axL.set_title('h<=100m')
    axC.set_title('coastal (10 km)')
    axD.set_title(f'difference from {LABELS[BASELINE]}')
    axL.legend(loc='best', fontsize=9)
    axC.legend(loc='best', fontsize=9)
    axD.legend(loc='best', fontsize=9)

    plt.tight_layout()
    out = f'{SAVEPATH}profile_zslice_100m_coastal_{var}.png'
    plt.savefig(out, bbox_inches='tight', dpi=800)
    plt.close(fig)
    print(f'saved -> {out}')
