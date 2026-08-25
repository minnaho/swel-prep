"""
Time- and horizontally-averaged depth profiles for every zsliced variable.
Reads zslice_profiles.npz (full domain) and zslice_profiles_coastal.npz
(10 km coastal mask) from ../postprocessing/.

For each variable, writes one PNG with two side-by-side panels:
  left  — full-domain mean
  right — coastal-mask mean
Four scenarios are overlaid as colored lines on each panel.

Output: ./figs/profile_zslice_<var>.png  (18 files total)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scenario_style as ss

NPZ_FULL    = '../postprocessing/zslice_profiles.npz'
NPZ_COASTAL = '../postprocessing/zslice_profiles_coastal.npz'

VARS_HIS = ['ptrace', 'rtrace', 'w', 'rho', 'u', 'v']
VARS_BGC = ['NO3', 'NH4', 'SPC', 'DIATC', 'DIAZC', 'TOTC',
            'SPCHL', 'DIATCHL', 'DIAZCHL', 'O2', 'DIC', 'DOC']
VARS_DIA = ['TOT_PROD']
TOTC_VARS = ['DIATC', 'DIAZC', 'SPC']

SCENARIOS = ['tideswec', 'tidesnowec', 'notidesnowec', 'notideswec', 'ampwec', 'tidesampwec']
BASELINE  = 'notidesnowec'
DIFF_SCENS = [s for s in SCENARIOS if s != BASELINE]
LABELS  = ss.LABELS
COLORS  = ss.COLORS
LSTYLES = ss.LSTYLES

# y-axis depth range per variable (bottom, top); None = use data min
_SURF150  = [-150, 0]
_SURF120  = [-120, 0]
_SURF200  = [-200, 0]
_MID500   = [-500, 0]
_s40   = [-40, 0]
_s80   = [-80, 0]
DEPTH_YLIM = {
    'ptrace':   _s80,
    'rtrace':   _s40,
    'w':        _MID500,
    'u':        _MID500,
    'v':        _MID500,
    'rho':      _SURF120,        # None for full depth
    'NO3':      _SURF150,
    'NH4':      _s80,
    'O2':       _MID500,
    'DIC':      _MID500,
    'DOC':      _MID500,
    'SPC':      _SURF150,
    'DIATC':    _SURF150,
    'DIAZC':    _SURF150,
    'SPCHL':    _SURF150,
    'DIATCHL':  _SURF150,
    'DIAZCHL':  _SURF150,
    'TOTC':     _SURF150,
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

ZONE_KEYS = [
    ('h0to50',   '0–50 m'),
    ('h50to200', '50–200 m'),
    ('h200p',    '200+ m'),
]

plt.rcParams.update({'font.size': 12})

full    = dict(np.load(NPZ_FULL,    allow_pickle=False))
coastal = dict(np.load(NPZ_COASTAL, allow_pickle=False))

# pre-compute TOTC = DIATC + DIAZC + SPC for full, coastal, and all zone prefixes
_zone_prefixes = [''] + [f'{zk}_' for zk, _ in ZONE_KEYS]
for scen in SCENARIOS:
    for pfx in _zone_prefixes:
        src_keys = [f'{pfx}{v}_{scen}' for v in TOTC_VARS]
        if all(k in full for k in src_keys):
            full[f'{pfx}TOTC_{scen}'] = sum(full[k] for k in src_keys)
        src_keys_c = [f'{v}_{scen}' for v in TOTC_VARS]   # coastal npz has no prefix
        if all(k in coastal for k in src_keys_c):
            coastal[f'TOTC_{scen}'] = sum(coastal[k] for k in src_keys_c)

SAVEPATH = './figs/profiles/'
os.makedirs(SAVEPATH, exist_ok=True)

for var in VARS_HIS + VARS_BGC + VARS_DIA:
    depth_key = 'depth_dia' if var in VARS_DIA else 'depth'
    depth = full[depth_key]

    ylim = DEPTH_YLIM.get(var, None)
    if ylim is None:
        ylim = [float(depth.min()), 0]

    units = VAR_UNITS.get(var, '')
    var_label = VAR_LABELS.get(var, var)
    xlabel = f'{var_label} [{units}]' if units else var_label
    scale = 86400.0 if var == 'TOT_PROD' else 1.0

    # ── full domain + coastal + difference ───────────────────────────────────
    fig, (axL, axR, axD) = plt.subplots(1, 3, figsize=(15, 8), sharey=True)

    base_key = f'{var}_{BASELINE}'
    for scen in SCENARIOS:
        key = f'{var}_{scen}'
        if key not in full:
            continue
        prof_full    = full[key] * scale
        prof_coastal = (coastal[key] * scale) if key in coastal else np.full_like(prof_full, np.nan)
        lw = ss.lw(scen, base_lw=1.5)
        axL.plot(prof_full,    depth, color=COLORS[scen], linestyle=LSTYLES[scen], label=LABELS[scen], linewidth=lw)
        axR.plot(prof_coastal, depth, color=COLORS[scen], linestyle=LSTYLES[scen], linewidth=lw)
        if scen != BASELINE and base_key in full:
            diff = (full[key] - full[base_key]) * scale
            axD.plot(diff, depth, color=COLORS[scen], linestyle=LSTYLES[scen],
                     label=f'{LABELS[scen]} − baseline', linewidth=lw)

    axL.set_ylim(ylim)
    for ax in (axL, axR, axD):
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3g'))
    for ax in (axL, axR):
        if var in XLIM:
            ax.set_xlim(XLIM[var])

    axL.set_ylabel('depth (m)')
    axL.set_title('full domain')
    axR.set_title('coastal (10 km)')
    axD.set_title(f'difference from {LABELS[BASELINE]}')
    axL.legend(loc='best', fontsize=12)
    axD.legend(loc='best', fontsize=12)

    plt.tight_layout()
    out = f'{SAVEPATH}profile_zslice_{var}.png'
    plt.savefig(out, bbox_inches='tight', dpi=800)
    plt.close(fig)
    print(f'saved -> {out}')

    # ── depth zones: row 0 = profiles, row 1 = differences ──────────────────
    n_zone = len(ZONE_KEYS)
    fig, axes = plt.subplots(2, n_zone, figsize=(5 * n_zone, 14), sharey=True)
    for col, (zkey, zlabel) in enumerate(ZONE_KEYS):
        ax   = axes[0, col]
        ax_d = axes[1, col]
        base_zk = f'{zkey}_{var}_{BASELINE}'
        for scen in SCENARIOS:
            zk = f'{zkey}_{var}_{scen}'
            if zk not in full:
                continue
            lw = ss.lw(scen, base_lw=1.5)
            ax.plot(full[zk] * scale, depth, color=COLORS[scen], linestyle=LSTYLES[scen],
                    label=LABELS[scen], linewidth=lw)
            if scen != BASELINE and base_zk in full:
                diff = (full[zk] - full[base_zk]) * scale
                ax_d.plot(diff, depth, color=COLORS[scen], linestyle=LSTYLES[scen],
                          linewidth=lw)
        for ax_ in (ax, ax_d):
            ax_.set_ylim(ylim)
            ax_.axvline(0, color='k', linewidth=0.5, alpha=0.5)
            ax_.set_xlabel(xlabel)
            ax_.grid(True, alpha=0.3)
            ax_.xaxis.set_major_locator(MaxNLocator(4))
            ax_.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3g'))
        if var in XLIM:
            ax.set_xlim(XLIM[var])
        ax.set_title(zlabel)
        ax_d.set_title(f'{zlabel} − baseline')
    axes[0, 0].set_ylabel('depth (m)')
    axes[1, 0].set_ylabel('depth (m)')
    axes[0, 0].legend(loc='best', fontsize=9)
    plt.tight_layout()
    out = f'{SAVEPATH}profile_zslice_{var}_depthzone.png'
    plt.savefig(out, bbox_inches='tight', dpi=800)
    plt.close(fig)
    print(f'saved -> {out}')

