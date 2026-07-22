"""
Time- and horizontally-averaged depth profiles for the DIAT/SP nutrient-
limitation and uptake diagnostics. Sibling of plot_profile_zslice.py,
restricted to the 3 scenarios covered by the bgc_dia_avg rerun
(tidesampwec, tidesnowec, notidesnowec) and reading
zslice_profiles_bgcdia.npz / zslice_profiles_bgcdia_coastal.npz
(written by postprocessing/profile_zslice_bgcdia.py) from ../postprocessing/.

For each variable, writes one PNG with three side-by-side panels:
  left   — full-domain mean
  middle — coastal-mask mean
  right  — difference from baseline (notidesnowec)
plus a depth-zone breakdown PNG (0-50 m / 50-200 m / 200+ m).

Output: ./figs/profile_zslice_bgcdia_<var>.png  (18 files)
        ./figs/profile_zslice_bgcdia_<var>_depthzone.png  (18 files)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scenario_style as ss

NPZ_FULL    = '../postprocessing/zslice_profiles_bgcdia.npz'
NPZ_COASTAL = '../postprocessing/zslice_profiles_bgcdia_coastal.npz'

DIAT_LIM_VARS    = ['DIAT_N_LIM', 'DIAT_FE_LIM', 'DIAT_PO4_LIM',
                     'DIAT_SIO3_LIM', 'DIAT_LIGHT_LIM', 'DIAT_P_LIM']
SP_LIM_VARS      = ['SP_N_LIM', 'SP_FE_LIM', 'SP_PO4_LIM',
                     'SP_LIGHT_LIM', 'SP_P_LIM']
DIAT_UPTAKE_VARS = ['DIAT_NO3_UPTAKE', 'DIAT_NH4_UPTAKE', 'DIAT_NO2_UPTAKE',
                     'DIAT_SI_UPTAKE']
SP_UPTAKE_VARS   = ['SP_NO3_UPTAKE', 'SP_NH4_UPTAKE', 'SP_NO2_UPTAKE']

LIM_VARS    = DIAT_LIM_VARS + SP_LIM_VARS
UPTAKE_VARS = DIAT_UPTAKE_VARS + SP_UPTAKE_VARS
VARS        = LIM_VARS + UPTAKE_VARS

SCENARIOS = ['tidesampwec', 'tidesnowec', 'notidesnowec']
BASELINE  = 'notidesnowec'
DIFF_SCENS = [s for s in SCENARIOS if s != BASELINE]
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
}

VAR_UNITS = {v: '' for v in LIM_VARS}
VAR_UNITS.update({v: r'mmol m$^{-2}$ s$^{-1}$' for v in UPTAKE_VARS})

# LIM factors are bounded [0, 1] by definition — fix the x-axis for
# interpretability across panels. UPTAKE vars are left data-driven.
XLIM = {v: [0, 1] for v in LIM_VARS}

ZONE_KEYS = [
    ('h0to50',   '0–50 m'),
    ('h50to200', '50–200 m'),
    ('h200p',    '200+ m'),
]

plt.rcParams.update({'font.size': 12})

full    = dict(np.load(NPZ_FULL,    allow_pickle=False))
coastal = dict(np.load(NPZ_COASTAL, allow_pickle=False))

SAVEPATH = './figs/bgcprofiles/'
os.makedirs(SAVEPATH, exist_ok=True)

for var in VARS:
    depth = full['depth']

    # zsliced grid only spans 0 to -200 m (the rerun's z-grid) — use full range
    ylim = [float(depth.min()), 0]

    var_label = VAR_LONG_NAME.get(var, var)
    units = VAR_UNITS.get(var, '')
    xlabel = f'{var_label} [{units}]' if units else var_label

    # ── full domain + coastal + difference ───────────────────────────────────
    fig, (axL, axR, axD) = plt.subplots(1, 3, figsize=(15, 8), sharey=True)

    base_key = f'{var}_{BASELINE}'
    for scen in SCENARIOS:
        key = f'{var}_{scen}'
        if key not in full:
            continue
        prof_full    = full[key]
        prof_coastal = coastal[key] if key in coastal else np.full_like(prof_full, np.nan)
        lw = ss.lw(scen, base_lw=1.5)
        axL.plot(prof_full,    depth, color=COLORS[scen], linestyle=LSTYLES[scen], label=LABELS[scen], linewidth=lw)
        axR.plot(prof_coastal, depth, color=COLORS[scen], linestyle=LSTYLES[scen], linewidth=lw)
        if scen != BASELINE and base_key in full:
            diff = full[key] - full[base_key]
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
    axL.legend(loc='best', fontsize=9)
    axD.legend(loc='best', fontsize=9)

    plt.tight_layout()
    out = f'{SAVEPATH}profile_zslice_bgcdia_{var}.png'
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
            ax.plot(full[zk], depth, color=COLORS[scen], linestyle=LSTYLES[scen],
                    label=LABELS[scen], linewidth=lw)
            if scen != BASELINE and base_zk in full:
                diff = full[zk] - full[base_zk]
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
    out = f'{SAVEPATH}profile_zslice_bgcdia_{var}_depthzone.png'
    plt.savefig(out, bbox_inches='tight', dpi=800)
    plt.close(fig)
    print(f'saved -> {out}')
