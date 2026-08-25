"""
RMSE(depth) profile of vertical velocity `w` on the shelf (eta_rho index >
186, h < 100 m), each line a comparison against the notidesnowec base case
-- see postprocessing/calc_w_rmse_wec_shelf.py for the calculation. Answers:
does WEC change the strength of tidal-bore-driven w, beyond the WEC-driven w
signature that already exists without tides?
  ampwec_notides -- WEC alone (tides off)      -- vs notidesampwec
  tides_nowec    -- tides alone (no WEC)       -- vs tidesnowec
  tides_ampwec   -- tides + WEC together       -- vs tidesampwec
If WEC specifically amplifies/dampens tidal-bore strength (an interaction,
not just two independent effects stacking), tides_ampwec won't just track
tides_nowec offset by ampwec_notides's own magnitude.

Std overlay: this version adds a second family of thin, semi-transparent
lines -- Delta-std(depth) = std(scenario) - std(notidesnowec), from
postprocessing/calc_w_std.py -- on the SAME x-axis as RMSE (same units).
RMSE alone can't tell "more variable w" apart from "same variability,
phase-shifted relative to the base"; both inflate RMSE identically. Where
RMSE is large but Delta-std ~ 0 at a given depth, the difference there is
redistribution/phase, not a change in variability magnitude. Delta-std can
be negative (RMSE cannot), so a vertical zero line is drawn and the x-axis
now generally extends below zero. See plot_w_rmse_wec_shelf.py for the
RMSE-only version of this plot (unmodified).

Each line's color/label come from scenario_style.py (the "other" scenario in
that comparison), same as plot_profile_drhodz_100m.py's diff panel -- so
these lines are visually consistent with every other scenario-comparison
plot in the project. The Delta-std line reuses the same scenario color and
line width as the RMSE line, forced solid regardless of that scenario's
usual linestyle, distinguished only by reduced alpha and no legend entry of
its own.

Reads: ../postprocessing/w_rmse_wec_shelf.npz, ../postprocessing/w_std.npz
  (std overlay is skipped with a warning if w_std.npz doesn't exist yet)
Output: ./figs/w_rmse_wec_shelf_std.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scenario_style as ss

NPZ = '../postprocessing/w_rmse_wec_shelf.npz'
STD_NPZ = '../postprocessing/w_std.npz'
SAVEPATH = './figs/'
REGION = 'shelf'

BASE_SCEN = 'notidesnowec'
COMPARISONS = {
    'ampwec_notides': 'notidesampwec',
    'tides_nowec':     'tidesnowec',
    'tides_ampwec':    'tidesampwec',
}

plt.rcParams.update({'font.size': 12})

data = dict(np.load(NPZ, allow_pickle=False))
depth = data['depth']

if os.path.exists(STD_NPZ):
    std = dict(np.load(STD_NPZ, allow_pickle=False))
else:
    print(f'WARNING: missing {STD_NPZ} -- run calc_w_std.py first; '
          f'plotting RMSE only')
    std = None

# only show the depth range that actually has data -- h < 100 m in the mask
# means nothing much below about -100 m can ever have a valid sample
valid = np.zeros(depth.shape, dtype=bool)
for key in COMPARISONS:
    valid |= np.isfinite(data[f'rmse_{key}'])
    if std is not None:
        valid |= np.isfinite(std[f'dstd_{REGION}_{key}'])
depth_lim = depth[valid].min() if valid.any() else -100.0

fig, ax = plt.subplots(figsize=(6, 8))
if std is not None:
    ax.axvline(0.0, color='gray', lw=0.8, alpha=0.6, zorder=1)
for key, scen in COMPARISONS.items():
    kw = ss.line_kwargs(scen, base_lw=2.0, label=f'{ss.label(scen)}')
    ax.plot(data[f'rmse_{key}'], depth, **kw)
    if std is not None:
        kw_std = ss.line_kwargs(scen, base_lw=2.0, alpha=0.55,
                                linestyle='-', label='_nolegend_')
        ax.plot(std[f'dstd_{REGION}_{key}'], depth, **kw_std)

ax.set_ylim([depth_lim, 0])
ax.set_xlabel(r'RMSE and $\Delta$std of $w$ vs no tides, no WEC (m s$^{-1}$)')
ax.set_ylabel('Depth (m)')
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.grid(True, alpha=0.3)
#ax.legend(loc='best', fontsize=10)

plt.tight_layout()
os.makedirs(SAVEPATH, exist_ok=True)
fname = f'{SAVEPATH}w_rmse_wec_shelf_std.png'
plt.savefig(fname, dpi=800, bbox_inches='tight')
plt.close(fig)
print(f'saved -> {fname}')
