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

Each line's color/linestyle/label/linewidth come from scenario_style.py
(the "other" scenario in that comparison), same as plot_profile_drhodz_100m.py's
diff panel -- so these lines are visually consistent with every other
scenario-comparison plot in the project.

Reads: ../postprocessing/w_rmse_wec_shelf.npz
Output: ./figs/w_rmse_wec_shelf.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scenario_style as ss

NPZ = '../postprocessing/w_rmse_wec_shelf.npz'
SAVEPATH = './figs/'

BASE_SCEN = 'notidesnowec'
COMPARISONS = {
    'ampwec_notides': 'notidesampwec',
    'tides_nowec':     'tidesnowec',
    'tides_ampwec':    'tidesampwec',
}

plt.rcParams.update({'font.size': 12})

data = dict(np.load(NPZ, allow_pickle=False))
depth = data['depth']

# only show the depth range that actually has data -- h < 100 m in the mask
# means nothing much below about -100 m can ever have a valid sample
valid = np.zeros(depth.shape, dtype=bool)
for key in COMPARISONS:
    valid |= np.isfinite(data[f'rmse_{key}'])
depth_lim = depth[valid].min() if valid.any() else -100.0

fig, ax = plt.subplots(figsize=(6, 8))
for key, scen in COMPARISONS.items():
    kw = ss.line_kwargs(scen, base_lw=2.0, label=f'{ss.label(scen)}')
    ax.plot(data[f'rmse_{key}'], depth, **kw)

ax.set_ylim([depth_lim, 0])
ax.set_xlabel(r'RMSE of $w$ vs no tides, no WEC (m s$^{-1}$)')
ax.set_ylabel('Depth (m)')
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs(SAVEPATH, exist_ok=True)
fname = f'{SAVEPATH}w_rmse_wec_shelf.png'
plt.savefig(fname, dpi=800, bbox_inches='tight')
plt.close(fig)
print(f'saved -> {fname}')
