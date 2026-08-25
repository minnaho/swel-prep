"""
2x4 grid combining the w/vort standalone RMSE/Delta-std profile plots and
the combined du/dz and drho/dz RMSE/std/RMS outputs into one figure, each
panel labeled (A)-(H) in its top-left corner:
  (A) w,          shelf     (plot_w_rmse_wec_shelf_std.py)
  (B) zeta/f,     shelf     (plot_vort_rmse_wec_shelf_std.py)
  (C) du/dz,      shelf     (calc_dudz_rmse_wec.py)
  (D) drho/dz,    shelf     (calc_drhodz_rmse_wec.py)
  (E) w,          offshore  (plot_w_rmse_wec_offshore_std.py)
  (F) zeta/f,     offshore  (plot_vort_rmse_wec_offshore_std.py)
  (G) du/dz,      offshore  (calc_dudz_rmse_wec.py)
  (H) drho/dz,    offshore  (calc_drhodz_rmse_wec.py)

Each panel is the same RMSE(depth)-vs-Delta-std(depth) comparison as its
standalone counterpart (see those scripts'/calc_dudz_rmse_wec.py's/
calc_drhodz_rmse_wec.py's docstrings for the full methodology) -- this
script just re-reads the existing npz outputs and lays them out together
for a single figure. The standalone scripts are left untouched.

w and zeta/f each have RMSE and std split across two separate npz files per
region (w_rmse_wec_<region>.npz + w_std.npz); du/dz and drho/dz each have
both already combined, region-qualified, in one npz (dudz_rmse_wec.npz /
drhodz_rmse_wec.npz), since calc_dudz_rmse_wec.py / calc_drhodz_rmse_wec.py
compute rmse/std/rms together in a single pass -- the two loader functions
below account for that difference in npz layout, but feed the same
plot_panel() so all 8 panels render identically.

Legend is drawn only in the bottom-left (w, offshore) panel, per the other
7 panels having no legend of their own.

Reads: ../postprocessing/w_rmse_wec_shelf.npz, w_rmse_wec_offshore.npz,
  vort_rmse_wec_shelf.npz, vort_rmse_wec_offshore.npz, w_std.npz, vort_std.npz,
  dudz_rmse_wec.npz, drhodz_rmse_wec.npz
  (a w/vort panel's Delta-std line is skipped with a warning if its std npz
  is missing, same as the standalone scripts)
Output: ./figs/rmse_std_grid.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scenario_style as ss

SAVEPATH = './figs/'

BASE_SCEN = 'notidesnowec'
COMPARISONS = {
    'ampwec_notides': 'notidesampwec',
    'tides_nowec':     'tidesnowec',
    'tides_ampwec':    'tidesampwec',
}

# (loader, loader_args, xlabel, title, show_legend)
PANELS = [
    ('w_vort', ('../postprocessing/w_rmse_wec_shelf.npz', '../postprocessing/w_std.npz', 'shelf'),
     r'RMSE and $\Delta$std of $w$ (m s$^{-1}$)', r'$w$, shelf', False),
    ('w_vort', ('../postprocessing/vort_rmse_wec_shelf.npz', '../postprocessing/vort_std.npz', 'shelf'),
     r'RMSE and $\Delta$std of $\zeta/f$', r'$\zeta/f$, shelf', False),
    ('combined', ('../postprocessing/dudz_rmse_wec.npz', 'shelf'),
     r'RMSE and $\Delta$std of $\partial u/\partial z$ (s$^{-1}$)',
     r'$\partial u/\partial z$, shelf', False),
    ('combined', ('../postprocessing/drhodz_rmse_wec.npz', 'shelf'),
     r'RMSE and $\Delta$std of $\partial \rho/\partial z$ (kg m$^{-4}$)',
     r'$\partial \rho/\partial z$, shelf', False),
    ('w_vort', ('../postprocessing/w_rmse_wec_offshore.npz', '../postprocessing/w_std.npz', 'offshore'),
     r'RMSE and $\Delta$std of $w$ (m s$^{-1}$)', r'$w$, offshore', True),
    ('w_vort', ('../postprocessing/vort_rmse_wec_offshore.npz', '../postprocessing/vort_std.npz', 'offshore'),
     r'RMSE and $\Delta$std of $\zeta/f$', r'$\zeta/f$, offshore', False),
    ('combined', ('../postprocessing/dudz_rmse_wec.npz', 'offshore'),
     r'RMSE and $\Delta$std of $\partial u/\partial z$ (s$^{-1}$)',
     r'$\partial u/\partial z$, offshore', False),
    ('combined', ('../postprocessing/drhodz_rmse_wec.npz', 'offshore'),
     r'RMSE and $\Delta$std of $\partial \rho/\partial z$ (kg m$^{-4}$)',
     r'$\partial \rho/\partial z$, offshore', False),
]

plt.rcParams.update({'font.size': 12})


def load_w_vort(rmse_npz, std_npz, region):
    data = dict(np.load(rmse_npz, allow_pickle=False))
    depth = data['depth']
    rmse = {key: data[f'rmse_{key}'] for key in COMPARISONS}
    if os.path.exists(std_npz):
        std = dict(np.load(std_npz, allow_pickle=False))
        dstd = {key: std[f'dstd_{region}_{key}'] for key in COMPARISONS}
    else:
        print(f'WARNING: missing {std_npz} -- run the matching calc_*_std.py '
              f'first; plotting RMSE only for {region}')
        dstd = None
    return depth, rmse, dstd


def load_combined(npz_path, region):
    data = dict(np.load(npz_path, allow_pickle=False))
    depth = data['depth']
    rmse = {key: data[f'rmse_{region}_{key}'] for key in COMPARISONS}
    dstd = {key: data[f'dstd_{region}_{key}'] for key in COMPARISONS}
    return depth, rmse, dstd


def plot_panel(ax, depth, rmse, dstd, xlabel, title, show_legend, panel_label):
    valid = np.zeros(depth.shape, dtype=bool)
    for key in COMPARISONS:
        valid |= np.isfinite(rmse[key])
        if dstd is not None:
            valid |= np.isfinite(dstd[key])
    depth_lim = depth[valid].min() if valid.any() else depth.min()

    if dstd is not None:
        ax.axvline(0.0, color='gray', lw=0.8, alpha=0.6, zorder=1)
    for key, scen in COMPARISONS.items():
        kw = ss.line_kwargs(scen, base_lw=2.0, label=f'{ss.label(scen)}')
        ax.plot(rmse[key], depth, **kw)
        if dstd is not None:
            kw_std = ss.line_kwargs(scen, base_lw=2.0, alpha=0.55,
                                    linestyle='-', label='_nolegend_')
            ax.plot(dstd[key], depth, **kw_std)

    ax.set_ylim([depth_lim, 0])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f'({panel_label})', transform=ax.transAxes,
            fontsize=14, fontweight='bold', ha='left', va='top')
    if show_legend:
        ax.legend(loc='upper right', fontsize=12)


LOADERS = {'w_vort': load_w_vort, 'combined': load_combined}

PANEL_LABELS = 'ABCDEFGH'

fig, axes = plt.subplots(2, 4, figsize=(19, 13))
for (kind, loader_args, xlabel, title, show_legend), ax, panel_label in zip(
        PANELS, axes.flat, PANEL_LABELS):
    depth, rmse, dstd = LOADERS[kind](*loader_args)
    plot_panel(ax, depth, rmse, dstd, xlabel, title, show_legend, panel_label)

axes[0, 0].set_ylabel('Depth (m)')
axes[1, 0].set_ylabel('Depth (m)')

for col in range(4):
    axes[0, col].set_xlabel('')
    if col > 0:
        axes[0, col].tick_params(axis='y', labelleft=False)
        axes[1, col].tick_params(axis='y', labelleft=False)

plt.tight_layout()
os.makedirs(SAVEPATH, exist_ok=True)
fname = f'{SAVEPATH}rmse_std_grid.png'
plt.savefig(fname, dpi=800, bbox_inches='tight')
plt.close(fig)
print(f'saved -> {fname}')
