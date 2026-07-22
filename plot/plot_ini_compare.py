"""
Compare initial conditions between tides and notides cases using zsliced files.
Averages each variable across the domain (masked to ocean cells) and plots
side-by-side profiles: both overlaid on the left, difference on the right.

Reads:
  tides:   /data/project3/minnaho/project9copy/swel/tides/mc60/rst/z_mc60_ini.20190415110120.nc
  notides: /data/project3/minnaho/project9copy/swel/notides/mc60/rst/z_mc60_ini.20190415110120.nc

Output: ./figs/ini_compare_<var>.png
"""

import os
import numpy as np
import seawater
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

TIDES_Z   = '/data/project3/minnaho/project9copy/swel/tides/mc60/rst/z_mc60_ini.20190415110120.nc'
NOTIDES_Z = '/data/project3/minnaho/project9copy/swel/notides/mc60/rst/z_mc60_ini.20190415110120.nc'
GRD       = 'mc60_grd.nc'

VAR_UNITS = {
    'temp':    '°C',
    'salt':    'PSU',
    'NO3':     'mmol N m⁻³',
    'PO4':     'mmol P m⁻³',
    'SiO3':    'mmol Si m⁻³',
    'Fe':      'mmol Fe m⁻³',
    'O2':      'mmol O2 m⁻³',
    'NH4':     'mmol N m⁻³',
    'DIC':     'mmol C m⁻³',
    'Alk':     'mmol m⁻³',
    'DOC':     'mmol C m⁻³',
    'DON':     'mmol N m⁻³',
    'DONR':    'mmol N m⁻³',
    'DOP':     'mmol P m⁻³',
    'DOPR':    'mmol P m⁻³',
    'DOFE':    'mmol Fe m⁻³',
    'NO2':     'mmol N m⁻³',
    'N2':      'mmol N2 m⁻³',
    'N2O':     'mmol N2O m⁻³',
    'SPC':     'mmol C m⁻³',
    'SPCHL':   'mg Chl m⁻³',
    'SPFE':    'mmol Fe m⁻³',
    'SPCACO3': 'mmol CaCO3 m⁻³',
    'DIATC':   'mmol C m⁻³',
    'DIATCHL': 'mg Chl m⁻³',
    'DIATFE':  'mmol Fe m⁻³',
    'DIATSI':  'mmol Si m⁻³',
    'DIAZC':   'mmol C m⁻³',
    'DIAZCHL': 'mg Chl m⁻³',
    'DIAZFE':  'mmol Fe m⁻³',
    'ZOOC':    'mmol C m⁻³',
}

_S150 = [-150, 0]
_S200 = [-200, 0]
_S500 = [-500, 0]
_S80  = [-80,  0]
_S40  = [-40,  0]

DEPTH_YLIM = {
    'temp':     _S500,
    'salt':     _S500,
    'sigma_t':  _S500,
    'NO3':      _S150,
    'PO4':      _S200,
    'SiO3':     _S200,
    'Fe':       _S200,
    'O2':       _S500,
    'NH4':      _S80,
    'DIC':      _S500,
    'Alk':      _S500,
    'DOC':      _S200,
    'DON':      _S200,
    'DONR':     _S200,
    'DOP':      _S200,
    'DOPR':     _S200,
    'DOFE':     _S200,
    'NO2':      _S200,
    'N2':       _S500,
    'N2O':      _S500,
    'SPC':      _S150,
    'SPCHL':    _S150,
    'SPFE':     _S150,
    'SPCACO3':  _S150,
    'DIATC':    _S150,
    'DIATCHL':  _S150,
    'DIATFE':   _S150,
    'DIATSI':   _S150,
    'DIAZC':    _S150,
    'DIAZCHL':  _S150,
    'DIAZFE':   _S150,
    'ZOOC':     _S200,
    'TOTC':     _S150,
}

os.makedirs('./figs/ini_compare', exist_ok=True)

grdnc    = Dataset(GRD)
mask_rho = np.array(grdnc['mask_rho'][:]).astype(float)
h        = np.array(grdnc['h'][:])               # bathymetry depth (positive, m)
mask_rho[mask_rho == 0] = np.nan

def make_h_mask(h_lo, h_hi):
    """Spatial mask: nan for land, nan where h <= h_lo or h > h_hi."""
    m = mask_rho.copy()
    m[(h <= h_lo) | (h > h_hi)] = np.nan
    return m

ZONE_DEFS = [
    ('0–50 m',   make_h_mask(0,   50)),
    ('50–200 m', make_h_mask(50,  200)),
    ('200+ m',   make_h_mask(200, np.inf)),
]



plt.rcParams.update({'font.size': 12})

with Dataset(TIDES_Z, 'r') as t_nc, Dataset(NOTIDES_Z, 'r') as nt_nc:
    depth = np.array(t_nc.variables['depth'][:])   # (n_z,) negative

    skip = {'depth', 'lon_rho', 'lat_rho', 'mask_rho', 'ocean_time',
            'tstart', 'tend', 'theta_s', 'theta_b', 'Tcline', 'hc',
            'sc_r', 'Cs_r'}

    vars_to_plot = [v for v in t_nc.variables
                    if v not in skip and v in nt_nc.variables]

    for var in vars_to_plot:
        arr_t  = np.squeeze(np.array(t_nc.variables[var][:]))
        arr_nt = np.squeeze(np.array(nt_nc.variables[var][:]))
        arr_t  = np.where(np.abs(arr_t)  > 1e30, np.nan, arr_t)
        arr_nt = np.where(np.abs(arr_nt) > 1e30, np.nan, arr_nt)

        prof_t  = np.nanmean(arr_t  * mask_rho[None, :, :], axis=(1, 2))
        prof_nt = np.nanmean(arr_nt * mask_rho[None, :, :], axis=(1, 2))
        diff    = prof_t - prof_nt

        units  = VAR_UNITS.get(var, '')
        ylabel = f'{var} [{units}]' if units else var
        ylim   = DEPTH_YLIM.get(var, [-500, 0])

        # ── profile figure ───────────────────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7), sharey=True)

        ax1.plot(prof_t,  depth, color='steelblue',  linewidth=1.8, label='tides')
        ax1.plot(prof_nt, depth, color='darkorange', linewidth=1.8,
                 linestyle='--', label='no tides')
        ax1.set_title('Profiles')
        ax1.legend(fontsize=10)

        ax2.plot(diff, depth, color='k', linewidth=1.8)
        ax2.axvline(0, color='k', linewidth=0.5, alpha=0.5)
        ax2.set_title('Tides − No tides')

        for ax in (ax1, ax2):
            ax.set_xlabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3g'))

        ax1.set_ylabel('Depth (m)')
        ax1.set_ylim(ylim)

        plt.tight_layout()
        out = f'./figs/ini_compare/{var}.png'
        plt.savefig(out, bbox_inches='tight', dpi=800)
        plt.close(fig)
        print(f'saved -> {out}')

        # ── depth-zone profile figure ────────────────────────────────────────
        n_zone = len(ZONE_DEFS)
        fig, axes = plt.subplots(2, n_zone, figsize=(5 * n_zone, 14), sharey=True)
        for col, (zlabel, h_mask) in enumerate(ZONE_DEFS):
            ax   = axes[0, col]
            ax_d = axes[1, col]
            prof_zt  = np.nanmean(arr_t  * h_mask[None, :, :], axis=(1, 2))
            prof_znt = np.nanmean(arr_nt * h_mask[None, :, :], axis=(1, 2))
            zdiff = prof_zt - prof_znt
            ax.plot(prof_zt,  depth, color='steelblue',  linewidth=1.8, label='tides')
            ax.plot(prof_znt, depth, color='darkorange', linewidth=1.8,
                    linestyle='--', label='no tides')
            ax_d.plot(zdiff, depth, color='k', linewidth=1.8)
            ax_d.axvline(0, color='k', linewidth=0.5, alpha=0.5)
            ax.set_title(zlabel)
            ax_d.set_title(f'{zlabel} − no tides')
            for ax_ in (ax, ax_d):
                ax_.set_xlabel(ylabel)
                ax_.set_ylim(ylim)
                ax_.grid(True, alpha=0.3)
                ax_.xaxis.set_major_locator(MaxNLocator(4))
                ax_.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3g'))
        axes[0, 0].set_ylabel('Depth (m)')
        axes[1, 0].set_ylabel('Depth (m)')
        axes[0, 0].legend(fontsize=10)

        plt.tight_layout()
        out2 = f'./figs/ini_compare/{var}_depthzone.png'
        plt.savefig(out2, bbox_inches='tight', dpi=800)
        plt.close(fig)
        print(f'saved -> {out2}')

    # ── total phytoplankton carbon (DIATC + DIAZC + SPC) ────────────────────
    phyto_vars = ['DIATC', 'DIAZC', 'SPC']
    if all(v in t_nc.variables and v in nt_nc.variables for v in phyto_vars):
        def load_sum(nc, varnames):
            s = None
            for v in varnames:
                arr = np.squeeze(np.array(nc.variables[v][:]))
                arr = np.where(np.abs(arr) > 1e30, np.nan, arr)
                s = arr if s is None else s + arr
            return s

        arr_t  = load_sum(t_nc,  phyto_vars)
        arr_nt = load_sum(nt_nc, phyto_vars)

        var    = 'TOTC'
        units  = 'mmol C m⁻³'
        ylabel = f'Total phyto C [{units}]'
        ylim   = DEPTH_YLIM.get(var, [-500, 0])

        prof_t  = np.nanmean(arr_t  * mask_rho[None, :, :], axis=(1, 2))
        prof_nt = np.nanmean(arr_nt * mask_rho[None, :, :], axis=(1, 2))
        diff    = prof_t - prof_nt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
        ax1.plot(prof_t,  depth, color='steelblue',  linewidth=1.8, label='tides')
        ax1.plot(prof_nt, depth, color='darkorange', linewidth=1.8,
                 linestyle='--', label='no tides')
        ax1.set_title('Profiles')
        ax1.legend(fontsize=10)
        ax2.plot(diff, depth, color='k', linewidth=1.8)
        ax2.axvline(0, color='k', linewidth=0.5, alpha=0.5)
        ax2.set_title('Tides − No tides')
        for ax in (ax1, ax2):
            ax.set_xlabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3g'))
        ax1.set_ylabel('Depth (m)')
        ax1.set_ylim(ylim)
        plt.tight_layout()
        out = f'./figs/ini_compare/{var}.png'
        plt.savefig(out, bbox_inches='tight', dpi=800)
        plt.close(fig)
        print(f'saved -> {out}')

        n_zone = len(ZONE_DEFS)
        fig, axes = plt.subplots(2, n_zone, figsize=(5 * n_zone, 14), sharey=True)
        for col, (zlabel, h_mask) in enumerate(ZONE_DEFS):
            ax   = axes[0, col]
            ax_d = axes[1, col]
            prof_zt  = np.nanmean(arr_t  * h_mask[None, :, :], axis=(1, 2))
            prof_znt = np.nanmean(arr_nt * h_mask[None, :, :], axis=(1, 2))
            zdiff = prof_zt - prof_znt
            ax.plot(prof_zt,  depth, color='steelblue',  linewidth=1.8, label='tides')
            ax.plot(prof_znt, depth, color='darkorange', linewidth=1.8,
                    linestyle='--', label='no tides')
            ax_d.plot(zdiff, depth, color='k', linewidth=1.8)
            ax_d.axvline(0, color='k', linewidth=0.5, alpha=0.5)
            ax.set_title(zlabel)
            ax_d.set_title(f'{zlabel} − no tides')
            for ax_ in (ax, ax_d):
                ax_.set_xlabel(ylabel)
                ax_.set_ylim(ylim)
                ax_.grid(True, alpha=0.3)
                ax_.xaxis.set_major_locator(MaxNLocator(4))
                ax_.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3g'))
        axes[0, 0].set_ylabel('Depth (m)')
        axes[1, 0].set_ylabel('Depth (m)')
        axes[0, 0].legend(fontsize=10)
        plt.tight_layout()
        out2 = f'./figs/ini_compare/{var}_depthzone.png'
        plt.savefig(out2, bbox_inches='tight', dpi=800)
        plt.close(fig)
        print(f'saved -> {out2}')

    # ── potential density (sigma-t) ──────────────────────────────────────────
    if 'temp' in t_nc.variables and 'salt' in t_nc.variables:
        def load_clean(nc, v):
            arr = np.squeeze(np.array(nc.variables[v][:]))
            return np.where(np.abs(arr) > 1e30, np.nan, arr)

        arr_t  = seawater.dens0(load_clean(t_nc,  'salt'), load_clean(t_nc,  'temp')) - 1000.0
        arr_nt = seawater.dens0(load_clean(nt_nc, 'salt'), load_clean(nt_nc, 'temp')) - 1000.0

        ylabel = r'$\sigma_t$ [kg m$^{-3}$]'
        ylim   = DEPTH_YLIM.get('sigma_t', [-500, 0])

        prof_t  = np.nanmean(arr_t  * mask_rho[None, :, :], axis=(1, 2))
        prof_nt = np.nanmean(arr_nt * mask_rho[None, :, :], axis=(1, 2))
        diff    = prof_t - prof_nt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
        ax1.plot(prof_t,  depth, color='steelblue',  linewidth=1.8, label='tides')
        ax1.plot(prof_nt, depth, color='darkorange', linewidth=1.8,
                 linestyle='--', label='no tides')
        ax1.set_title('Profiles')
        ax1.legend(fontsize=10)
        ax2.plot(diff, depth, color='k', linewidth=1.8)
        ax2.axvline(0, color='k', linewidth=0.5, alpha=0.5)
        ax2.set_title('Tides − No tides')
        for ax in (ax1, ax2):
            ax.set_xlabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3g'))
        ax1.set_ylabel('Depth (m)')
        ax1.set_ylim(ylim)
        plt.tight_layout()
        out = './figs/ini_compare/sigma_t.png'
        plt.savefig(out, bbox_inches='tight', dpi=800)
        plt.close(fig)
        print(f'saved -> {out}')

        n_zone = len(ZONE_DEFS)
        fig, axes = plt.subplots(2, n_zone, figsize=(5 * n_zone, 14), sharey=True)
        for col, (zlabel, h_mask) in enumerate(ZONE_DEFS):
            ax   = axes[0, col]
            ax_d = axes[1, col]
            prof_zt  = np.nanmean(arr_t  * h_mask[None, :, :], axis=(1, 2))
            prof_znt = np.nanmean(arr_nt * h_mask[None, :, :], axis=(1, 2))
            zdiff = prof_zt - prof_znt
            ax.plot(prof_zt,  depth, color='steelblue',  linewidth=1.8, label='tides')
            ax.plot(prof_znt, depth, color='darkorange', linewidth=1.8,
                    linestyle='--', label='no tides')
            ax_d.plot(zdiff, depth, color='k', linewidth=1.8)
            ax_d.axvline(0, color='k', linewidth=0.5, alpha=0.5)
            ax.set_title(zlabel)
            ax_d.set_title(f'{zlabel} − no tides')
            for ax_ in (ax, ax_d):
                ax_.set_xlabel(ylabel)
                ax_.set_ylim(ylim)
                ax_.grid(True, alpha=0.3)
                ax_.xaxis.set_major_locator(MaxNLocator(4))
                ax_.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3g'))
        axes[0, 0].set_ylabel('Depth (m)')
        axes[1, 0].set_ylabel('Depth (m)')
        axes[0, 0].legend(fontsize=10)
        plt.tight_layout()
        out2 = './figs/ini_compare/sigma_t_depthzone.png'
        plt.savefig(out2, bbox_inches='tight', dpi=800)
        plt.close(fig)
        print(f'saved -> {out2}')
