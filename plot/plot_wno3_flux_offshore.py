"""
h>100m-restricted version of plot_wno3_flux.py -- offshore counterpart of
plot_wno3_flux_100m.py, same 4-panel layout (time series, envelope, PDF, box
plot) for w'NO3' eddy flux at 10m, 20m, and 30m, plus the Akt*dNO3/dz
diffusive flux at 20m and 30m, all reading from the offshore-only (h>100m)
npz files written by calc_wno3_flux_10m_offshore.py /
calc_wno3_flux_20m_offshore.py / calc_wno3_flux_30m_offshore.py /
calc_akt_dno3dz_20m_offshore.py / calc_akt_dno3dz_30m_offshore.py instead of
the shelf-restricted ones. The diffusive-flux entries use the exact same
4-panel machinery even though they aren't an eddy correlation -- their npz
schema was deliberately built to match (see calc_akt_dno3dz_30m_100m.py's
docstring), and their sign is flipped at plot time (scale_ts/scale_pdf =
-1.0) to the standard downgradient-diffusion convention rather than
rerunning the calc scripts (see the AktdNO3dz_30m entry below).

No ptrace/rtrace entries here (unlike plot_wno3_flux_100m.py) -- no offshore
wptrace/wrtrace npz data exists yet (only the shelf-restricted _100m/_20m_100m
files do); add calc_wtrace_flux_offshore.py-style calc scripts first if those
are wanted here too.

Layout: row 1 = time series (full width), row 2 = envelope (full width),
row 3 = PDF (wide) + box plot (narrow), the box plot next to the PDF since
both summarize the same flux-value distribution (box plot: per-timestep
domain-mean values only, see the mean-time-series note below; PDF: every
unmasked pixel at every timestep, pooled).

Output: ./figs/wno3_flux_offshore.png, wno3_flux_20m_offshore.png,
        wno3_flux_30m_offshore.png, akt_dno3dz_flux_20m_offshore.png,
        akt_dno3dz_flux_30m_offshore.png
"""

import sys
sys.path.append('/data/project3/minnaho/global/')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import pyfuncs as pf
import scenario_style as ss

SCENARIOS = ['notidesnowec', 'ampwec', 'tidesnowec', 'tidesampwec']

labels  = {n: ss.label(n) for n in SCENARIOS}
colors  = {n: ss.color(n) for n in SCENARIOS}
lstyles = {n: ss.ls(n)    for n in SCENARIOS}

# ── per-tracer config ─────────────────────────────────────────────────────────
# ts_npz  : NPZ with the time series (key 'time_series' or 'ts_mean')
# ts_key  : key name for the mean time series
# env_npz : NPZ with ts_min / ts_max / bin_centers / pdf
# scale_ts  : multiply time series values for readability
# scale_pdf : multiply bin_centers for readability
# units   : physical unit string for axis labels
# math    : LaTeX math symbol for the variable
# ylim_ts/ylim_env/ylim_pdf/ylim_box : [ymin, ymax] applied to that panel via
#   set_ylim, or None for matplotlib's auto range (default). Independent per
#   panel/tracer -- e.g. AktdNO3dz_20m's box plot can have its own range
#   without touching NO3_30m's time series. ylim_pdf is on a log y-axis, so
#   both bounds must be positive if set.

TRACERS = {
    'NO3': dict(
        ts_npz   = lambda n: f'../postprocessing/wno3_flux_offshore_{n}.npz',
        ts_key   = 'time_series',
        env_npz  = lambda n: f'../postprocessing/wno3_env_offshore_{n}.npz',
        scale_ts  = 1.0,
        scale_pdf = 1.0,
        units    = r'mmol N m$^{-2}$ s$^{-1}$',
        math     = r"$w'NO_3'$ (10 m, h>100 m)",
        outfile  = './figs/wno3_flux_offshore.png',
        ylim_ts  = None,
        ylim_env = None,
        ylim_pdf = None,
        ylim_box = None,
    ),
    'NO3_20m': dict(
        ts_npz   = lambda n: f'../postprocessing/wno3_flux_20m_offshore_{n}.npz',
        ts_key   = 'time_series',
        env_npz  = lambda n: f'../postprocessing/wno3_env_20m_offshore_{n}.npz',
        scale_ts  = 1.0,
        scale_pdf = 1.0,
        units    = r'mmol N m$^{-2}$ s$^{-1}$',
        math     = r"$w'NO_3^-'$",
        outfile  = './figs/wno3_flux_20m_offshore.png',
        ylim_ts  = None,
        ylim_env = None,
        ylim_pdf = None,
        ylim_box = None,
    ),
    'NO3_30m': dict(
        ts_npz   = lambda n: f'../postprocessing/wno3_flux_30m_offshore_{n}.npz',
        ts_key   = 'time_series',
        env_npz  = lambda n: f'../postprocessing/wno3_env_30m_offshore_{n}.npz',
        scale_ts  = 1.0,
        scale_pdf = 1.0,
        units    = r'mmol N m$^{-2}$ s$^{-1}$',
        math     = r"$w'NO_3^-'$",
        outfile  = './figs/wno3_flux_30m_offshore.png',
        ylim_ts  = None,
        ylim_env = None,
        ylim_pdf = None,
        ylim_box = None,
    ),
    'AktdNO3dz_20m': dict(
        ts_npz   = lambda n: f'../postprocessing/akt_dno3dz_flux_20m_offshore_{n}.npz',
        ts_key   = 'time_series',
        env_npz  = lambda n: f'../postprocessing/akt_dno3dz_env_20m_offshore_{n}.npz',
        # same sign-flip reasoning as AktdNO3dz_30m below
        scale_ts  = -1.0,
        scale_pdf = -1.0,
        units    = r'mmol N m$^{-2}$ s$^{-1}$',
        math     = r"$-A_{kt}\,\frac{\partial NO_3^-}{\partial z}$",
        outfile  = './figs/akt_dno3dz_flux_20m_offshore.png',
        ylim_ts  = None,
        ylim_env = None,
        ylim_pdf = None,
        ylim_box = None,
    ),
    'AktdNO3dz_30m': dict(
        ts_npz   = lambda n: f'../postprocessing/akt_dno3dz_flux_30m_offshore_{n}.npz',
        ts_key   = 'time_series',
        env_npz  = lambda n: f'../postprocessing/akt_dno3dz_env_30m_offshore_{n}.npz',
        # calc_akt_dno3dz_30m_offshore.py stores the literal Akt*dNO3/dz
        # product (no sign flip, see that script's docstring) -- flipped to
        # the standard downgradient-diffusion convention (F = -Akt*dNO3/dz,
        # z positive up: positive = upward nutrient supply, matching
        # w'NO3''s sign convention) here instead of rerunning the calc script
        scale_ts  = -1.0,
        scale_pdf = -1.0,
        units    = r'mmol N m$^{-2}$ s$^{-1}$',
        math     = r"$-A_{kt}\,\frac{\partial NO_3^-}{\partial z}$",
        outfile  = './figs/akt_dno3dz_flux_30m_offshore.png',
        ylim_ts  = None,
        ylim_env = None,
        ylim_pdf = None,
        ylim_box = None,
    ),
}

axfont = 16

for tracer, cfg in TRACERS.items():
    sc  = cfg['scale_ts']
    scp = cfg['scale_pdf']
    # abs() -- a scale of -1.0 (sign flip only, e.g. AktdNO3dz_30m) isn't a
    # magnitude rescale and shouldn't trigger the x10^-n offset label; log10
    # of a negative number is also undefined
    ts_pow  = int(np.log10(abs(sc)))  if abs(sc)  != 1.0 else None
    pdf_pow = int(np.log10(abs(scp))) if abs(scp) != 1.0 else None

    ts_ylabel  = (f"{cfg['math']} mean\n"
                  + (f"($\\times 10^{{-{ts_pow}}}$ " if ts_pow else '(')
                  + f"{cfg['units']})")
    env_ylabel = (f"{cfg['math']} envelope\n"
                  + (f"($\\times 10^{{-{ts_pow}}}$ " if ts_pow else '(')
                  + f"{cfg['units']})")
    pdf_xlabel = (f"{cfg['math']} "
                  + (f"($\\times 10^{{-{pdf_pow}}}$ " if pdf_pow else '(')
                  + f"{cfg['units']})")

    fig = plt.figure(figsize=[14, 14])
    gs  = gridspec.GridSpec(3, 2, width_ratios=[4, 1], hspace=0.35, wspace=0.08,
                            figure=fig)
    ax_ts  = fig.add_subplot(gs[0, :])
    ax_env = fig.add_subplot(gs[1, :])
    ax_pdf = fig.add_subplot(gs[2, 0])
    ax_box = fig.add_subplot(gs[2, 1])

    ts_data = {}   # collect for box plot

    for name in SCENARIOS:
        d_ts  = np.load(cfg['ts_npz'](name),  allow_pickle=True)
        d_env = np.load(cfg['env_npz'](name), allow_pickle=True)
        lbl   = labels[name]
        clr   = colors[name]
        ls    = lstyles[name]
        lw_ts  = ss.lw(name, base_lw=1.5)
        lw_env = ss.lw(name, base_lw=1.2)
        lw_pdf = ss.lw(name, base_lw=1.5)

        # time series — skip first 12 steps (spin-up / notides_wec artifact)
        ts    = d_ts[cfg['ts_key']][12:] * sc
        times = pf.numdate(d_ts['ocean_times'][12:], 'seconds since 1995-01-01')
        ax_ts.plot(times, ts, color=clr, linestyle=ls, linewidth=lw_ts, label=lbl)
        ts_data[name] = ts

        # envelope — a negative scale_ts flips which raw bound is the min/max,
        # so swap them after scaling or fill_between (and the max/min lines)
        # would invert
        t_env    = pf.numdate(d_env['ocean_times'][12:], 'seconds since 1995-01-01')
        ts_min_raw = d_env['ts_min'][12:] * sc
        ts_max_raw = d_env['ts_max'][12:] * sc
        ts_min, ts_max = (ts_max_raw, ts_min_raw) if sc < 0 else (ts_min_raw, ts_max_raw)
        ax_env.fill_between(t_env, ts_min, ts_max, color=clr, alpha=0.1, linewidth=0)
        ax_env.plot(t_env, ts_max, color=clr, linestyle=ls, linewidth=lw_env, label=lbl)
        ax_env.plot(t_env, ts_min, color=clr, linestyle=ls, linewidth=lw_env)

        # PDF — peak normalization
        pdf_norm = d_env['pdf'] / np.max(d_env['pdf'])
        ax_pdf.plot(d_env['bin_centers'] * scp, pdf_norm,
                    color=clr, linestyle=ls, linewidth=lw_pdf, label=lbl)

    # box and whisker
    ABBREV_LABELS = {
        'notidesnowec': 'NT\nNW',
        'ampwec':       'NT\nAW',
        'tidesnowec':   'T\nNW',
        'tidesampwec':  'T\nAW',
    }
    short_labels = [ABBREV_LABELS[n] for n in SCENARIOS]

    print(f'\n[{tracer}] box-plot stats (per-timestep domain-mean, scaled by {sc:g}):')
    for name in SCENARIOS:
        vals = ts_data[name]
        print(f'  {labels[name]:<22s} mean={np.mean(vals):.4e}  median={np.median(vals):.4e}')

    for i, name in enumerate(SCENARIOS):
        clr = colors[name]
        bp = ax_box.boxplot(ts_data[name], positions=[i], widths=0.55,
                            patch_artist=True, showmeans=True,
                            meanprops={'marker': 'o', 'markerfacecolor': clr,
                                       'markeredgecolor': 'k', 'markersize': 6},
                            medianprops={'color': 'k', 'linewidth': 1.5},
                            boxprops={'facecolor': clr, 'alpha': 0.5},
                            whiskerprops={'color': clr, 'linewidth': 1.2},
                            capprops={'color': clr, 'linewidth': 1.2},
                            flierprops={'marker': '.', 'markerfacecolor': clr,
                                        'markersize': 2, 'alpha': 0.3})
    ax_box.set_xticks(range(len(SCENARIOS)))
    ax_box.set_xticklabels(short_labels, fontsize=axfont)
    ax_box.axhline(0, color='k', linewidth=0.5)
    # no longer adjacent to ax_ts, so give the box plot its own visible
    # y-axis (on the right, since it now sits to the right of ax_pdf)
    ax_box.yaxis.tick_right()
    ax_box.tick_params(axis='y', which='major', labelsize=axfont)
    ax_box.tick_params(axis='x', which='major', labelsize=axfont)
    # fixed ticks match the NO3 tracers' native (unscaled, |sc|=1.0) value
    # range -- AktdNO3dz's sc=-1.0 is a sign flip only, same native scale as
    # w'NO3', so it gets this too
    if abs(sc) == 1.0:
        ax_box.set_yticks([-0.0003, -0.0002, -0.0001, 0, 0.0001, 0.0002, 0.0003])
        # show the shared x10^-4 offset at the top of the y-axis (mantissa
        # ticks -3..3) instead of repeating "0.0001"-style decimals on
        # every tick -- only makes sense at this native NO3 scale
        for ax in (ax_ts, ax_box):
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.get_offset_text().set_fontsize(axfont)
    # explicit ylim wins over the fixed-tick range above if both are set
    if cfg['ylim_box'] is not None:
        ax_box.set_ylim(cfg['ylim_box'])

    # time series formatting
    ax_ts.axhline(0, color='k', linewidth=0.5)
    ax_ts.set_ylabel(ts_ylabel, fontsize=axfont)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax_ts.tick_params(axis='both', which='major', labelsize=axfont)
    if cfg['ylim_ts'] is not None:
        ax_ts.set_ylim(cfg['ylim_ts'])

    # envelope formatting
    ax_env.axhline(0, color='k', linewidth=0.5)
    ax_env.set_ylabel(env_ylabel, fontsize=axfont)
    ax_env.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax_env.tick_params(axis='both', which='major', labelsize=axfont)
    if cfg['ylim_env'] is not None:
        ax_env.set_ylim(cfg['ylim_env'])

    # PDF formatting
    ax_pdf.set_yscale('log')
    ax_pdf.set_xlabel(pdf_xlabel, fontsize=axfont)
    ax_pdf.set_ylabel('Normalized PDF', fontsize=axfont)
    ax_pdf.tick_params(axis='both', which='major', labelsize=axfont)
    ax_pdf.legend(loc='upper right', fontsize=axfont)
    if cfg['ylim_pdf'] is not None:
        ax_pdf.set_ylim(cfg['ylim_pdf'])

    # panel labels -- inside the axes (top-left, in data coords) rather
    # than set_title(loc='left'), which sat above the frame and got
    # overlapped by ax_ts/ax_box's x10^-4 offset text (that text renders
    # just above the axes, same spot set_title used). A white backing box
    # keeps (C) legible against ax_pdf's upper-left legend too.
    for ax, letter in zip((ax_ts, ax_env, ax_pdf, ax_box), 'ABCD'):
        ax.text(0.02, 0.98, f'({letter})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))

    plt.savefig(cfg['outfile'], bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f"saved {cfg['outfile']}")
