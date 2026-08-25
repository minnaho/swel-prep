"""
Daily bar(w'NO3') version of plot_wno3_flux_100m.py's NO3_20m panel -- same
4-panel layout (time series, envelope, PDF, box plot) as the hourly script,
but reading the daily-averaged npz files written by
calc_wno3_flux_20m_100m_daily.py (day-local mean decomposition, see that
script's docstring) instead of the hourly whole-record-mean ones.

Scoped to NO3_20m only, same reasoning as plot_wno3_flux_20m_offshore.py:
no daily counterpart exists yet for NO3 at 10m or ptrace/rtrace, only
calc_wno3_flux_20m_100m_daily.py was built. Single-entry TRACERS dict so
adding those siblings later (if their daily calc scripts get built) is a
one-entry addition.

Layout: row 1 = time series (full width), row 2 = envelope (full width),
row 3 = PDF (wide) + box plot (narrow) -- matches plot_wno3_flux_100m.py's
current layout exactly (box next to PDF, abbreviated box x-labels at
axfont, fixed box y-ticks + shared x10^-4 offset text on ax_ts/ax_box,
panel labels (A)-(D) inside the axes with a white backing box).

Spin-up skip: the hourly script drops the first 12 (hourly) steps for a
known spin-up/notides_wec artifact. The daily npz only has ~10-24 values,
so an analogous "skip 12" would gut most of the record -- skips only the
first DAY (index 0) instead, per explicit confirmation (calc_wno3_flux_20m_100m_daily.py
already NaNs out a degenerate <2-sample boundary day on its own; this
additionally guards a partial-but-valid first day still touched by the
same spin-up transient).

Output: ./figs/wno3_flux_20m_100m_daily.png
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

TRACERS = {
    'NO3_20m_daily': dict(
        ts_npz   = lambda n: f'../postprocessing/wno3_flux_20m_100m_daily_{n}.npz',
        ts_key   = 'time_series',
        env_npz  = lambda n: f'../postprocessing/wno3_env_20m_100m_daily_{n}.npz',
        scale_ts  = 1.0,
        scale_pdf = 1.0,
        units    = r'mmol N m$^{-2}$ s$^{-1}$',
        math     = r"$\overline{w'NO_3'}$",
        outfile  = './figs/wno3_flux_20m_100m_daily.png',
    ),
}

axfont = 16

for tracer, cfg in TRACERS.items():
    sc  = cfg['scale_ts']
    scp = cfg['scale_pdf']
    ts_pow  = int(np.log10(sc))  if sc  != 1.0 else None
    pdf_pow = int(np.log10(scp)) if scp != 1.0 else None

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

        # time series — skip first day (spin-up / notides_wec artifact,
        # daily analog of the hourly script's [12:] skip)
        ts    = d_ts[cfg['ts_key']][1:] * sc
        times = pf.numdate(d_ts['ocean_times'][1:], 'seconds since 1995-01-01')
        ax_ts.plot(times, ts, color=clr, linestyle=ls, linewidth=lw_ts, label=lbl)
        ts_data[name] = ts[~np.isnan(ts)]

        # envelope
        t_env   = pf.numdate(d_env['ocean_times'][1:], 'seconds since 1995-01-01')
        ts_min  = d_env['ts_min'][1:] * sc
        ts_max  = d_env['ts_max'][1:] * sc
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
    # fixed ticks match the NO3 tracer's native (unscaled, sc=1.0) value
    # range, same as plot_wno3_flux_100m.py
    if sc == 1.0:
        #ax_box.set_yticks([-0.0003, -0.0002, -0.0001, 0, 0.0001, 0.0002, 0.0003])
        # show the shared x10^-4 offset at the top of the y-axis (mantissa
        # ticks -3..3) instead of repeating "0.0001"-style decimals on
        # every tick
        for ax in (ax_ts, ax_box):
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.get_offset_text().set_fontsize(axfont)

    # time series formatting
    ax_ts.axhline(0, color='k', linewidth=0.5)
    ax_ts.set_ylabel(ts_ylabel, fontsize=axfont)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax_ts.tick_params(axis='both', which='major', labelsize=axfont)

    # envelope formatting
    ax_env.axhline(0, color='k', linewidth=0.5)
    ax_env.set_ylabel(env_ylabel, fontsize=axfont)
    ax_env.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax_env.tick_params(axis='both', which='major', labelsize=axfont)

    # PDF formatting
    ax_pdf.set_yscale('log')
    ax_pdf.set_xlabel(pdf_xlabel, fontsize=axfont)
    ax_pdf.set_ylabel('Normalized PDF', fontsize=axfont)
    ax_pdf.tick_params(axis='both', which='major', labelsize=axfont)
    # bbox_to_anchor nudges the legend down slightly so it doesn't overlap
    # the (C) panel label reserved at the very top-left corner
    ax_pdf.legend(loc='upper left', bbox_to_anchor=(0.0, 0.93), fontsize=axfont)

    # panel labels -- inside the axes (top-left, in data coords), same
    # placement/reasoning as plot_wno3_flux_100m.py
    for ax, letter in zip((ax_ts, ax_env, ax_pdf, ax_box), 'ABCD'):
        ax.text(0.02, 0.98, f'({letter})', transform=ax.transAxes,
                fontsize=14, fontweight='bold', ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))

    plt.savefig(cfg['outfile'], bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f"saved {cfg['outfile']}")
